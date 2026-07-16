import os
import sys
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.models import mobilefacenet as MobileFaceNet
from src.quantizer import quantize_model

# =========================================
# CONFIG
# =========================================
FP32_PATH = os.path.join(PROJECT_ROOT, "weights/FP32/mobilefacenet_fp32.pth")
SAVE_PATH = os.path.join(PROJECT_ROOT, "weights/ptq/mobilefacenet_q6_ptq.pth")

WEIGHT_BIT = 6
ACT_BIT = 6
CALIB_BATCHES = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

class SyntheticFaceDataset(Dataset):
    def __init__(self, num_classes=5, num_images_per_class=10, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []
        for c in range(num_classes):
            for _ in range(num_images_per_class):
                img = Image.fromarray((torch.rand(3, 112, 112).permute(1, 2, 0).numpy() * 255).astype('uint8'))
                self.data.append(img)
                self.labels.append(c)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# =========================================
# 1. LOAD FP32 MODEL
# =========================================
# Detect embedding size from state dict to be fully robust
embedding_size = 128
if os.path.exists(FP32_PATH):
    print("Loading backbone weights from:", FP32_PATH)
    state = torch.load(FP32_PATH, map_location="cpu")
    # remove module. prefix if exists
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    for k in new_state:
        if "output_layer.linear.weight" in k:
            embedding_size = new_state[k].shape[0]
            break
    model_fp32 = MobileFaceNet(embedding_size=embedding_size)
    model_fp32.load_state_dict(new_state, strict=False)
else:
    print("[WARNING] Teacher FP32 weights not found at:", FP32_PATH)
    print("Initializing FP32 model randomly for structure checkout.")
    model_fp32 = MobileFaceNet(embedding_size=embedding_size)

model_fp32.to(device)
model_fp32.eval()

print("Loaded FP32 MobileFaceNet model")

# =========================================
# 2. QUANTIZE MODEL
# =========================================
model_q = quantize_model(
    model_fp32,
    weight_bit=WEIGHT_BIT,
    act_bit=ACT_BIT
).to(device)

model_q.eval()

print("Quantized model created")

# =========================================
# 3. CALIBRATION DATA
# =========================================
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

DATA_ROOT = os.path.join(PROJECT_ROOT, "data/final_dataset")

if not os.path.exists(DATA_ROOT) or len(os.listdir(DATA_ROOT)) == 0:
    print("[INFO] Real training image dataset folder not found. Running PTQ calibration using synthetic face images.")
    calib_dataset = SyntheticFaceDataset(num_classes=5, num_images_per_class=20, transform=transform)
else:
    print("Loading dataset from:", DATA_ROOT)
    calib_dataset = datasets.ImageFolder(
        DATA_ROOT,
        transform=transform
    )

calib_loader = DataLoader(
    calib_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

# =========================================
# 4. CALIBRATION
# =========================================
print("Running calibration...")

with torch.no_grad():
    for i, (img, _) in enumerate(calib_loader):
        img = img.to(device)
        _ = model_q(img)

        if i >= CALIB_BATCHES:
            break

print("Calibration done")

# =========================================
# 5. FREEZE OBSERVER
# =========================================
def freeze_observer(model):
    for module in model.modules():
        if hasattr(module, 'disable_observer'):
            module.disable_observer()

freeze_observer(model_q)

print("Observer frozen")

# =========================================
# 6. SAVE
# =========================================
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
torch.save(model_q.state_dict(), SAVE_PATH)

print(f"Saved quantized model to {SAVE_PATH}")