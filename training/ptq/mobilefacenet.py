import sys
sys.path.append("/kaggle/input/project/QuantFace")

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ✅ backbone của bạn
from backbones.mobilefacenet import MobileFaceNet, quantize_model

# =========================================
# CONFIG
# =========================================
FP32_PATH = "/kaggle/input/datasets/nhihong159/pretrain-model/Mobilefacenet.pth"
SAVE_PATH = "mobilefacenet_q6_ptq.pth"

WEIGHT_BIT = 6
ACT_BIT = 6
CALIB_BATCHES = 50   # ⚠️ chỉ cần ít batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================
# 1. LOAD FP32 MODEL
# =========================================
model_fp32 = MobileFaceNet(embedding_size=128)
state = torch.load(FP32_PATH, map_location="cpu")
model_fp32.load_state_dict(state)
model_fp32.to(device)
model_fp32.eval()

print("Loaded FP32 model")

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
# 3. CALIBRATION DATA (rất quan trọng)
# =========================================
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ⚠️ dùng dataset train hoặc 1 subset nhỏ
calib_dataset = datasets.ImageFolder(
    "/kaggle/input/datasets/nhihong159/final-data/final_dataset",
    transform=transform
)

calib_loader = DataLoader(
    calib_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

# =========================================
# 4. RUN CALIBRATION
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
# 5. FREEZE OBSERVER (QUAN TRỌNG)
# =========================================
def freeze_observer(model):
    for module in model.modules():
        if hasattr(module, 'disable_observer'):
            module.disable_observer()

freeze_observer(model_q)

print("Observer frozen")

# =========================================
# 6. SAVE MODEL
# =========================================
torch.save(model_q.state_dict(), SAVE_PATH)

print(f"Saved quantized model to {SAVE_PATH}")