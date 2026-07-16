import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.models import mobilefacenet as MobileFaceNet
from src.quantizer import quantize_model

np.bool = bool

# =========================================
# CONFIG
# =========================================
FP32_PATH = os.path.join(PROJECT_ROOT, "weights/FP32/mobilefacenet_fp32.pth")
SAVE_DIR = os.path.join(PROJECT_ROOT, "weights/w0-3p")

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_BIT = 6
ACT_BIT = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

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
# DATASET (10k SUBSET)
# =========================================
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

DATA_ROOT = os.path.join(PROJECT_ROOT, "data/final_dataset")

if not os.path.exists(DATA_ROOT) or len(os.listdir(DATA_ROOT)) == 0:
    print("[INFO] Real training image dataset folder not found. Running training using synthetic face images.")
    trainset = SyntheticFaceDataset(num_classes=5, num_images_per_class=20, transform=transform)
    train_loader = DataLoader(trainset, batch_size=4, shuffle=True, drop_last=True)
    EPOCHS = 2
else:
    print("Loading dataset from:", DATA_ROOT)
    full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
    # Select subset
    np.random.seed(42) # set seed for deterministic subset choice
    subset_indices = np.random.choice(len(full_dataset), min(10000, len(full_dataset)), replace=False)
    trainset = Subset(full_dataset, subset_indices)
    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True,
        drop_last=True
    )

print("Subset size:", len(trainset))

# =========================================
# LOAD TEACHER & DETECT EMBEDDING SIZE
# =========================================
embedding_size = 128
if os.path.exists(FP32_PATH):
    print("Loading teacher weights from:", FP32_PATH)
    state = torch.load(FP32_PATH, map_location="cpu")
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
    for k in new_state:
        if "output_layer.linear.weight" in k:
            embedding_size = new_state[k].shape[0]
            break
else:
    print("[WARNING] Teacher FP32 weights not found at:", FP32_PATH)
    new_state = None

print("Detected embedding size:", embedding_size)

# =========================================
# TEACHER (FP32)
# =========================================
teacher = MobileFaceNet(embedding_size=embedding_size)
if new_state is not None:
    teacher.load_state_dict(new_state, strict=False)
teacher = teacher.to(device)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

# =========================================
# STUDENT (QUANTIZED)
# =========================================
student_fp32 = MobileFaceNet(embedding_size=embedding_size)
if new_state is not None:
    student_fp32.load_state_dict(new_state, strict=False)

student = quantize_model(
    student_fp32,
    weight_bit=WEIGHT_BIT,
    act_bit=ACT_BIT
).to(device)

# =========================================
# OPTIMIZER
# =========================================
optimizer = torch.optim.SGD(
    student.parameters(),
    lr=LR,
    momentum=0.9,
    weight_decay=5e-4
)

criterion = torch.nn.MSELoss()

# =========================================
# METRICS
# =========================================
def compute_metrics(feat_s, feat_t, threshold=0.5):
    cos_sim = (feat_s * feat_t).sum(dim=1)
    preds = (cos_sim > threshold).float()
    labels = torch.ones_like(preds)

    acc = (preds == labels).float().mean().item()
    tar = preds.mean().item()

    return acc, tar

FP32_ACC = 0.99

# =========================================
# TRAIN LOOP
# =========================================
for epoch in range(EPOCHS):
    student.train()
    teacher.eval()

    total_loss = 0
    total_acc = 0
    total_tar = 0

    for img, _ in train_loader:
        img = img.to(device)

        with torch.no_grad():
            feat_teacher = F.normalize(teacher(img), dim=1)

        feat_student = F.normalize(student(img), dim=1)

        loss = criterion(feat_student, feat_teacher)
        acc, tar = compute_metrics(feat_student, feat_teacher)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        total_tar += tar

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    avg_tar = total_tar / len(train_loader)

    gap = (FP32_ACC - avg_acc) / FP32_ACC

    print(
        f"[MobileFaceNet Q6] Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
        f"TAR: {avg_tar:.4f} | Gap: {gap:.4f}"
    )

    # Save checkpoint
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(SAVE_DIR, f"mobilefacenet_q6_epoch{epoch}.pth")
    torch.save(student.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")