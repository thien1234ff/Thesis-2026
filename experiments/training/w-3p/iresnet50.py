import os
import sys
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)

from src.models import iresnet50
from src.quantizer import quantize_model

np.bool = bool

# =========================================
# CONFIG
# =========================================
FP32_PATH = os.path.join(PROJECT_ROOT, "weights/FP32/iresnet50_fp32.pth")
SAVE_DIR = os.path.join(PROJECT_ROOT, "weights/w-3p")

BATCH_SIZE = 32
EPOCHS = 30
BASE_LR = 1e-3
BETA = 300
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
# DATASET
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
    trainset = datasets.ImageFolder(DATA_ROOT, transform=transform)
    train_loader = DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 if os.name != 'nt' else 0,
        pin_memory=True,
        drop_last=True
    )

print("Identities:", len(trainset.classes) if hasattr(trainset, 'classes') else 5)
print("Images:", len(trainset))

# =========================================
# LOAD TEACHER
# =========================================
teacher = iresnet50(num_features=512)
if os.path.exists(FP32_PATH):
    teacher.load_state_dict(torch.load(FP32_PATH, map_location="cpu"))
else:
    print("[WARNING] Teacher FP32 weights not found at:", FP32_PATH)
teacher = teacher.to(device)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

# =========================================
# LOAD STUDENT (QUANTIZED)
# =========================================
student_fp32 = iresnet50(num_features=512)
if os.path.exists(FP32_PATH):
    student_fp32.load_state_dict(torch.load(FP32_PATH, map_location="cpu"))

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
    lr=BASE_LR,
    momentum=0.9,
    weight_decay=5e-4
)

# =========================================
# LR SCHEDULER (3 PHASE)
# =========================================
def lr_schedule_3_phase(epoch):
    if epoch < 5:
        start_mult = 1e-6 / BASE_LR
        end_mult = 1.0
        return start_mult + (end_mult - start_mult) * (epoch / 4.0)

    elif epoch < 25:
        return 1.0

    else:
        curr = epoch - 25
        total = 5
        cos_out = 0.5 * (1 + math.cos(math.pi * curr / total))
        min_lr_mult = 1e-5 / 1e-3
        return min_lr_mult + (1.0 - min_lr_mult) * cos_out

scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule_3_phase)

criterion = torch.nn.MSELoss()

# =========================================
# METRICS FUNCTION
# =========================================
def compute_metrics(feat_s, feat_t, threshold=0.5):
    cos_sim = (feat_s * feat_t).sum(dim=1)
    preds = (cos_sim > threshold).float()
    labels = torch.ones_like(preds)

    acc = (preds == labels).float().mean().item()
    tar = preds.mean().item()

    return acc, tar

# =========================================
# FREEZE FUNCTIONS
# =========================================
def freeze_observer(model):
    for name, p in model.named_parameters():
        if 'observer' in name.lower() or 'quant' in name.lower() or 'scale' in name.lower():
            p.requires_grad = False

    for module in model.modules():
        if hasattr(module, 'disable_observer'):
            module.disable_observer()

# =========================================
# TRAIN LOOP
# =========================================
for epoch in range(EPOCHS):
    student.train()
    teacher.eval()

    # Phase freezing check
    if epoch >= 5:
        freeze_observer(student)
        phase_name = "Main/Cool-down"
    else:
        phase_name = "Warm-up"

    total_loss = 0
    total_acc = 0
    total_tar = 0

    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)

        # Teacher
        with torch.no_grad():
            feat_teacher = F.normalize(teacher(img), dim=1)

        # Student
        feat_student = F.normalize(student(img), dim=1)

        # Loss
        mse_loss_raw = criterion(feat_student, feat_teacher)
        loss = BETA * mse_loss_raw

        # Metrics
        acc, tar = compute_metrics(feat_student, feat_teacher)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc
        total_tar += tar

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    avg_tar = total_tar / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Phase: {phase_name:15s} | "
        f"Loss: {avg_loss:.4f} | "
        f"Acc: {avg_acc*100:.2f}% | "
        f"TAR: {avg_tar*100:.2f}% | "
        f"LR: {optimizer.param_groups[0]['lr']:.6f}"
    )

    # Save checkpoint
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(SAVE_DIR, f"iresnet50_q{WEIGHT_BIT}_epoch{epoch}.pth")
    torch.save(student.state_dict(), checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")