import sys
sys.path.append("/kaggle/input/quantfacemodel/QuantFace")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import os
import math
import numpy as np

from backbones.iresnet import iresnet18, quantize_model

np.bool = bool

# =========================================
# CONFIG
# =========================================
DATA_ROOT = "/kaggle/input/datasets/hhongeeee/final-dataset/final_dataset"
FP32_PATH = "/kaggle/input/datasets/hhongeeee/ms1mv2-pretrain/181952backbone.pth"

BATCH_SIZE = 128
EPOCHS = 30
BASE_LR = 1e-3
BETA = 300
WEIGHT_BIT = 6
ACT_BIT = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================================
# DATASET
# =========================================
transform = transforms.Compose([
    transforms.Resize((112,112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

trainset = datasets.ImageFolder(DATA_ROOT, transform=transform)

train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

print("Identities:", len(trainset.classes))
print("Images:", len(trainset))

# =========================================
# LOAD TEACHER
# =========================================
teacher = iresnet18(num_features=512)
teacher.load_state_dict(torch.load(FP32_PATH))
teacher = teacher.to(device)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

# =========================================
# LOAD STUDENT (QUANTIZED)
# =========================================
student_fp32 = iresnet18(num_features=512)
student_fp32.load_state_dict(torch.load(FP32_PATH))

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
# TRACKING
# =========================================
step_losses = []
step_accs = []
step_tars = []

epoch_losses = []
epoch_accs = []
epoch_tars = []

# =========================================
# TRAIN LOOP
# =========================================
global_step = 0

for epoch in range(EPOCHS):

    student.train()
    teacher.eval()

    if epoch >= 5:
        freeze_observer(student)
        phase_name = "Main/Cool-down"
    else:
        phase_name = "Warm-up"

    total_loss = 0
    total_acc = 0
    total_tar = 0

    for img, _ in train_loader:
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

        # Logging
        step_losses.append(loss.item())
        step_accs.append(acc)
        step_tars.append(tar)

        total_loss += loss.item()
        total_acc += acc
        total_tar += tar

        global_step += 1

        if global_step % 100 == 0:
            print(f"Pha: {phase_name} | Epoch [{epoch+1}/{EPOCHS}] | Step {global_step} | "
                  f"MSE: {mse_loss_raw.item():.6f} | Loss: {loss.item():.4f} | "
                  f"Acc: {acc:.4f} | TAR: {tar:.4f}")

    scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_acc / len(train_loader)
    avg_tar = total_tar / len(train_loader)

    epoch_losses.append(avg_loss)
    epoch_accs.append(avg_acc)
    epoch_tars.append(avg_tar)

    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | "
          f"Acc: {avg_acc:.4f} | TAR: {avg_tar:.4f} | "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")

    torch.save(student.state_dict(), f"iresnet18_q{WEIGHT_BIT}_epoch{epoch}.pth")

print("Training finished.")

# =========================================
# SAVE METRICS
# =========================================
np.save("metrics.npy", {
    "loss": epoch_losses,
    "acc": epoch_accs,
    "tar": epoch_tars
})

# =========================================
# PLOT
# =========================================
plt.figure(figsize=(15,5))

# Loss
plt.subplot(1,3,1)
plt.plot(epoch_losses)
plt.title("Loss per Epoch")

# Accuracy
plt.subplot(1,3,2)
plt.plot(epoch_accs)
plt.title("Accuracy per Epoch")

# TAR
plt.subplot(1,3,3)
plt.plot(epoch_tars)
plt.title("TAR per Epoch")

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()