import sys
sys.path.append("/kaggle/input/quantfacemodel/QuantFace")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# =========================================
# IMPORT iResNet18
# =========================================
from backbones.iresnet import iresnet18, quantize_model

np.bool = bool

# =========================================
# CONFIG
# =========================================
DATA_ROOT = "/kaggle/input/datasets/hhongeeee/final-dataset/final_dataset"
FP32_PATH = "/kaggle/input/datasets/hhongeeee/pretrain-iresnet18/iresnet18.pth"

BATCH_SIZE = 128
EPOCHS = 30
LR = 1e-3
WEIGHT_BIT = 6
ACT_BIT = 6

# iResNet18 embedding size
EMBEDDING_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================================
# DATASET
# =========================================
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

full_dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)

subset_indices = np.random.choice(len(full_dataset), 10000, replace=False)
trainset = Subset(full_dataset, subset_indices)

train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)

print("Subset size:", len(trainset))

# =========================================
# LOAD TEACHER (FP32 iResNet18)
# =========================================
teacher = iresnet18(num_features=EMBEDDING_SIZE)
teacher.load_state_dict(torch.load(FP32_PATH, map_location="cpu"))
teacher = teacher.to(device)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

# =========================================
# LOAD STUDENT (Q6)
# =========================================
student_fp32 = iresnet18(num_features=EMBEDDING_SIZE)
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

# =========================================
# TRACKING
# =========================================
epoch_losses = []
epoch_accs = []
epoch_tars = []
relative_gaps = []

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

    epoch_losses.append(avg_loss)
    epoch_accs.append(avg_acc)
    epoch_tars.append(avg_tar)
    relative_gaps.append(gap)

    print(f"[iResNet18 Q6] Epoch {epoch+1}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | "
          f"TAR: {avg_tar:.4f} | Gap: {gap:.4f}")

    torch.save(student.state_dict(), f"iresnet18_q6_epoch{epoch}.pth")

print("Training finished.")

# =========================================
# SAVE METRICS
# =========================================
np.save("iresnet18_metrics.npy", {
    "loss": epoch_losses,
    "acc": epoch_accs,
    "tar": epoch_tars,
    "gap": relative_gaps
})

# =========================================
# PLOT
# =========================================
plt.figure(figsize=(20,5))

plt.subplot(1,4,1)
plt.plot(epoch_losses)
plt.title("Loss")

plt.subplot(1,4,2)
plt.plot(epoch_accs)
plt.title("Accuracy")

plt.subplot(1,4,3)
plt.plot(epoch_tars)
plt.title("TAR")

plt.subplot(1,4,4)
plt.plot(relative_gaps)
plt.title("Relative Gap")

plt.tight_layout()
plt.savefig("iresnet18_training_metrics.png")
plt.show()