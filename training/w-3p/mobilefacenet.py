import sys
sys.path.append("/kaggle/input/quantfacemodel/QuantFace")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import numpy as np

from Training.mobilefacenet import MobileFaceNet, quantize_model

np.bool = bool

# =========================================
# CONFIG
# =========================================
DATA_ROOT = "/kaggle/input/datasets/hhongeeee/dulieucuatoi"
FP32_PATH = "/kaggle/input/datasets/hhongeeee/mobilefacenet_pretrain.pth"

BATCH_SIZE = 128
EPOCHS = 30
BASE_LR = 2e-3
BETA = 300
WEIGHT_BIT = 6
ACT_BIT = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# =========================================
# LOAD CHECKPOINT
# =========================================
state = torch.load(FP32_PATH, map_location="cpu")

# remove "module."
new_state = {}
for k, v in state.items():
    new_state[k.replace("module.", "")] = v

# detect embedding size automatically
embedding_size = None
for k in new_state:
    if "output_layer.linear.weight" in k:
        embedding_size = new_state[k].shape[0]
        break

if embedding_size is None:
    raise ValueError("Cannot detect embedding_size from checkpoint")

print("Detected embedding size:", embedding_size)

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
# TEACHER (FP32)
# =========================================
teacher = MobileFaceNet(embedding_size=embedding_size)
teacher.load_state_dict(new_state, strict=True)
teacher = teacher.to(device)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

# =========================================
# STUDENT (QAT) - GIỮ LOGIC CŨ
# =========================================
student_fp32 = MobileFaceNet(embedding_size=embedding_size)

student_fp32.load_state_dict(new_state)  # giữ logic init từ pretrained

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
# LR SCHEDULER (GIỮ NGUYÊN)
# =========================================
def lr_schedule(epoch):
    if epoch < 5:
        return (epoch + 1) / 5
    elif epoch < 25:
        return 1.0
    else:
        return 0.1

scheduler = LambdaLR(optimizer, lr_lambda=lr_schedule)

criterion = torch.nn.MSELoss()

# =========================================
# TRAIN LOOP (GIỮ NGUYÊN LOGIC)
# =========================================
for epoch in range(EPOCHS):

    student.train()
    teacher.eval()

    total_loss = 0

    for img, _ in train_loader:
        img = img.to(device)

        with torch.no_grad():
            feat_teacher = F.normalize(teacher(img), dim=1)

        feat_student = F.normalize(student(img), dim=1)

        loss = BETA * criterion(feat_student, feat_teacher)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1} | Loss {total_loss/len(train_loader):.4f}")

    torch.save(student.state_dict(),
               f"mobilefacenet_q{WEIGHT_BIT}_epoch{epoch}.pth")

print("Training finished")