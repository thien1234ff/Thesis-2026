# ============================================================
# FULL EVAL SCRIPT: FP32 vs Q16 (iresnet50)
# ============================================================

import os
import io
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve

# ✅ đổi sang iresnet50
from backbones.iresnet import iresnet50, quantize_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 1. LOAD BIN (InsightFace format)
# ============================================================

def decode_bin_image(bin_item):
    img = Image.open(io.BytesIO(bin_item)).convert("RGB")
    img = img.resize((112, 112), Image.BILINEAR)
    return img


def load_bin(bin_path):
    with open(bin_path, "rb") as f:
        bins, issame = pickle.load(f, encoding="bytes")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    images = []
    for b in bins:
        img = decode_bin_image(b)
        images.append(transform(img))

    images = torch.stack(images)
    issame = np.asarray(issame)

    return images, issame


# ============================================================
# 2. EXTRACT EMBEDDINGS (WITH FLIP TEST)
# ============================================================

@torch.no_grad()
def extract_embeddings(model, images, batch_size=64, use_flip=True):  # ⚠️ giảm batch
    model.eval()
    embeddings = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)

        feat = model(batch)

        if use_flip:
            flipped = torch.flip(batch, dims=[3])
            feat_flip = model(flipped)
            feat = feat + feat_flip

        feat = F.normalize(feat, dim=1)
        embeddings.append(feat.cpu())

    return torch.cat(embeddings, dim=0).numpy()


# ============================================================
# 3. 10-FOLD VERIFICATION
# ============================================================

def evaluate_10fold(embeddings, issame, folds=10):
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]

    n = len(issame)
    indices = np.arange(n)
    fold_size = n // folds

    accs = []
    thresholds = np.linspace(-1.0, 1.0, 1000)

    for i in range(folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < folds - 1 else n

        test_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        sims_train = np.sum(emb1[train_idx] * emb2[train_idx], axis=1)
        labels_train = issame[train_idx]

        best_acc = 0
        best_th = 0

        for th in thresholds:
            acc = np.mean((sims_train > th) == labels_train)
            if acc > best_acc:
                best_acc = acc
                best_th = th

        sims_test = np.sum(emb1[test_idx] * emb2[test_idx], axis=1)
        labels_test = issame[test_idx]

        acc = np.mean((sims_test > best_th) == labels_test)
        accs.append(acc)

    return np.mean(accs), np.std(accs)


# ============================================================
# 4. TAR@FAR
# ============================================================

def tar_at_far(embeddings, issame, target_far=1e-4):
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    labels = issame.astype(int)

    scores = np.sum(emb1 * emb2, axis=1)

    fpr, tpr, _ = roc_curve(labels, scores)

    if target_far > max(fpr):
        return 0.0

    return np.interp(target_far, fpr, tpr)


# ============================================================
# 5. LOAD MODEL (iresnet50)
# ============================================================

def load_backbone(weight_path, bit="fp32"):

    if bit == "fp32":
        model = iresnet50(num_features=512)
    else:
        model = quantize_model(
            iresnet50(num_features=512),
            weight_bit=bit,
            act_bit=bit
        )

    state = torch.load(weight_path, map_location="cpu")

    # remove backbone. prefix if exists
    if any(k.startswith("backbone.") for k in state.keys()):
        state = {k.replace("backbone.", ""): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    return model


# ============================================================
# 6. EVALUATION WRAPPER
# ============================================================

def eval_model(weight_path, name, bit="fp32"):

    print(f"\n==============================")
    print(f"Evaluating: {name}")
    print(f"==============================")

    model = load_backbone(weight_path, bit)

    datasets = {
        "LFW": "/kaggle/input/datasets/debarghamitraroy/msm1-retinaface-t1/eval/lfw.bin",
        "CFP_FP": "/kaggle/input/datasets/debarghamitraroy/msm1-retinaface-t1/eval/cfp_fp.bin",
        "AGEDB_30": "/kaggle/input/datasets/debarghamitraroy/msm1-retinaface-t1/eval/agedb_30.bin",
        "CALFW": "/kaggle/input/datasets/debarghamitraroy/msm1-retinaface-t1/eval/calfw.bin",
        "CPLFW": "/kaggle/input/datasets/debarghamitraroy/msm1-retinaface-t1/eval/cplfw.bin",
    }

    for ds, bin_path in datasets.items():
        images, issame = load_bin(bin_path)

        emb = extract_embeddings(model, images)

        mean, std = evaluate_10fold(emb, issame)
        tar = tar_at_far(emb, issame, target_far=1e-4)

        print(
            f"{ds:10s}: "
            f"Acc={mean*100:.3f}% ± {std*100:.3f}% | "
            f"TAR@1e-4={tar*100:.3f}%"
        )


# ============================================================
# 7. RUN COMPARISON
# ============================================================

FP32_PATH = "/kaggle/input/datasets/hhongeeee/ms1mv2-pretrain/iresnet50.pth"
Q16_PATH = "/kaggle/input/datasets/hhongeeee/beta300/iresnet50_q6_epoch29.pth"

eval_model(FP32_PATH, "FP32", bit="fp32")
eval_model(Q16_PATH, "Q6", bit=6)