# ============================================================
# FULL EVAL SCRIPT: FP32 vs Q16 (MobileFaceNet)
# ============================================================

import os
import io
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from src.models import mobilefacenet as MobileFaceNet
from src.quantizer import quantize_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ============================================================
# 1. LOAD BIN
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
# 2. EXTRACT EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embeddings(model, images, batch_size=128, use_flip=True):
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
# 5. LOAD MODEL (MobileFaceNet)
# ============================================================

def load_backbone(weight_path, bit="fp32"):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weight path not found: {weight_path}")

    # Detect embedding size from state dict to be fully robust
    state = torch.load(weight_path, map_location="cpu")
    
    # remove module. prefix if exists
    new_state = {}
    for k, v in state.items():
        new_state[k.replace("module.", "")] = v
        
    embedding_size = 512
    for k in new_state:
        if "output_layer.linear.weight" in k:
            embedding_size = new_state[k].shape[0]
            break

    if bit == "fp32":
        model = MobileFaceNet(embedding_size=embedding_size)
    else:
        model = quantize_model(
            MobileFaceNet(embedding_size=embedding_size),
            weight_bit=bit,
            act_bit=bit
        )

    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()

    return model


# ============================================================
# 6. EVAL
# ============================================================

def eval_model(weight_path, name, bit="fp32"):
    print(f"\n==============================")
    print(f"Evaluating: {name}")
    print(f"==============================")

    try:
        model = load_backbone(weight_path, bit)
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("Please place the pre-trained weights in the weights/ directory as described in the README.")
        return

    datasets = {
        "LFW": os.path.join(PROJECT_ROOT, "data/lfw.bin"),
        "CFP_FP": os.path.join(PROJECT_ROOT, "data/cfp_fp.bin"),
        "AGEDB_30": os.path.join(PROJECT_ROOT, "data/agedb_30.bin"),
        "CALFW": os.path.join(PROJECT_ROOT, "data/calfw.bin"),
        "CPLFW": os.path.join(PROJECT_ROOT, "data/cplfw.bin"),
    }

    for ds, bin_path in datasets.items():
        if not os.path.exists(bin_path):
            print(f"[INFO] Dataset binary for {ds} not found at {bin_path}. Skipping.")
            continue
            
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
# 7. RUN
# ============================================================

if __name__ == "__main__":
    FP32_PATH = os.path.join(PROJECT_ROOT, "weights/FP32/mobilefacenet_fp32.pth")
    Q6_PATH = os.path.join(PROJECT_ROOT, "weights/w-3p/mobilefacenet_q6-w3p.pth")

    print("Running evaluation comparison for MobileFaceNet (FP32 vs Q6)...")
    eval_model(FP32_PATH, "FP32 (Teacher)", bit="fp32")
    eval_model(Q6_PATH, "Q6 Student (Proposed)", bit=6)