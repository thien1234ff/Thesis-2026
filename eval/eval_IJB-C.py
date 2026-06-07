import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import roc_curve
from skimage import transform as trans
from concurrent.futures import ThreadPoolExecutor

from insightface.app import FaceAnalysis
from backbones.iresnet import iresnet18, quantize_model

# =========================
# CONFIG
# =========================
DATA_DIR = "/kaggle/input/datasets/hhongeeee/testijb-c/IJBC"
IMG_DIR = os.path.join(DATA_DIR, "loose_crop")
META_DIR = os.path.join(DATA_DIR, "meta")

MODEL_PATH = "/kaggle/input/datasets/hhongeeee/beta001/iresnet18_q6_epoch29 (1).pth"

ALIGN_DIR = "/kaggle/working/aligned"

BATCH_SIZE = 512
NUM_WORKERS = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD MODEL
# =========================
print("Loading model...")

model_fp32 = iresnet18(num_features=512)
model = quantize_model(model_fp32, weight_bit=6, act_bit=6)

state = torch.load(MODEL_PATH, map_location="cpu")

if any(k.startswith("backbone.") for k in state.keys()):
    state = {k.replace("backbone.", ""): v for k, v in state.items()}

model.load_state_dict(state, strict=False)
model = model.to(device)
model.eval()

print("Model loaded")

# =========================
# RETINAFACE
# =========================
print("Loading RetinaFace...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU nhưng nhanh hơn

# =========================
# ALIGN TEMPLATE
# =========================
src = np.array([
    [30.2946, 51.6963],
    [65.5318, 51.5014],
    [48.0252, 71.7366],
    [33.5493, 92.3655],
    [62.7299, 92.2041]
], dtype=np.float32)
src[:, 0] += 8.0

# =========================
# DETECT + ALIGN
# =========================
def detect_align_save(name):
    img_path = os.path.join(IMG_DIR, name)
    save_path = os.path.join(ALIGN_DIR, name)

    if os.path.exists(save_path):
        return

    img = cv2.imread(img_path)
    if img is None:
        return

    faces = face_app.get(img)
    if len(faces) == 0:
        return

    face = max(faces, key=lambda x: x.det_score)
    landmark = face.kps.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(landmark, src)
    M = tform.params[0:2, :]

    aligned = cv2.warpAffine(img, M, (112, 112))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, aligned)

# =========================
# LOAD META
# =========================
def load_meta():
    templates, medias, names = [], [], []

    with open(os.path.join(META_DIR, "ijbc_face_tid_mid.txt")) as f:
        for line in f:
            name, tid, mid = line.strip().split()
            names.append(name)
            templates.append(int(tid))
            medias.append(int(mid))

    pairs = []
    with open(os.path.join(META_DIR, "ijbc_template_pair_label.txt")) as f:
        for line in f:
            t1, t2, label = line.strip().split()
            pairs.append((int(t1), int(t2), int(label)))

    return names, np.array(templates), np.array(medias), pairs

# =========================
# STEP 1: ALIGN ALL
# =========================
def run_alignment(names):
    print("Running alignment (multi-thread)...")

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        list(tqdm(executor.map(detect_align_save, names), total=len(names)))

# =========================
# STEP 2: EXTRACT FEATURES (GPU)
# =========================
@torch.no_grad()
def extract_features(names):
    feats = []
    batch_imgs = []

    for name in tqdm(names):
        path = os.path.join(ALIGN_DIR, name)

        img = cv2.imread(path)
        if img is None:
            feats.append(np.zeros(512))
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_flip = np.fliplr(img).copy()

        img = np.transpose(img, (2, 0, 1))
        img_flip = np.transpose(img_flip, (2, 0, 1))

        batch_imgs.append(img)
        batch_imgs.append(img_flip)

        if len(batch_imgs) >= BATCH_SIZE:
            batch = torch.from_numpy(np.array(batch_imgs)).float().to(device)
            batch = (batch / 255.0 - 0.5) / 0.5

            feat = model(batch)
            feat = F.normalize(feat, dim=1)

            feat = feat.reshape(-1, 2, 512)
            feat = feat[:, 0] + feat[:, 1]
            feat = F.normalize(feat, dim=1)

            feats.extend(feat.cpu().numpy())
            batch_imgs = []

    if len(batch_imgs) > 0:
        batch = torch.from_numpy(np.array(batch_imgs)).float().to(device)
        batch = (batch / 255.0 - 0.5) / 0.5

        feat = model(batch)
        feat = F.normalize(feat, dim=1)

        feat = feat.reshape(-1, 2, 512)
        feat = feat[:, 0] + feat[:, 1]
        feat = F.normalize(feat, dim=1)

        feats.extend(feat.cpu().numpy())

    return np.array(feats)

# =========================
# TEMPLATE POOLING
# =========================
def build_templates(feats, templates, medias):
    unique_templates = np.unique(templates)
    template_feats = np.zeros((len(unique_templates), 512))

    for i, tid in enumerate(unique_templates):
        idx = np.where(templates == tid)[0]
        face_feats = feats[idx]
        face_medias = medias[idx]

        media_feats = []
        for m in np.unique(face_medias):
            midx = np.where(face_medias == m)[0]
            media_feats.append(np.mean(face_feats[midx], axis=0, keepdims=True))

        media_feats = np.concatenate(media_feats, axis=0)
        template_feats[i] = np.sum(media_feats, axis=0)

    template_feats /= np.linalg.norm(template_feats, axis=1, keepdims=True)
    return template_feats, unique_templates

# =========================
# EVAL
# =========================
def evaluate(template_feats, template_ids, pairs):
    template2id = {tid: i for i, tid in enumerate(template_ids)}

    scores, labels = [], []

    for t1, t2, label in pairs:
        if t1 not in template2id or t2 not in template2id:
            continue

        sim = np.dot(template_feats[template2id[t1]],
                     template_feats[template2id[t2]])

        scores.append(sim)
        labels.append(label)

    return np.array(scores), np.array(labels)

# =========================
# TAR@FAR
# =========================
def tar_at_far(scores, labels, far):
    fpr, tpr, _ = roc_curve(labels, scores)
    return np.interp(far, fpr, tpr)

# =========================
# RUN
# =========================
names, templates, medias, pairs = load_meta()

run_alignment(names)  # chạy 1 lần thôi

print("Extracting features...")
feats = extract_features(names)

print("Building templates...")
template_feats, template_ids = build_templates(feats, templates, medias)

print("Evaluating...")
scores, labels = evaluate(template_feats, template_ids, pairs)

print("\n===== RESULT =====")
for far in [1e-4, 1e-5, 1e-6]:
    print(f"TAR@{far}: {tar_at_far(scores, labels, far):.4f}")