import io
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve

# ============================================================
# 1. LOAD INSIGHTFACE BIN BINARY FORMAT
# ============================================================

def decode_bin_image(bin_item):
    """Decodes raw bytes image from bin file to PIL RGB image."""
    img = Image.open(io.BytesIO(bin_item)).convert("RGB")
    img = img.resize((112, 112), Image.BILINEAR)
    return img


def load_bin(bin_path):
    """Loads a verification bin file (e.g. lfw.bin) and prepares it as a tensor.
    
    Args:
        bin_path (str): Absolute path to the .bin file.
        
    Returns:
        tuple: (images_tensor, is_same_array)
    """
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
# 2. EXTRACT EMBEDDINGS (WITH OPTIONAL FLIP TEST)
# ============================================================

@torch.no_grad()
def extract_embeddings(model, images, batch_size=128, use_flip=True, device="cuda"):
    """Extracts normalization embeddings from the model.
    
    Args:
        model (nn.Module): The evaluation backbone.
        images (torch.Tensor): Preprocessed images of shape (N, 3, 112, 112).
        batch_size (int): Size of batches for forward pass.
        use_flip (bool): Enable horizontal flip test augmentation.
        device (str): Device to run inference on.
        
    Returns:
        np.ndarray: Extracted embeddings of shape (N, Dim).
    """
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
# 3. 10-FOLD VERIFICATION ACCURACY
# ============================================================

def evaluate_10fold(embeddings, issame, folds=10):
    """Performs 10-fold cross-validation to find the optimal cosine threshold
    and evaluate verification accuracy.
    
    Args:
        embeddings (np.ndarray): Extracted embeddings of shape (2*N, Dim).
        issame (np.ndarray): Boolean array of matches of shape (N,).
        folds (int): Number of folds for cross-validation.
        
    Returns:
        tuple: (mean_accuracy, std_accuracy)
    """
    # Split into pairs
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

        # Train: Find best threshold
        sims_train = np.sum(emb1[train_idx] * emb2[train_idx], axis=1)
        labels_train = issame[train_idx]

        best_acc = 0
        best_th = 0

        for th in thresholds:
            acc = np.mean((sims_train > th) == labels_train)
            if acc > best_acc:
                best_acc = acc
                best_th = th

        # Test: Evaluate accuracy using the best threshold
        sims_test = np.sum(emb1[test_idx] * emb2[test_idx], axis=1)
        labels_test = issame[test_idx]

        acc = np.mean((sims_test > best_th) == labels_test)
        accs.append(acc)

    return float(np.mean(accs)), float(np.std(accs))


# ============================================================
# 4. TAR@FAR METRIC
# ============================================================

def tar_at_far(embeddings, issame, target_far=1e-4):
    """Computes True Acceptance Rate (TAR) at a target False Acceptance Rate (FAR).
    
    Args:
        embeddings (np.ndarray): Extracted embeddings of shape (2*N, Dim).
        issame (np.ndarray): Boolean array of matches of shape (N,).
        target_far (float): Target FAR threshold (e.g. 1e-4).
        
    Returns:
        float: Computed TAR value.
    """
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    labels = issame.astype(int)

    # Cosine similarities
    scores = np.sum(emb1 * emb2, axis=1)

    # Compute ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)

    if target_far > max(fpr):
        return 0.0

    # Interpolate to find TPR (TAR) at target FPR (FAR)
    return float(np.interp(target_far, fpr, tpr))
