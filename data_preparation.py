"""
data_preparation.py
====================
- Reads JPG images from named subfolders inside SOURCE_DIR
- Each folder name = class label
- Extracts features from the OBJECT REGION ONLY (not background)
  using background subtraction with a computed per-image background estimate
- DATA AUGMENTATION: brightness/contrast/flip/rotation variants
- Performs Train/Test split (80/20, stratified)
"""

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ─── Configuration ────────────────────────────────────────────────────────────
SOURCE_DIR  = r"E:\photos ml"
DATASET_DIR = r"E:\photos ml\project\dataset"
IMG_SIZE    = 128
TEST_SIZE   = 0.20
RANDOM_SEED = 42

CLASS_NAMES = [
    "ac remote",
    "earbuds",
    "geometry box",
    "mobile,phone",
    "notebook",
    "pen",
    "scale",
    "stapler",
    "water bottle",
    "whitener",
]
# ──────────────────────────────────────────────────────────────────────────────


def _augment(img):
    """Random brightness, contrast, flip, rotation."""
    delta = np.random.randint(-50, 51)
    img = np.clip(img.astype(np.int32) + delta, 0, 255).astype(np.uint8)
    alpha = np.random.uniform(0.7, 1.35)
    img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)
    angle = np.random.uniform(-12, 12)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return img


def _find_object_crop(img):
    """
    Detect the main object in the image and return a tight crop around it.
    Uses corner pixels as background sample, then finds the largest
    region that differs from that background.
    Falls back to center 60% crop if detection fails.
    """
    h, w = img.shape[:2]

    # Sample background from corners (away from where object is held)
    corner_size = min(h, w) // 8
    corners = [
        img[:corner_size, :corner_size],
        img[:corner_size, w-corner_size:],
        img[h-corner_size:, :corner_size],
        img[h-corner_size:, w-corner_size:],
    ]
    bg_color = np.median(
        np.vstack([c.reshape(-1, 3) for c in corners]), axis=0
    ).astype(np.float32)

    # Find pixels that differ from background
    diff = np.abs(img.astype(np.float32) - bg_color).mean(axis=2)
    thresh = max(18, diff.std() * 0.8)
    mask = (diff > thresh).astype(np.uint8)

    # Clean the mask
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k2)

    # Find contours and get the largest one
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if cnts:
        largest = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        min_area = h * w * 0.03   # at least 3% of frame
        if area > min_area:
            x, y, bw, bh = cv2.boundingRect(largest)
            # Add 8% padding
            pad_x = int(bw * 0.08)
            pad_y = int(bh * 0.08)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + bw + pad_x)
            y2 = min(h, y + bh + pad_y)
            cropped = img[y1:y2, x1:x2]
            if cropped.size > 0:
                return cropped

    # Fallback: center 70% crop
    mh, mw_m = int(h * 0.15), int(w * 0.15)
    return img[mh:h-mh, mw_m:w-mw_m]


def extract_features(img, augment=False):
    """
    Extract features from one image (numpy BGR array).
    Automatically crops to the object region first, then computes
    colour + shape + texture features.
    """
    if augment:
        img = _augment(img)

    # ── Step 1: find and crop to object ──────────────────────────────────────
    obj_crop = _find_object_crop(img)

    # ── Step 2: resize to fixed working resolution ────────────────────────────
    r    = cv2.resize(obj_crop, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)

    feats = []

    # 1. Saturation histogram ×5 — main discriminator (colourful vs grey/black)
    s_hist = cv2.calcHist([hsv], [1], None, [64], [0, 256]).flatten()
    s_hist /= (s_hist.sum() + 1e-7)
    feats.append(np.tile(s_hist, 5))

    # 2. Hue histogram ×5 — what colour the object actually is
    h_hist = cv2.calcHist([hsv], [0], None, [64], [0, 180]).flatten()
    h_hist /= (h_hist.sum() + 1e-7)
    feats.append(np.tile(h_hist, 5))

    # 3. Edge grid 8×8 ×5 — shape complexity per region
    edges = cv2.Canny(gray, 30, 100)
    step  = IMG_SIZE // 8
    edge_grid = np.array(
        [edges[i*step:(i+1)*step, j*step:(j+1)*step].mean() / 255.0
         for i in range(8) for j in range(8)], dtype="float32")
    feats.append(np.tile(edge_grid, 5))

    # 4. BGR colour histograms
    for ch in range(3):
        hist = cv2.calcHist([r], [ch], None, [32], [0, 256]).flatten()
        hist /= (hist.sum() + 1e-7)
        feats.append(hist)

    # 5. Equalised grayscale 32×32 (shape regardless of brightness)
    gray_eq = cv2.equalizeHist(gray)
    feats.append(cv2.resize(gray_eq, (32, 32)).flatten().astype("float32") / 255.0)

    # 6. Sobel direction histograms (edge orientations)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sx_h, _ = np.histogram(np.clip(sx, -255, 255), bins=32, range=(-255, 255), density=True)
    sy_h, _ = np.histogram(np.clip(sy, -255, 255), bins=32, range=(-255, 255), density=True)
    feats.append(sx_h.astype("float32"))
    feats.append(sy_h.astype("float32"))

    # 7. Laplacian texture grid 4×4
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    s4  = IMG_SIZE // 4
    lg  = np.array([lap[i*s4:(i+1)*s4, j*s4:(j+1)*s4].mean()
                    for i in range(4) for j in range(4)], dtype="float32")
    lg /= (lg.max() + 1e-7)
    feats.append(lg)

    return np.concatenate(feats).astype("float32")


def build_dataset():
    """Load all images + augmented copies. Test split = originals only."""
    orig_X, orig_y, filenames, label_names = [], [], [], []

    for class_id, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(folder):
            print(f"  [WARN] Folder not found: {folder}")
            continue

        jpgs = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
        print(f"  [{class_id:>2}] {class_name:<20}  {len(jpgs)} images")

        for fname in jpgs:
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            feat = extract_features(img, augment=False)
            orig_X.append(feat)
            orig_y.append(class_id)
            filenames.append(fname)
            label_names.append(class_name)

    orig_X = np.array(orig_X)
    orig_y = np.array(orig_y)

    # Split originals — test set stays augmentation-free
    X_tr, X_te, y_tr, y_te = train_test_split(
        orig_X, orig_y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=orig_y,
    )

    # Augment training split only
    np.random.seed(RANDOM_SEED)
    aug_X, aug_y = list(X_tr), list(y_tr)

    for class_id, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(SOURCE_DIR, class_name)
        if not os.path.isdir(folder):
            continue
        for fname in sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")]):
            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue
            for _ in range(5):
                aug_X.append(extract_features(img, augment=True))
                aug_y.append(class_id)

    return np.array(aug_X), X_te, np.array(aug_y), y_te, filenames, label_names, orig_y


def prepare():
    os.makedirs(DATASET_DIR, exist_ok=True)

    print(f"\nLoading images from: {SOURCE_DIR}")
    print("Object auto-crop + augmentation (5x) enabled\n")

    X_train, X_test, y_train, y_test, filenames, label_names, orig_y = build_dataset()

    df = pd.DataFrame({"filename": filenames, "label_id": orig_y,
                       "label_name": label_names})
    df.to_csv(os.path.join(DATASET_DIR, "labels.csv"), index=False)

    np.save(os.path.join(DATASET_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATASET_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(DATASET_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATASET_DIR, "y_test.npy"),  y_test)

    print(f"\n  Train : {X_train.shape[0]} samples (incl. augmented)")
    print(f"  Test  : {X_test.shape[0]}  samples (originals only)")
    print(f"  Features : {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test


def load_splits():
    X_train = np.load(os.path.join(DATASET_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATASET_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATASET_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATASET_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare()
