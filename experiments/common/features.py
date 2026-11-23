"""Hand-made features for the classical methods (Lab 8 style)."""
import cv2
import numpy as np


def lbp_image(gray):
    """Basic 8-neighbour LBP code (0..255) for every pixel. The 1-pixel border is 0."""
    g = gray.astype(np.int16)
    c = g[1:-1, 1:-1]
    h, w = c.shape
    code = np.zeros((h, w), np.uint8)
    neighbours = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    for bit, (dy, dx) in enumerate(neighbours):
        n = g[dy:dy + h, dx:dx + w]
        code |= ((n >= c).astype(np.uint8) << bit)
    return np.pad(code, 1)


def histogram(values, bins, max_value):
    """Normalised histogram of integer values in [0, max_value)."""
    hist = np.bincount((values.ravel().astype(np.int32) * bins) // max_value, minlength=bins)[:bins]
    return hist / max(values.size, 1)


def window_features(patch_bgr):
    """Features of one detector window: HSV mean/std, LBP histogram (2x2 grid), edge density."""
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    lbp = lbp_image(gray)
    edges = cv2.Canny(gray, 80, 160)
    return window_features_from_maps(hsv.reshape(patch_bgr.shape), lbp, edges)


def window_features_from_maps(hsv, lbp, edges):
    """Same as window_features, but from maps that are already computed (faster for sliding windows)."""
    hsv = hsv.reshape(-1, 3)
    feats = [hsv.mean(0) / 255.0, hsv.std(0) / 255.0]
    h, w = lbp.shape
    for gy in range(2):
        for gx in range(2):
            cell = lbp[gy * h // 2:(gy + 1) * h // 2, gx * w // 2:(gx + 1) * w // 2]
            feats.append(histogram(cell, 16, 256))
    feats.append([edges.mean() / 255.0])
    return np.concatenate(feats).astype(np.float32)


def patch_feature_maps(crop_bgr, patch=8):
    """Features for every patch x patch block of a crop, all at once.
    Per block: LAB mean/std (6), hue histogram (8), LBP histogram (8), edge density (1), position x, y (2) = 25.
    Returns an array (rows, cols, 25)."""
    h, w = crop_bgr.shape[:2]
    rows = h // patch
    cols = w // patch

    def blocks(a):
        # (H, W, C) -> (rows, cols, patch * patch, C)
        a = a[:rows * patch, :cols * patch]
        a = a.reshape(rows, patch, cols, patch, -1).transpose(0, 2, 1, 3, 4)
        return a.reshape(rows, cols, patch * patch, -1)

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB).astype(np.float32) / 255.0
    hue = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[..., 0]
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    lbp = lbp_image(gray)
    edges = (cv2.Canny(gray, 80, 160) > 0).astype(np.float32)

    lab_b = blocks(lab)
    hue_onehot = np.eye(8, dtype=np.float32)[(hue.astype(np.int32) * 8) // 180]
    lbp_onehot = np.eye(8, dtype=np.float32)[lbp.astype(np.int32) // 32]

    ys, xs = np.mgrid[0:rows, 0:cols].astype(np.float32)
    pos = np.stack([(xs + 0.5) / cols, (ys + 0.5) / rows], -1)

    feats = [
        lab_b.mean(2),
        lab_b.std(2),
        blocks(hue_onehot).mean(2),
        blocks(lbp_onehot).mean(2),
        blocks(edges[..., None]).mean(2),
        pos,
    ]
    return np.concatenate(feats, -1)


def block_labels(label, patch=8, n_classes=11):
    """Most common class in every patch x patch block. Returns (rows, cols)."""
    h, w = label.shape
    rows = h // patch
    cols = w // patch
    a = label[:rows * patch, :cols * patch].reshape(rows, patch, cols, patch).transpose(0, 2, 1, 3)
    a = a.reshape(rows, cols, patch * patch)
    counts = np.eye(n_classes, dtype=np.int32)[a].sum(2)
    return counts.argmax(-1)


def landmark_features(crop_bgr):
    """32 x 32 gray pixels + LBP histogram (4 x 4 grid, 16 bins) of a landmark crop."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    lbp = lbp_image(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA))
    hists = []
    for gy in range(4):
        for gx in range(4):
            hists.append(histogram(lbp[gy * 16:(gy + 1) * 16, gx * 16:(gx + 1) * 16], 16, 256))
    return np.concatenate([small.ravel()] + hists).astype(np.float32)
