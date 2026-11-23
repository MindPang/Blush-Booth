"""Reading LaPa and making the model inputs (scenes for the detector, crops for landmarks / parser).

This is a smaller version of the old facekit/datasets.py. The crop and scene code is the same
as before (256 x 192 scenes, 160 landmark crops, 256 parser crops), only the cache and the
big Dataset classes are gone. The experiments read LaPa directly.
"""
from pathlib import Path

import cv2
import numpy as np

from .geometry import box_centre_size, face_box, stretch_matrix, transform_points, warp
from .layout import BACKGROUND, HAIR, N_POINTS

IMG_EXTS = (".jpg", ".jpeg", ".png")

# detector scene size (same as the old project)
SCENE_W = 256
SCENE_H = 192
STRIDE = 8
GRID_W = SCENE_W // STRIDE
GRID_H = SCENE_H // STRIDE
SCORE_THR = 0.40   # old decision X20, chosen on val


# ------------------------------------------------------------------ reading LaPa

def read_image(path, flags=cv2.IMREAD_COLOR):
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def list_names(lapa_root, split):
    """All image names (no extension) of one split, sorted."""
    folder = Path(lapa_root) / split / "images"
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)


def image_path(lapa_root, split, name):
    folder = Path(lapa_root) / split / "images"
    for ext in IMG_EXTS:
        if (folder / (name + ext)).exists():
            return folder / (name + ext)
    return folder / (name + ".jpg")


def read_landmarks(path):
    """One LaPa landmark file -> (106, 2) array."""
    lines = Path(path).read_text().split("\n")
    n = int(lines[0])
    if n != N_POINTS:
        raise ValueError(f"{path}: expected {N_POINTS} points, file says {n}")
    return np.array([line.split()[:2] for line in lines[1:1 + n]], dtype=np.float64)


def read_sample(lapa_root, split, name, with_label=True):
    """Returns (BGR image, points (106, 2), label map or None)."""
    folder = Path(lapa_root) / split
    img = read_image(image_path(lapa_root, split, name))
    points = read_landmarks(folder / "landmarks" / (name + ".txt"))
    label = None
    if with_label:
        label = read_image(folder / "labels" / (name + ".png"), cv2.IMREAD_GRAYSCALE)
    return img, points, label


# ------------------------------------------------------------------ detector scenes

def load_backgrounds(folder, max_side=384):
    """Face-free background photos (backgrounds/<split>/), longest side at most 384."""
    paths = sorted(p for p in Path(folder).glob("*") if p.suffix.lower() in IMG_EXTS)
    if not paths:
        raise FileNotFoundError(f"no background photos in {folder}")
    out = []
    for p in paths:
        img = read_image(p)
        s = max_side / max(img.shape[:2])
        if s < 1:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        out.append(img)
    return out


def background_crop(bg_img, rng):
    """Random part of a background photo, resized to the scene size."""
    h, w = bg_img.shape[:2]
    scale = np.sqrt(rng.uniform(0.25, 1.0))
    crop_w = max(4, int(w * scale))
    crop_h = max(4, int(h * scale))
    x0 = rng.randint(w - crop_w + 1)
    y0 = rng.randint(h - crop_h + 1)
    bg = cv2.resize(bg_img[y0:y0 + crop_h, x0:x0 + crop_w], (SCENE_W, SCENE_H), interpolation=cv2.INTER_AREA)
    if rng.uniform() < 0.5:
        bg = np.ascontiguousarray(bg[:, ::-1])
    return bg


def face_scene(img, pts, bg_img, rng):
    """Paste the LaPa face onto a background (same steps as the old make_scene).
    Returns (scene 256 x 192, true face box)."""
    bg = background_crop(bg_img, rng)
    centre, box_w, _ = box_centre_size(face_box(pts))
    face_w = SCENE_W * np.exp(rng.uniform(np.log(0.08), np.log(0.75)))
    angle = rng.uniform(-25, 25)
    target = np.array([rng.uniform(0.05, 0.95) * SCENE_W, rng.uniform(0.10, 0.90) * SCENE_H])
    matrix = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])), angle, face_w / box_w)
    matrix[:, 2] = target - matrix[:, :2] @ centre

    face = warp(img, matrix, (SCENE_W, SCENE_H), antialias=True)
    mask = cv2.warpAffine(np.ones(img.shape[:2], np.float32), matrix, (SCENE_W, SCENE_H), flags=cv2.INTER_LINEAR)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
    mask = cv2.GaussianBlur(mask, (0, 0), 2)[..., None]
    out = np.clip(face * mask + bg * (1 - mask), 0, 255).astype(np.uint8)
    return out, face_box(transform_points(pts, matrix))


def encode_target(box):
    """CenterNet training target for one scene (heatmap, size/offset, mask)."""
    hm = np.zeros((1, GRID_H, GRID_W), np.float32)
    reg = np.zeros((4, GRID_H, GRID_W), np.float32)
    mask = np.zeros((1, GRID_H, GRID_W), np.float32)
    if box is None:
        return hm, reg, mask
    (cx, cy), w, h = box_centre_size(box)
    gx = cx / STRIDE
    gy = cy / STRIDE
    ix = int(np.clip(np.floor(gx), 0, GRID_W - 1))
    iy = int(np.clip(np.floor(gy), 0, GRID_H - 1))
    sigma = max(0.6, 0.08 * min(w, h) / STRIDE)
    yy, xx = np.mgrid[0:GRID_H, 0:GRID_W]
    hm[0] = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * sigma ** 2))
    hm[0, iy, ix] = 1.0
    reg[:, iy, ix] = [np.log(max(w, 1e-3) / STRIDE), np.log(max(h, 1e-3) / STRIDE), gx - ix, gy - iy]
    mask[0, iy, ix] = 1.0
    return hm, reg, mask


def decode(out):
    """CenterNet output -> one box + score per image (highest heatmap peak)."""
    import torch

    with torch.no_grad():
        hm = torch.sigmoid(out[:, 0:1])
        peaks = hm * (torch.nn.functional.max_pool2d(hm, 3, 1, 1) == hm).float()
        n = out.shape[0]
        score, idx = peaks.reshape(n, -1).max(1)
        iy = idx // GRID_W
        ix = idx % GRID_W
        rows = torch.arange(n, device=out.device)
        w = torch.exp(out[rows, 1, iy, ix].clamp(-6, 6)) * STRIDE
        h = torch.exp(out[rows, 2, iy, ix].clamp(-6, 6)) * STRIDE
        cx = (ix.float() + out[rows, 3, iy, ix]) * STRIDE
        cy = (iy.float() + out[rows, 4, iy, ix]) * STRIDE
        boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return boxes.cpu().numpy().astype(np.float64), score.cpu().numpy().astype(np.float64)


# ------------------------------------------------------------------ parser crops

def label_face_box(label):
    """Face box from the label map: every pixel that is not background and not hair."""
    ys, xs = np.nonzero((label != BACKGROUND) & (label != HAIR))
    if len(xs) == 0:
        h, w = label.shape
        return np.array([0.0, 0.0, w - 1.0, h - 1.0])
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float64)


def parser_matrix(box, pads, angle, size):
    """Grow the box (pads = left, right, top, bottom as fractions), rotate by -angle, stretch to size x size."""
    x0, y0, x1, y1 = box
    w = max(x1 - x0, 1.0)
    h = max(y1 - y0, 1.0)
    x0, x1 = x0 - pads[0] * w, x1 + pads[1] * w
    y0, y1 = y0 - pads[2] * h, y1 + pads[3] * h
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return stretch_matrix(centre, x1 - x0, y1 - y0, angle, size, size)
