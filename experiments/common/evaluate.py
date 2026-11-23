"""The same evaluation for every experiment of a model, so the numbers can be compared."""
import time

import numpy as np

from facekit.geometry import box_iou
from facekit.layout import CLASS_NAMES, HAIR, LIP_CLASSES, N_CLASSES, REGIONS, SKIN
from facekit.metrics import confusion, iou_f1, merge_classes, nme_per_point


# ------------------------------------------------------------------ face detection

def run_detector(detect_fn, scenes, boxes):
    """Run detect_fn(scene) -> (box, score) on every scene. Returns the raw results + ms per frame."""
    scores = []
    ious = []
    has_face = []
    start = time.perf_counter()
    for scene, gt in zip(scenes, boxes):
        box, score = detect_fn(scene)
        if box is None:
            score = 0.0
        scores.append(float(score))
        has_face.append(gt is not None)
        if gt is not None and box is not None:
            ious.append(box_iou(box, gt))
        elif gt is not None:
            ious.append(0.0)
        else:
            ious.append(np.nan)
    ms = 1000 * (time.perf_counter() - start) / max(len(scenes), 1)
    return {"scores": np.array(scores), "ious": np.array(ious), "has_face": np.array(has_face, bool)}, ms


def detection_metrics(raw, thr):
    """recall (IoU > 0.5 and score >= thr), mean IoU of the best box, false alarm rate on face-free scenes."""
    face = raw["has_face"]
    scores = raw["scores"]
    ious = raw["ious"]
    recall = float(((ious[face] > 0.5) & (scores[face] >= thr)).mean()) if face.any() else float("nan")
    mean_iou = float(ious[face].mean()) if face.any() else float("nan")
    false_alarm = float((scores[~face] >= thr).mean()) if (~face).any() else 0.0
    return [recall, mean_iou, false_alarm]


# ------------------------------------------------------------------ face landmarks

def run_landmarks(predict_fn, crops, gt_points):
    """predict_fn(crop) -> (106, 2) points in crop pixels. Returns (predictions, ms per crop)."""
    preds = []
    start = time.perf_counter()
    for crop in crops:
        preds.append(predict_fn(crop))
    ms = 1000 * (time.perf_counter() - start) / max(len(crops), 1)
    return np.array(preds), ms


def landmark_metrics(preds, gt_points, fail_thr=0.10):
    """NME (% of the outer eye corner distance) over all 106 points, eye points, lip points + failure rate."""
    err = nme_per_point(preds, gt_points)
    per_face = err.mean(1)
    nme_all = float(100 * per_face.mean())
    nme_eyes = float(100 * err[:, REGIONS["eyes"]].mean())
    nme_lips = float(100 * err[:, REGIONS["lips"]].mean())
    failure = float((per_face > fail_thr).mean())
    return [nme_all, nme_eyes, nme_lips, failure]


# ------------------------------------------------------------------ face parser

def run_parser(parse_fn, crops, labels):
    """parse_fn(crop) -> (11, H, W) scores or (H, W) labels. Returns (confusion matrix, ms per crop)."""
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    total = 0.0
    for crop, label in zip(crops, labels):
        start = time.perf_counter()
        out = np.asarray(parse_fn(crop))
        total += time.perf_counter() - start
        if out.ndim == 3:
            out = out.argmax(0)
        cm += confusion(out, label)
    return cm, 1000 * total / max(len(crops), 1)


def parser_metrics(cm):
    """mIoU over the 11 classes, lips IoU (upper + lower lip merged), hair IoU, skin IoU."""
    iou, _ = iou_f1(cm)
    lips_cm, lips_idx = merge_classes(cm, LIP_CLASSES)
    lips_iou = iou_f1(lips_cm)[0][lips_idx]
    miou = float(np.nanmean(iou))
    return [miou, float(lips_iou), float(iou[HAIR]), float(iou[SKIN])]


def per_class_text(cm):
    iou, _ = iou_f1(cm)
    short = ["bg", "skin", "lbrow", "rbrow", "leye", "reye", "nose", "ulip", "mouth", "llip", "hair"]
    assert len(short) == len(CLASS_NAMES)
    return " ".join(f"{n}={v:.2f}" for n, v in zip(short, iou))
