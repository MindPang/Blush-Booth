"""D1 (lecture IP07, Lab 7): template matching with an average face.

1. Average grayscale face: crop the true face box of every train image, resize to 32 x 40, take the mean.
2. Search the scene at 5 scales with cv2.matchTemplate and keep the best match.
3. Two match scores are compared: SSD (TM_SQDIFF_NORMED, score = 1 - value) and
   normalised correlation (TM_CCOEFF_NORMED). One row per score and threshold.

    python experiments/face_detection/classical/d1_template_matching.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from experiments.common import data, evaluate, results  # noqa: E402
from facekit import datasets as D  # noqa: E402
from facekit.geometry import face_box, letterbox, unletterbox_box  # noqa: E402

EXP_ID = "D1"
TEMPLATE_W = 32
TEMPLATE_H = 40
WINDOW_WIDTHS = [28, 44, 70, 110, 172]        # 5 scales: face width in scene pixels
METHODS = {"ncc": cv2.TM_CCOEFF_NORMED, "ssd": cv2.TM_SQDIFF_NORMED}
# the two scores have a different range, so each gets its own thresholds
THRESHOLDS = {"ncc": [0.3, 0.4, 0.5, 0.6, 0.7], "ssd": [0.80, 0.85, 0.90, 0.93, 0.96]}


def average_face(args):
    """Mean of all train face crops (gray, histogram equalised)."""
    total = np.zeros((TEMPLATE_H, TEMPLATE_W), np.float64)
    names = data.train_names(args)
    for name in names:
        img, pts, _ = D.read_sample(args.lapa, "train", name, with_label=False)
        x0, y0, x1, y1 = face_box(pts).astype(int)
        h, w = img.shape[:2]
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, w - 1), min(y1, h - 1)
        gray = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(cv2.resize(gray, (TEMPLATE_W, TEMPLATE_H), interpolation=cv2.INTER_AREA))
        total += gray
    return (total / len(names)).astype(np.uint8)


def load_model(path, device=None):
    saved = np.load(path)
    return {"template": saved["template"], "method": str(saved["method"])}


def save_model(model, path):
    np.savez(path, template=model["template"], method=model["method"])


def detect(model, frame):
    """Best match over the 5 scales -> (box, score)."""
    scene, s = letterbox(frame, D.SCENE_W, D.SCENE_H)
    gray = cv2.equalizeHist(cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY))
    method = METHODS[model["method"]]
    best_score = -1.0
    best_box = None
    for win_w in WINDOW_WIDTHS:
        f = TEMPLATE_W / win_w          # resize the scene so the face becomes template size
        small = cv2.resize(gray, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        if small.shape[0] < TEMPLATE_H or small.shape[1] < TEMPLATE_W:
            continue
        res = cv2.matchTemplate(small, model["template"], method)
        if model["method"] == "ssd":
            res = 1.0 - res
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            x, y = max_loc
            best_box = np.array([x, y, x + TEMPLATE_W, y + TEMPLATE_H], np.float64) / f
    if best_box is None:
        return None, 0.0
    return unletterbox_box(best_box, s), float(best_score)


def main():
    args = data.get_args("D1 template matching")
    data.set_seed(args.seed)

    start = time.perf_counter()
    template = average_face(args)
    train_time = time.perf_counter() - start
    scenes, boxes = data.detection_eval_scenes(args)

    for name in METHODS:
        if args.variant and args.variant != name:
            continue
        model = {"template": template, "method": name}
        save_model(model, data.model_path(args, EXP_ID, name, ".npz"))
        raw, ms = evaluate.run_detector(lambda img: detect(model, img), scenes, boxes)
        for thr in THRESHOLDS[name]:
            metrics = evaluate.detection_metrics(raw, thr)
            results.write_row("detection", args, EXP_ID, "classical", "lecture IP07 / Lab 7", name,
                              f"thr={thr} scales=5 template={TEMPLATE_W}x{TEMPLATE_H}", metrics,
                              train_time, ms, "average gray face, no learning")


if __name__ == "__main__":
    main()
