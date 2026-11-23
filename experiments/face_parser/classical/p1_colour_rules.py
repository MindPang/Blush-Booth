"""P1 (lecture): colour rules + thresholding + morphology. Only skin, lips and hair (+ background).

  lips: a* channel of LAB (red-ness), Otsu threshold, only inside the mouth area.
        The mouth area is where lips are in at least 5% of the train crops (a fixed prior map).
        Above the middle row of the found lips = upper lip, below = lower lip.
  skin: YCrCb range Cr 133..173, Cb 77..127 (the usual skin box from the lecture).
  hair: dark pixels (V < 90 in HSV) that are not skin, in the top half of the crop or at the sides.
  cleanup: morphological open + close (5 x 5), compared with no cleanup.
All the other classes are never predicted, so their IoU is 0 (the per-class IoU is in the notes).

    python experiments/face_parser/classical/p1_colour_rules.py
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
from facekit.layout import D_LIP, HAIR, MOUTH, N_CLASSES, SKIN, U_LIP  # noqa: E402

EXP_ID = "P1"
VARIANTS = ["no_cleanup", "open_close"]


def mouth_prior(train_items):
    """Pixels where at least 5% of the train crops have lips (or inner mouth)."""
    total = None
    count = 0
    for _, label in train_items:
        lips = np.isin(label, [U_LIP, MOUTH, D_LIP]).astype(np.float64)
        if total is None:
            total = lips
        else:
            total += lips
        count += 1
    return total / count >= 0.05


def clean(mask, kernel):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def rules_labels(model, crop):
    """Label map (H, W) from the colour rules."""
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mouth_area = model["mouth_area"]
    if mouth_area.shape != (h, w):
        mouth_area = cv2.resize(mouth_area.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

    # lips: Otsu on a* inside the mouth area
    lips = np.zeros((h, w), np.uint8)
    a_values = lab[..., 1][mouth_area]
    if a_values.size > 0:
        thr, _ = cv2.threshold(a_values.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lips = ((lab[..., 1] > thr) & mouth_area).astype(np.uint8)

    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]
    skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8)

    ys, xs = np.mgrid[0:h, 0:w]
    top_or_side = (ys < 0.45 * h) | (xs < 0.15 * w) | (xs > 0.85 * w)
    hair = ((hsv[..., 2] < 90) & (skin == 0) & top_or_side).astype(np.uint8)

    if model["cleanup"] == "open_close":
        kernel = np.ones((5, 5), np.uint8)
        lips = clean(lips, kernel)
        skin = clean(skin, kernel)
        hair = clean(hair, kernel)

    out = np.zeros((h, w), np.uint8)
    out[hair > 0] = HAIR
    out[skin > 0] = SKIN
    if lips.any():
        middle = np.median(ys[lips > 0])
        out[(lips > 0) & (ys < middle)] = U_LIP
        out[(lips > 0) & (ys >= middle)] = D_LIP
    return out


def parse_crop(model, crop):
    """(11, H, W) one-hot scores, so the app can use it like the CNN parsers."""
    labels = rules_labels(model, crop)
    return np.eye(N_CLASSES, dtype=np.float32)[labels].transpose(2, 0, 1)


def load_model(path, device=None):
    saved = np.load(path)
    return {"mouth_area": saved["mouth_area"], "cleanup": str(saved["cleanup"])}


def main():
    args = data.get_args("P1 colour rules")
    start = time.perf_counter()
    area = mouth_prior(data.parser_train_crops(args))
    train_time = time.perf_counter() - start
    crops, labels = data.parser_eval_crops(args)

    for variant in VARIANTS:
        if args.variant and args.variant != variant:
            continue
        model = {"mouth_area": area, "cleanup": variant}
        np.savez(data.model_path(args, EXP_ID, variant, ".npz"), mouth_area=area, cleanup=variant)
        cm, ms = evaluate.run_parser(lambda c: rules_labels(model, c), crops, labels)
        results.write_row("parser", args, EXP_ID, "classical", "lecture (colour spaces, Otsu, morphology)",
                          variant, "lips=a*+Otsu skin=YCrCb hair=dark V<90", evaluate.parser_metrics(cm),
                          train_time, ms, "only skin/lips/hair; per-class IoU: " + evaluate.per_class_text(cm))


if __name__ == "__main__":
    main()
