"""L2 (lecture, corner detection): Harris corners for the 4 eye corners.

A search window (12% of the crop) is put where the mean shape (L1) has each eye corner.
Inside the window the strongest Harris response is taken as the corner.
"weighted" also multiplies the response with a Gaussian around the mean position, so far-away
corners (eyebrows, glasses) win less often.
Only 4 points are found, so it is scored on those 4 points against L1 on the same 4 points.
metric_1 (106-point NME) stays empty, so select_best.py never picks it as the landmark model.

    python experiments/face_landmark/classical/l2_harris.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from experiments.common import data, results  # noqa: E402
from experiments.face_landmark.classical.l1_mean_shape import mean_shape  # noqa: E402
from facekit.layout import L_EYE_INNER, L_EYE_OUTER, R_EYE_INNER, R_EYE_OUTER  # noqa: E402

EXP_ID = "L2"
CORNERS = [R_EYE_OUTER, R_EYE_INNER, L_EYE_INNER, L_EYE_OUTER]
VARIANTS = ["plain", "weighted"]


def eye_corners(crop, start_points, weighted, window=0.12):
    """Move the 4 start points to the strongest Harris corner in a window around each one."""
    size = crop.shape[0]
    gray = np.float32(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    response = cv2.cornerHarris(gray, 3, 3, 0.04)
    half = int(window * size / 2)
    out = []
    for x, y in start_points:
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(cx - half, 0), min(cx + half + 1, size)
        y0, y1 = max(cy - half, 0), min(cy + half + 1, size)
        if x1 <= x0 or y1 <= y0:
            out.append((x, y))
            continue
        r = response[y0:y1, x0:x1].copy()
        if weighted:
            ys, xs = np.mgrid[y0:y1, x0:x1]
            r = r * np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * (half / 2) ** 2))
        iy, ix = np.unravel_index(np.argmax(r), r.shape)
        out.append((x0 + ix, y0 + iy))
    return np.array(out, np.float64)


def nme_4(preds, gt):
    """NME on the 4 corners, normalised by the outer eye corner distance (points 0 and 3 here)."""
    err = np.linalg.norm(np.asarray(preds) - gt, axis=-1)
    eye_dist = np.linalg.norm(gt[:, 0] - gt[:, 3], axis=-1).clip(1e-6)
    return float(100 * (err / eye_dist[:, None]).mean())


def main():
    args = data.get_args("L2 Harris eye corners")
    start = time.perf_counter()
    train_points = np.array([pts for _, pts in data.landmark_train_crops(args)])
    mean_uv = mean_shape(train_points)
    train_time = time.perf_counter() - start

    crops, points = data.landmark_eval_crops(args)
    start_points = mean_uv[CORNERS] * data.LMK_SIZE - 0.5
    gt = points[:, CORNERS]
    l1_preds = np.repeat(start_points[None], len(crops), 0)
    l1_nme = nme_4(l1_preds, gt)

    for variant in VARIANTS:
        if args.variant and args.variant != variant:
            continue
        preds = []
        t0 = time.perf_counter()
        for crop in crops:
            preds.append(eye_corners(crop, start_points, variant == "weighted"))
        ms = 1000 * (time.perf_counter() - t0) / max(len(crops), 1)
        nme4 = nme_4(preds, gt)
        results.write_row("landmark", args, EXP_ID, "classical", "lecture (Harris corners)", variant,
                          "points=4 eye corners window=12%", [None, nme4, None, None], train_time, ms,
                          f"4 eye corners only; L1 mean shape on the same 4 points = {l1_nme:.3f}")


if __name__ == "__main__":
    main()
