"""P2: K-means clustering on LAB colour + (x, y) position, K = 6 / 8 / 11.

Train: take 300 random pixels from every train crop, features = (L, a, b, x, y) all scaled to 0..1.
Run K-means on all of them. Every cluster gets the most common true class of its train pixels.
Test: every pixel goes to its nearest cluster centre and gets that cluster's class.

    python experiments/face_parser/classical/p2_kmeans.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.cluster import MiniBatchKMeans  # noqa: E402

from experiments.common import data, evaluate, results  # noqa: E402
from facekit.layout import N_CLASSES  # noqa: E402

EXP_ID = "P2"
K_VALUES = [6, 8, 11]
PIXELS_PER_IMAGE = 300


def pixel_features(crop):
    """(H * W, 5) features: L, a, b, x, y in 0..1."""
    h, w = crop.shape[:2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32) / 255.0
    ys, xs = np.mgrid[0:h, 0:w]
    pos = np.stack([xs.ravel() / (w - 1), ys.ravel() / (h - 1)], 1).astype(np.float32)
    return np.concatenate([lab, pos], 1)


def sample_pixels(train_items, seed):
    """300 random pixels (features + true class) from every train crop."""
    rng = np.random.RandomState(seed)
    X = []
    y = []
    for crop, label in train_items:
        feats = pixel_features(crop)
        pick = rng.choice(len(feats), PIXELS_PER_IMAGE, replace=False)
        X.append(feats[pick])
        y.append(label.ravel()[pick])
    return np.concatenate(X), np.concatenate(y)


def train_kmeans(X, y, k, seed):
    km = MiniBatchKMeans(n_clusters=k, random_state=seed, batch_size=4096, n_init=3)
    clusters = km.fit_predict(X)
    cluster_class = np.zeros(k, np.int64)
    for c in range(k):
        members = y[clusters == c]
        if len(members) > 0:
            cluster_class[c] = np.bincount(members, minlength=N_CLASSES).argmax()
    return {"kmeans": km, "cluster_class": cluster_class}


def kmeans_labels(model, crop):
    h, w = crop.shape[:2]
    clusters = model["kmeans"].predict(pixel_features(crop))
    return model["cluster_class"][clusters].reshape(h, w).astype(np.uint8)


def parse_crop(model, crop):
    labels = kmeans_labels(model, crop)
    return np.eye(N_CLASSES, dtype=np.float32)[labels].transpose(2, 0, 1)


def load_model(path, device=None):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = data.get_args("P2 K-means on LAB + position")
    start = time.perf_counter()
    X, y = sample_pixels(data.parser_train_crops(args), args.seed)
    sample_time = time.perf_counter() - start
    crops, labels = data.parser_eval_crops(args)

    for k in K_VALUES:
        name = f"K={k}"
        if args.variant and args.variant != name:
            continue
        start = time.perf_counter()
        model = train_kmeans(X, y, k, args.seed)
        train_time = sample_time + time.perf_counter() - start
        with open(data.model_path(args, EXP_ID, f"K{k}", ".pkl"), "wb") as f:
            pickle.dump(model, f)
        cm, ms = evaluate.run_parser(lambda c: kmeans_labels(model, c), crops, labels)
        results.write_row("parser", args, EXP_ID, "classical", "lecture (clustering)", name,
                          f"features=LAB+xy pixels/image={PIXELS_PER_IMAGE}", evaluate.parser_metrics(cm),
                          train_time, ms, "cluster -> majority train class; per-class IoU: "
                          + evaluate.per_class_text(cm))


if __name__ == "__main__":
    main()
