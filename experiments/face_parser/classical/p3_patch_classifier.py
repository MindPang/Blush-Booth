"""P3 (Lab 8): classify 8 x 8 patches with hand-made features.

Features per patch (25): LAB mean/std, hue histogram (8), LBP histogram (8), edge density, position (x, y).
Train: 100 random patches per train crop (label = most common class in the patch).
Classifiers: Random Forest, kNN, Decision Tree. SVM is skipped: too slow with ~200k patches.
3-fold StratifiedKFold on 20k train patches first (Lab 8 style), then the test on the 500 val crops
(all 32 x 32 patches of every crop, the patch probabilities are resized to the full 256 x 256).

    python experiments/face_parser/classical/p3_patch_classifier.py
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
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from experiments.common import data, evaluate, results  # noqa: E402
from experiments.common.features import block_labels, patch_feature_maps  # noqa: E402
from facekit.layout import N_CLASSES  # noqa: E402

EXP_ID = "P3"
PATCH = 8
PATCHES_PER_IMAGE = 100
MAX_KNN_SAMPLES = 30000     # kNN has to compare with every train patch, so it gets fewer


def make_classifiers(seed):
    return {
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1, random_state=seed),
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "DecisionTree": DecisionTreeClassifier(max_depth=20, random_state=seed),
    }


def training_patches(train_items, rng):
    X = []
    y = []
    for crop, label in train_items:
        feats = patch_feature_maps(crop, PATCH).reshape(-1, 25)
        classes = block_labels(label, PATCH).ravel()
        pick = rng.choice(len(feats), min(PATCHES_PER_IMAGE, len(feats)), replace=False)
        X.append(feats[pick])
        y.append(classes[pick])
    return np.concatenate(X), np.concatenate(y)


def parse_crop(model, crop):
    """(11, H, W) class probabilities: classify every patch, then resize the patch map to the crop size."""
    h, w = crop.shape[:2]
    feats = patch_feature_maps(crop, PATCH)
    rows, cols = feats.shape[:2]
    prob = model.predict_proba(feats.reshape(-1, feats.shape[-1]))
    full = np.zeros((rows * cols, N_CLASSES), np.float32)
    full[:, model.classes_] = prob           # classes never seen in training stay 0
    full = full.reshape(rows, cols, N_CLASSES)
    scores = cv2.resize(full, (w, h), interpolation=cv2.INTER_LINEAR)
    return scores.transpose(2, 0, 1)


def load_model(path, device=None):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = data.get_args("P3 patch features + classifier")
    data.set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    start = time.perf_counter()
    X, y = training_patches(data.parser_train_crops(args), rng)
    feature_time = time.perf_counter() - start
    print(f"{len(y)} train patches, {X.shape[1]} features")
    crops, labels = data.parser_eval_crops(args)

    cv_pick = rng.choice(len(y), min(20000, len(y)), replace=False)
    n_splits = 3
    for name, clf in make_classifiers(args.seed).items():
        if args.variant and args.variant != name:
            continue
        # classes with fewer patches than folds would break StratifiedKFold, so they are left out of the CV
        counts = np.bincount(y[cv_pick], minlength=N_CLASSES)
        ok = np.isin(y[cv_pick], np.nonzero(counts >= n_splits)[0])
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        cv_acc = cross_val_score(clf, X[cv_pick][ok], y[cv_pick][ok], cv=cv, scoring="accuracy")

        Xc, yc = X, y
        if name == "kNN" and len(y) > MAX_KNN_SAMPLES:
            pick = rng.choice(len(y), MAX_KNN_SAMPLES, replace=False)
            Xc, yc = X[pick], y[pick]
        start = time.perf_counter()
        clf.fit(Xc, yc)
        train_time = feature_time + time.perf_counter() - start
        with open(data.model_path(args, EXP_ID, name, ".pkl"), "wb") as f:
            pickle.dump(clf, f)

        cm, ms = evaluate.run_parser(lambda c: parse_crop(clf, c), crops, labels)
        results.write_row("parser", args, EXP_ID, "classical", "Lab 8", name,
                          f"patch={PATCH} patches/image={PATCHES_PER_IMAGE} features=25",
                          evaluate.parser_metrics(cm), train_time, ms,
                          f"{n_splits}-fold CV patch accuracy {cv_acc.mean():.3f}; {len(yc)} train patches; "
                          "per-class IoU: " + evaluate.per_class_text(cm))


if __name__ == "__main__":
    main()
