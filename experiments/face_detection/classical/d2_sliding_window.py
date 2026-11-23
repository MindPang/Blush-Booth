"""D2 (Lab 8): sliding window + hand-made features + a classifier.

Windows: 3 face widths (36, 72, 144 px in the 256 x 192 scene), aspect 32 x 40, step = 1/4 window.
Features per window: HSV mean/std, LBP histogram (2 x 2 cells), edge density (Canny).
Classifiers: kNN, SVM (RBF), Random Forest. 5-fold StratifiedKFold on the train windows first (like Lab 8),
then the real test: detect on the 500 + 500 val scenes. NMS, then the best box is the answer.

    python experiments/face_detection/classical/d2_sliding_window.py
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
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from experiments.common import data, evaluate, results  # noqa: E402
from experiments.common.features import lbp_image, window_features_from_maps  # noqa: E402
from facekit import datasets as D  # noqa: E402
from facekit.geometry import box_iou, letterbox, unletterbox_box  # noqa: E402

EXP_ID = "D2"
WIN_W = 32
WIN_H = 40
WINDOW_WIDTHS = [36, 72, 144]
THRESHOLDS = [0.5, 0.7, 0.9]
MAX_SVM_SAMPLES = 6000      # RBF SVM gets slow with many samples, so it gets a random part of the windows


def make_classifiers(seed):
    return {
        "kNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
        # CalibratedClassifierCV turns the SVM margin into a probability (0..1) like the other two
        "SVM": make_pipeline(StandardScaler(), CalibratedClassifierCV(SVC(kernel="rbf", C=1.0), cv=3, ensemble=False)),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed),
    }


def all_windows(scene):
    """Every window of every scale. Returns (features (N, F), boxes (N, 4) in scene pixels)."""
    feats = []
    boxes = []
    for win_w in WINDOW_WIDTHS:
        f = WIN_W / win_w
        small = cv2.resize(scene, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        if h < WIN_H or w < WIN_W:
            continue
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lbp = lbp_image(gray)
        edges = cv2.Canny(gray, 80, 160)
        step = WIN_W // 4
        for y in range(0, h - WIN_H + 1, step):
            for x in range(0, w - WIN_W + 1, step):
                feats.append(window_features_from_maps(hsv[y:y + WIN_H, x:x + WIN_W],
                                                       lbp[y:y + WIN_H, x:x + WIN_W],
                                                       edges[y:y + WIN_H, x:x + WIN_W]))
                boxes.append(np.array([x, y, x + WIN_W, y + WIN_H], np.float64) / f)
    return np.array(feats), np.array(boxes)


def training_windows(scene_list, rng):
    """Positives: windows with IoU > 0.6 with the true box (up to 3 per scene).
    Negatives: random windows with IoU < 0.3 (5 per face scene, 10 per empty scene)."""
    X = []
    y = []
    for scene, gt in scene_list:
        feats, wins = all_windows(scene)
        if gt is None:
            pick = rng.choice(len(wins), min(10, len(wins)), replace=False)
            X += list(feats[pick])
            y += [0] * len(pick)
            continue
        ious = np.array([box_iou(b, gt) for b in wins])
        pos = np.argsort(-ious)[:3]
        pos = pos[ious[pos] > 0.6]
        neg = np.nonzero(ious < 0.3)[0]
        neg = rng.choice(neg, min(5, len(neg)), replace=False)
        X += list(feats[pos]) + list(feats[neg])
        y += [1] * len(pos) + [0] * len(neg)
    return np.array(X), np.array(y)


def nms(boxes, scores, iou_thr=0.3):
    """Plain non-maximum suppression. Returns the kept indexes, best first."""
    order = list(np.argsort(-scores))
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if box_iou(boxes[i], boxes[j]) < iou_thr]
    return keep


def load_model(path, device=None):
    with open(path, "rb") as f:
        return pickle.load(f)


def detect(model, frame):
    scene, s = letterbox(frame, D.SCENE_W, D.SCENE_H)
    feats, wins = all_windows(scene)
    scores = model.predict_proba(feats)[:, 1]
    # only the top 50 windows go into NMS (it is a simple O(n^2) loop)
    top = np.argsort(-scores)[:50]
    keep = nms(wins[top], scores[top])
    best = top[keep[0]]
    return unletterbox_box(wins[best], s), float(scores[best])


def main():
    args = data.get_args("D2 sliding window + classifier")
    data.set_seed(args.seed)
    rng = np.random.RandomState(args.seed)

    print("making train windows ...")
    start = time.perf_counter()
    X, y = training_windows(data.detection_train_scenes(args), rng)
    feature_time = time.perf_counter() - start
    print(f"{len(y)} windows ({y.sum()} faces), {X.shape[1]} features")
    scenes, boxes = data.detection_eval_scenes(args)

    n_splits = min(5, int(min(np.bincount(y))))
    for name, clf in make_classifiers(args.seed).items():
        if args.variant and args.variant != name:
            continue
        Xc, yc = X, y
        if name == "SVM" and len(y) > MAX_SVM_SAMPLES:
            pick = rng.choice(len(y), MAX_SVM_SAMPLES, replace=False)
            Xc, yc = X[pick], y[pick]

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        cv_acc = cross_val_score(clf, Xc, yc, cv=cv, scoring="accuracy")

        start = time.perf_counter()
        clf.fit(Xc, yc)
        train_time = feature_time + time.perf_counter() - start
        with open(data.model_path(args, EXP_ID, name, ".pkl"), "wb") as f:
            pickle.dump(clf, f)

        raw, ms = evaluate.run_detector(lambda img: detect(clf, img), scenes, boxes)
        for thr in THRESHOLDS:
            metrics = evaluate.detection_metrics(raw, thr)
            results.write_row("detection", args, EXP_ID, "classical", "Lab 8", name,
                              f"thr={thr} scales=3 features=HSV+LBP+edges", metrics, train_time, ms,
                              f"{n_splits}-fold CV window accuracy {cv_acc.mean():.3f}; {len(yc)} train windows")


if __name__ == "__main__":
    main()
