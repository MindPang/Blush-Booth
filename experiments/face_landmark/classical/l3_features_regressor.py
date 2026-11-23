"""L3 (Lab 8 style, regression): hand-made features -> PCA 50 -> 212 numbers (106 x, y).

Features: 32 x 32 gray pixels + LBP histograms (4 x 4 grid, 16 bins) = 1280 numbers, then
StandardScaler + PCA to 50. Regressors: kNN, Random Forest, Ridge. 5-fold KFold on the train crops
first (NME), then the test on the 500 val crops.

    python experiments/face_landmark/classical/l3_features_regressor.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from experiments.common import data, evaluate, results  # noqa: E402
from experiments.common.features import landmark_features  # noqa: E402

EXP_ID = "L3"
N_PCA = 50


def make_regressors(seed, n_pca):
    def pipe(reg):
        return make_pipeline(StandardScaler(), PCA(n_components=n_pca, random_state=seed), reg)
    return {
        "kNN": pipe(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "RandomForest": pipe(RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=seed)),
        "Ridge": pipe(Ridge(alpha=1.0)),
    }


def predict_points(model, crop):
    size = crop.shape[0]
    uv = model.predict(landmark_features(crop)[None])[0].reshape(-1, 2)
    return uv * size - 0.5


def load_model(path, device=None):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = data.get_args("L3 features + PCA + regressor")
    start = time.perf_counter()
    X = []
    train_points = []
    for crop, pts in data.landmark_train_crops(args):
        X.append(landmark_features(crop))
        train_points.append(pts)
    X = np.array(X)
    train_points = np.array(train_points)
    Y = ((train_points + 0.5) / data.LMK_SIZE).reshape(len(X), -1)
    feature_time = time.perf_counter() - start
    n_pca = min(N_PCA, len(X) // 2, X.shape[1])
    crops, points = data.landmark_eval_crops(args)

    for name, reg in make_regressors(args.seed, n_pca).items():
        if args.variant and args.variant != name:
            continue
        # k-fold on the train crops (Lab 8 style); NME in the crop
        fold_nme = []
        for tr, te in KFold(n_splits=5, shuffle=True, random_state=args.seed).split(X):
            reg.fit(X[tr], Y[tr])
            pred = reg.predict(X[te]).reshape(len(te), -1, 2) * data.LMK_SIZE - 0.5
            fold_nme.append(evaluate.landmark_metrics(pred, train_points[te])[0])

        start = time.perf_counter()
        reg.fit(X, Y)
        train_time = feature_time + time.perf_counter() - start
        with open(data.model_path(args, EXP_ID, name, ".pkl"), "wb") as f:
            pickle.dump(reg, f)

        preds, ms = evaluate.run_landmarks(lambda c: predict_points(reg, c), crops, points)
        results.write_row("landmark", args, EXP_ID, "classical", "Lab 8 (regression)", name,
                          f"features=gray32+LBP pca={n_pca}", evaluate.landmark_metrics(preds, points),
                          train_time, ms, f"5-fold CV NME {np.mean(fold_nme):.3f}")


if __name__ == "__main__":
    main()
