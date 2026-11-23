"""L1 (baseline, no learning): put the average face shape in the crop.

The mean of the 106 points (in crop coordinates 0..1) over the train crops is the answer for every face.
Every other landmark method has to beat this.

    python experiments/face_landmark/classical/l1_mean_shape.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402

from experiments.common import data, evaluate, results  # noqa: E402

EXP_ID = "L1"


def mean_shape(train_points, size=data.LMK_SIZE):
    """Mean of the points in 0..1 crop units."""
    return ((train_points + 0.5) / size).mean(0)


def predict_points(model, crop):
    size = crop.shape[0]
    return model["mean_uv"] * size - 0.5


def load_model(path, device=None):
    return {"mean_uv": np.load(path)["mean_uv"]}


def main():
    args = data.get_args("L1 mean shape")
    start = time.perf_counter()
    train_points = np.array([pts for _, pts in data.landmark_train_crops(args)])
    model = {"mean_uv": mean_shape(train_points)}
    train_time = time.perf_counter() - start
    np.savez(data.model_path(args, EXP_ID, "mean_shape", ".npz"), mean_uv=model["mean_uv"])

    crops, points = data.landmark_eval_crops(args)
    preds, ms = evaluate.run_landmarks(lambda c: predict_points(model, c), crops, points)
    results.write_row("landmark", args, EXP_ID, "classical", "baseline (no learning)", "mean_shape",
                      f"crop={data.LMK_SIZE}", evaluate.landmark_metrics(preds, points), train_time, ms,
                      "mean of the train shapes in the crop")


if __name__ == "__main__":
    main()
