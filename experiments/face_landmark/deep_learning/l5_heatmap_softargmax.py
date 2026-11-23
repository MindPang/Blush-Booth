"""L5: the old LandmarkNet (5.9 M parameters), trained from scratch.

A small U-shaped network gives one heatmap per point (106 maps at 1/4 of the 160 crop). Soft-argmax
turns each heatmap into an (x, y) point: softmax over the map, then the weighted mean position.
So it is trained directly on the point positions (L1 / eye distance), but it still "looks" for each point.

    python experiments/face_landmark/deep_learning/l5_heatmap_softargmax.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from experiments.common import cnn, data, evaluate, results  # noqa: E402
from facekit.losses import landmark_loss  # noqa: E402
from facekit.models import LandmarkNet, count_params, load_checkpoint, pick_device, save_checkpoint  # noqa: E402

EXP_ID = "L5"


def load_model(path, device="cpu"):
    return load_checkpoint(path, device)[0]


@torch.no_grad()
def predict_points(model, crop):
    device = next(model.parameters()).device
    uv = model(cnn.to_input(crop).to(device))[0].cpu().numpy().astype("float64")
    return uv * crop.shape[0] - 0.5


def loss_fn(model, batch, device):
    pred = model(cnn.to_input(batch["image"]).to(device))
    return landmark_loss(pred, batch["uv"].to(device))


def main():
    args = data.get_args("L5 LandmarkNet heatmaps + soft-argmax from scratch")
    data.set_seed(args.seed)
    device = pick_device(args.device)

    model = LandmarkNet()
    print(f"LandmarkNet: {count_params(model) / 1e6:.2f} M parameters, device {device}")
    losses, train_time = cnn.train_cnn(model, data.LandmarkTrainSet(args), loss_fn, args, device)
    save_checkpoint(data.model_path(args, EXP_ID, "heatmap", ".pt"), model, "landmarks",
                    {"size": data.LMK_SIZE, "epochs": args.epochs})

    crops, points = data.landmark_eval_crops(args)
    preds, ms = evaluate.run_landmarks(lambda c: predict_points(model, c), crops, points)
    results.write_row("landmark", args, EXP_ID, "deep learning", "existing model, from scratch", "heatmap",
                      f"epochs={args.epochs} batch={args.batch} lr={args.lr} loss=L1",
                      evaluate.landmark_metrics(preds, points), train_time, ms,
                      f"{device}; " + cnn.loss_note(losses))


if __name__ == "__main__":
    main()
