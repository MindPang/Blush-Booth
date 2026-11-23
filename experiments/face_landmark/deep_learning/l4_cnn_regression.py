"""L4: small CNN that outputs the 212 numbers (106 x, y) directly (no heatmaps).

5 blocks of conv3x3 (stride 2) - BN - ReLU: 160 -> 80 -> 40 -> 20 -> 10 -> 5 pixels,
then flatten -> FC 256 -> FC 212. Loss: L1 distance / eye distance (same landmark loss as L5).

    python experiments/face_landmark/deep_learning/l4_cnn_regression.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from experiments.common import cnn, data, evaluate, results  # noqa: E402
from facekit.layout import N_POINTS  # noqa: E402
from facekit.losses import landmark_loss  # noqa: E402
from facekit.models import conv_bn_relu, count_params, pick_device  # noqa: E402

EXP_ID = "L4"


class SmallRegressionCNN(nn.Module):
    def __init__(self, size=160):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(3, 32, 3, 2),
            conv_bn_relu(32, 64, 3, 2),
            conv_bn_relu(64, 128, 3, 2),
            conv_bn_relu(128, 256, 3, 2),
            conv_bn_relu(256, 256, 3, 2),
        )
        side = size // 32
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * side * side, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, N_POINTS * 2),
        )

    def forward(self, x):
        out = self.head(self.features(x))
        return out.reshape(-1, N_POINTS, 2) + 0.5     # start around the middle of the crop


def load_model(path, device="cpu"):
    model = SmallRegressionCNN()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict_points(model, crop):
    device = next(model.parameters()).device
    uv = model(cnn.to_input(crop).to(device))[0].cpu().numpy().astype("float64")
    return uv * crop.shape[0] - 0.5


def loss_fn(model, batch, device):
    pred = model(cnn.to_input(batch["image"]).to(device))
    return landmark_loss(pred, batch["uv"].to(device))


def main():
    args = data.get_args("L4 small CNN, direct regression")
    data.set_seed(args.seed)
    device = pick_device(args.device)

    model = SmallRegressionCNN()
    print(f"SmallRegressionCNN: {count_params(model) / 1e6:.2f} M parameters, device {device}")
    losses, train_time = cnn.train_cnn(model, data.LandmarkTrainSet(args), loss_fn, args, device)
    path = data.model_path(args, EXP_ID, "cnn_regression", ".pt")
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}}, path)

    crops, points = data.landmark_eval_crops(args)
    preds, ms = evaluate.run_landmarks(lambda c: predict_points(model, c), crops, points)
    results.write_row("landmark", args, EXP_ID, "deep learning", "new small CNN, from scratch", "cnn_regression",
                      f"epochs={args.epochs} batch={args.batch} lr={args.lr} loss=L1",
                      evaluate.landmark_metrics(preds, points), train_time, ms,
                      f"{device}; " + cnn.loss_note(losses))


if __name__ == "__main__":
    main()
