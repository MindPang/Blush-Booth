"""L6 (beyond class): MobileNetV2 pretrained on ImageNet (torchvision) + a regression head.

Why: MobileNetV2 is made for phones (inverted residual blocks with depthwise convs, about 2.2 M parameters
without the classifier), so it is fast, and the ImageNet weights already know edges / textures.
We remove the ImageNet classifier: features (1280 x 5 x 5 for a 160 crop) -> global average pool ->
dropout -> Linear(1280, 212). Same loss as L4/L5 (L1 / eye distance), same optimizer and schedule.
Input uses ImageNet normalisation.

    python experiments/face_landmark/deep_learning/l6_mobilenetv2.py
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
from facekit.models import count_params, pick_device  # noqa: E402

EXP_ID = "L6"


class MobileNetLandmarks(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v2(weights=weights).features
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, N_POINTS * 2))

    def forward(self, x):
        f = self.features(x).mean(dim=(2, 3))         # global average pool
        return self.head(f).reshape(-1, N_POINTS, 2) + 0.5


def load_model(path, device="cpu"):
    model = MobileNetLandmarks(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict_points(model, crop):
    device = next(model.parameters()).device
    uv = model(cnn.to_input(crop, "imagenet").to(device))[0].cpu().numpy().astype("float64")
    return uv * crop.shape[0] - 0.5


def loss_fn(model, batch, device):
    pred = model(cnn.to_input(batch["image"], "imagenet").to(device))
    return landmark_loss(pred, batch["uv"].to(device))


def main():
    def extra(ap):
        ap.add_argument("--no-pretrained", action="store_true", help="random weights (only to test the code offline)")
    args = data.get_args("L6 MobileNetV2 pretrained + regression head", extra=extra)
    data.set_seed(args.seed)
    device = pick_device(args.device)

    model = MobileNetLandmarks(pretrained=not args.no_pretrained)
    print(f"MobileNetV2 landmarks: {count_params(model) / 1e6:.2f} M parameters, device {device}")
    losses, train_time = cnn.train_cnn(model, data.LandmarkTrainSet(args), loss_fn, args, device)
    path = data.model_path(args, EXP_ID, "mobilenetv2", ".pt")
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}}, path)

    crops, points = data.landmark_eval_crops(args)
    preds, ms = evaluate.run_landmarks(lambda c: predict_points(model, c), crops, points)
    note = f"{device}; " + cnn.loss_note(losses)
    if args.no_pretrained:
        note = "NOT PRETRAINED; " + note
    results.write_row("landmark", args, EXP_ID, "deep learning", "pretrained ImageNet (torchvision), fine-tuned",
                      "mobilenetv2", f"epochs={args.epochs} batch={args.batch} lr={args.lr} loss=L1",
                      evaluate.landmark_metrics(preds, points), train_time, ms, note)


if __name__ == "__main__":
    main()
