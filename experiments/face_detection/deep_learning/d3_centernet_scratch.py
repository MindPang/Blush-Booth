"""D3: the old CenterNet-style DetectorNet (1.4 M parameters), trained from scratch.

Input: 256 x 192 scene. Output: 5 maps at 1/8 size (face-centre heatmap, log width, log height, x/y offset).
Loss: focal loss on the heatmap + L1 on size and offset (same as the old project).
Short run: 4 epochs on 2,000 train images, new random scenes every epoch.

    python experiments/face_detection/deep_learning/d3_centernet_scratch.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from experiments.common import cnn, data, evaluate, results  # noqa: E402
from facekit import datasets as D  # noqa: E402
from facekit.geometry import letterbox, unletterbox_box  # noqa: E402
from facekit.losses import detector_loss  # noqa: E402
from facekit.models import DetectorNet, count_params, load_checkpoint, pick_device, save_checkpoint  # noqa: E402

EXP_ID = "D3"
THRESHOLDS = [0.3, 0.4, 0.5]


def load_model(path, device="cpu"):
    return load_checkpoint(path, device)[0]


@torch.no_grad()
def detect(model, frame):
    scene, s = letterbox(frame, D.SCENE_W, D.SCENE_H)
    device = next(model.parameters()).device
    out = model(cnn.to_input(scene).to(device))
    boxes, scores = D.decode(out)
    return unletterbox_box(boxes[0], s), float(scores[0])


def loss_fn(model, batch, device):
    out = model(cnn.to_input(batch["image"]).to(device))
    loss, _ = detector_loss(out, batch["hm"].to(device), batch["reg"].to(device), batch["mask"].to(device))
    return loss


def main():
    args = data.get_args("D3 CenterNet-style detector from scratch")
    data.set_seed(args.seed)
    device = pick_device(args.device)

    model = DetectorNet()
    print(f"DetectorNet: {count_params(model) / 1e6:.2f} M parameters, device {device}")
    losses, train_time = cnn.train_cnn(model, data.DetectionTrainSet(args), loss_fn, args, device)
    save_checkpoint(data.model_path(args, EXP_ID, "centernet", ".pt"), model, "detector",
                    {"epochs": args.epochs, "lr": args.lr})

    scenes, boxes = data.detection_eval_scenes(args)
    raw, ms = evaluate.run_detector(lambda img: detect(model, img), scenes, boxes)
    for thr in THRESHOLDS:
        metrics = evaluate.detection_metrics(raw, thr)
        results.write_row("detection", args, EXP_ID, "deep learning", "existing model, from scratch", "centernet",
                          f"thr={thr} epochs={args.epochs} batch={args.batch} lr={args.lr}", metrics,
                          train_time, ms, f"{device}; " + cnn.loss_note(losses))


if __name__ == "__main__":
    main()
