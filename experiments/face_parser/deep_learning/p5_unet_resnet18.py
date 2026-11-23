"""P5 (beyond class): U-Net with a ResNet18 encoder pretrained on ImageNet (segmentation_models_pytorch).

Why: the encoder already knows edges, textures and shapes from 1.2 M ImageNet photos, so with only
2,000 face crops it may learn faster than P4. Same idea as P4: encoder -> decoder with skip connections,
11 class scores per pixel, same loss (CE + Dice), same optimizer and schedule. Input uses ImageNet
normalisation. Needs:  pip install segmentation-models-pytorch

    python experiments/face_parser/deep_learning/p5_unet_resnet18.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from experiments.common import cnn, data, evaluate, results  # noqa: E402
from facekit.layout import N_CLASSES  # noqa: E402
from facekit.losses import ce_dice_loss  # noqa: E402
from facekit.models import count_params, pick_device  # noqa: E402

EXP_ID = "P5"


def build(pretrained=True):
    import segmentation_models_pytorch as smp
    weights = "imagenet" if pretrained else None
    return smp.Unet(encoder_name="resnet18", encoder_weights=weights, classes=N_CLASSES)


def load_model(path, device="cpu"):
    model = build(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    return model.to(device).eval()


@torch.no_grad()
def parse_crop(model, crop):
    device = next(model.parameters()).device
    return model(cnn.to_input(crop, "imagenet").to(device))[0].float().cpu().numpy()


def loss_fn(model, batch, device):
    logits = model(cnn.to_input(batch["image"], "imagenet").to(device))
    loss, _ = ce_dice_loss(logits, batch["label"].to(device))
    return loss


def main():
    def extra(ap):
        ap.add_argument("--no-pretrained", action="store_true", help="random encoder (only to test the code offline)")
    args = data.get_args("P5 U-Net with pretrained ResNet18 encoder", extra=extra)
    data.set_seed(args.seed)
    device = pick_device(args.device)

    model = build(pretrained=not args.no_pretrained)
    print(f"U-Net ResNet18: {count_params(model) / 1e6:.2f} M parameters, device {device}")
    losses, train_time = cnn.train_cnn(model, data.ParserTrainSet(args), loss_fn, args, device)
    path = data.model_path(args, EXP_ID, "unet_resnet18", ".pt")
    torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()}}, path)

    crops, labels = data.parser_eval_crops(args)
    cm, ms = evaluate.run_parser(lambda c: parse_crop(model, c), crops, labels)
    results.write_row("parser", args, EXP_ID, "deep learning", "pretrained ImageNet encoder (smp), fine-tuned",
                      "unet_resnet18", f"epochs={args.epochs} batch={args.batch} lr={args.lr} loss=CE+Dice",
                      evaluate.parser_metrics(cm), train_time, ms,
                      f"{device}; " + ("NOT PRETRAINED; " if args.no_pretrained else "") + cnn.loss_note(losses) + "; per-class IoU: " + evaluate.per_class_text(cm))


if __name__ == "__main__":
    main()
