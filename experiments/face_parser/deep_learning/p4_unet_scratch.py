"""P4: the old U-Net face parser (base width 32, about 5.1 M parameters), trained from scratch.

Encoder: 5 levels of (conv3x3-BN-ReLU) x 2 with max-pool between them. Decoder: transposed conv up,
concatenate the skip connection, (conv3x3-BN-ReLU) x 2. Last 1x1 conv gives 11 class scores per pixel.
Loss: cross entropy + soft Dice (same as the old project). Crop 256 x 256.

    python experiments/face_parser/deep_learning/p4_unet_scratch.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from experiments.common import cnn, data, evaluate, results  # noqa: E402
from facekit.losses import ce_dice_loss  # noqa: E402
from facekit.models import UNet, count_params, load_checkpoint, pick_device, save_checkpoint  # noqa: E402

EXP_ID = "P4"


def load_model(path, device="cpu"):
    return load_checkpoint(path, device)[0]


@torch.no_grad()
def parse_crop(model, crop):
    device = next(model.parameters()).device
    return model(cnn.to_input(crop).to(device))[0].float().cpu().numpy()


def loss_fn(model, batch, device):
    logits = model(cnn.to_input(batch["image"]).to(device))
    loss, _ = ce_dice_loss(logits, batch["label"].to(device))
    return loss


def main():
    args = data.get_args("P4 U-Net from scratch")
    data.set_seed(args.seed)
    device = pick_device(args.device)

    model = UNet(base=32)
    print(f"UNet: {count_params(model) / 1e6:.2f} M parameters, device {device}")
    losses, train_time = cnn.train_cnn(model, data.ParserTrainSet(args), loss_fn, args, device)
    save_checkpoint(data.model_path(args, EXP_ID, "unet", ".pt"), model, "parser",
                    {"base": 32, "size": data.PAR_SIZE, "epochs": args.epochs})

    crops, labels = data.parser_eval_crops(args)
    cm, ms = evaluate.run_parser(lambda c: parse_crop(model, c), crops, labels)
    results.write_row("parser", args, EXP_ID, "deep learning", "existing model, from scratch", "unet",
                      f"base=32 epochs={args.epochs} batch={args.batch} lr={args.lr} loss=CE+Dice",
                      evaluate.parser_metrics(cm), train_time, ms,
                      f"{device}; " + cnn.loss_note(losses) + "; per-class IoU: " + evaluate.per_class_text(cm))


if __name__ == "__main__":
    main()
