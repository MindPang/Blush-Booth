"""One training loop for all the CNN experiments, so they all get the same optimizer and schedule:
AdamW (weight decay 1e-4) + OneCycle (10% warm-up, then cosine down), batch 16, lr 1e-3, 4 epochs."""
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


def to_input(images, norm="minus1_1"):
    """uint8 BGR images (B, H, W, 3) (numpy or tensor) -> float tensor (B, 3, H, W), RGB.
    norm="minus1_1": like the old project ([-1, 1]).  norm="imagenet": for the pretrained models."""
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    rgb = images[..., ::-1].astype(np.float32) / 255.0
    if norm == "imagenet":
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    else:
        rgb = rgb * 2.0 - 1.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))


def worker_init(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)


def train_cnn(model, dataset, loss_fn, args, device):
    """Train `model` for args.epochs. loss_fn(model, batch, device) returns the loss.
    Returns (list of mean train loss per epoch, training time in seconds)."""
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        drop_last=len(dataset) > args.batch, worker_init_fn=worker_init)
    steps = max(len(loader) * args.epochs, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)

    model.to(device)
    losses = []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        count = 0
        for batch in loader:
            loss = loss_fn(model, batch, device)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
            count += 1
        losses.append(total / max(count, 1))
        print(f"  epoch {epoch + 1}/{args.epochs}  train loss {losses[-1]:.4f}  "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
    model.eval()
    return losses, time.perf_counter() - start


def loss_note(losses):
    return "train loss " + " > ".join(f"{v:.3f}" for v in losses)
