import torch
import torch.nn as nn
import torch.nn.functional as F

from .layout import N_CLASSES, N_POINTS

HEATMAP_BIAS = -2.19  # sigmoid(-2.19) is about 0.1 (section 5.4)


def conv_bn_relu(cin, cout, k=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    """Residual block: conv3x3 -> BN -> ReLU -> conv3x3 -> BN, add the shortcut, then ReLU."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)), inplace=True)
        y = self.bn2(self.conv2(y))
        if self.short is None:
            shortcut = x
        else:
            shortcut = self.short(x)
        return F.relu(y + shortcut, inplace=True)


def stage(cin, cout, n, stride=2):
    """n ResBlocks in a row. The first one changes the size/channels."""
    blocks = [ResBlock(cin, cout, stride)]
    for _ in range(n - 1):
        blocks.append(ResBlock(cout, cout))
    return nn.Sequential(*blocks)


class DetectorNet(nn.Module):
    """CenterNet-style detector for one face, stride 8, 5 output channels (section 5.3/5.4)."""

    STRIDE = 8

    def __init__(self):
        super().__init__()
        self.stem = conv_bn_relu(3, 24, 3, 2)
        self.s2 = stage(24, 32, 1)
        self.s3 = stage(32, 64, 2)
        self.s4 = stage(64, 96, 2)
        self.s5 = stage(96, 160, 2)
        self.lat3 = nn.Conv2d(64, 64, 1)
        self.lat4 = nn.Conv2d(96, 64, 1)
        self.lat5 = nn.Conv2d(160, 64, 1)
        self.head = nn.Sequential(conv_bn_relu(64, 64), conv_bn_relu(64, 64), nn.Conv2d(64, 5, 1))
        with torch.no_grad():
            self.head[-1].bias.zero_()
            self.head[-1].bias[0] = HEATMAP_BIAS

    def forward(self, x):
        c3 = self.s3(self.s2(self.stem(x)))
        c4 = self.s4(c3)
        c5 = self.s5(c4)
        p = self.lat5(c5)
        p = self.lat4(c4) + F.interpolate(p, size=c4.shape[-2:], mode="nearest")
        p = self.lat3(c3) + F.interpolate(p, size=c3.shape[-2:], mode="nearest")
        return self.head(p)


def soft_argmax(heatmaps):
    """Turn (B, K, H, W) heatmap logits into (B, K, 2) points (x, y) in [0, 1]."""
    b, k, h, w = heatmaps.shape
    prob = F.softmax(heatmaps.reshape(b, k, h * w), dim=-1).reshape(b, k, h, w)
    xs = (torch.arange(w, device=heatmaps.device, dtype=prob.dtype) + 0.5) / w
    ys = (torch.arange(h, device=heatmaps.device, dtype=prob.dtype) + 0.5) / h
    x = (prob.sum(2) * xs).sum(-1)
    y = (prob.sum(3) * ys).sum(-1)
    return torch.stack([x, y], dim=-1)


def upsample_and_concat(x, skip):
    x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
    return torch.cat([x, skip], 1)


class LandmarkNet(nn.Module):
    """Small U-shaped network -> 106 heatmaps at 1/4 size -> soft-argmax (section 6.3)."""

    def __init__(self, n_points=N_POINTS):
        super().__init__()
        self.stem = nn.Sequential(conv_bn_relu(3, 32, 3, 2), ResBlock(32, 32))
        self.e1 = stage(32, 64, 2)
        self.e2 = stage(64, 128, 2)
        self.e3 = stage(128, 192, 2)
        self.e4 = stage(192, 256, 2)
        self.d3 = ResBlock(256 + 192, 160)
        self.d2 = ResBlock(160 + 128, 128)
        self.d1 = ResBlock(128 + 64, 96)
        self.head = nn.Conv2d(96, n_points, 1)

    def heatmaps(self, x):
        s = self.stem(x)
        e1 = self.e1(s)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        d = self.d3(upsample_and_concat(e4, e3))
        d = self.d2(upsample_and_concat(d, e2))
        d = self.d1(upsample_and_concat(d, e1))
        return self.head(d)

    def forward(self, x):
        return soft_argmax(self.heatmaps(x))


class ConvBlock(nn.Sequential):
    """(conv3x3 -> BN -> ReLU) two times."""

    def __init__(self, cin, cout):
        super().__init__(conv_bn_relu(cin, cout), conv_bn_relu(cout, cout))


class UNet(nn.Module):
    """U-Net face parser with base width 32, about 5.1 M parameters (section 7.2)."""

    def __init__(self, n_classes=N_CLASSES, base=32):
        super().__init__()
        channels = [base, 2 * base, 4 * base, 8 * base, 8 * base]

        self.enc = nn.ModuleList()
        cin = 3
        for c in channels:
            self.enc.append(ConvBlock(cin, c))
            cin = c

        self.ups = nn.ModuleList()
        self.dec = nn.ModuleList()
        for skip_c in reversed(channels[:-1]):
            self.ups.append(nn.ConvTranspose2d(cin, skip_c, 2, 2))
            self.dec.append(ConvBlock(2 * skip_c, skip_c))
            cin = skip_c

        self.head = nn.Conv2d(cin, n_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)

        for up, dec, skip in zip(self.ups, self.dec, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], 1))
        return self.head(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def build_model(kind, config=None):
    if config is None:
        config = {}
    if kind == "detector":
        return DetectorNet()
    if kind == "landmarks":
        return LandmarkNet()
    if kind == "parser":
        return UNet(base=int(config.get("base", 32)))
    raise ValueError(kind)


def save_checkpoint(path, model, kind, config=None, extra=None):
    """Save the weights plus a small dict that says how to rebuild the model.
    Same format as the old project, so the old runs/*/best.pt files still load."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"kind": kind, "config": config or {}, "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
    if extra:
        ckpt["extra"] = extra
    torch.save(ckpt, path)


def load_checkpoint(path, device="cpu"):
    """Load a DetectorNet / LandmarkNet / UNet checkpoint (old or new)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["kind"], ckpt.get("config", {}))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


def pick_device(name="auto"):
    """GPU if there is one (CUDA or Apple MPS), otherwise CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
