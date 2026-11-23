import cv2
import numpy as np

from .layout import REGIONS

# BGR colour for each class:
# background, skin, l brow, r brow, l eye, r eye, nose, upper lip, mouth, lower lip, hair
PALETTE = np.array([
    [0, 0, 0], [128, 192, 255], [0, 128, 0], [0, 255, 0], [255, 0, 0], [255, 128, 128],
    [0, 255, 255], [0, 0, 255], [128, 0, 128], [0, 0, 160], [64, 64, 128]], np.uint8)

REGION_COLOURS = {
    "contour": (255, 255, 0),
    "brows": (0, 255, 0),
    "nose": (0, 255, 255),
    "eyes": (255, 0, 0),
    "lips": (0, 0, 255),
}


def colourise(label):
    """Turn a label map into a colour picture."""
    return PALETTE[np.clip(label, 0, len(PALETTE) - 1)]


def overlay_label(img, label, alpha=0.5):
    return cv2.addWeighted(img, 1 - alpha, colourise(label), alpha, 0)


def draw_points(img, pts, numbers=False, colour=None, r=1):
    """Draw the 106 points, each face part in its own colour (or all in `colour`)."""
    out = img.copy()
    for region, idx in REGIONS.items():
        c = colour or REGION_COLOURS[region]
        for i in idx:
            x, y = pts[i]
            cv2.circle(out, (int(round(x)), int(round(y))), r, c, -1, cv2.LINE_AA)
            if numbers:
                cv2.putText(out, str(i), (int(x) + 2, int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, c, 1, cv2.LINE_AA)
    return out


def draw_box(img, box, colour=(0, 255, 0), thickness=1, text=None):
    out = img.copy()
    x0, y0, x1, y1 = [int(round(v)) for v in box]
    cv2.rectangle(out, (x0, y0), (x1, y1), colour, thickness, cv2.LINE_AA)
    if text:
        cv2.putText(out, text, (x0, max(10, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
    return out


def grid(images, cols=4, pad=2):
    """Put a list of images (gray or BGR) next to each other in a grid."""
    if not images:
        return np.zeros((8, 8, 3), np.uint8)

    imgs = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        imgs.append(im)

    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, np.uint8)
    for k, im in enumerate(imgs):
        r, c = divmod(k, cols)
        if im.shape[:2] != (h, w):
            im = cv2.resize(im, (w, h))
        top = pad + r * (h + pad)
        left = pad + c * (w + pad)
        canvas[top:top + h, left:left + w] = im
    return canvas


def tensor_to_bgr(t):
    """Opposite of geometry.to_tensor, so I can draw a tensor."""
    a = t.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip((a + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return img[..., ::-1].copy()
