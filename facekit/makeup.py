import copy
import json
import re
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .geometry import box_iou, face_box, invert, warp


LAYERS = ["lips", "blush", "shadow", "liner", "lashes", "skin", "lens", "brows", "teeth", "hair",
          "mustache", "beard"]
KEYS = {str((i + 1) % 10): layer for i, layer in enumerate(LAYERS[:10])}

OVERLAYS = ("mustache", "beard")
REMOVED_LAYERS = ("wig",)  # X48: removed, so it gets dropped from old saved looks

ORDER = ["hair", "skin", "blush", "shadow", "brows", "lens", "teeth", "lips", "liner", "lashes",
         "beard", "mustache"]

STYLES = {
    "lips": ["full", "ombre"],
    "blush": ["cheek", "high"],
    "shadow": ["lid", "lid+lower"],
    "liner": ["line", "wing"],
    "lashes": ["natural", "volume", "cat eye", "doll", "wispy"],
    "mustache": ["classic", "pencil", "handlebar"],
    "beard": ["full", "goatee", "chin strap", "stubble"],
}

# Lash shapes: (how many lashes, base length as a fraction of face width, length(p), lean(p)).
# p goes from 0 at the inner corner to 1 at the outer corner. The count and base length come
# from 12.4, the length / lean curves are my own choice.
LASH_SHAPES = {
    "natural": (14, 0.026, lambda p: 0.6 + 0.7 * p, lambda p: 10 + 30 * p),
    "volume": (20, 0.032, lambda p: 0.6 + 0.7 * p, lambda p: 10 + 30 * p),
    "cat eye": (18, 0.032, lambda p: 0.4 + 1.1 * p ** 1.5, lambda p: 15 + 45 * p),
    "doll": (18, 0.034, lambda p: 0.6 + 0.6 * np.sin(np.pi * p), lambda p: 10 * (p - 0.5)),
    "wispy": (24, 0.030, lambda p: 0.6 + 0.4 * p, lambda p: 10 + 25 * p),
}

EXTRA_KEYS = {
    "lips": "gloss",
    "shadow": "shimmer",
    "skin": "smooth",   # skin smoothing, has its own strength (X42)
    "hair": "lift",
}

# colour swatches (BGR). The names are shown in the UI.
PRESETS = {
    "lips": {"berry": (95, 55, 170), "coral-rose": (110, 100, 225), "red": (40, 30, 190), "nude": (120, 135, 190),
             "plum": (90, 40, 120), "pink": (150, 110, 230)},
    "blush": {"rosy pink": (150, 120, 230), "light pink": (180, 160, 240), "peach": (130, 160, 245),
              "coral": (110, 130, 240)},
    "shadow": {"pink": (170, 140, 225), "peach-pink": (150, 160, 235), "brown": (60, 90, 140),
               "bronze": (50, 110, 170), "plum": (110, 60, 120), "grey": (110, 110, 110)},
    "liner": {"dark": (25, 20, 20), "brown": (30, 50, 80), "navy": (80, 30, 20)},
    "lashes": {"black": (15, 15, 15), "dark brown": (25, 35, 55), "brown": (40, 60, 90)},
    "skin": {"neutral": (170, 190, 225), "light": (190, 205, 235), "warm": (150, 180, 220), "tan": (120, 155, 200)},
    "lens": {"light honey brown": (60, 130, 190), "grey": (150, 150, 140), "blue": (190, 140, 80),
             "green": (90, 150, 90), "dark brown": (30, 50, 90)},
    "brows": {"dark brown": (30, 40, 60), "black brown": (20, 25, 35), "soft brown": (60, 80, 110)},
    "teeth": {"white": (245, 245, 245), "natural white": (225, 235, 240), "bright": (255, 255, 255)},
    "hair": {"brown": (40, 70, 120), "burgundy": (60, 30, 120), "ash": (120, 120, 110), "blonde": (110, 190, 230),
             "black": (20, 20, 20), "pink": (180, 120, 230)},
    "mustache": {"black": (22, 22, 26), "dark brown": (30, 40, 60), "brown": (45, 70, 110),
                 "blonde": (105, 165, 205), "grey": (150, 150, 150), "ginger": (40, 90, 170)},
    "beard": {"black": (22, 22, 26), "dark brown": (30, 40, 60), "brown": (45, 70, 110),
              "blonde": (105, 165, 205), "grey": (150, 150, 150), "ginger": (40, 90, 170)},
}


def make_layer(color, k, blur, style=None, **extra):
    """Settings for one layer: colour, strength k, blur size, and maybe a style and extras."""
    layer = {"color": tuple(color), "k": float(k), "blur": int(blur)}
    if style is not None:
        layer["style"] = style
    layer.update(extra)
    return layer


P = PRESETS

# The built-in looks (12.5).
# The user asked for lower k on hair, blush and lens so it looks more natural (decision X41).
# project.md 12.5 had hair off, blush 0.28 / 0.15, lens 0 / 0.75.
LOOKS = {
    "soft pink": {
        "lips": make_layer(P["lips"]["berry"], 0.60, 7, "ombre", gloss=0.3),
        "blush": make_layer(P["blush"]["rosy pink"], 0.14, 41, "high"),
        "shadow": make_layer(P["shadow"]["pink"], 0.35, 21, "lid+lower", shimmer=0.2),
        "liner": make_layer(P["liner"]["dark"], 0.65, 1, "wing"),
        "lashes": make_layer(P["lashes"]["black"], 0.70, 1, "volume"),
        "skin": make_layer(P["skin"]["neutral"], 0.40, 15, smooth=0.40),
        "lens": make_layer(P["lens"]["light honey brown"], 0.20, 3),
        "brows": make_layer(P["brows"]["dark brown"], 0.30, 5),
        "teeth": make_layer(P["teeth"]["white"], 0.60, 3),
        "hair": make_layer(P["hair"]["brown"], 0.29, 11, lift=0.8),
        "mustache": make_layer(P["mustache"]["dark brown"], 0.0, 3, "classic"),  # off by default (X46)
        "beard": make_layer(P["beard"]["dark brown"], 0.0, 3, "full"),
    },
    "natural": {
        "lips": make_layer(P["lips"]["coral-rose"], 0.55, 7, "ombre", gloss=0.5),
        "blush": make_layer(P["blush"]["light pink"], 0.14, 41, "high"),
        "shadow": make_layer(P["shadow"]["peach-pink"], 0.25, 21, "lid+lower", shimmer=0.25),
        "liner": make_layer(P["liner"]["brown"], 0.35, 1, "line"),
        "lashes": make_layer(P["lashes"]["dark brown"], 0.55, 1, "natural"),
        "skin": make_layer(P["skin"]["light"], 0.70, 15, smooth=0.70),
        "lens": make_layer(P["lens"]["light honey brown"], 0.20, 3),
        "brows": make_layer(P["brows"]["soft brown"], 0.25, 5),
        "teeth": make_layer(P["teeth"]["white"], 0.55, 3),
        "hair": make_layer(P["hair"]["brown"], 0.29, 11, lift=0.8),
        "mustache": make_layer(P["mustache"]["dark brown"], 0.0, 3, "classic"),
        "beard": make_layer(P["beard"]["dark brown"], 0.0, 3, "full"),
    },
}
BUILTIN_LOOKS = tuple(LOOKS)

# The k a layer gets when you turn it on from k = 0. 13.1 says 0.6, but hair, blush and
# lens use the lower values the user picked (X41).
ON_K = {"hair": 0.29, "blush": 0.14, "lens": 0.20, "mustache": 0.90, "beard": 0.85}

EYE = {
    "R": dict(upper=[66, 67, 68, 69, 70], lower=[70, 71, 72, 73, 66], outer=66, inner=70, pupil=104,
              brow_lower=[33, 41, 40, 39, 38, 37], brow=list(range(33, 42))),
    "L": dict(upper=[75, 76, 77, 78, 79], lower=[79, 80, 81, 82, 75], outer=79, inner=75, pupil=105,
              brow_lower=[42, 50, 49, 48, 47, 46], brow=list(range(42, 51))),
}
LIP_CLASSES = (7, 9)
SKIN_CLASSES = (1, 6)
BROW_CLASSES = (2, 3)
MOUTH_CLASS = 8
HAIR_CLASS = 10


def face_width(pts):
    """Distance between the two ends of the jaw line (points 0 and 32)."""
    return float(np.linalg.norm(pts[0] - pts[32]))


def odd(n):
    """Round to an odd number (at least 1), for blur kernel sizes."""
    return max(1, int(round(n)) | 1)


def soft_alpha(mask, blur, fw):
    """Blur the mask edge so the colour fades out (12.2)."""
    alpha = mask.astype(np.float32)
    size = odd(blur * fw / 220.0)
    if size > 1:
        return cv2.GaussianBlur(alpha, (size, size), 0)
    return alpha


def blend(img, coloured, alpha, k):
    """12.2: result = original * (1 - alpha * k) + coloured * (alpha * k)."""
    a = (alpha * k)[..., None]
    return img * (1 - a) + coloured * a


def poly_mask(shape, poly, value=1):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(np.asarray(poly)).astype(np.int32)], value)
    return mask.astype(bool)


def lab_of(bgr):
    """LAB value of one BGR colour."""
    return cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def to_lab(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


def from_lab(lab):
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def recolour_lab(img, mask, color, keep_ab=0.3, l_mix=0.0):
    """Change the colour in LAB."""
    lab = to_lab(img)
    target = lab_of(color)
    mask = mask.astype(bool)
    if mask.any():
        mean_ab = lab[..., 1:][mask].mean(0)
        lab[..., 1:] = target[1:] + keep_ab * (lab[..., 1:] - mean_ab)
    lab[..., 0] = (1 - l_mix) * lab[..., 0] + l_mix * target[0]
    return from_lab(lab)


def eye_polygon(pts, side):
    eye = EYE[side]
    return pts[eye["upper"] + eye["lower"][1:-1]]


def brow_y_at(pts, side, xs):
    """y of the bottom edge of the brow at the given x positions."""
    brow = pts[EYE[side]["brow_lower"]]
    order = np.argsort(brow[:, 0])
    return np.interp(xs, brow[order, 0], brow[order, 1])


def lips_mask(pts, labels, shape, use_unet):
    if use_unet and labels is not None:
        return np.isin(labels, LIP_CLASSES)
    return poly_mask(shape, pts[84:96]) & ~poly_mask(shape, pts[96:104])


def blush_mask(pts, labels, shape, style, use_unet):
    fw = face_width(pts)
    mask = np.zeros(shape[:2], np.uint8)
    eye_mid = (pts[74] + pts[83]) / 2
    for jaw_i, nose_i, eye_i in ((4, 56, 74), (28, 62, 83)):
        centre = 0.5 * pts[jaw_i] + 0.5 * pts[nose_i]
        axes = np.array([0.13, 0.085]) * fw
        if style == "high":
            outward = np.sign(centre[0] - eye_mid[0])
            centre = centre + np.array([0.03 * fw * outward, 0.0]) + 0.25 * (pts[eye_i] - centre)
            axes = axes * 1.25
        eye_line = pts[83] - pts[74]
        angle = np.degrees(np.arctan2(eye_line[1], eye_line[0]))
        cv2.ellipse(mask, (int(centre[0]), int(centre[1])), (int(axes[0]), int(axes[1])), angle, 0, 360, 1, -1)

    if use_unet and labels is not None:
        skin = np.isin(labels, (1,))
    else:
        skin = poly_mask(shape, pts[0:33])
    return mask.astype(bool) & skin


def shadow_mask(pts, shape, style):
    fw = face_width(pts)
    mask = np.zeros(shape[:2], bool)
    for side in ("R", "L"):
        upper_lid = pts[EYE[side]["upper"]]
        brow_y = brow_y_at(pts, side, upper_lid[:, 0])
        top = upper_lid.copy()
        top[:, 1] = upper_lid[:, 1] + 0.35 * (brow_y - upper_lid[:, 1])
        mask |= poly_mask(shape, np.vstack([upper_lid, top[::-1]]))
        if style == "lid+lower":
            band = np.zeros(shape[:2], np.uint8)
            thickness = max(1, int(round(0.018 * fw)))
            cv2.polylines(band, [np.round(pts[EYE[side]["lower"]]).astype(np.int32)], False, 1,
                          thickness, cv2.LINE_AA)
            mask |= band.astype(bool)
        # never colour the eye itself
        mask &= ~poly_mask(shape, eye_polygon(pts, side))
    return mask


def liner_mask(pts, shape, style):
    fw = face_width(pts)
    mask = np.zeros(shape[:2], np.uint8)
    thickness = max(1, int(round(0.008 * fw)))
    for side in ("R", "L"):
        eye = EYE[side]
        cv2.polylines(mask, [np.round(pts[eye["upper"]]).astype(np.int32)], False, 1, thickness, cv2.LINE_AA)
        if style == "wing":
            corner = pts[eye["outer"]]
            inner = pts[eye["inner"]]
            outward = (corner - inner) / max(np.linalg.norm(corner - inner), 1e-6)
            up = np.array([outward[1], -outward[0]])
            if np.dot(up, pts[eye["brow"]].mean(0) - corner) < 0:
                up = -up
            tip = corner + outward * 0.055 * fw + up * 0.03 * fw
            if side == "R":
                near = pts[eye["upper"][1]]
            else:
                near = pts[eye["upper"][-2]]
            cv2.fillPoly(mask, [np.round(np.array([corner, tip, near])).astype(np.int32)], 1)
    return mask.astype(bool)


def resample_line(poly, n):
    """n points spread evenly along a polyline, plus where they are (0..1) along it."""
    seg_len = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    dist = np.r_[0, np.cumsum(seg_len)]
    targets = (np.arange(n) + 0.5) / n * dist[-1]
    points = np.c_[np.interp(targets, dist, poly[:, 0]), np.interp(targets, dist, poly[:, 1])]
    return points, targets / max(dist[-1], 1e-6)


def lashes_mask(pts, shape, style):
    fw = face_width(pts)
    count, base, length_fn, lean_fn = LASH_SHAPES[style]
    mask = np.zeros(shape[:2], np.uint8)
    thickness = max(1, int(round(0.004 * fw)))
    for side in ("R", "L"):
        eye = EYE[side]
        lid = pts[eye["upper"]]
        if np.linalg.norm(lid[0] - pts[eye["inner"]]) > 1e-6:
            lid = lid[::-1]
        roots, positions = resample_line(lid, count)

        tangent = np.gradient(roots, axis=0)
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-6)
        normal = np.c_[tangent[:, 1], -tangent[:, 0]]
        if np.mean(normal[:, 1]) > 0:  # image y goes down, so up means negative y
            normal = -normal
        outward = pts[eye["outer"]] - pts[eye["inner"]]
        outward /= max(np.linalg.norm(outward), 1e-6)

        for i in range(len(roots)):
            root = roots[i]
            p = positions[i]
            length = base * fw * length_fn(p)
            if style == "wispy" and i % 4 == 3:
                length *= 1.7
            lean = np.radians(lean_fn(p))
            direction = np.cos(lean) * normal[i] + np.sin(lean) * outward
            direction /= max(np.linalg.norm(direction), 1e-6)
            tip = root + direction * length
            cv2.line(mask, tuple(np.round(root).astype(int)), tuple(np.round(tip).astype(int)), 1,
                     thickness, cv2.LINE_AA)
    return mask.astype(bool)


def skin_mask(pts, labels, shape, use_unet):
    if use_unet and labels is not None:
        return np.isin(labels, SKIN_CLASSES)
    face = poly_mask(shape, np.vstack([pts[0:33], pts[46:41:-1], pts[37:32:-1]]))
    for side in ("R", "L"):
        face &= ~poly_mask(shape, eye_polygon(pts, side))
        brow_hull = cv2.convexHull(pts[EYE[side]["brow"]].astype(np.float32)).reshape(-1, 2)
        face &= ~poly_mask(shape, brow_hull)
    return face & ~poly_mask(shape, pts[84:96])


def lens_masks(pts, shape):
    """Returns (iris without the pupil, darker outer ring), both cut by the eyelids."""
    iris = np.zeros(shape[:2], np.uint8)
    ring = np.zeros(shape[:2], np.uint8)
    for side in ("R", "L"):
        eye = EYE[side]
        centre = tuple(np.round(pts[eye["pupil"]]).astype(int))
        radius = 0.22 * np.linalg.norm(pts[eye["outer"]] - pts[eye["inner"]])

        iris_one = np.zeros(shape[:2], np.uint8)
        cv2.circle(iris_one, centre, max(1, int(round(radius))), 1, -1, cv2.LINE_AA)
        cv2.circle(iris_one, centre, max(1, int(round(0.4 * radius))), 0, -1, cv2.LINE_AA)

        ring_one = np.zeros(shape[:2], np.uint8)
        cv2.circle(ring_one, centre, max(1, int(round(radius))), 1, -1, cv2.LINE_AA)
        cv2.circle(ring_one, centre, max(1, int(round(0.8 * radius))), 0, -1, cv2.LINE_AA)

        eye_area = poly_mask(shape, eye_polygon(pts, side))
        iris |= (iris_one.astype(bool) & eye_area).astype(np.uint8)
        ring |= (ring_one.astype(bool) & eye_area).astype(np.uint8)
    return iris.astype(bool), ring.astype(bool)


# (low, high): no lens below low (closed eye), full lens above high. I picked these by eye on val (X54).
LENS_OPEN = (0.45, 0.60)


def iris_visible(pts, side):
    """How much of the lens circle is not hidden by the eyelids."""
    eye = EYE[side]
    radius = 0.22 * float(np.linalg.norm(pts[eye["outer"]] - pts[eye["inner"]]))
    if radius < 0.5:
        return 0.0
    scale = 4.0
    centre = pts[eye["pupil"]]
    origin = centre - 1.2 * radius
    n = int(np.ceil(2.4 * radius * scale))
    disc = np.zeros((n, n), np.uint8)
    eye_area = np.zeros((n, n), np.uint8)
    cv2.circle(disc, tuple(np.round((centre - origin) * scale).astype(int)), int(round(radius * scale)), 1, -1)
    cv2.fillPoly(eye_area, [np.round((eye_polygon(pts, side) - origin) * scale).astype(np.int32)], 1)
    return float((disc & eye_area).sum() / max(int(disc.sum()), 1))


def fade_between(value, low, high):
    """0 below low, 1 above high, a straight line in between."""
    return float(np.clip((value - low) / max(high - low, 1e-6), 0, 1))


def lens_open(pts):
    """{'R': w, 'L': w}: 1 = eye open enough to show the lens, 0 = closed (X54)."""
    if LENS_OPEN is None:
        return {"R": 1.0, "L": 1.0}
    low, high = LENS_OPEN
    result = {}
    for side in ("R", "L"):
        result[side] = fade_between(iris_visible(pts, side), low, high)
    return result


def brows_mask(pts, labels, shape, use_unet):
    if use_unet and labels is not None:
        return np.isin(labels, BROW_CLASSES)
    mask = np.zeros(shape[:2], bool)
    for side in ("R", "L"):
        brow_hull = cv2.convexHull(pts[EYE[side]["brow"]].astype(np.float32)).reshape(-1, 2)
        mask |= poly_mask(shape, brow_hull)
    return mask


def teeth_mask(img, pts, labels, use_unet):
    """Teeth = bright and not red pixels inside the mouth (Otsu threshold on L)."""
    if use_unet and labels is not None:
        mouth = labels == MOUTH_CLASS
    else:
        mouth = poly_mask(img.shape, pts[96:104])
    if mouth.sum() < 20:
        return np.zeros(img.shape[:2], bool)
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    a = lab[..., 1]
    threshold, _ = cv2.threshold(L[mouth].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mouth & (L > threshold) & (a < 145)


#
# How the mustache works:
# I draw it once as a template in a "mouth frame" where the mouth corners 84 / 90 are at
# (-1, 0) and (1, 0) and y goes down. Then every frame it gets moved onto the face, so it
# follows the tilt and size of the face.
#
# In the LaPa mean shape (300 train photos) the bottom of the nose (point 60) is at
# y = -0.76 in this frame, and the top of the upper lip is about -0.23 in the middle.
# The frame height follows the real nose-to-mouth gap but clamped to 0.6-1.4 x the
# average gap. Without the clamp, when the head is bowed or turned the nose can be level
# with the mouth corners and the mustache got squashed flat (saw this on a test photo, X48).
#
# The hair texture is lots of thin strands growing down and outwards, darker at the edges.
#
# X48 (user): the mustache was too close to the lip, so the whole template is drawn
# MUSTACHE_LIFT units higher (0.10 = 5% of the mouth width).
MUSTACHE_LIFT = 0.10
NOSE_GAP = 0.76


def mustache_affine(pts):
    """2x3 matrix from mouth-frame units to the image."""
    centre = (pts[84] + pts[90]) / 2
    x_axis = (pts[90] - pts[84]) / 2
    half_width = max(float(np.linalg.norm(x_axis)), 1e-6)
    up = np.array([x_axis[1], -x_axis[0]]) / half_width  # towards the nose on an upright face
    ratio = np.dot(pts[60] - centre, up) / half_width / NOSE_GAP
    ratio = float(np.clip(ratio, 0.6, 1.4))
    down = -up * half_width * ratio
    return np.c_[x_axis, down, centre - MUSTACHE_LIFT * down]


def spline(ctrl, closed=True, n=12):
    """Catmull-Rom curve through the control points, so I get smooth outlines from a few points."""
    pts = np.asarray(ctrl, float)
    if closed:
        pts = np.vstack([pts[-1], pts, pts[0], pts[1]])
    else:
        pts = np.vstack([pts[0], pts, pts[-1]])
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    pieces = []
    for p0, p1, p2, p3 in zip(pts[:-3], pts[1:-2], pts[2:-1], pts[3:]):
        piece = 0.5 * (2 * p1
                       + (p2 - p0) * t
                       + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                       + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)
        pieces.append(piece)
    if not closed:
        pieces.append(pts[-2][None])
    return np.vstack(pieces)


def taper_stroke(curve, w0, w1):
    """Outline of a stroke along `curve` that gets thinner from width w0 to w1 (for mustache tips)."""
    curve = np.asarray(curve, float)
    direction = np.gradient(curve, axis=0)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True).clip(1e-9)
    widths = np.linspace(w0, w1, len(curve))[:, None]
    offset = np.c_[-direction[:, 1], direction[:, 0]] * widths / 2
    return np.vstack([curve + offset, (curve - offset)[::-1]])


class Canvas:
    """An image that covers x in [x0, x1], y in [y0, y1] (template units), with S pixels per unit."""

    def __init__(self, x0, x1, y0, y1, S):
        self.x0 = x0
        self.y0 = y0
        self.S = S
        self.w = int(round((x1 - x0) * S))
        self.h = int(round((y1 - y0) * S))

    def px(self, poly):
        """Unit coordinates -> pixel coordinates."""
        return np.round((np.asarray(poly, float) - [self.x0, self.y0]) * self.S).astype(np.int32)

    def fill(self, mask, poly, value=1):
        cv2.fillPoly(mask, [self.px(poly)], value, cv2.LINE_AA)


def strand_shade(canvas, mask, flow, n, length, seed):
    """Brightness map (around 1.0) made of n thin hair strands."""
    rng = np.random.RandomState(seed)
    shade = np.ones((canvas.h, canvas.w), np.float32)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return shade

    pick = rng.randint(0, len(xs), n)
    p = np.c_[xs[pick] / canvas.S + canvas.x0, ys[pick] / canvas.S + canvas.y0]

    steps = 8
    path = [p]
    for _ in range(steps):
        p = p + flow(p[:, 0], p[:, 1]) * (length / steps)
        path.append(p)
    path = np.stack(path, 1)

    brightness = rng.choice(np.linspace(0.7, 1.3, 7), n)
    for v in np.unique(brightness):  # draw all strands with the same brightness at once (faster)
        lines = [canvas.px(q) for q in path[brightness == v]]
        cv2.polylines(shade, lines, False, float(v), 1, cv2.LINE_AA)
    return cv2.GaussianBlur(shade, (3, 3), 0)


def finish_texture(canvas, mask, shade, edge, noise_seed):
    """Make the edges darker (looks fuller), add some soft noise and a soft alpha."""
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5) / canvas.S
    shade = shade * (0.6 + 0.4 * np.clip(dist / edge, 0, 1))
    rng = np.random.RandomState(noise_seed)
    noise = cv2.GaussianBlur(rng.normal(0, 1, (canvas.h, canvas.w)).astype(np.float32), (0, 0), 0.15 * canvas.S)
    shade = shade * (1 + 0.08 * noise / max(float(noise.std()), 1e-6))
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)
    return np.dstack([alpha, shade]).astype(np.float32)


def mirror(poly):
    return [(-x, y) for x, y in poly]


@lru_cache(maxsize=None)
def mustache_template(style):
    """Draw one mustache style in the mouth frame (corners at x = -1 and 1)."""
    canvas = Canvas(-2.0, 2.0, -1.2, 0.5, 120)
    mask = np.zeros((canvas.h, canvas.w), np.uint8)

    if style == "classic":
        top = [(-1.22, 0.12), (-1.0, -0.33), (-0.7, -0.64), (-0.3, -0.68), (0, -0.64), (0.3, -0.68),
               (0.7, -0.64), (1.0, -0.33), (1.22, 0.12)]
        bottom = [(1.05, 0.03), (0.8, -0.19), (0.45, -0.31), (0.15, -0.37), (0, -0.34), (-0.15, -0.37),
                  (-0.45, -0.31), (-0.8, -0.19), (-1.05, 0.03)]
        canvas.fill(mask, spline(top, False, 8).tolist() + spline(bottom, False, 8).tolist())
    elif style == "pencil":
        half = [(0.07, -0.50), (0.40, -0.45), (0.75, -0.28), (0.97, -0.08), (0.95, -0.02), (0.72, -0.18),
                (0.40, -0.32), (0.07, -0.38)]
        canvas.fill(mask, half)
        canvas.fill(mask, mirror(half))
    elif style == "handlebar":
        top = [(-0.95, -0.40), (-0.6, -0.58), (-0.25, -0.64), (0, -0.60), (0.25, -0.64), (0.6, -0.58),
               (0.95, -0.40)]
        bottom = [(0.95, -0.15), (0.6, -0.27), (0.3, -0.35), (0, -0.33), (-0.3, -0.35), (-0.6, -0.27),
                  (-0.95, -0.15)]
        canvas.fill(mask, spline(top, False, 8).tolist() + spline(bottom, False, 8).tolist())
        tip = spline([(0.80, -0.30), (1.15, -0.22), (1.45, -0.34), (1.62, -0.58), (1.52, -0.78),
                      (1.38, -0.74)], False, 10)
        canvas.fill(mask, taper_stroke(tip, 0.30, 0.05))
        canvas.fill(mask, mirror(taper_stroke(tip, 0.30, 0.05)))
    else:
        raise ValueError(style)

    def flow(x, y):
        d = np.c_[0.9 * x, np.ones_like(x)]
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    shade = strand_shade(canvas, mask, flow, 2500, 0.16, seed=21)
    return finish_texture(canvas, mask, shade, 0.10, 22), canvas


def place_template(tpl, unit_matrix, shape):
    """Warp a template (alpha, shade) onto the image."""
    both, canvas = tpl
    to_units = np.array([[1 / canvas.S, 0, canvas.x0], [0, 1 / canvas.S, canvas.y0], [0, 0, 1]])
    matrix = unit_matrix @ to_units
    corners = np.c_[[0, canvas.w, 0, canvas.w], [0, 0, canvas.h, canvas.h], np.ones(4)] @ matrix.T
    xa, ya = np.maximum(np.floor(corners.min(0)).astype(int), 0)
    xb, yb = np.minimum(np.ceil(corners.max(0)).astype(int) + 1, [shape[1], shape[0]])
    if xb <= xa or yb <= ya:
        return None
    matrix[:, 2] -= [xa, ya]
    both = warp(both, matrix, (int(xb - xa), int(yb - ya)), value=0, antialias=True)
    return both[..., 0], both[..., 1], (int(ya), int(yb), int(xa), int(xb))


LIP_LABELS = (7, 8, 9)
# (low, high) from the val study X50: high = 1st percentile on clean val, low = high / 2
LIP_SEEN = (0.324, 0.647)


def lip_fraction(pts, labels):
    """How much of the outer lip outline (84-95) the parser labels as lips or mouth."""
    if labels is None:
        return None
    mask = poly_mask(labels.shape, pts[84:96])
    if not mask.any():
        return 0.0
    return float(np.isin(labels[mask], LIP_LABELS).mean())


def lip_visibility(pts, labels):
    """1 = lips are seen (show the mustache), 0 = not seen (hide it), fades in between (X50)."""
    frac = lip_fraction(pts, labels)
    if frac is None or LIP_SEEN is None:
        return 1.0
    low, high = LIP_SEEN
    return fade_between(frac, low, high)


EYE_LABELS = (4, 5)
# (low, high): chosen on val in X51 (balanced accuracy; the rule I fixed before gave 0, see X51)
EYE_SEEN = (0.10, 0.20)
EYE_LAYERS = ("shadow", "lens", "liner", "lashes")  # hidden on an eye the parser can't see (X51, X53)


def eye_fraction(pts, labels, side):
    """How much of one eye outline the parser labels as eye."""
    if labels is None:
        return None
    mask = poly_mask(labels.shape, eye_polygon(pts, side))
    if mask.sum() < 4:
        mask = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if not mask.any():
        return 0.0
    return float(np.isin(labels[mask], EYE_LABELS).mean())


def eye_visibility(pts, labels):
    """{'R': v, 'L': v}: 1 = eye seen, 0 = covered, fades in between (X51)."""
    result = {}
    for side in ("R", "L"):
        frac = eye_fraction(pts, labels, side)
        if frac is None or EYE_SEEN is None:
            result[side] = 1.0
        else:
            low, high = EYE_SEEN
            result[side] = fade_between(frac, low, high)
    return result


def side_weights(pts, vis, box):
    """Weight map over the box (ya, yb, xa, xb), one value for each eye."""
    ya, yb, xa, xb = box
    right_pupil = pts[EYE["R"]["pupil"]]
    left_pupil = pts[EYE["L"]["pupil"]]
    axis = left_pupil - right_pupil
    middle = (left_pupil + right_pupil) / 2
    X, Y = np.meshgrid(np.arange(xa, xb), np.arange(ya, yb))
    on_right_side = (X - middle[0]) * axis[0] + (Y - middle[1]) * axis[1] < 0
    return np.where(on_right_side, vis["R"], vis["L"]).astype(np.float32)


def overlay_alpha(name, settings, shape, pts, fw):
    """(alpha 0..1, shade ~1, window) of the mustache over an image of this shape, or None."""
    template = mustache_template(settings.get("style", "classic"))
    result = place_template(template, mustache_affine(pts), shape)
    if result is None:
        return None
    alpha, shade, window = result
    return np.clip(alpha, 0, 1), np.where(alpha > 1e-3, shade, 1.0), window


def scene_light(img, pts):
    """Face brightness / 150 (clipped). In a dark scene the mustache gets darker too, like real hair."""
    box = np.clip(np.round(face_box(pts)).astype(int), 0, [img.shape[1], img.shape[0]] * 2)
    face = img[box[1]:box[3], box[0]:box[2]]
    if face.size == 0:
        return 1.0
    y = cv2.cvtColor(np.clip(face, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)[..., 0]
    y = y[poly_mask(face.shape, pts[0:33] - box[:2])]
    if y.size:
        mean_y = y.mean()
    else:
        mean_y = 150
    return float(np.clip(mean_y / 150, 0.45, 1.15))


def hair_colour(color, shade, light):
    """Colour times the strand shading, plus a bit more contrast so black hair still shows strands."""
    sh = shade[..., None]
    return np.clip((np.asarray(color, np.float32) * sh + 35 * (sh - 1)) * light, 0, 255)


#
# The mustache is one fixed shape, but a beard has to follow the jaw line, so I build
# its outline every frame from the landmarks: the jaw contour (0-32, chin is 16) pushed
# outwards a bit (more at the chin), the mouth corners and the lower lip (91-95).
# The strand texture is cached in a "face frame" (origin = mouth centre, unit = half the
# face width) and warped with the face. The mustache is drawn on top of it.
# Shapes, widths, strand counts and the visibility thresholds are my own choices (not tested on val).
#
# Top of the full beard / stubble / chin strap: this fraction of the way from the mouth
# centre (0) up to the nose bottom (1, point 60). The jaw point at that height is
# interpolated so the top moves smoothly between frames. (User: starting at jaw points
# 2 / 30, about nose level, was too high.)
BEARD_TOP = 0.45
GOATEE_JAW = (11, 21)
BEARD_OK_LABELS = (1, 6, 10)
BEARD_SEEN = (0.35, 0.60)


def face_axes(pts):
    """Returns (unit x along the face (jaw 0 -> 32), unit down (nose -> chin), face width, mouth centre)."""
    x_axis = pts[32] - pts[0]
    fw = max(float(np.linalg.norm(x_axis)), 1e-6)
    x_axis = x_axis / fw
    down = np.array([-x_axis[1], x_axis[0]])
    if np.dot(pts[16] - pts[54], down) < 0:
        down = -down
    mouth_centre = (pts[84] + pts[90]) / 2
    return x_axis, down, fw, mouth_centre


def jaw_normals(pts):
    """Unit normals of the jaw contour 0-32, pointing outwards (away from the nose)."""
    jaw = pts[0:33]
    tangent = np.gradient(jaw, axis=0)
    normals = np.c_[-tangent[:, 1], tangent[:, 0]] / np.linalg.norm(tangent, axis=1, keepdims=True).clip(1e-9)
    pointing_in = np.sum(normals * (jaw - pts[54]), axis=1) < 0
    normals[pointing_in] *= -1
    return normals


def jaw_at(arr, idx):
    """Points (or normals) of the jaw contour 0-32 at in-between (fractional) indices."""
    i = np.arange(33)
    return np.c_[np.interp(idx, i, arr[:, 0]), np.interp(idx, i, arr[:, 1])]


def jaw_index_at(pts, height, side):
    """Fractional jaw index where the jaw first gets down to `height`."""
    _, down, _, centre = face_axes(pts)
    if side == 0:
        idx = np.arange(0, 17)
    else:
        idx = np.arange(32, 15, -1)
    heights = (pts[idx] - centre) @ down
    if heights[0] >= height:
        return float(idx[0])
    for a, b, ha, hb in zip(idx[:-1], idx[1:], heights[:-1], heights[1:]):
        if hb >= height:
            return float(a + (b - a) * (height - ha) / max(hb - ha, 1e-6))
    return 16.0


def beard_outline(pts, style):
    """Closed outline (N x 2, image coords) of one beard style."""
    jaw = pts[0:33]
    normals_all = jaw_normals(pts)
    x_axis, down, fw, centre = face_axes(pts)

    top_height = BEARD_TOP * float((pts[60] - centre) @ down)  # negative = above the mouth
    lo = jaw_index_at(pts, top_height, 0)
    hi = jaw_index_at(pts, top_height, 1)

    if np.linalg.norm(pts[84] - jaw[0]) < np.linalg.norm(pts[90] - jaw[0]):
        corner_a, corner_b = 84, 90
    else:
        corner_a, corner_b = 90, 84
    if corner_b == 90:
        lower_lip = pts[91:96]
    else:
        lower_lip = pts[95:90:-1]
    mouth_w = float(np.linalg.norm(pts[90] - pts[84]))
    under_lip = (lower_lip + down * 0.03 * fw).tolist()

    def near_corner(k, side):
        return pts[k] + side * x_axis * 0.10 * mouth_w + down * 0.04 * mouth_w

    def corner_side(k):
        return np.sign(np.dot(pts[k] - centre, x_axis))

    def along(i0, i1):
        return np.linspace(i0, i1, max(int(abs(i1 - i0) * 2) + 2, 8))

    def normals_at(idx):
        n = jaw_at(normals_all, idx)
        return n / np.linalg.norm(n, axis=1, keepdims=True).clip(1e-9)

    def outer_jaw(i0, i1, base, chin_extra, down_extra):
        idx = along(i0, i1)
        chin_weight = np.clip(1 - np.abs(idx - 16) / 14.0, 0, 1) ** 2
        push_out = normals_at(idx) * (fw * (base + chin_extra * chin_weight))[:, None]
        push_down = down * (fw * down_extra * chin_weight)[:, None]
        return jaw_at(jaw, idx) + push_out + push_down

    if style in ("full", "stubble"):
        if style == "stubble":
            outer = outer_jaw(lo, hi, 0.012, 0.0, 0.0)
        else:
            outer = outer_jaw(lo, hi, 0.02, 0.05, 0.03)

        def cheek(k, side):
            i = jaw_index_at(pts, float((pts[k] - centre) @ down), side)
            return 0.55 * jaw_at(jaw, [i])[0] + 0.45 * pts[k]

        def top_end(i):
            return jaw_at(jaw, [i])[0] - normals_at([i])[0] * 0.025 * fw

        poly = (outer.tolist()
                + [top_end(hi), cheek(corner_b, 1), near_corner(corner_b, corner_side(corner_b))]
                + under_lip
                + [near_corner(corner_a, corner_side(corner_a)), cheek(corner_a, 0), top_end(lo)])
    elif style == "goatee":
        lo, hi = GOATEE_JAW
        outer = outer_jaw(lo, hi, 0.015, 0.035, 0.02)
        poly = (outer.tolist()
                + [near_corner(corner_b, corner_side(corner_b))]
                + under_lip
                + [near_corner(corner_a, corner_side(corner_a))])
    elif style == "chin strap":
        idx = along(lo, hi)
        outer = outer_jaw(lo, hi, 0.012, 0.015, 0.0)
        taper = np.clip(np.minimum(idx - lo, hi - idx) / 2.0, 0.2, 1.0)
        inner = jaw_at(jaw, idx) - normals_at(idx) * (0.055 * fw * taper)[:, None]
        poly = outer.tolist() + inner[::-1].tolist()
    else:
        raise ValueError(style)
    return spline(np.asarray(poly, float), True, 3)


@lru_cache(maxsize=None)
def beard_texture(kind):
    """Beard texture in the face frame (origin = mouth centre, unit = half the face width, y down)."""
    canvas = Canvas(-1.5, 1.5, -1.2, 1.6, 110)
    full = np.ones((canvas.h, canvas.w), np.uint8)
    rng = np.random.RandomState(41)

    def flow(x, y):
        d = np.c_[0.35 * x, np.ones_like(x)]
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    if kind == "strands":
        shade = strand_shade(canvas, full, flow, 9000, 0.10, seed=42)
        cover = np.ones_like(shade)
    elif kind == "stubble":
        dots = strand_shade(canvas, full, flow, 16000, 0.02, seed=43)
        cover = np.clip(np.abs(dots - 1) * 3.0, 0, 1)
        cover = 0.25 + 0.75 * cv2.GaussianBlur(cover, (3, 3), 0)
        shade = 0.85 + 0.15 * rng.uniform(0, 1, (canvas.h, canvas.w)).astype(np.float32)
    else:
        raise ValueError(kind)

    noise = cv2.GaussianBlur(rng.normal(0, 1, (canvas.h, canvas.w)).astype(np.float32), (0, 0), 0.12 * canvas.S)
    shade = shade * (1 + 0.08 * noise / max(float(noise.std()), 1e-6))
    return np.dstack([cover, shade]).astype(np.float32), canvas


def beard_visibility(pts, labels, mask, window, fw):
    """Where the beard is allowed to show."""
    if labels is None:
        return 1.0, 1.0
    ya, yb, xa, xb = window
    inside = poly_mask((yb - ya, xb - xa), pts[0:33] - [xa, ya]).astype(np.uint8)
    e = odd(0.05 * fw)
    inner = cv2.erode(inside, np.ones((e, e), np.uint8)).astype(bool)
    ok = np.isin(labels[ya:yb, xa:xb], BEARD_OK_LABELS)
    allowed = (~inner | ok).astype(np.float32)

    area = inner & mask
    if area.sum() > 20:
        seen = float(ok[area].mean())
    else:
        seen = 1.0
    low, high = BEARD_SEEN
    b = odd(0.02 * fw)
    return cv2.GaussianBlur(allowed, (b, b), 0), float(np.clip((seen - low) / (high - low), 0, 1))


def smoothstep(v):
    q = np.clip(v, 0, 1)
    return q * q * (3 - 2 * q)


def beard_alpha(settings, shape, pts, labels=None):
    """(alpha 0..1, shade ~1, window (ya, yb, xa, xb)) of the beard over an image of this shape, or None."""
    style = settings.get("style", "full")
    poly = beard_outline(pts, style)
    x_axis, down, fw, centre = face_axes(pts)

    pad = int(0.03 * fw) + 3
    xa, ya = np.maximum(np.floor(poly.min(0)).astype(int) - pad, 0)
    xb, yb = np.minimum(np.ceil(poly.max(0)).astype(int) + pad + 1, [shape[1], shape[0]])
    if xb - xa < 3 or yb - ya < 3:
        return None
    window = (int(ya), int(yb), int(xa), int(xb))
    h = yb - ya
    w = xb - xa

    mask_u8 = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask_u8, [np.round(poly - [xa, ya]).astype(np.int32)], 1, cv2.LINE_AA)
    mask = mask_u8.astype(bool)
    if not mask.any():
        return None

    if style == "stubble":
        both, canvas = beard_texture("stubble")
    else:
        both, canvas = beard_texture("strands")
    unit_matrix = np.c_[x_axis * fw / 2, down * fw / 2, centre]  # face-frame units -> image
    matrix = unit_matrix @ np.array([[1 / canvas.S, 0, canvas.x0], [0, 1 / canvas.S, canvas.y0], [0, 0, 1]])
    matrix[:, 2] -= [xa, ya]
    both = warp(both, matrix, (int(w), int(h)), value=0, antialias=True)
    cover = both[..., 0]
    shade = np.where(both[..., 1] > 1e-3, both[..., 1], 1.0)

    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(0.008 * fw, 0.8))
    dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5) / fw
    shade = shade * (0.72 + 0.28 * np.clip(dist / 0.05, 0, 1))
    if style == "stubble":
        alpha = soft * cover
    else:
        shade = 1 + 1.5 * (shade - 1)
        alpha = np.clip(soft + 4.8 * (shade - 1) * soft * (1 - soft), 0, 1) * cover

    X, Y = np.meshgrid(np.arange(xa, xb, dtype=np.float32), np.arange(ya, yb, dtype=np.float32))
    face_y = ((X - centre[0]) * down[0] + (Y - centre[1]) * down[1]) / (fw / 2)
    top_y = float(np.min((poly - centre) @ down)) / (fw / 2)
    alpha = alpha * smoothstep((face_y - top_y) / 0.22)
    if style in ("full", "stubble"):
        alpha = alpha * (0.7 + 0.3 * smoothstep((face_y + 0.1) / 0.35))

    vis, fade = beard_visibility(pts, labels, mask, window, fw)
    alpha = alpha * vis * fade
    return np.clip(alpha, 0, 1).astype(np.float32), shade.astype(np.float32), window


def mask_bbox(mask, pad, shape):
    """Box (y0, y1, x0, x1) around the True pixels plus some padding, or None if empty."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    H, W = shape[:2]
    return (max(0, ys.min() - pad), min(H, ys.max() + pad + 1),
            max(0, xs.min() - pad), min(W, xs.max() + pad + 1))


def crop_valid_map(matrix, size, shape, offset, fw):
    """0..1 map of where a parser crop (size x size) covers the frame."""
    inv_matrix = invert(matrix).copy()
    inv_matrix[:, 2] -= offset
    ones = np.ones((size, size), np.uint8)
    covered = cv2.warpAffine(ones, inv_matrix, (shape[1], shape[0]), flags=cv2.INTER_NEAREST).astype(np.float32)
    e = odd(0.15 * fw)
    covered = cv2.erode(covered, np.ones((e, e), np.uint8))
    return cv2.GaussianBlur(covered, (2 * e + 1, 2 * e + 1), 0)


def layer_mask(name, settings, img, pts, labels, hair_labels, use_unet):
    """Mask for one layer over the face area. For the lens also returns the darker ring."""
    shape = img.shape
    if name == "hair":
        # union of the normal crop and the bigger crop, this was best on val (X37, X39)
        if not use_unet or (hair_labels is None and labels is None):
            return None, None
        mask = np.zeros(shape[:2], bool)
        for lab in (labels, hair_labels):
            if lab is not None:
                mask |= lab == HAIR_CLASS
        return mask, None
    if name == "skin":
        return skin_mask(pts, labels, shape, use_unet), None
    if name == "blush":
        return blush_mask(pts, labels, shape, settings.get("style", "cheek"), use_unet), None
    if name == "shadow":
        return shadow_mask(pts, shape, settings.get("style", "lid")), None
    if name == "brows":
        return brows_mask(pts, labels, shape, use_unet), None
    if name == "lens":
        iris, ring = lens_masks(pts, shape)
        return iris | ring, ring
    if name == "teeth":
        return teeth_mask(img, pts, labels, use_unet), None
    if name == "lips":
        return lips_mask(pts, labels, shape, use_unet), None
    if name == "liner":
        return liner_mask(pts, shape, settings.get("style", "line")), None
    if name == "lashes":
        return lashes_mask(pts, shape, settings.get("style", "natural")), None
    return None, None


def adapt_strength(k, orig, mask):
    """12.2: blush and eyeshadow get stronger on darker skin or in dim light (uses cheek Y in YCrCb)."""
    Y = cv2.cvtColor(np.clip(orig, 0, 255).astype(np.uint8), cv2.COLOR_BGR2YCrCb)[..., 0][mask].mean()
    return k * float(np.clip(1.3 - 0.45 * (Y - 70) / 110, 0.85, 1.3))


def smooth_skin(sub, fw):
    """Skin smoothing that keeps edges (bilateral filter, sized for the face width)."""
    d = max(odd(11 * fw / 220), 5)
    img = np.clip(sub, 0, 255).astype(np.uint8)
    return cv2.bilateralFilter(img, d, 45, max(3.0, 11 * fw / 220)).astype(np.float32)


HAIR_LIFT = 0.8  # default "lift" for the hair layer (old saved looks don't have one)


def lift_hair(sub, mask, color, amount):
    """Lighten (bleach) dark hair before the hair colour goes on (user request)."""
    if amount <= 0 or mask.sum() < 20:
        return sub
    lab = to_lab(sub)
    L = lab[..., 0]
    target = lab_of(color)
    low, median, high = np.percentile(L[mask], [5, 50, 95])
    gap = max(float(target[0]) - median, 0.0)
    ease = float(np.clip(gap / 120.0, 0, 1))
    goal = median + amount * gap * ease * ease * (3 - 2 * ease)
    if goal <= median + 2:
        return sub

    eps = 6.0
    # power so that the 95% / 5% lightness ratio becomes about 1.8
    power = np.log(1.8) / max(np.log((high + eps) / (low + eps)), 1e-3)
    power = float(np.clip(power, 0.25, 1.0))
    new_L = goal * ((L + eps) / (median + eps)) ** power

    # soft roll-off near white instead of a hard clip, so highlights don't burn out
    knee = 225.0
    over = np.maximum(new_L - knee, 0)
    new_L = np.where(new_L > knee, knee + (255 - knee) * (1 - np.exp(-over / (255 - knee))), new_L)
    lab[..., 0] = np.maximum(L, new_L)

    strength = float(np.clip((goal - median) / 120.0, 0, 1))
    warm = np.array([132.0, 142.0], np.float32)
    ab = 128 + (1 - 0.6 * strength) * (lab[..., 1:] - 128)
    lab[..., 1:] = ab + strength * 0.7 * (0.5 * (warm + target[1:]) - ab)
    return from_lab(lab)


def layer_colour(name, settings, sub, mask, ring, fw):
    """The fully coloured version of `sub` (float BGR) for one layer."""
    color = settings["color"]

    if name == "hair":
        return recolour_lab(sub, mask, color, keep_ab=0.3, l_mix=0.2)

    if name == "skin":
        lab = to_lab(sub)
        target = lab_of(color)
        lab[..., 0] += 8
        lab[..., 1:] = 0.75 * lab[..., 1:] + 0.25 * target[1:]
        return from_lab(lab)

    if name in ("blush", "liner", "lashes"):
        flat = np.empty_like(sub)
        flat[:] = color
        return flat

    if name == "shadow":
        hsv = cv2.cvtColor(np.clip(sub, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        target_hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)
        hsv[..., 0] = target_hsv[0]
        hsv[..., 1] = target_hsv[1]
        shimmer = float(settings.get("shimmer", 0))
        hsv[..., 2] = np.clip(hsv[..., 2] * (1 + 0.6 * shimmer), 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if name == "brows":
        return recolour_lab(sub, mask, color, keep_ab=0.3, l_mix=0.25)

    if name == "lens":
        coloured = recolour_lab(sub, mask & ~ring, color, keep_ab=0.2, l_mix=0.4)
        coloured[ring] *= 0.75
        return coloured

    if name == "teeth":
        # a*/b* go towards the chosen colour. Light colours brighten, darker ones also pull L down (X49).
        lab = to_lab(sub)
        target = lab_of(color)
        lab[..., 1:] = target[1:] + 0.2 * (lab[..., 1:] - target[1:])
        L = lab[..., 0] + 18
        pull = float(np.clip((235 - target[0]) / 235, 0, 0.6))
        lab[..., 0] = L + pull * (target[0] - L)
        return from_lab(lab)

    if name == "lips":
        coloured = recolour_lab(sub, mask, color, keep_ab=0.3)
        gloss = float(settings.get("gloss", 0))
        if gloss > 0 and mask.sum() > 10:
            lab = to_lab(coloured)
            L = lab[..., 0]
            p80 = np.percentile(L[mask], 80)
            lab[..., 0] = L + gloss * 60 * np.clip((L - p80) / max(L[mask].max() - p80, 1), 0, 1)
            coloured = from_lab(lab)
        return coloured

    raise ValueError(name)


def paint_overlay(name, settings, k, roi, orig, p, lab_c, fw, use_unet):
    """Draw the mustache or beard onto roi. Returns (alpha, box in roi) or None if nothing was drawn."""
    if name == "mustache":
        if use_unet:
            vis = lip_visibility(p, lab_c)  # hide it when the lips can't be seen (X50)
        else:
            vis = 1.0
        if vis <= 0:
            return None
        k = k * vis
        result = overlay_alpha(name, settings, roi.shape, p, fw)
    else:
        if use_unet:
            result = beard_alpha(settings, roi.shape, p, lab_c)
        else:
            result = beard_alpha(settings, roi.shape, p, None)

    if result is None:
        return None
    box = mask_bbox(result[0] > 4e-3, 1, result[0].shape)
    if box is None:
        return None

    alpha, shade, (win_y, _, win_x, _) = result
    ya, yb, xa, xb = box
    alpha = alpha[ya:yb, xa:xb]
    shade = shade[ya:yb, xa:xb]
    ya, yb, xa, xb = ya + win_y, yb + win_y, xa + win_x, xb + win_x
    colour = hair_colour(settings["color"], shade, scene_light(orig, p))
    roi[ya:yb, xa:xb] = blend(roi[ya:yb, xa:xb], colour, alpha, k)
    return alpha * k, (ya, yb, xa, xb)


def paint(frame, pts, labels, hair_labels, look, use_unet=True, hidden=frozenset(), hair_crop=None,
          want_masks=False):
    """Put on all the visible layers in the 12.3 order."""
    H, W = frame.shape[:2]

    fb = face_box(pts)
    bw = fb[2] - fb[0]
    bh = fb[3] - fb[1]
    x0 = int(max(0, fb[0] - 0.6 * bw))
    y0 = int(max(0, fb[1] - 0.6 * bh))
    x1 = int(min(W, fb[2] + 0.6 * bw))
    y1 = int(min(H, fb[3] + 0.6 * bh))
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    orig = roi.copy()
    p = pts - [x0, y0]

    if labels is not None:
        lab_c = labels[y0:y1, x0:x1]
    else:
        lab_c = None
    if hair_labels is not None:
        hair_c = hair_labels[y0:y1, x0:x1]
    else:
        hair_c = lab_c

    fw = face_width(p)
    alphas = {}

    for name in ORDER:
        settings = look[name]
        k = float(min(settings["k"], 1.0))
        if name == "skin":
            smooth = float(np.clip(settings.get("smooth", 0.0), 0, 1))
        else:
            smooth = 0.0
        if (k <= 0 and smooth <= 0) or name in hidden:
            continue

        # mustache / beard: a generated hair texture placed with the landmarks (X46)
        if name in OVERLAYS:
            drawn = paint_overlay(name, settings, k, roi, orig, p, lab_c, fw, use_unet)
            if drawn is not None and want_masks:
                alpha, (ya, yb, xa, xb) = drawn
                full = np.zeros((H, W), np.float32)
                full[y0 + ya:y0 + yb, x0 + xa:x0 + xb] = alpha
                alphas[name] = full
            continue

        eye_vis = None
        if use_unet and name in EYE_LAYERS:
            eye_vis = eye_visibility(p, lab_c)
        if name == "lens":
            # no lens on a closed eye, faded on an almost closed one (X54)
            open_amount = lens_open(p)
            new_vis = {}
            for side in open_amount:
                if eye_vis is None:
                    new_vis[side] = open_amount[side]
                else:
                    new_vis[side] = eye_vis[side] * open_amount[side]
            eye_vis = new_vis
        if eye_vis is not None and max(eye_vis.values()) <= 0:
            continue  # both eyes covered (X51)

        mask, ring = layer_mask(name, settings, roi, p, lab_c, hair_c, use_unet)
        if mask is None or not mask.any():
            continue

        blur_size = odd(settings["blur"] * fw / 220.0)
        box = mask_bbox(mask, blur_size + 2, roi.shape)
        ya, yb, xa, xb = box
        sub = roi[ya:yb, xa:xb]
        sub_mask = mask[ya:yb, xa:xb]

        if name in ("blush", "shadow"):
            k = min(adapt_strength(k, orig[ya:yb, xa:xb], sub_mask), 1.0)
        alpha = soft_alpha(sub_mask, settings["blur"], fw)
        if eye_vis is not None and min(eye_vis.values()) < 1:
            alpha = alpha * side_weights(p, eye_vis, box)  # one eye covered or fading (X51)

        if smooth > 0:
            # skin smoothing with its own strength, before the foundation colour (X42)
            sub = blend(sub, smooth_skin(sub, fw), alpha, smooth)

        if name == "hair":
            if hair_crop is not None:
                valid = crop_valid_map(hair_crop[0], hair_crop[1], roi.shape, [x0, y0], fw)
                alpha = alpha * valid[ya:yb, xa:xb]
            lift = float(np.clip(settings.get("lift", HAIR_LIFT), 0, 1))
            if lift > 0:
                sub = blend(sub, lift_hair(sub, sub_mask, settings["color"], lift), alpha, 1.0)

        if ring is None:
            sub_ring = None
        else:
            sub_ring = ring[ya:yb, xa:xb]
        colour = layer_colour(name, settings, sub, sub_mask, sub_ring, fw)

        if name == "lips" and settings.get("style") == "ombre":
            line = np.zeros(sub_mask.shape, np.uint8)
            cv2.polylines(line, [np.round(p[96:104] - [xa, ya]).astype(np.int32)], True, 1, 1)
            dist = cv2.distanceTransform((1 - line).astype(np.uint8), cv2.DIST_L2, 3)
            max_dist = max(float(dist[sub_mask].max()), 1.0)
            alpha = alpha * np.clip(1.0 - 0.65 * dist / max_dist, 0.35, 1.0)

        roi[ya:yb, xa:xb] = blend(sub, colour, alpha, k)

        if want_masks:
            full = np.zeros((H, W), np.float32)
            full[y0 + ya:y0 + yb, x0 + xa:x0 + xb] = alpha * max(k, smooth)
            alphas[name] = full

    out = frame.copy()
    out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out, alphas


HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def bgr_to_hex(c):
    b, g, r = [int(v) for v in c]
    return f"#{r:02X}{g:02X}{b:02X}"


def hex_to_bgr(h):
    return (int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16))


def validate_look(look):
    """List of problems with a look (empty list = ok)."""
    errors = []
    for name in LAYERS:
        layer = look.get(name)
        if not isinstance(layer, dict):
            errors.append(f"missing layer {name}")
            continue
        c = layer.get("color")
        if not (isinstance(c, (tuple, list)) and len(c) == 3 and all(0 <= int(v) <= 255 for v in c)):
            errors.append(f"{name}: bad colour {c!r}")
        if not (0 <= float(layer.get("k", -1)) <= 1):
            errors.append(f"{name}: k must be 0-1")
        blur = layer.get("blur", 0)
        if not (isinstance(blur, int) and 1 <= blur <= 61 and blur % 2 == 1):
            errors.append(f"{name}: blur must be odd 1-61")
        if name in STYLES and layer.get("style") not in STYLES[name]:
            errors.append(f"{name}: unknown style {layer.get('style')!r}")
        if name == "skin" and "smooth" in layer and not (0 <= float(layer["smooth"]) <= 1):
            errors.append("skin: smooth must be 0-1")
        if name == "hair" and "lift" in layer and not (0 <= float(layer["lift"]) <= 1):
            errors.append("hair: lift must be 0-1")
    return errors


def look_to_json(look):
    """Same look but with colours as hex strings (for saving)."""
    out = {}
    for name, layer in look.items():
        new_layer = {}
        for key, value in layer.items():
            if key != "color":
                new_layer[key] = value
        new_layer["color"] = bgr_to_hex(layer["color"])
        out[name] = new_layer
    return out


def look_from_json(data):
    out = {}
    for name, layer in data.items():
        layer = dict(layer)
        if not (isinstance(layer.get("color"), str) and HEX.match(layer["color"])):
            raise ValueError(f"{name}: colour must be hex like #EBA0AF")
        layer["color"] = hex_to_bgr(layer["color"])
        out[name] = layer
    return out


def compare_image(before, after, look_name="", vertical=False):
    """One image with Before | After next to each other (or on top of each other), and the look name under it."""
    h, w = before.shape[:2]
    font_scale = max(0.5, w / 640)
    thickness = max(1, int(round(2 * font_scale)))

    halves = []
    for img, title in ((before, "Before"), (after, "After")):
        im = img.copy()
        (text_w, text_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(im, (0, 0), (text_w + 16, text_h + 16), (0, 0, 0), -1)
        cv2.putText(im, title, (8, text_h + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                    thickness, cv2.LINE_AA)
        halves.append(im)

    if vertical:
        body = np.vstack(halves)
    else:
        body = np.hstack(halves)

    caption_h = int(36 * font_scale)
    caption = np.full((caption_h, body.shape[1], 3), 255, np.uint8)
    if look_name:
        text = f"Look: {look_name}"
    else:
        text = ""
    cv2.putText(caption, text, (10, int(caption_h * 0.7)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * font_scale, (40, 40, 40), thickness, cv2.LINE_AA)
    return np.vstack([body, caption])


class MakeupSession:
    """Holds the models, the tracking and the current settings."""

    def __init__(self, pipeline, hair_parser=None, look="soft pink",
                 looks_file="looks/user_looks.json", hair_pad=0.45):
        self.pipe = pipeline
        self.hair_parser = hair_parser
        self.hair_pad = hair_pad
        self.looks_file = Path(looks_file)
        self.looks = copy.deepcopy(LOOKS)
        self.warnings = self.load_user_looks()
        if look not in self.looks:
            raise ValueError(f"unknown look {look!r}; choose from {list(self.looks)}")
        self.look_name = look
        self.settings = copy.deepcopy(self.looks[look])
        self.selected = "lips"
        self.use_unet = pipeline.par is not None
        self.hidden = set()
        self.prev_pts = None


    def load_user_looks(self):
        """Read the saved looks file. Returns a list of warnings for looks that were skipped."""
        warnings = []
        if not self.looks_file.exists():
            return warnings
        try:
            data = json.loads(self.looks_file.read_text())
        except Exception as e:  # noqa: BLE001
            return [f"could not read {self.looks_file}: {e}"]

        for name, look_data in data.get("looks", {}).items():
            if name in BUILTIN_LOOKS:
                warnings.append(f"skipped saved look {name!r}: same name as a built-in look")
                continue
            try:
                look = look_from_json(look_data)
            except ValueError as e:
                warnings.append(f"skipped saved look {name!r}: {e}")
                continue

            if isinstance(look.get("skin"), dict) and "smooth" not in look["skin"]:
                look["skin"]["smooth"] = float(look["skin"].get("k", 0))
            for extra in OVERLAYS:
                # looks saved before X46 have no mustache / beard: add them switched off
                look.setdefault(extra, copy.deepcopy(LOOKS[BUILTIN_LOOKS[0]][extra]))
            for gone in REMOVED_LAYERS:
                look.pop(gone, None)  # looks saved while the wig existed (X46-X47)

            errors = validate_look(look)
            if errors:
                warnings.append(f"skipped saved look {name!r}: {'; '.join(errors)}")
                continue
            self.looks[name] = look
        return warnings

    def write_user_looks(self):
        user_looks = {}
        for name, look in self.looks.items():
            if name not in BUILTIN_LOOKS:
                user_looks[name] = look_to_json(look)
        self.looks_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.looks_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"looks": user_looks}, indent=2))
        tmp.replace(self.looks_file)

    def list_looks(self):
        return list(self.looks)

    def load_look(self, name):
        self.look_name = name
        self.settings = copy.deepcopy(self.looks[name])
        self.hidden = set()

    def next_look(self):
        names = self.list_looks()
        i = names.index(self.look_name)
        self.load_look(names[(i + 1) % len(names)])

    def reset(self):
        self.load_look(self.look_name)

    def save_look(self, name, overwrite=False):
        """Save the current settings as a look."""
        name = name.strip()
        if not name:
            return "empty"
        if name in BUILTIN_LOOKS:
            return "builtin"
        if name in self.looks and not overwrite:
            return "exists"
        look = copy.deepcopy(self.settings)
        assert not validate_look(look), validate_look(look)
        self.looks[name] = look
        self.look_name = name
        self.write_user_looks()
        return "saved"

    def delete_look(self, name):
        if name in BUILTIN_LOOKS:
            return "builtin"
        if name not in self.looks:
            return "missing"
        del self.looks[name]
        self.write_user_looks()
        if self.look_name == name:
            self.load_look(BUILTIN_LOOKS[0])
        return "deleted"


    def turn_on(self, layer):
        if not self.is_active(layer):
            self.settings[layer]["k"] = ON_K.get(layer, 0.6)
        self.hidden.discard(layer)

    def set_layer(self, layer):
        self.selected = layer
        self.turn_on(layer)  # 13.1: picking a layer that is off turns it on

    def set_color(self, bgr, layer=None):
        if layer is None:
            layer = self.selected
        self.settings[layer]["color"] = tuple(int(v) for v in bgr)
        self.turn_on(layer)

    def set_style(self, style, layer=None):
        if layer is None:
            layer = self.selected
        if style not in STYLES.get(layer, []):
            raise ValueError(f"{layer} has no style {style!r}")
        self.settings[layer]["style"] = style
        self.turn_on(layer)

    def set_strength(self, k, layer=None):
        if layer is None:
            layer = self.selected
        self.settings[layer]["k"] = float(np.clip(k, 0, 1))
        if k > 0:
            self.hidden.discard(layer)

    def set_smooth(self, v):
        """Skin smoothing strength, separate from the foundation strength k (X42)."""
        self.settings["skin"]["smooth"] = float(np.clip(v, 0, 1))
        if v > 0:
            self.hidden.discard("skin")

    def set_lift(self, v):
        """How much dark hair gets lightened before the hair colour (see lift_hair). 0 = off."""
        self.settings["hair"]["lift"] = float(np.clip(v, 0, 1))
        self.hidden.discard("hair")

    def is_active(self, layer):
        settings = self.settings[layer]
        if settings["k"] > 0:
            return True
        return layer == "skin" and settings.get("smooth", 0) > 0

    def toggle_layer(self, layer):
        """Hide or show a layer without losing its settings (13.1)."""
        if layer in self.hidden:
            self.hidden.discard(layer)
        elif not self.is_active(layer):
            self.turn_on(layer)
        else:
            self.hidden.add(layer)

    def is_on(self, layer):
        return self.is_active(layer) and layer not in self.hidden


    def analyse(self, frame, track=True):
        """Run the models on one frame: box, points, labels (and the bigger hair labels if hair is on)."""
        result = {"box": None, "score": None, "points": None, "labels": None, "hair_labels": None,
                  "hair_crop": None}
        box, score = self.pipe.detect(frame)
        result["box"] = box
        result["score"] = score
        if box is None:
            self.prev_pts = None
            return result

        # keep tracking if the face is still in about the same place, otherwise start again from the box
        if track and self.prev_pts is not None and box_iou(face_box(self.prev_pts), box) > 0.5:
            pts = self.pipe.landmarks_from_points(frame, self.prev_pts)
        else:
            pts = self.pipe.landmarks_from_box(frame, box, 2)
        if track:
            self.prev_pts = pts
        else:
            self.prev_pts = None
        result["points"] = pts

        if self.pipe.par is not None:
            result["labels"] = self.pipe.parse(frame, pts)[0]
            if self.is_on("hair"):
                hair_labels, matrix = self.pipe.parse(frame, pts, self.hair_pad, model=self.hair_parser)
                result["hair_labels"] = hair_labels
                result["hair_crop"] = (matrix, self.pipe.parser_size)
        return result

    def process(self, frame, track=True, want_masks=False, analysis=None):
        """Returns {'before', 'after', 'masks' (only if want_masks), 'points', 'box', 'labels', 'face', 'analysis'}."""
        if analysis is not None and (analysis["hair_labels"] is not None or not self.is_on("hair")
                                     or analysis["points"] is None or self.pipe.par is None):
            r = analysis
        else:
            r = self.analyse(frame, track)
        out = {
            "before": frame,
            "after": frame.copy(),
            "masks": {},
            "points": r["points"],
            "box": r["box"],
            "score": r["score"],
            "labels": r["labels"],
            "face": r["points"] is not None,
            "analysis": r,
        }
        if r["points"] is None:
            return out
        out["after"], out["masks"] = paint(frame, r["points"], r["labels"], r["hair_labels"], self.settings,
                                           self.use_unet, frozenset(self.hidden), r["hair_crop"], want_masks)
        return out

    def compare_image(self, before, after, vertical=False):
        return compare_image(before, after, self.look_name, vertical)
