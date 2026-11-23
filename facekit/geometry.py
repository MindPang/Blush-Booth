import cv2
import numpy as np

from .layout import L_EYE_INNER, L_EYE_OUTER, R_EYE_INNER, R_EYE_OUTER

# How much to grow the landmark box so it covers the whole LaPa face (up to the hairline).
BOX_EXT_SIDE = 0.06
BOX_EXT_TOP = 0.34
BOX_EXT_BOTTOM = 0.01


def face_box(points):
    """Get the face box (x0, y0, x1, y1) from the 106 landmark points."""
    points = np.asarray(points, dtype=np.float64)
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    w = x1 - x0
    h = y1 - y0
    return np.array([
        x0 - BOX_EXT_SIDE * w,
        y0 - BOX_EXT_TOP * h,
        x1 + BOX_EXT_SIDE * w,
        y1 + BOX_EXT_BOTTOM * h,
    ])


def box_centre_size(box):
    """Return (centre, width, height) of a box."""
    box = np.asarray(box, dtype=np.float64)
    centre = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    width = box[2] - box[0]
    height = box[3] - box[1]
    return centre, width, height


def eye_centres(points):
    """Return (right eye centre, left eye centre)."""
    points = np.asarray(points, dtype=np.float64)
    right_eye = (points[R_EYE_OUTER] + points[R_EYE_INNER]) / 2
    left_eye = (points[L_EYE_OUTER] + points[L_EYE_INNER]) / 2
    return right_eye, left_eye


def roll_angle(points):
    """How much the head is tilted, in degrees (angle of the line between the eyes)."""
    right_eye, left_eye = eye_centres(points)
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    return float(np.degrees(np.arctan2(dy, dx)))


def rotation_part(angle_deg):
    """2x2 matrix that turns a line at angle_deg back to horizontal."""
    a = np.radians(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    return np.array([[cos_a, sin_a], [-sin_a, cos_a]])


def crop_matrix(centre, side, angle, out_size):
    """2x3 affine matrix for a square crop."""
    rot_scale = rotation_part(angle) * (out_size / float(side))
    middle = (out_size - 1) / 2.0
    shift = np.array([middle, middle]) - rot_scale @ np.asarray(centre, dtype=np.float64)
    return np.hstack([rot_scale, shift[:, None]])


def stretch_matrix(centre, w, h, angle, out_w, out_h):
    """2x3 affine matrix that rotates around `centre` and stretches a w x h box to out_w x out_h."""
    scale = np.diag([out_w / float(w), out_h / float(h)])
    rot_scale = scale @ rotation_part(angle)
    middle = np.array([(out_w - 1) / 2.0, (out_h - 1) / 2.0])
    shift = middle - rot_scale @ np.asarray(centre, dtype=np.float64)
    return np.hstack([rot_scale, shift[:, None]])


def transform_points(points, matrix):
    """Move points with a 2x3 affine matrix (new = points @ A.T + t)."""
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]


def invert(matrix):
    """Inverse of a 2x3 affine matrix."""
    return cv2.invertAffineTransform(np.asarray(matrix, dtype=np.float64))


def warp(img, matrix, out_wh, label=False, border=cv2.BORDER_CONSTANT, value=0, antialias=False):
    """cv2.warpAffine with the right settings."""
    if label:
        return cv2.warpAffine(img, matrix, out_wh, flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    matrix = np.asarray(matrix, dtype=np.float64)
    if antialias:
        scale = float(np.sqrt(abs(np.linalg.det(matrix[:, :2]))))
        if scale < 0.75:
            h, w = img.shape[:2]
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            sx = new_w / w
            sy = new_h / h
            # a point q in the small image is the point p = (q + 0.5) / s - 0.5 in the original,
            # so fix the matrix so it starts from the small image instead
            new_a = matrix[:, :2] @ np.diag([1 / sx, 1 / sy])
            new_t = matrix[:, :2] @ np.array([0.5 / sx - 0.5, 0.5 / sy - 0.5]) + matrix[:, 2]
            matrix = np.hstack([new_a, new_t[:, None]])

    return cv2.warpAffine(img, matrix, out_wh, flags=cv2.INTER_LINEAR, borderMode=border, borderValue=value)


def upright_box(points, angle=None):
    """Face box measured after turning the face upright."""
    points = np.asarray(points, dtype=np.float64)
    if angle is None:
        angle = roll_angle(points)
    rot = rotation_part(angle)
    mean_point = points.mean(axis=0)
    upright_points = (points - mean_point) @ rot.T
    centre, w, h = box_centre_size(face_box(upright_points))
    centre = rot.T @ centre + mean_point
    return centre, w, h, angle


def landmark_crop_params(points, scale=1.2):
    """Crop settings from landmark points (section 6.1)."""
    centre, w, h, angle = upright_box(points)
    return centre, scale * max(w, h), angle


def box_crop_params(box, scale=1.2):
    """Crop settings from a detector box, for a new face (section 6.1). Angle is 0."""
    centre, w, h = box_centre_size(box)
    return centre, scale * max(w, h), 0.0


def to_tensor(img_bgr):
    """Turn a uint8 BGR image (H, W, 3) into a float tensor (3, H, W) in [-1, 1]."""
    import torch

    rgb = img_bgr[..., ::-1].astype(np.float32)
    rgb = rgb / 127.5 - 1.0
    chw = np.ascontiguousarray(rgb.transpose(2, 0, 1))
    return torch.from_numpy(chw)


def resize_scale_points(points, s):
    """Where points end up after cv2.resize by a factor s (pixel-centre convention)."""
    return (np.asarray(points, dtype=np.float64) + 0.5) * s - 0.5


def letterbox(frame, out_w=256, out_h=192, pad=114):
    """Resize the frame to fit in out_w x out_h without stretching it."""
    h, w = frame.shape[:2]
    s = min(out_w / w, out_h / h)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))
    if s < 1:
        interp = cv2.INTER_AREA
    else:
        interp = cv2.INTER_LINEAR
    canvas = np.full((out_h, out_w, 3), pad, dtype=np.uint8)
    canvas[:new_h, :new_w] = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    return canvas, s


def unletterbox_box(box, s):
    """Move a box from the letterboxed frame back to the original image."""
    corners = np.asarray(box, dtype=np.float64).reshape(2, 2)
    return resize_scale_points(corners, 1.0 / s).reshape(4)


def box_iou(a, b):
    """Intersection over union of two boxes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def flip_points(points, width, perm):
    """Mirror the points left-right and reorder them (section 2.3)."""
    out = np.asarray(points, dtype=np.float64).copy()
    out[:, 0] = (width - 1) - out[:, 0]
    return out[perm]
