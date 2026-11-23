import json
from pathlib import Path

import numpy as np

N_POINTS = 106
N_CLASSES = 11

# Which points belong to which part of the face.
# "R" means the person's right, which is the left side of the image (if not mirrored).
REGIONS = {
    "contour": list(range(0, 33)),
    "brows": list(range(33, 51)),
    "nose": list(range(51, 66)),
    "eyes": list(range(66, 84)) + [104, 105],
    "lips": list(range(84, 104)),
}
R_EYE_OUTER, R_EYE_INNER, R_EYE_CENTRE = 66, 70, 74
L_EYE_INNER, L_EYE_OUTER, L_EYE_CENTRE = 75, 79, 83
R_PUPIL, L_PUPIL = 104, 105

CLASS_NAMES = [
    "background", "skin", "left brow (image left)", "right brow (image right)", "left eye (image left)",
    "right eye (image right)", "nose", "upper lip", "inner mouth", "lower lip", "hair",
]
BACKGROUND, SKIN, L_BROW, R_BROW, L_EYE, R_EYE, NOSE, U_LIP, MOUTH, D_LIP, HAIR = range(11)
FACE_CLASSES = list(range(1, 11))
LIP_CLASSES = (U_LIP, D_LIP)       # these two are merged for the "lips IoU" (section 7.5)

LABEL_FLIP = np.array([0, 1, 3, 2, 5, 4, 6, 7, 8, 9, 10], dtype=np.uint8)


def spec_flip_perm():
    """Build the flip permutation (FLIP_PERM) from the table in section 2.1."""
    perm = np.arange(N_POINTS)

    def swap(a, b):
        perm[a] = b
        perm[b] = a

    for i in range(33):
        perm[i] = 32 - i
    for j in range(5):
        swap(33 + j, 46 - j)
    for j in range(4):
        swap(38 + j, 50 - j)
    for j in range(11):
        perm[55 + j] = 65 - j
    for j in range(5):
        swap(66 + j, 79 - j)
    for j in range(3):
        swap(71 + j, 82 - j)
    swap(R_EYE_CENTRE, L_EYE_CENTRE)
    for j in range(7):
        perm[84 + j] = 90 - j
    for j in range(5):
        perm[91 + j] = 95 - j
    for j in range(5):
        perm[96 + j] = 100 - j
    for j in range(3):
        perm[101 + j] = 103 - j
    swap(R_PUPIL, L_PUPIL)
    return perm


FLIP_PERM = spec_flip_perm()


def is_valid_flip_perm(perm):
    """Check that perm uses every point once and flipping twice gives back the start."""
    perm = np.asarray(perm)
    if perm.shape != (N_POINTS,):
        return False
    if sorted(perm.tolist()) != list(range(N_POINTS)):
        return False
    return bool(np.all(perm[perm] == np.arange(N_POINTS)))


def normalise_shapes(shapes):
    """Take out position, tilt and size from a batch of face shapes."""
    shapes = np.asarray(shapes, dtype=np.float64)
    right_eye = (shapes[:, R_EYE_OUTER] + shapes[:, R_EYE_INNER]) / 2
    left_eye = (shapes[:, L_EYE_OUTER] + shapes[:, L_EYE_INNER]) / 2
    middle = (right_eye + left_eye) / 2
    eye_vec = left_eye - right_eye
    dist = np.linalg.norm(eye_vec, axis=1, keepdims=True).clip(1e-6)
    cos_a = eye_vec[:, 0:1] / dist
    sin_a = eye_vec[:, 1:2] / dist
    x = shapes[..., 0] - middle[:, 0:1]
    y = shapes[..., 1] - middle[:, 1:2]
    rotated = np.stack([cos_a * x + sin_a * y, -sin_a * x + cos_a * y], axis=-1)
    return rotated / dist[..., None]


# Which part turns into which part when the face is mirrored (from the section 2.1 table).
# I only match points inside the mirror part so points that are really close together
# (like pupil 104 and eye centre 74) don't get mixed up.
MIRROR_PARTS = [
    (range(0, 33), range(0, 33)),
    (range(33, 42), range(42, 51)),
    (range(42, 51), range(33, 42)),
    (range(51, 66), range(51, 66)),
    (range(66, 75), range(75, 84)),
    (range(75, 84), range(66, 75)),
    (range(84, 96), range(84, 96)),
    (range(96, 104), range(96, 104)),
    ([R_PUPIL], [L_PUPIL]),
    ([L_PUPIL], [R_PUPIL]),
]


def derive_flip_perm(shapes):
    """Work out the flip permutation from real data."""
    mean_shape = normalise_shapes(shapes).mean(0)
    mirrored = mean_shape * np.array([-1.0, 1.0])
    perm = np.full(N_POINTS, -1)
    for src, dst in MIRROR_PARTS:
        src = list(src)
        dst = np.array(list(dst))
        dists = np.linalg.norm(mirrored[src][:, None, :] - mean_shape[dst][None, :, :], axis=-1)
        perm[src] = dst[dists.argmin(1)]
    return perm


def load_flip_perm(data_root=None):
    """Use <data_root>/flip_perm.json (checked on the data) if it exists, otherwise the table above."""
    if data_root is not None:
        path = Path(data_root) / "flip_perm.json"
        if path.exists():
            perm = np.array(json.loads(path.read_text())["flip_perm"], dtype=np.int64)
            if not is_valid_flip_perm(perm):
                raise ValueError(f"{path} does not hold a valid flip permutation")
            return perm
    return FLIP_PERM.copy()


assert is_valid_flip_perm(FLIP_PERM)
