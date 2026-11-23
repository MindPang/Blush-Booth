"""The model pipeline used by the app: detector -> crop -> landmarks -> parser.

Which model is used for each step is written in best_models.json (made by
experiments/select_best.py and experiments/full_train.py). Every experiment script has the same
small functions, so the pipeline does not care which approach won:

    detector module:  load_model(path, device), detect(model, frame) -> (box, score)
    landmark module:  load_model(path, device), predict_points(model, crop) -> (106, 2) in crop pixels
    parser module:    load_model(path, device), parse_crop(model, crop) -> (11, H, W) class scores
"""
import importlib
import json
from pathlib import Path

import cv2
import numpy as np

from .datasets import parser_matrix
from .geometry import box_crop_params, crop_matrix, face_box, invert, landmark_crop_params, transform_points, warp

ROOT = Path(__file__).resolve().parent.parent
SHORT_CONFIG = ROOT / "best_models.json"            # short-run winners (select_best.py)
FULL_CONFIG = ROOT / "model" / "best_models.json"   # same winners trained on all images (full_train.py)


def default_config():
    """The fully trained models once full_train.py has made them, otherwise the short-run winners."""
    return FULL_CONFIG if FULL_CONFIG.exists() else SHORT_CONFIG


DEFAULT_CONFIG = default_config()

# what to tell the user when a model has not been trained yet
HOW_TO_TRAIN = {
    "detector": "python experiments/full_train.py --task detection",
    "landmark": "python experiments/full_train.py --task landmark",
    "parser": "python experiments/full_train.py --task parser",
}


class MissingModel(Exception):
    """A model file in best_models.json does not exist yet."""


def find_file(rel_path):
    """Paths in best_models.json are relative to the project folder."""
    p = Path(rel_path)
    if p.is_absolute():
        return p
    return ROOT / p


def load_part(name, entry, device):
    """Import the experiment script named in the config and load its saved model."""
    if not entry or not entry.get("weights"):
        reason = "not in best_models.json"
    elif not find_file(entry["weights"]).exists():
        reason = entry["weights"] + " not found"
    else:
        reason = None
    if reason:
        raise MissingModel(f"No trained {name} yet ({reason}). Run the {name} experiments, "
                           f"then experiments/select_best.py, then: {HOW_TO_TRAIN[name]}")
    module = importlib.import_module(entry["module"])
    return module, module.load_model(str(find_file(entry["weights"])), device)


class FacePipeline:
    def __init__(self, config=None, device="cpu"):
        from .models import pick_device

        if config is None:
            config = default_config()
        config = Path(config)
        self.config_path = config
        cfg = {}
        if config.exists():
            cfg = json.loads(config.read_text())
        self.cfg = cfg
        self.device = pick_device(str(device))
        self.warnings = []
        if not config.exists():
            self.warnings.append(f"No trained models yet ({config.name} not found). Run the short experiments, "
                                 "then python experiments/select_best.py")

        # every part is optional, so the app always starts (it just finds no face without a detector)
        self.det_module, self.det = self.try_load("detector", cfg)
        self.score_thr = float(cfg.get("detector", {}).get("threshold", 0.40))

        self.lmk_module, self.lmk = self.try_load("landmark", cfg)
        self.lmk_size = int(cfg.get("landmark", {}).get("crop_size", 160))

        # without a parser the makeup only uses the landmarks
        self.par_module, self.par = self.try_load("parser", cfg)
        self.parser_size = int(cfg.get("parser", {}).get("crop_size", 256))

        # the bigger hair crop can use its own parser, otherwise the normal parser is used
        self.hair_par = None
        hair = cfg.get("hair_parser")
        if hair and hair.get("weights") and find_file(hair["weights"]).exists() and self.par is not None:
            self.hair_par = load_part("parser", hair, self.device)[1]

    def try_load(self, name, cfg):
        try:
            return load_part(name, cfg.get(name), self.device)
        except MissingModel as e:
            if cfg:
                self.warnings.append(str(e))
            return None, None

    @classmethod
    def from_config(cls, config=None, device="cpu"):
        return cls(config, device)

    # ---------------------------------------------------------- 1. face detection
    def detect(self, frame):
        if self.det is None or self.lmk is None:
            return None, 0.0     # no trained detector / landmark model yet
        box, score = self.det_module.detect(self.det, frame)
        if box is None or score < self.score_thr:
            return None, score
        return np.asarray(box, np.float64), score

    # ---------------------------------------------------------- 2. landmarks
    def landmarks_from_crop(self, frame, centre, side, angle):
        matrix = crop_matrix(centre, side, angle, self.lmk_size)
        crop = warp(frame, matrix, (self.lmk_size, self.lmk_size), antialias=True)
        crop_points = self.lmk_module.predict_points(self.lmk, crop)
        return transform_points(crop_points, invert(matrix))

    def landmarks_from_box(self, frame, box, upright_passes=2):
        """New face: crop from the detector box first, then redo it upright a few times."""
        points = self.landmarks_from_crop(frame, *box_crop_params(box))
        for _ in range(upright_passes):
            points = self.landmarks_from_crop(frame, *landmark_crop_params(points))
        return points

    def landmarks_from_points(self, frame, prev_pts):
        """Tracking: the crop comes from the points of the last frame."""
        return self.landmarks_from_crop(frame, *landmark_crop_params(prev_pts))

    # ---------------------------------------------------------- 3. face parsing
    def parse(self, frame, pts, pad=0.2, model=None):
        """Label map for the whole frame. Returns (labels, crop matrix)."""
        size = self.parser_size
        matrix = parser_matrix(face_box(pts), np.full(4, pad), 0.0, size)
        crop = warp(frame, matrix, (size, size), border=cv2.BORDER_REFLECT, antialias=True)
        if model is None:
            model = self.par
        scores = np.asarray(self.par_module.parse_crop(model, crop), np.float32)

        # warp the class scores back, but only inside the box the crop covers (faster)
        h, w = frame.shape[:2]
        inv_matrix = invert(matrix)
        n = size - 1
        corners = transform_points(np.array([[0, 0], [n, 0], [0, n], [n, n]], np.float64), inv_matrix)
        x0, y0 = np.floor(corners.min(0)).astype(int).clip(0, [w, h])
        x1, y1 = (np.ceil(corners.max(0)).astype(int) + 1).clip(0, [w, h])

        out = np.zeros((h, w), np.uint8)
        if x1 > x0 and y1 > y0:
            box_matrix = inv_matrix.copy()
            box_matrix[:, 2] -= [x0, y0]
            warped = []
            for i in range(len(scores)):
                if i == 0:
                    fill = 1e4     # outside the crop = background
                else:
                    fill = -1e4
                warped.append(cv2.warpAffine(scores[i], box_matrix, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                                             borderMode=cv2.BORDER_CONSTANT, borderValue=fill))
            out[y0:y1, x0:x1] = np.stack(warped).argmax(0)
        return out, matrix

    # ---------------------------------------------------------- everything at once (for scripts)
    def process(self, frame, prev_pts=None, upright_passes=2):
        result = {"box": None, "score": None, "points": None, "labels": None}
        result["box"], result["score"] = self.detect(frame)
        if prev_pts is not None:
            result["points"] = self.landmarks_from_points(frame, prev_pts)
        elif result["box"] is not None:
            result["points"] = self.landmarks_from_box(frame, result["box"], upright_passes)
        if result["points"] is not None and self.par is not None:
            result["labels"], _ = self.parse(frame, result["points"])
        return result
