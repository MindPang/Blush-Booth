"""Shared data code for all experiments: options, subset lists, eval sets and training sets.

Every experiment uses the same data:
  short run train = first 2,000 LaPa train images after shuffling with seed 0 (list saved in subsets/)
  short run eval  = fixed 500 val images (+500 face-free scenes from backgrounds/ for detection)
  test split      = only used by experiments/test_once.py at the very end
"""
import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from facekit import datasets as D  # noqa: E402
from facekit.geometry import crop_matrix, landmark_crop_params, transform_points, warp  # noqa: E402

SUBSET_DIR = Path(__file__).resolve().parent / "subsets"
OLD_ROOT = Path(r"E:\Blush Booth")

LMK_SIZE = 160      # landmark crop (same as old project)
PAR_SIZE = 256      # parser crop (same as old project)


def default_dir(name):
    """LaPa/ or backgrounds/ in this folder if it is there, otherwise the one in the old project."""
    if (ROOT / name).exists() or not (OLD_ROOT / name).exists():
        return str(ROOT / name)
    return str(OLD_ROOT / name)


def get_args(description, epochs=4, extra=None):
    """The same command line options for every experiment script."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--lapa", default=default_dir("LaPa"), help="LaPa folder with train/ val/ test/")
    ap.add_argument("--backgrounds", default=default_dir("backgrounds"), help="face-free photos (detection only)")
    ap.add_argument("--train-images", type=int, default=2000, help="0 = all train images (full training)")
    ap.add_argument("--val-images", type=int, default=500)
    ap.add_argument("--full", action="store_true", help="full training: all train images (= --train-images 0)")
    ap.add_argument("--split", default="val", help="val for experiments, test only through test_once.py")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=epochs, help="CNN only")
    ap.add_argument("--batch", type=int, default=16, help="CNN only")
    ap.add_argument("--lr", type=float, default=1e-3, help="CNN only")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (CNN only)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--variant", default=None, help="run only this classifier / variant (name from results.csv)")
    ap.add_argument("--save", default=None, help="where to save the trained model")
    ap.add_argument("--csv", default=None, help="results file (default: results.csv of the model folder)")
    ap.add_argument("--quick", action="store_true", help="tiny run to check the code (40 train, 20 val, 1 epoch)")
    if extra:
        extra(ap)
    args = ap.parse_args()
    if args.full:
        args.train_images = 0
    if args.quick:
        args.train_images = 40
        args.val_images = 20
        args.epochs = 1
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def is_fake(lapa):
    return (Path(lapa) / "FAKE_DATA.txt").exists()


def run_note(args):
    """Text for the notes column so quick / fake runs are never picked by select_best.py."""
    parts = []
    if args.quick:
        parts.append("QUICK")
    if is_fake(args.lapa):
        parts.append("FAKE DATA")
    if args.train_images == 0:
        parts.append("FULL TRAIN")
    return " ".join(parts)


# ------------------------------------------------------------------ subset lists

def subset_names(lapa, split, n, seed=0):
    """First n names after shuffling with `seed`. The list is saved in subsets/ the first time.
    n = 0 means all images (not shuffled)."""
    names = D.list_names(lapa, split)
    if not names:
        raise SystemExit(f"no images in {Path(lapa) / split / 'images'} (use --lapa)")
    if n == 0 or n >= len(names):
        return names

    tag = "fake_" if is_fake(lapa) else ""
    path = SUBSET_DIR / f"{tag}{split}_{n}_seed{seed}.txt"
    if path.exists():
        saved = path.read_text().split()
        if all((Path(lapa) / split / "landmarks" / (s + ".txt")).exists() for s in saved[:5]):
            return saved

    order = list(names)
    random.Random(seed).shuffle(order)
    chosen = order[:n]
    SUBSET_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(chosen) + "\n")
    print(f"saved subset list {path}")
    return chosen


def train_names(args):
    return subset_names(args.lapa, "train", args.train_images, 0)


def eval_names(args):
    if args.split == "test":
        # the final test uses the whole test split
        n = 0 if not args.quick else args.val_images
        return subset_names(args.lapa, "test", n, 0)
    return subset_names(args.lapa, "val", args.val_images, 0)


# ------------------------------------------------------------------ face detection data

def detection_eval_scenes(args):
    """Fixed eval scenes: one face scene per eval image + the same number of face-free scenes.
    Returns (list of scenes, list of boxes or None)."""
    names = eval_names(args)
    bg_split = "test" if args.split == "test" else "val"
    bgs = D.load_backgrounds(Path(args.backgrounds) / bg_split)
    scenes = []
    boxes = []
    for i, name in enumerate(names):
        rng = np.random.RandomState(1000 + i)
        img, pts, _ = D.read_sample(args.lapa, args.split, name, with_label=False)
        scene, box = D.face_scene(img, pts, bgs[rng.randint(len(bgs))], rng)
        scenes.append(scene)
        boxes.append(box)
    for i in range(len(names)):
        rng = np.random.RandomState(5000 + i)
        scenes.append(D.background_crop(bgs[rng.randint(len(bgs))], rng))
        boxes.append(None)
    return scenes, boxes


def detection_train_scenes(args, n_empty=None, seed=0):
    """One fixed face scene per train image + some empty scenes (classical detectors, YOLO).
    It yields (scene, box) one at a time, so even all 18,168 images do not have to fit in memory."""
    names = train_names(args)
    bgs = D.load_backgrounds(Path(args.backgrounds) / "train")
    if n_empty is None:
        n_empty = len(names) // 4
    for i, name in enumerate(names):
        rng = np.random.RandomState(seed * 100000 + i)
        img, pts, _ = D.read_sample(args.lapa, "train", name, with_label=False)
        yield D.face_scene(img, pts, bgs[rng.randint(len(bgs))], rng)
    for i in range(n_empty):
        rng = np.random.RandomState(seed * 100000 + 50000 + i)
        yield D.background_crop(bgs[rng.randint(len(bgs))], rng), None


class DetectionTrainSet:
    """New random scenes every epoch (CNN training). 15% of the scenes have no face."""

    def __init__(self, args, no_face_p=0.15):
        self.lapa = args.lapa
        self.names = train_names(args)
        self.bg_dir = Path(args.backgrounds) / "train"
        self.bgs = None
        self.no_face_p = no_face_p

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        if self.bgs is None:
            self.bgs = D.load_backgrounds(self.bg_dir)   # loaded here so it also works with workers
        rng = np.random
        bg = self.bgs[rng.randint(len(self.bgs))]
        if rng.uniform() < self.no_face_p:
            scene = D.background_crop(bg, rng)
            box = None
        else:
            img, pts, _ = D.read_sample(self.lapa, "train", self.names[i], with_label=False)
            scene, box = D.face_scene(img, pts, bg, rng)
        if rng.uniform() < 0.5:
            scene = np.ascontiguousarray(scene[:, ::-1])
            if box is not None:
                box = np.array([D.SCENE_W - 1 - box[2], box[1], D.SCENE_W - 1 - box[0], box[3]])
        scene = colour_jitter(scene, rng)
        hm, reg, mask = D.encode_target(box)
        if box is None:
            box = np.full(4, -1.0)
        return {"image": scene, "hm": hm, "reg": reg, "mask": mask, "box": box.astype(np.float32)}


# ------------------------------------------------------------------ landmark data

def landmark_crop(img, pts, rng, shift=0.05, scale=0.05, rotate=5.0, size=LMK_SIZE):
    """Upright face crop from the true points with a small random shift / scale / turn.
    Returns (crop, points in crop pixels)."""
    centre, side, angle = landmark_crop_params(pts)
    centre = centre + rng.uniform(-shift, shift, 2) * side
    side = side * rng.uniform(1 - scale, 1 + scale)
    angle = angle + rng.uniform(-rotate, rotate)
    matrix = crop_matrix(centre, side, angle, size)
    crop = warp(img, matrix, (size, size), antialias=True)
    return crop, transform_points(pts, matrix)


def landmark_eval_crops(args):
    """Fixed eval crops. The random shift for image i always uses seed 2000 + i, so every
    experiment gets exactly the same crops. Returns (crops (N,160,160,3), points (N,106,2))."""
    names = eval_names(args)
    crops = []
    points = []
    for i, name in enumerate(names):
        img, pts, _ = D.read_sample(args.lapa, args.split, name, with_label=False)
        crop, crop_pts = landmark_crop(img, pts, np.random.RandomState(2000 + i))
        crops.append(crop)
        points.append(crop_pts)
    return np.array(crops), np.array(points)


def landmark_train_crops(args):
    """One fixed crop per train image (classical methods), same small shift as eval.
    Yields (crop, points in crop pixels) one at a time."""
    for i, name in enumerate(train_names(args)):
        img, pts, _ = D.read_sample(args.lapa, "train", name, with_label=False)
        yield landmark_crop(img, pts, np.random.RandomState(300000 + i))


class LandmarkTrainSet:
    """Random crops every epoch for CNN training (bigger shift / scale / turn, like the old project)."""

    def __init__(self, args):
        self.lapa = args.lapa
        self.names = train_names(args)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = np.random
        img, pts, _ = D.read_sample(self.lapa, "train", self.names[i], with_label=False)
        crop, crop_pts = landmark_crop(img, pts, rng, shift=0.08, scale=0.15, rotate=20)
        crop = colour_jitter(crop, rng)
        uv = (crop_pts + 0.5) / LMK_SIZE
        return {"image": crop, "uv": uv.astype(np.float32)}


# ------------------------------------------------------------------ parser data

def parser_crop(img, label, pads, angle, size=PAR_SIZE):
    matrix = D.parser_matrix(D.label_face_box(label), pads, angle, size)
    crop = warp(img, matrix, (size, size), border=cv2.BORDER_REFLECT, antialias=True)
    crop_label = warp(label, matrix, (size, size), label=True)
    return crop, crop_label


def parser_eval_crops(args):
    """Fixed eval crops: label face box + 20% on every side. Returns (crops, labels)."""
    return parser_fixed_crops(args.lapa, args.split, eval_names(args))


def parser_train_crops(args):
    """The same kind of fixed crop for every train image (classical methods). Yields (crop, label)."""
    for name in train_names(args):
        img, _, label = D.read_sample(args.lapa, "train", name)
        yield parser_crop(img, label, np.full(4, 0.2), 0.0)


def parser_fixed_crops(lapa, split, names):
    crops = []
    labels = []
    for name in names:
        img, _, label = D.read_sample(lapa, split, name)
        crop, crop_label = parser_crop(img, label, np.full(4, 0.2), 0.0)
        crops.append(crop)
        labels.append(crop_label)
    return np.array(crops), np.array(labels)


class ParserTrainSet:
    """Random padding and turn every epoch for CNN training (same ranges as the old project)."""

    def __init__(self, args):
        self.lapa = args.lapa
        self.names = train_names(args)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        rng = np.random
        img, _, label = D.read_sample(self.lapa, "train", self.names[i])
        crop, crop_label = parser_crop(img, label, rng.uniform(0.05, 0.35, 4), rng.uniform(-20, 20))
        if rng.uniform() < 0.5:
            from facekit.layout import LABEL_FLIP
            crop = np.ascontiguousarray(crop[:, ::-1])
            crop_label = np.ascontiguousarray(LABEL_FLIP[crop_label[:, ::-1]])
        crop = colour_jitter(crop, rng)
        return {"image": crop, "label": crop_label.astype(np.int64)}


# ------------------------------------------------------------------ small helpers

def colour_jitter(img, rng):
    """Simple brightness / contrast change."""
    alpha = rng.uniform(0.8, 1.2)
    beta = rng.uniform(-20, 20)
    return np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)


TASK_FOLDERS = {"D": "face_detection", "P": "face_parser", "L": "face_landmark"}


def checkpoint_name(exp_id, variant, seed, tag=""):
    """File name (without extension) of a saved model, e.g. D2_RandomForest_s0."""
    name = f"{exp_id}_{variant}_s{seed}{tag}"
    for ch in " =/\\:,":
        name = name.replace(ch, "")
    return name


def checkpoint_dir(exp_id):
    """experiments/<model folder>/checkpoints/"""
    return ROOT / "experiments" / TASK_FOLDERS[exp_id[0]] / "checkpoints"


def model_path(args, exp_id, variant, ext):
    """Where to save a trained model: --save if given, else
    experiments/<model folder>/checkpoints/<exp>_<variant>_s<seed><ext> (_quick / _full added for those runs)."""
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    tag = ""
    if args.quick:
        tag = "_quick"
    if args.train_images == 0:
        tag = "_full"
    folder = checkpoint_dir(exp_id)
    folder.mkdir(parents=True, exist_ok=True)
    return folder / (checkpoint_name(exp_id, variant, args.seed, tag) + ext)
