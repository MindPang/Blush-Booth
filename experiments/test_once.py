"""Evaluate the final model of one part on the TEST split, once. Logged in records/evaluation_runs.csv.

    python experiments/test_once.py --task detection
    python experiments/test_once.py --task landmark
    python experiments/test_once.py --task parser
    python experiments/test_once.py --task end2end      (whole app pipeline on the test photos)

It refuses to test the same weights twice (use --again only if you really have to, and say so in the report).
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import csv
import importlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from experiments.common import data, evaluate  # noqa: E402
from experiments.common.results import METRIC_NAMES  # noqa: E402
from facekit.models import pick_device  # noqa: E402

from facekit.pipeline import default_config  # noqa: E402

CONFIG = default_config()  # model/best_models.json (fully trained) once full_train.py has run
RECORDS = ROOT / "records" / "evaluation_runs.csv"
CONFIG_KEY = {"detection": "detector", "parser": "parser", "landmark": "landmark"}
FIELDS = ["time", "task", "exp_id", "classifier", "weights", "split", "n_images", "metrics", "ms_per_image", "notes"]


def already_tested(task, weights):
    if not RECORDS.exists():
        return False
    with open(RECORDS, newline="") as f:
        for row in csv.DictReader(f):
            if row["task"] == task and row["weights"] == weights and row["split"] == "test":
                return True
    return False


def log(row):
    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RECORDS.exists()
    with open(RECORDS, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if is_new:
            w.writeheader()
        w.writerow(row)
    print(f"logged in {RECORDS}")


def test_part(args, task, config):
    entry = config[CONFIG_KEY[task]]
    weights = entry["weights"]
    if already_tested(task, weights) and not args.again:
        sys.exit(f"{weights} was already tested on the test split (see {RECORDS}). Use --again only if needed.")
    module = importlib.import_module(entry["module"])
    model = module.load_model(str(ROOT / weights), pick_device(args.device))

    if task == "detection":
        scenes, boxes = data.detection_eval_scenes(args)
        raw, ms = evaluate.run_detector(lambda img: module.detect(model, img), scenes, boxes)
        metrics = evaluate.detection_metrics(raw, float(entry.get("threshold", 0.4)))
        n = len(scenes)
    elif task == "landmark":
        crops, points = data.landmark_eval_crops(args)
        preds, ms = evaluate.run_landmarks(lambda c: module.predict_points(model, c), crops, points)
        metrics = evaluate.landmark_metrics(preds, points)
        n = len(crops)
    else:
        crops, labels = data.parser_eval_crops(args)
        cm, ms = evaluate.run_parser(lambda c: module.parse_crop(model, c), crops, labels)
        metrics = evaluate.parser_metrics(cm)
        n = len(crops)

    text = "; ".join(f"{name}={v:.4f}" for name, v in zip(METRIC_NAMES[task], metrics))
    print(f"TEST {task} {entry['exp_id']} {entry['classifier']}: {text}, {ms:.2f} ms/image")
    log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": task, "exp_id": entry["exp_id"],
         "classifier": entry["classifier"], "weights": weights, "split": "test", "n_images": n,
         "metrics": text, "ms_per_image": f"{ms:.2f}", "notes": data.run_note(args)})


def test_end2end(args):
    """The app pipeline on the full test photos: is the face found, and how good are the points / labels?"""
    from facekit import datasets as D
    from facekit.geometry import box_iou, face_box
    from facekit.metrics import confusion
    from facekit.pipeline import FacePipeline

    pipe = FacePipeline(CONFIG, args.device)
    if pipe.det is None or pipe.lmk is None:
        sys.exit("\n".join(pipe.warnings))
    names = data.eval_names(args)
    found = 0
    nme = []
    cm = np.zeros((11, 11), np.int64)
    start = time.perf_counter()
    for name in names:
        img, pts, label = D.read_sample(args.lapa, "test", name)
        res = pipe.process(img)
        if res["box"] is None or box_iou(res["box"], face_box(pts)) < 0.5:
            continue
        found += 1
        nme.append(evaluate.landmark_metrics(res["points"][None], pts[None])[0])
        if res["labels"] is not None:
            cm += confusion(res["labels"], label)
    ms = 1000 * (time.perf_counter() - start) / max(len(names), 1)
    text = f"found={found / max(len(names), 1):.4f}; nme_found_%={np.mean(nme) if nme else float('nan'):.4f}"
    if cm.sum() > 0:
        text += f"; miou_11={evaluate.parser_metrics(cm)[0]:.4f}"
    print(f"TEST end2end: {text}, {ms:.1f} ms per photo (new face, no tracking)")
    log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "task": "end2end", "exp_id": "", "classifier": "",
         "weights": CONFIG.relative_to(ROOT).as_posix(), "split": "test", "n_images": len(names), "metrics": text,
         "ms_per_image": f"{ms:.2f}", "notes": data.run_note(args)})


def main():
    def extra(ap):
        ap.add_argument("--task", required=True, choices=["detection", "landmark", "parser", "end2end"])
        ap.add_argument("--again", action="store_true")
    args = data.get_args("Test the final model once", extra=extra)
    args.split = "test"
    if not CONFIG.exists():
        sys.exit("best_models.json not found: run select_best.py and full_train.py first")
    config = json.loads(CONFIG.read_text())
    if args.task == "end2end":
        test_end2end(args)
    elif CONFIG_KEY[args.task] not in config:
        sys.exit(f"no {args.task} model in best_models.json yet")
    else:
        test_part(args, args.task, config)


if __name__ == "__main__":
    main()
