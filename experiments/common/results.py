"""Write one row per run to results.csv (same columns for all three models)."""
import csv
from pathlib import Path

COLUMNS = ["exp_id", "approach", "source", "classifier", "settings", "train_images", "seed",
           "metric_1", "metric_2", "metric_3", "metric_4", "train_time_s", "ms_per_image", "notes"]

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
CSV_FILES = {
    "detection": EXPERIMENTS_DIR / "face_detection" / "results.csv",
    "parser": EXPERIMENTS_DIR / "face_parser" / "results.csv",
    "landmark": EXPERIMENTS_DIR / "face_landmark" / "results.csv",
}

# what metric_1 .. metric_4 mean for each model
METRIC_NAMES = {
    "detection": ["recall", "mean_iou", "false_alarm_rate", "unused"],
    "parser": ["miou_11", "lips_iou", "hair_iou", "skin_iou"],
    "landmark": ["nme_106_%", "nme_eyes_%", "nme_lips_%", "failure_rate"],
}


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_row(task, args, exp_id, approach, source, classifier, settings, metrics, train_time_s,
              ms_per_image, notes=""):
    """Append one row. metrics = list of 4 numbers (None = empty)."""
    path = Path(args.csv) if args.csv else CSV_FILES[task]
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0

    from .data import run_note, train_names
    train_images = len(train_names(args))
    extra = run_note(args)
    if extra:
        notes = (extra + "; " + notes).strip("; ")
    if args.split != "val":
        notes = (f"split={args.split}; " + notes).strip("; ")

    metrics = list(metrics) + [None] * (4 - len(metrics))
    row = [exp_id, approach, source, classifier, settings, train_images, args.seed] + \
          [fmt(m) for m in metrics[:4]] + [fmt(float(train_time_s)), fmt(float(ms_per_image)), notes]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(COLUMNS)
        writer.writerow(row)
    names = METRIC_NAMES[task]
    shown = ", ".join(f"{n}={fmt(m)}" for n, m in zip(names, metrics) if m is not None)
    print(f"[{exp_id} {classifier}] {settings} -> {shown}, {ms_per_image:.2f} ms/image  (row added to {path})")


def read_rows(task):
    path = CSV_FILES[task]
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
