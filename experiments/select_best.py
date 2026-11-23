"""Pick the best model of each part from the short runs, with the rule we fixed BEFORE the runs.

  detection: highest recall, with false alarms <= 3% and <= 15 ms per frame
  parser:    highest mIoU, with lips IoU within 0.02 of the best lips IoU and <= 30 ms per crop
  landmark:  lowest NME, with <= 15 ms per crop
  If the top two are within 1 point, the faster one wins (and both should be rerun with seeds 1 and 2).

Only val rows of short runs count (QUICK, FAKE DATA, FULL TRAIN and test rows are skipped).
Rows of the same experiment with different seeds are averaged.
Writes best_models.json (the app reads it) and experiments/selection_report.txt.

    python experiments/select_best.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.common.data import checkpoint_dir, checkpoint_name  # noqa: E402
from experiments.common.results import METRIC_NAMES, read_rows  # noqa: E402

CONFIG = ROOT / "best_models.json"
REPORT = ROOT / "experiments" / "selection_report.txt"

MODULES = {
    "D1": "experiments.face_detection.classical.d1_template_matching",
    "D2": "experiments.face_detection.classical.d2_sliding_window",
    "D3": "experiments.face_detection.deep_learning.d3_centernet_scratch",
    "D4": "experiments.face_detection.deep_learning.d4_yolov8n",
    "P1": "experiments.face_parser.classical.p1_colour_rules",
    "P2": "experiments.face_parser.classical.p2_kmeans",
    "P3": "experiments.face_parser.classical.p3_patch_classifier",
    "P4": "experiments.face_parser.deep_learning.p4_unet_scratch",
    "P5": "experiments.face_parser.deep_learning.p5_unet_resnet18",
    "L1": "experiments.face_landmark.classical.l1_mean_shape",
    "L3": "experiments.face_landmark.classical.l3_features_regressor",
    "L4": "experiments.face_landmark.deep_learning.l4_cnn_regression",
    "L5": "experiments.face_landmark.deep_learning.l5_heatmap_softargmax",
    "L6": "experiments.face_landmark.deep_learning.l6_mobilenetv2",
}

# "1 point": recall / mIoU are 0..1 so 1 point = 0.01. NME is already in %, and the numbers are small
# (about 1.5 - 10), so we use 0.1 NME-% as "1 point" there. Change it here if the team decides otherwise.
TOLERANCE = {"detection": 0.01, "parser": 0.01, "landmark": 0.10}
CONFIG_KEY = {"detection": "detector", "parser": "parser", "landmark": "landmark"}

lines = []


def say(text=""):
    print(text)
    lines.append(text)


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def usable(row, include_quick):
    notes = row.get("notes", "")
    if "FULL TRAIN" in notes or "split=test" in notes:
        return False
    if not include_quick and ("QUICK" in notes or "FAKE DATA" in notes):
        return False
    return True


def group_rows(rows):
    """Average the rows of the same experiment / classifier / settings over the seeds."""
    groups = {}
    for r in rows:
        key = (r["exp_id"], r["classifier"], r["settings"])
        groups.setdefault(key, []).append(r)
    out = []
    for (exp_id, classifier, settings), members in groups.items():
        g = {"exp_id": exp_id, "classifier": classifier, "settings": settings,
             "seeds": sorted(int(m["seed"]) for m in members)}
        for k in ("metric_1", "metric_2", "metric_3", "metric_4", "ms_per_image"):
            values = [number(m[k]) for m in members if number(m[k]) is not None]
            g[k] = sum(values) / len(values) if values else None
        out.append(g)
    return out


def rank(task, groups):
    """Returns (sorted list of candidates that pass the limits, text that explains the limits)."""
    if task == "detection":
        ok = [g for g in groups if g["metric_1"] is not None and g["metric_3"] is not None
              and g["metric_3"] <= 0.03 and g["ms_per_image"] <= 15]
        ok.sort(key=lambda g: -g["metric_1"])
        return ok, "false alarms <= 3% and <= 15 ms/frame, highest recall"
    if task == "parser":
        ok = [g for g in groups if g["metric_1"] is not None and g["ms_per_image"] <= 30]
        if ok:
            best_lips = max(g["metric_2"] for g in ok)
            ok = [g for g in ok if g["metric_2"] >= best_lips - 0.02]
        ok.sort(key=lambda g: -g["metric_1"])
        return ok, "<= 30 ms/crop, lips IoU within 0.02 of the best, highest mIoU"
    ok = [g for g in groups if g["metric_1"] is not None and g["ms_per_image"] <= 15]
    ok.sort(key=lambda g: g["metric_1"])
    return ok, "<= 15 ms/crop, lowest NME (106 points)"


def find_checkpoint(exp_id, classifier, seed, include_quick=False):
    """The saved short-run model of this experiment (any extension)."""
    folder = checkpoint_dir(exp_id)
    tags = [""]
    if include_quick:
        tags.append("_quick")
    for tag in tags:
        name = checkpoint_name(exp_id, classifier, seed, tag)
        for ext in (".pt", ".pkl", ".npz"):
            if (folder / (name + ext)).exists():
                return (folder / (name + ext)).relative_to(ROOT).as_posix()
    return None


def threshold_of(settings):
    for part in settings.split():
        if part.startswith("thr="):
            return float(part[4:])
    return 0.40


def show(task, g):
    names = METRIC_NAMES[task]
    values = []
    for i, n in enumerate(names):
        v = g[f"metric_{i + 1}"]
        if v is not None:
            values.append(f"{n}={v:.4f}")
    return (f"{g['exp_id']:3s} {g['classifier']:15s} {g['settings'][:45]:45s} " + " ".join(values)
            + f"  {g['ms_per_image']:.1f} ms  seeds={g['seeds']}")


def choose(task, include_quick):
    rows = [r for r in read_rows(task) if usable(r, include_quick)]
    groups = [g for g in group_rows(rows) if g["exp_id"] in MODULES]
    say(f"===== {task}: {len(groups)} candidate runs")
    if not groups:
        say("  no results yet - run the experiments first")
        return None
    for g in groups:
        say("  " + show(task, g))

    ranked, rule = rank(task, groups)
    say(f"  rule: {rule}")
    if not ranked:
        say("  WARNING: no run meets the limits, so I take the best main metric without the limits")
        ranked = sorted([g for g in groups if g["metric_1"] is not None], key=lambda g: g["metric_1"],
                        reverse=(task != "landmark"))
        if not ranked:
            return None

    # keep only the best row of each model (e.g. the best threshold), so the top two are different models
    seen = set()
    unique = []
    for g in ranked:
        if (g["exp_id"], g["classifier"]) not in seen:
            seen.add((g["exp_id"], g["classifier"]))
            unique.append(g)
    ranked = unique

    best = ranked[0]
    if len(ranked) > 1:
        second = ranked[1]
        if abs(best["metric_1"] - second["metric_1"]) <= TOLERANCE[task]:
            say(f"  top two are within 1 point: {best['exp_id']} {best['classifier']} and "
                f"{second['exp_id']} {second['classifier']} -> the faster one wins")
            if len(best["seeds"]) < 3 or len(second["seeds"]) < 3:
                say("  NOTE: rerun both with --seed 1 and --seed 2, then run select_best.py again")
            if second["ms_per_image"] < best["ms_per_image"]:
                best = second
    say(f"  WINNER: {show(task, best)}")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-quick", action="store_true", help="also use QUICK / FAKE rows (only to test the code)")
    args = ap.parse_args()

    config = {}
    if CONFIG.exists():
        config = json.loads(CONFIG.read_text())

    for task in ("detection", "parser", "landmark"):
        best = choose(task, args.include_quick)
        say()
        if best is None:
            continue
        key = CONFIG_KEY[task]
        old = config.get(key, {})
        if old.get("stage") == "full" and old.get("exp_id") == best["exp_id"] \
                and old.get("classifier") == best["classifier"]:
            say(f"  {key}: same winner as the fully trained model already in best_models.json, kept it")
            continue
        weights = find_checkpoint(best["exp_id"], best["classifier"], best["seeds"][0], args.include_quick)
        if weights is None:
            say(f"  WARNING: no saved short-run model found for {best['exp_id']} {best['classifier']}")
        entry = {
            "exp_id": best["exp_id"],
            "classifier": best["classifier"],
            "settings": best["settings"],
            "seed": best["seeds"][0],
            "module": MODULES[best["exp_id"]],
            "weights": weights,
            "stage": "short run (2,000 train images) - run experiments/full_train.py next",
            "val_metrics": {n: best[f"metric_{i + 1}"] for i, n in enumerate(METRIC_NAMES[task])},
            "ms_per_image": best["ms_per_image"],
        }
        if task == "detection":
            entry["threshold"] = threshold_of(best["settings"])
        if task == "landmark":
            entry["crop_size"] = 160
        if task == "parser":
            entry["crop_size"] = 256
        config[key] = entry

    CONFIG.write_text(json.dumps(config, indent=2))
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"wrote {CONFIG} and {REPORT}")


if __name__ == "__main__":
    main()
