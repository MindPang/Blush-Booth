"""Train the winner of one part on ALL LaPa train images (18,168), then point best_models.json at it.

It runs the winner's own experiment script with --full, so the code is exactly the same as in the
short run. Everything a full run makes goes in model/ (the short-run files are not touched):
    model/<part>/            the trained weights, results.csv (val metrics) and train_log.txt (console output)
    model/training_runs.csv  one line per full run
    model/best_models.json   copy of best_models.json pointing at the full weights
                             (the root best_models.json keeps the short-run weights)

    python experiments/full_train.py --task detection
    python experiments/full_train.py --task landmark --epochs 20 --workers 4
    python experiments/full_train.py --task parser
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "best_models.json"              # winners of the short runs (read only here)
OUT = ROOT / "model"                            # all full-training output
RECORDS = OUT / "training_runs.csv"
FULL_CONFIG = OUT / "best_models.json"

CONFIG_KEY = {"detection": "detector", "parser": "parser", "landmark": "landmark"}
# default full-training epochs (the old project used 30 / 40 / 12; less here so it fits in the time we have)
FULL_EPOCHS = {"detection": 20, "landmark": 20, "parser": 8}
EXTENSIONS = {"D1": ".npz", "D2": ".pkl", "P1": ".npz", "P2": ".pkl", "P3": ".pkl", "L1": ".npz", "L3": ".pkl"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(CONFIG_KEY))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lapa", default=None)
    ap.add_argument("--backgrounds", default=None)
    ap.add_argument("--quick", action="store_true", help="only to test the code")
    args = ap.parse_args()

    if not CONFIG.exists():
        sys.exit("best_models.json not found: run the experiments and experiments/select_best.py first")
    config = json.loads(CONFIG.read_text())
    key = CONFIG_KEY[args.task]
    if key not in config:
        sys.exit(f"no {key} chosen yet: run experiments/select_best.py first")
    entry = config[key]

    exp_id = entry["exp_id"]
    script = ROOT / (entry["module"].replace(".", "/") + ".py")
    epochs = args.epochs or FULL_EPOCHS[args.task]
    ext = EXTENSIONS.get(exp_id, ".pt")
    task_dir = OUT / args.task
    task_dir.mkdir(parents=True, exist_ok=True)
    save = task_dir / f"{exp_id}_{entry['classifier']}_full{ext}".replace("=", "")
    log_path = task_dir / "train_log.txt"

    cmd = [sys.executable, str(script), "--full", "--variant", entry["classifier"], "--seed", str(entry["seed"]),
           "--epochs", str(epochs), "--workers", str(args.workers), "--device", args.device, "--save", str(save),
           "--csv", str(task_dir / "results.csv")]
    if args.batch:
        cmd += ["--batch", str(args.batch)]
    if args.lapa:
        cmd += ["--lapa", args.lapa]
    if args.backgrounds:
        cmd += ["--backgrounds", args.backgrounds]
    if args.quick:
        cmd += ["--quick"]
    print("running:", " ".join(cmd))
    start = time.perf_counter()
    with open(log_path, "a", encoding="utf-8") as log:      # show the output and keep a copy
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')}  {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = proc.wait()
    seconds = time.perf_counter() - start
    status = "ok" if returncode == 0 and save.exists() else "failed"

    RECORDS.parent.mkdir(parents=True, exist_ok=True)
    is_new = not RECORDS.exists()
    with open(RECORDS, "a", newline="") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["time", "task", "exp_id", "classifier", "epochs", "status", "seconds", "weights", "command"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), args.task, exp_id, entry["classifier"], epochs, status,
                    f"{seconds:.0f}", save.relative_to(ROOT).as_posix(), " ".join(cmd)])

    if status != "ok":
        sys.exit(f"full training failed, {FULL_CONFIG.relative_to(ROOT).as_posix()} was not changed")
    if args.quick:
        print("quick test only: best_models.json was not changed")
        return
    # model/best_models.json: start from it if an earlier full run made it, else from the short-run file
    full_config = json.loads(FULL_CONFIG.read_text()) if FULL_CONFIG.exists() else config
    entry["weights"] = save.relative_to(ROOT).as_posix()
    entry["stage"] = "full"
    entry["full_epochs"] = epochs
    full_config[key] = entry
    FULL_CONFIG.write_text(json.dumps(full_config, indent=2))
    print(f"done in {seconds / 60:.1f} min. {FULL_CONFIG.relative_to(ROOT).as_posix()} now uses {entry['weights']}")
    print(f"next: python experiments/test_once.py --task {args.task}")


if __name__ == "__main__":
    main()
