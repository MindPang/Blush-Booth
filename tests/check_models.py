"""Load every saved model (experiments/*/checkpoints and runs/) and run it once through the app pipeline
functions on one photo. Use it after the experiments to see that every model can be used by the app.

    python tests/check_models.py --image some_face.jpg
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import argparse
import importlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from experiments.select_best import MODULES  # noqa: E402
from facekit.pipeline import FacePipeline  # noqa: E402

KEYS = {"D": "detector", "L": "landmark", "P": "parser"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"could not read {args.image}")

    files = sorted(ROOT.glob("experiments/*/checkpoints/*.*")) + sorted(ROOT.glob("runs/*/*.*"))
    found = {"detector": [], "landmark": [], "parser": []}
    for f in files:
        exp_id = f.name.split("_")[0]
        if exp_id in MODULES:
            found[KEYS[exp_id[0]]].append((exp_id, f))
    if not any(found.values()):
        sys.exit("no saved models found yet")

    # the first model of each part is used as the "other" parts when one part is tested
    first = {k: v[0] for k, v in found.items() if v}
    for part, items in found.items():
        for exp_id, f in items:
            config = {}
            for k, (e, path) in first.items():
                config[k] = {"module": MODULES[e], "weights": str(path), "threshold": 0.0}
            config[part] = {"module": MODULES[exp_id], "weights": str(f), "threshold": 0.0}
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
                json.dump(config, tmp)
            try:
                pipe = FacePipeline(tmp.name, args.device)
                res = pipe.process(frame)
                n_pts = 0 if res["points"] is None else len(res["points"])
                labels = "none" if res["labels"] is None else str(sorted(set(res["labels"].ravel().tolist())))
                print(f"OK    {part:8s} {f.relative_to(ROOT)}: box={res['box'] is not None} points={n_pts} "
                      f"label classes={labels}")
            except Exception as e:  # noqa: BLE001 - show every problem
                print(f"ERROR {part:8s} {f.relative_to(ROOT)}: {e!r}")
            Path(tmp.name).unlink()


if __name__ == "__main__":
    main()
