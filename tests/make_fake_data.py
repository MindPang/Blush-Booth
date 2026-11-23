"""Make a small fake LaPa folder + fake backgrounds to check that every script runs.
The numbers you get on it mean nothing (the notes column says FAKE DATA).

    python tests/make_fake_data.py --out fake_data
    python experiments/face_detection/classical/d1_template_matching.py --lapa fake_data/LaPa --backgrounds fake_data/backgrounds --quick
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_lapa import make_fake_lapa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fake_data")
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--val", type=int, default=24)
    ap.add_argument("--test", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out)
    make_fake_lapa(out / "LaPa", {"train": args.train, "val": args.val, "test": args.test})
    rng = np.random.RandomState(0)
    for split in ("train", "val", "test"):
        folder = out / "backgrounds" / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(6):
            img = rng.randint(0, 255, (240, 320, 3)).astype(np.uint8)
            img = cv2.GaussianBlur(img, (0, 0), 3)
            cv2.imwrite(str(folder / f"bg_{i}.jpg"), img)
    print(f"fake data in {out}")


if __name__ == "__main__":
    main()
