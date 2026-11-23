"""Step 1: make and save the fixed image lists that every experiment uses.

    python experiments/common/make_subsets.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.data import SUBSET_DIR, get_args, subset_names  # noqa: E402


def main():
    args = get_args("Make the subset lists (train 2000 seed 0, val 500 seed 0).")
    train = subset_names(args.lapa, "train", args.train_images, 0)
    val = subset_names(args.lapa, "val", args.val_images, 0)
    print(f"LaPa: {args.lapa}")
    print(f"train subset: {len(train)} images, val subset: {len(val)} images")
    print(f"lists are in {SUBSET_DIR}")


if __name__ == "__main__":
    main()
