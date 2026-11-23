"""Run every experiment script on tiny fake data (--quick) to check that nothing crashes.
Rows go to tests/smoke_results.csv (not the real results.csv). Takes a few minutes on a CPU.

    python tests/smoke_test.py                 (D4/P5/L6 need internet for the pretrained weights)
    python tests/smoke_test.py --offline       (P5/L6 with random weights, D4 skipped)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "tests" / "fake_data"

SCRIPTS = [
    "experiments/face_detection/classical/d1_template_matching.py",
    "experiments/face_detection/classical/d2_sliding_window.py",
    "experiments/face_detection/deep_learning/d3_centernet_scratch.py",
    "experiments/face_detection/deep_learning/d4_yolov8n.py",
    "experiments/face_parser/classical/p1_colour_rules.py",
    "experiments/face_parser/classical/p2_kmeans.py",
    "experiments/face_parser/classical/p3_patch_classifier.py",
    "experiments/face_parser/deep_learning/p4_unet_scratch.py",
    "experiments/face_parser/deep_learning/p5_unet_resnet18.py",
    "experiments/face_landmark/classical/l1_mean_shape.py",
    "experiments/face_landmark/classical/l2_harris.py",
    "experiments/face_landmark/classical/l3_features_regressor.py",
    "experiments/face_landmark/deep_learning/l4_cnn_regression.py",
    "experiments/face_landmark/deep_learning/l5_heatmap_softargmax.py",
    "experiments/face_landmark/deep_learning/l6_mobilenetv2.py",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()
    if not (FAKE / "LaPa").exists():
        subprocess.run([sys.executable, str(ROOT / "tests" / "make_fake_data.py"), "--out", str(FAKE)], check=True)
    csv = ROOT / "tests" / "smoke_results.csv"
    failed = []
    for script in SCRIPTS:
        cmd = [sys.executable, str(ROOT / script), "--quick", "--lapa", str(FAKE / "LaPa"),
               "--backgrounds", str(FAKE / "backgrounds"), "--csv", str(csv)]
        if args.offline and "d4_" in script:
            print("skip (offline):", script)
            continue
        if args.offline and ("p5_" in script or "l6_" in script):
            cmd.append("--no-pretrained")
        print("\n>>>", script, flush=True)
        if subprocess.run(cmd, cwd=str(ROOT)).returncode != 0:
            failed.append(script)
    print("\nfailed:", failed if failed else "none")
    print(f"rows in {csv}")


if __name__ == "__main__":
    main()
