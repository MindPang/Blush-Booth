"""D4 (beyond class): YOLOv8n, pretrained on COCO by Ultralytics, fine-tuned on our LaPa scenes.

Why: YOLOv8n is a small, fast one-stage detector (about 3.2 M parameters). It already knows general
image features from COCO, so a short fine-tune on our scenes may beat a detector trained from scratch.
We train only one class ("face"), input 256, 4 epochs, AdamW lr 1e-3 with a cosine schedule (close to the
other CNNs). Needs:  pip install ultralytics

The scenes are written to experiments/face_detection/yolo_data/ in the YOLO format
(images/*.jpg + labels/*.txt with "0 cx cy w h", all in 0..1).

    python experiments/face_detection/deep_learning/d4_yolov8n.py
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid OMP Error #15 (duplicate libiomp5md.dll)

import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from experiments.common import data, evaluate, results  # noqa: E402
from facekit import datasets as D  # noqa: E402

EXP_ID = "D4"
THRESHOLDS = [0.3, 0.4, 0.5]
YOLO_DIR = Path(__file__).resolve().parents[1] / "yolo_data"


def write_yolo_split(folder, scene_list):
    (folder / "images").mkdir(parents=True, exist_ok=True)
    (folder / "labels").mkdir(parents=True, exist_ok=True)
    for i, (scene, box) in enumerate(scene_list):
        cv2.imwrite(str(folder / "images" / f"{i:05d}.jpg"), scene)
        text = ""
        if box is not None:
            x0, y0, x1, y1 = np.clip(box, [0, 0, 0, 0], [D.SCENE_W, D.SCENE_H, D.SCENE_W, D.SCENE_H])
            cx = (x0 + x1) / 2 / D.SCENE_W
            cy = (y0 + y1) / 2 / D.SCENE_H
            w = (x1 - x0) / D.SCENE_W
            h = (y1 - y0) / D.SCENE_H
            text = f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
        (folder / "labels" / f"{i:05d}.txt").write_text(text)


def yolo_dir(args):
    """experiments/face_detection/yolo_data/ for short runs; next to the saved model when --save is given
    (full training saves in model/detection/, so its YOLO copy goes to model/detection/yolo_data/)."""
    if args.save:
        return Path(args.save).resolve().parent / "yolo_data"
    return YOLO_DIR


def make_dataset(args, eval_scenes, eval_boxes, folder):
    """Train scenes (fixed, one per train image + 25% empty) and the val scenes, in YOLO format."""
    if folder.exists():
        shutil.rmtree(folder)
    write_yolo_split(folder / "train", data.detection_train_scenes(args, seed=args.seed))
    write_yolo_split(folder / "val", zip(eval_scenes, eval_boxes))
    yaml = folder / "faces.yaml"
    yaml.write_text(f"path: \"{folder.as_posix()}\"\ntrain: train/images\nval: val/images\nnames:\n  0: face\n")
    return yaml


def load_model(path, device="cpu"):
    from ultralytics import YOLO
    model = YOLO(path)
    model.my_device = str(device)
    return model


def detect(model, frame):
    """Best box of YOLO (in frame pixels) and its confidence."""
    device = getattr(model, "my_device", "cpu")
    if device == "cuda":
        device = 0
    out = model.predict(frame, imgsz=256, conf=0.01, max_det=5, verbose=False, device=device)[0]
    if len(out.boxes) == 0:
        return None, 0.0
    best = int(out.boxes.conf.argmax())
    box = out.boxes.xyxy[best].cpu().numpy().astype(np.float64)
    return box, float(out.boxes.conf[best])


def main():
    args = data.get_args("D4 YOLOv8n pretrained + fine-tuned")
    data.set_seed(args.seed)
    from ultralytics import YOLO
    from facekit.models import pick_device

    device = str(pick_device(args.device))
    yolo_device = 0 if device == "cuda" else device

    scenes, boxes = data.detection_eval_scenes(args)
    folder = yolo_dir(args)
    yaml = make_dataset(args, scenes, boxes, folder)

    model = YOLO("yolov8n.pt")       # downloads the COCO weights the first time
    start = time.perf_counter()
    model.train(data=str(yaml), imgsz=256, epochs=args.epochs, batch=args.batch, single_cls=True,
                optimizer="AdamW", lr0=args.lr, cos_lr=True, seed=args.seed, device=yolo_device,
                workers=args.workers, project=str(folder / "runs"), name="train", exist_ok=True,
                plots=False, verbose=False)
    train_time = time.perf_counter() - start

    last = folder / "runs" / "train" / "weights" / "last.pt"   # last epoch, like the other CNNs
    save_path = data.model_path(args, EXP_ID, "yolov8n", ".pt")
    shutil.copy(last, save_path)

    model = load_model(str(save_path), device)
    raw, ms = evaluate.run_detector(lambda img: detect(model, img), scenes, boxes)
    for thr in THRESHOLDS:
        metrics = evaluate.detection_metrics(raw, thr)
        results.write_row("detection", args, EXP_ID, "deep learning", "pretrained COCO (ultralytics), fine-tuned",
                          "yolov8n", f"thr={thr} epochs={args.epochs} imgsz=256 lr={args.lr}", metrics,
                          train_time, ms, device)


if __name__ == "__main__":
    main()
