# Blush Booth - virtual makeup try-on

A real-time virtual makeup app (lips, blush, eyeshadow, eyeliner, lashes, foundation, brows, hair colour and more).
It runs locally with your webcam or on a photo, either in a browser page or in an OpenCV window.
Nothing is uploaded; all pictures stay on your computer.

Work flow:

```
camera frame -> face detector (YOLOv8n) -> landmark model (heatmap CNN, 106 points) -> face parser (U-Net, 11 regions) -> makeup drawn on the regions
```
<img width="3840" height="2160" alt="Blush_Booth_App_Workflow" src="https://github.com/user-attachments/assets/e8617195-1726-4272-9437-47094ddbdeb2" />

The trained weights are included in this repository, so **you can try the app without training anything**.

---

## 1. Try the app (for everyone)

### Try it online
Open **https://blush-booth.netlify.app** in your browser, no install needed. The page downloads the models to your
browser the first time it loads (this can take a moment), so you need an internet connection for that first visit.
Allow camera access, or choose a photo.

For the full local version (Python app, saved looks, OpenCV window), follow the steps below.

### Download
```bash
git clone https://github.com/MindPang/Blush-Booth
cd Blush-Booth
```
(or click **Code -> Download ZIP** on GitHub and unzip it).

### Install
Python 3.9+ is needed.
```bash
pip install -r requirements.txt
```
Tip: use a virtual environment (`python -m venv venv`, then activate it). With an NVIDIA GPU, install the CUDA build
of PyTorch first (see pytorch.org); the app also runs fine on CPU.

### Run

Browser UI (recommended):
```bash
python ui/server.py
```
This opens http://127.0.0.1:5050. Click **Use camera** or **Choose photo**, pick a look, and adjust the layers.
Options: `--port 8000`, `--look natural`, `--no-browser`, `--device cpu`, `--camera 1`.
More about the UI is in [ui/README.md](ui/README.md).

OpenCV window (webcam):
```bash
python makeup_app.py
```

Make up one photo and save it:
```bash
python makeup_app.py --image my_photo.jpg --out result.png
```
(`--vertical` puts Before above After for portrait photos; `--list-looks` lists the saved looks.)

Debug overlays: add `--show-box`, `--show-dots`, `--show-label` to see the detection box, landmarks and parsed regions.

Check that the models load: `python tests/check_models.py --image my_photo.jpg`

**Note:** the first run of the browser UI loads fonts from Google Fonts (falls back to a system font offline).
The pretrained weights needed by the app are already in `model/`; no model download happens at run time.

---

## 2. Where the models come from

### Data
[LaPa](https://github.com/JDAI-CV/lapa-dataset) face parsing dataset: 18,168 train / 2,000 val / 2,000 test images,
each with 11-class parsing labels and 106 landmarks. Face-free **background** images are used as negatives for
detection (train/val/test). The data is **not** in this repository (too large). Put it next to the code:

```
LaPa/{train,val,test}/{images,labels,landmarks}
backgrounds/{train,val,test}
```
or pass `--lapa PATH --backgrounds PATH` to any script (e.g. `--lapa dataset/LaPa --backgrounds dataset/backgrounds`).

### Method: pick every model by a fair experiment
There are three tasks. For each one we ran the same ladder of methods on the **same data subset** with a selection
rule written **before** the runs.

| Task | Classical (lecture) | CNN from scratch | Pretrained |
|---|---|---|---|
| Detection | D1 template matching (SSD/NCC), D2 sliding window + HSV/LBP/edges + kNN/SVM/RF | D3 CenterNet-style | **D4 YOLOv8n** |
| Parser (11 classes) | P1 colour rules, P2 K-means (LAB+xy), P3 patch features + RF/kNN/DecisionTree | P4 U-Net | **P5 U-Net + ResNet18 encoder** |
| Landmarks (106 pts) | L1 mean shape, L2 Harris eye corners, L3 gray+LBP+PCA -> kNN/RF/Ridge | L4 CNN regression, **L5 heatmap + soft-argmax** | L6 MobileNetV2 |

Bold = winner.

**Protocol (all experiments)**
- Train: first 2,000 LaPa train images after shuffling with seed 0 (lists in `experiments/common/subsets/`).
- Evaluation: a fixed 500 val images (+500 face-free scenes for detection). The test split is touched only once, at the end.
- CNNs: 4 epochs, AdamW lr 1e-3, weight decay 1e-4, OneCycle, batch 16. Classical: same images, k-fold CV, then scored on val.
- Metrics: defined in the [experiment reference](#experiment-reference) below.

**Selection rules (val only, fixed in advance)**

| Task | Rule |
|---|---|
| Detection | highest recall with false alarms <= 3% and <= 15 ms/frame |
| Parser | highest mIoU with lips IoU within 0.02 of the best and <= 30 ms/crop |
| Landmarks | lowest NME with <= 15 ms/crop |

Ties within 1 point (0.01 recall/mIoU, 0.1 NME-%; set in `TOLERANCE` in `experiments/select_best.py`) go to the faster model; if the top two are that close the two are
rerun with more seeds. All candidates and the decisions are in [experiments/selection_report.txt](experiments/selection_report.txt).

### Decision (short runs, 2,000 train images, val set)

| Task | Winner | Result | Runner-ups |
|---|---|---|---|
| Detection | D4 YOLOv8n | recall 0.954, false alarms 1.8%, 6.6 ms | D3 0.63-0.78 recall; D2 <= 0.36; D1 ~0.10 |
| Parser | P5 U-Net + ResNet18 | mIoU 0.788, lips IoU 0.724, 5.5 ms | P4 0.370; P3 <= 0.241; P1/P2 ~0.12 |
| Landmarks | L5 heatmap + soft-argmax | NME 6.02%, failures 6.6%, 4.3 ms | L6 8.82%; L4 11.04%; L3 >= 13.46%; L1 22.7% |

Classical methods are fine for teaching but cannot handle LaPa's variety of pose, hair and lighting; pretrained
encoders (P5, D4) and heatmap regression (L5) win clearly while staying far under the speed limits.

### Full training of the winners (all 18,168 train images)

The three winners are retrained with the same code (`full_train.py` runs each winner's script with `--full`) on all train images, for more epochs than the 4-epoch short runs (20 / 8 / 20). Only after that is the test split used, once per model (`test_once.py` refuses to test the same weights twice unless `--again` is given).

| Task | Epochs | Val result | **Test result (used once)** |
|---|---|---|---|
| Detection D4 | 20 | recall 0.974, false alarms 0.2% | recall 0.972, mean IoU 0.849, false alarms 1.2% |
| Parser P5 | 8 | mIoU 0.847, lips 0.779 | mIoU 0.851, lips 0.793, hair 0.916, skin 0.946 |
| Landmarks L5 | 20 | NME 1.95%, failures 0.4% | NME 1.45%, failures 0.45% |
| End to end | - | - | face found 98.2%, NME 1.41%, mIoU 0.842, about 69 ms per frame |

These weights are the three `.pt` files in `model/`, selected by `model/best_models.json`.
Test numbers come from `records/evaluation_runs.csv`; training logs are in `model/*/train_log.txt` and `model/training_runs.csv`.

### Experiment reference

Every experiment run appends one row to the `results.csv` of its task (`experiments/face_*/results.csv`) with the columns
`exp_id, approach, source, classifier, settings, train_images, seed, metric_1..metric_4, train_time_s, ms_per_image, notes`.

| Task | metric_1 | metric_2 | metric_3 | metric_4 |
|---|---|---|---|---|
| Detection | recall (IoU > 0.5 and score >= threshold) | mean IoU of the best box | false-alarm rate on face-free scenes | - |
| Parser | mIoU over 11 classes | lips IoU (upper + lower lip merged) | hair IoU | skin IoU |
| Landmarks | NME, 106 points (% of eye distance) | NME eye points | NME lip points | failure rate (NME > 10%) |

Details worth knowing:
- L2 (Harris) only finds the 4 eye corners, so its `metric_1` is empty and `metric_2` is the NME of those 4 corners.
- P1 uses a* + Otsu for lips, a YCrCb box for skin and dark pixels for hair, with and without open/close clean-up; P2 runs K-means on LAB + (x, y) with K = 6 / 8 / 11.
- Run all experiments on the same PC: the millisecond limits in the selection rules depend on the machine.

---

## 3. Reproduce the training

Needs LaPa + backgrounds (see above) and a GPU is strongly recommended. Paths use `/`; on Windows `\` also works.

```bash
# 0. optional code check (tiny fake data, ~3 min)
python tests/smoke_test.py --offline

# 1. fixed training/eval subset lists
python experiments/common/make_subsets.py

# 2. classical experiments
python experiments/face_detection/classical/d1_template_matching.py
python experiments/face_detection/classical/d2_sliding_window.py
python experiments/face_parser/classical/p1_colour_rules.py
python experiments/face_parser/classical/p2_kmeans.py
python experiments/face_parser/classical/p3_patch_classifier.py
python experiments/face_landmark/classical/l1_mean_shape.py
python experiments/face_landmark/classical/l2_harris.py
python experiments/face_landmark/classical/l3_features_regressor.py

# 3. CNNs from scratch and pretrained (add --workers 4 for faster loading)
python experiments/face_detection/deep_learning/d3_centernet_scratch.py --workers 4
python experiments/face_detection/deep_learning/d4_yolov8n.py --workers 4
python experiments/face_parser/deep_learning/p4_unet_scratch.py --workers 4
python experiments/face_parser/deep_learning/p5_unet_resnet18.py --workers 4
python experiments/face_landmark/deep_learning/l4_cnn_regression.py --workers 4
python experiments/face_landmark/deep_learning/l5_heatmap_softargmax.py --workers 4
python experiments/face_landmark/deep_learning/l6_mobilenetv2.py --workers 4

# 4. apply the selection rule (writes root best_models.json + experiments/selection_report.txt)
python experiments/select_best.py

# 5. train the winners on all images (writes model/*/ *.pt and model/best_models.json)
python experiments/full_train.py --task detection --workers 4
python experiments/full_train.py --task landmark  --workers 4
python experiments/full_train.py --task parser    --workers 4

# 6. evaluate once on the test split
python experiments/test_once.py --task detection
python experiments/test_once.py --task landmark
python experiments/test_once.py --task parser
python experiments/test_once.py --task end2end
```

`run_short_experiments.bat` runs steps 1-4 in one go (about 2-3 hours on a GPU, Windows only).

Useful options on every experiment script: `--quick` (tiny test run), `--variant NAME` (one classifier only),
`--seed N`, `--epochs N`, `--batch N`, `--device cpu`, `--lapa PATH`, `--backgrounds PATH`.
`full_train.py` defaults to 20 epochs for detection and landmarks and 8 for the parser (`--epochs N` to change).

Pretrained starting weights (YOLOv8n on COCO, ResNet18 and MobileNetV2 on ImageNet) are downloaded automatically the
first time D4, P5 or L6 are trained; this needs internet and is not needed just to run the app.

---

## 4. Repository layout

```
makeup_app.py            OpenCV app (camera or --image photo)
ui/                      browser UI (server.py, actions.py, static/)
facekit/                 makeup drawing, pipeline, models (DetectorNet, LandmarkNet, UNet), geometry, losses, metrics
looks/user_looks.json    saved looks
model/                   trained weights (3 .pt files) + best_models.json (full-training winners)
best_models.json         short-run winners (written by select_best.py; fallback if model/best_models.json is missing)
dataset/                 optional: LaPa/ and backgrounds/ may live here (see section 2)
experiments/             all 15 experiments, select_best.py, full_train.py, test_once.py, selection_report.txt
records/                 evaluation runs (test results)
tests/                   smoke test, model check, makeup tests
```

The app imports the winning model's code from `experiments/` (named in `model/best_models.json`), so keep that folder.
If a model file is missing the app still starts and prints which step to run.

## Credits
LaPa dataset (JD AI Research). YOLOv8 by Ultralytics. U-Net via `segmentation-models-pytorch`. MobileNetV2/ResNet18 from torchvision.
