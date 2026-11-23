# Revised Blush Booth - virtual makeup try-on

Same app and same UI as the old Blush Booth. What changed is the model part: every model is now chosen by
a fair experiment (lecture method -> Lab 8 classifiers -> beyond class), all run on the same data, with a
selection rule fixed before the runs. The winners are trained fully and the app and the browser server
load them from `model/best_models.json` (made by `full_train.py`). If full training has not been run yet,
they fall back to the short-run winners in `best_models.json`. Use `--config best_models.json` to force
the short-run models.

## What changed compared with the old project

| | Old Blush Booth | Revised |
|---|---|---|
| UI (`makeup_app.py`, `ui/`, `facekit/makeup.py`, `looks/`) | - | the same (only the model loading changed: `--config best_models.json` instead of `--det/--lmk/--par`) |
| Models | 3 CNNs fixed in advance | 15 experiments, winner picked by a rule (`experiments/select_best.py`) |
| Data code | big cache step + Dataset classes | reads LaPa directly (`experiments/common/data.py`), same crops / scene sizes as before |
| Pipeline | `facekit/pipeline.py` tied to the 3 CNNs | small pipeline that works with any winner (classical or CNN) |
| Missing models | crash | the app starts and tells you which step to run |
| Web (Netlify / ONNX) version | yes | not carried over (only the local app and the browser UI with `ui/server.py`) |

## Setup

```bat
cd "E:\Revised Blush Booth"
pip install -r requirements.txt
```
For a GPU, install the CUDA build of PyTorch first (see pytorch.org). The pretrained models (D4, P5, L6)
download their weights from the internet the first time they run.

**Data (needed):** LaPa (`train/ val/ test/`, each with `images/ labels/ landmarks/`) and the face-free
`backgrounds/` (`train/ val/ test/`). Either copy both folders into `E:\Revised Blush Booth\`, or leave them
where they are: if there is no `LaPa\` / `backgrounds\` in this folder, the scripts automatically use
`E:\Blush Booth\LaPa` and `E:\Blush Booth\backgrounds`. You can also give the path: `--lapa "D:\data\LaPa"`.

**Models:** there are none at the start. Nothing from the old `runs/` is used. You get models by running the
steps below. The app already works after step 3 (with the short-run models).

## Run order (Windows)

Times are rough guesses for one PC with an NVIDIA GPU (CPU only is much slower for the CNNs).
Measure the first step and scale. `--workers 4` makes the CNN data loading faster (use 0 if it gives errors).

| Step | Command | Rough time |
|---|---|---|
| 0. check the code (optional) | `python tests\smoke_test.py --offline` | 3 min |
| 1. subset lists | `python experiments\common\make_subsets.py` | < 1 min |
| 2a. classical detection | `python experiments\face_detection\classical\d1_template_matching.py` then `...\d2_sliding_window.py` | 3 min + 10-15 min |
| 2b. classical parser | `python experiments\face_parser\classical\p1_colour_rules.py`, `p2_kmeans.py`, `p3_patch_classifier.py` | 3 + 5 + 10 min |
| 2c. classical landmarks | `python experiments\face_landmark\classical\l1_mean_shape.py`, `l2_harris.py`, `l3_features_regressor.py` | 2 + 2 + 5 min |
| 3a. CNNs from scratch | `d3_centernet_scratch.py`, `p4_unet_scratch.py`, `l4_cnn_regression.py`, `l5_heatmap_softargmax.py` (each with `--workers 4`) | 5-15 min each |
| 3b. pretrained (optional) | `d4_yolov8n.py`, `p5_unet_resnet18.py`, `l6_mobilenetv2.py` | 5-15 min each |
| 4. pick winners | `python experiments\select_best.py` | seconds |
| 5. full training | `python experiments\full_train.py --task detection --workers 4` (same for `landmark`, `parser`) | CNN: 20-80 min each, classical: 5-20 min |
| 6. test once | `python experiments\test_once.py --task detection` (and `landmark`, `parser`, `end2end`) | 2-5 min each |
| 7. app | `python ui\server.py` or `python makeup_app.py` | - |

`run_short_experiments.bat` runs steps 1-4 in one go (about 2-3 hours in total on a GPU).

**If time is short:** after step 4 the app already uses the short-run winners; once `full_train.py` has
written `model/best_models.json` the app switches to the fully trained models by itself. `full_train.py` takes
`--epochs N` (defaults: detection 20, landmark 20, parser 8). Lower it, e.g. `--epochs 6`, to finish
in time, and write the number of epochs in the report. Steps 2-3 can run at the same time in two
windows if the PC has enough memory.

Multi-seed rule: if `select_best.py` says the top two are within 1 point, rerun those two scripts with
`--seed 1` and `--seed 2` (same command + `--seed 1`), then run `select_best.py` again (it averages seeds).

Useful options for every experiment script: `--quick` (tiny test run), `--variant NAME` (only one classifier,
e.g. `--variant RandomForest`), `--seed N`, `--epochs N`, `--batch N`, `--device cpu`, `--lapa PATH`.

## The experiments

All experiments use the same data: train = first 2,000 LaPa train images after shuffling with seed 0
(list saved in `experiments/common/subsets/`), eval = fixed 500 val images (+500 face-free scenes for
detection). The test split is only used by `test_once.py`. CNNs: 4 epochs, AdamW lr 1e-3, weight decay
1e-4, OneCycle schedule, batch 16. Classical: features from the same 2,000 images, k-fold CV on train
(Lab 8 style, shown in the notes column), then scored on the 500 val images.

Every run adds one row to `results.csv` in its model folder:
`exp_id, approach, source, classifier, settings, train_images, seed, metric_1..4, train_time_s, ms_per_image, notes`

| Model | metric_1 | metric_2 | metric_3 | metric_4 |
|---|---|---|---|---|
| face_detection | recall (IoU > 0.5 and score >= thr) | mean IoU of the best box | false-alarm rate on face-free scenes | - |
| face_parser | mIoU over 11 classes | lips IoU (upper + lower merged) | hair IoU | skin IoU |
| face_landmark | NME 106 points (% of eye distance) | NME eye points | NME lip points | failure rate (NME > 10%) |

(L2 Harris only finds 4 eye corners, so its metric_2 is the NME of those 4 corners and metric_1 is empty.)

| ID | Detection | ID | Parser | ID | Landmarks |
|---|---|---|---|---|---|
| D1 | template matching, average face, 5 scales, SSD vs NCC (lecture, Lab 7) | P1 | colour rules: a* + Otsu (lips), YCrCb (skin), dark pixels (hair), open/close | L1 | mean shape (baseline) |
| D2 | sliding window + HSV/LBP/edge features + kNN / SVM / Random Forest (Lab 8) | P2 | K-means on LAB + (x, y), K = 6/8/11 | L2 | Harris corners for the 4 eye corners |
| D3 | old CenterNet-style DetectorNet, from scratch | P3 | 8 x 8 patch features + Random Forest / kNN / Decision Tree (Lab 8) | L3 | gray + LBP -> PCA 50 -> kNN / Random Forest / Ridge |
| D4 | YOLOv8n pretrained, fine-tuned | P4 | old U-Net, from scratch, CE + Dice | L4 | small CNN, direct regression |
| | | P5 | U-Net with pretrained ResNet18 encoder | L5 | old LandmarkNet, heatmaps + soft-argmax |
| | | | | L6 | MobileNetV2 pretrained + regression head |

**Selection rule (val only, fixed before the runs):**
detection = highest recall with false alarms <= 3% and <= 15 ms/frame;
parser = highest mIoU with lips IoU within 0.02 of the best and <= 30 ms/crop;
landmark = lowest NME with <= 15 ms/crop; within 1 point -> the faster one.
"1 point" = 0.01 for recall / mIoU and 0.1 NME-% for landmarks (set in `TOLERANCE` in `select_best.py`).
Run all experiments on the same PC, because the ms limits depend on the machine.
The choice and all candidates are written to `experiments/selection_report.txt`.

## The pretrained models (for the Q&A)

We use them only with retraining on our LaPa data (as the course allows).

**YOLOv8n (D4, detection).** A one-stage detector from Ultralytics, the smallest ("nano") version, about
3.2 M parameters. Backbone: CSP-Darknet style conv blocks (C2f) that make feature maps at 3 sizes;
neck: a feature pyramid (FPN + PAN) that mixes the small and big maps; head: anchor-free, for every grid cell
it predicts a class score ("face" or not) + the box (distances from the cell to the 4 sides). Overlapping boxes are removed with
NMS. It was pretrained on COCO (80 everyday classes). We fine-tune it with one class "face" on the same
256-px scenes as D3 for 4 epochs; we keep the box with the highest confidence.

**U-Net with ResNet18 encoder (P5, parser, segmentation_models_pytorch).** Same U shape as our P4: an
encoder that makes the image smaller and the features deeper, and a decoder that makes it big again, with
skip connections that copy the encoder maps to the decoder so the borders stay sharp. The difference: the
encoder is ResNet18 (residual blocks, 11 M parameters) pretrained on ImageNet, so it already knows edges and
textures. The decoder and the 11-class output are new and trained on LaPa with the same CE + Dice loss.

**MobileNetV2 (L6, landmarks, torchvision).** A small CNN made for phones (about 2.2 M parameters without
the classifier). Its "inverted residual" blocks first widen the channels with a 1x1 conv, then use a cheap
depthwise 3x3 conv, then narrow again with a 1x1 conv, with a shortcut around the block. We take the
ImageNet-pretrained feature part, average-pool it to 1280 numbers and add one linear layer that outputs the
212 numbers (106 x, y). Trained with the same L1 / eye-distance loss as L4 and L5.

## How the app uses the winners

`model/best_models.json` (fully trained, written by `full_train.py`) or, before full training,
`best_models.json` (short runs, written by `select_best.py`) says for each part which
experiment script and which saved file to use. Every experiment script has the same small functions:
`load_model(path, device)` and `detect(model, frame)` / `predict_points(model, crop)` / `parse_crop(model, crop)`.
`facekit/pipeline.py` calls them: detector -> landmark crop (160) -> parser crop (256), exactly like before.
If a model is missing the app still starts and prints e.g. "No trained detector yet ... run ...".
`python tests\check_models.py --image some_photo.jpg` loads every saved model through the app pipeline.

## Folder layout

```
makeup_app.py            OpenCV app (camera or --image photo)
ui/                      browser UI (server.py + static/)
facekit/                 makeup.py (makeup drawing), pipeline.py, models.py (DetectorNet, LandmarkNet, UNet),
                         datasets.py (LaPa reading, scenes, crops), geometry.py, layout.py, losses.py, metrics.py, viz.py
looks/user_looks.json    saved looks
best_models.json         made by select_best.py / full_train.py (not there at the start)
experiments/
  common/                data.py (subsets, eval sets), features.py, evaluate.py, results.py, cnn.py, make_subsets.py
  face_detection/        classical/ (d1, d2)  deep_learning/ (d3, d4)  results.csv  checkpoints/
  face_parser/           classical/ (p1, p2, p3)  deep_learning/ (p4, p5)  results.csv  checkpoints/
  face_landmark/         classical/ (l1, l2, l3)  deep_learning/ (l4, l5, l6)  results.csv  checkpoints/
  select_best.py  full_train.py  test_once.py
runs/<task>/             fully trained winners (made by full_train.py)
records/                 training_runs.csv, evaluation_runs.csv
tests/                   smoke_test.py, make_fake_data.py, fake_lapa.py, check_models.py, test_makeup.py
```

Rows marked `QUICK` or `FAKE DATA` in the notes are code tests and are ignored by `select_best.py`.
