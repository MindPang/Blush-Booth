@echo off
REM Short runs of all experiments (same 2,000 train / 500 val images), then pick the winners.
REM Run from the project folder:  run_short_experiments.bat
REM Set WORKERS=0 if you get DataLoader errors. Add --lapa "E:\Blush Booth\LaPa" if LaPa is somewhere else.
set WORKERS=4

python experiments\common\make_subsets.py || goto :error

REM ---- classical (lecture / Lab 8) ----
python experiments\face_detection\classical\d1_template_matching.py || goto :error
python experiments\face_detection\classical\d2_sliding_window.py || goto :error
python experiments\face_parser\classical\p1_colour_rules.py || goto :error
python experiments\face_parser\classical\p2_kmeans.py || goto :error
python experiments\face_parser\classical\p3_patch_classifier.py || goto :error
python experiments\face_landmark\classical\l1_mean_shape.py || goto :error
python experiments\face_landmark\classical\l2_harris.py || goto :error
python experiments\face_landmark\classical\l3_features_regressor.py || goto :error

REM ---- CNNs (4 epochs, same optimizer / schedule) ----
python experiments\face_detection\deep_learning\d3_centernet_scratch.py --workers %WORKERS% || goto :error
python experiments\face_parser\deep_learning\p4_unet_scratch.py --workers %WORKERS% || goto :error
python experiments\face_landmark\deep_learning\l4_cnn_regression.py --workers %WORKERS% || goto :error
python experiments\face_landmark\deep_learning\l5_heatmap_softargmax.py --workers %WORKERS% || goto :error

REM ---- beyond class (pretrained, needs internet the first time) ----
python experiments\face_detection\deep_learning\d4_yolov8n.py --workers %WORKERS%
python experiments\face_parser\deep_learning\p5_unet_resnet18.py --workers %WORKERS%
python experiments\face_landmark\deep_learning\l6_mobilenetv2.py --workers %WORKERS%

python experiments\select_best.py
echo.
echo Done. Winners are in best_models.json (the app can use them now).
echo Next: python experiments\full_train.py --task detection   (and landmark, parser)
goto :eof

:error
echo A step failed - see the message above.
exit /b 1
