#!/bin/bash
set -e
DATA_ROOT=/path/to/dataset_root   # must contain train/, val/, test/ with class folders
OUT=checkpoints

# Example training (repeat per model)
python train.py --data_root $DATA_ROOT --model efficientnet_b4 --epochs 20 --batch_size 16 --out $OUT

# Evaluate CEF
python evaluate_cef.py --data_root $DATA_ROOT --model efficientnet_b4 --weights $OUT/efficientnet_b4_best.pt --batch_size 8 --steps 10
