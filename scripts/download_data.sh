#!/usr/bin/env bash
# Download COCO val2017 images and annotations, then generate the INT8 calibration set.
set -euo pipefail

DATA_DIR="${COCO_DATA_DIR:-./data}"

echo "Downloading COCO val2017 images (~1 GB)..."
wget -q --show-progress http://images.cocodataset.org/zips/val2017.zip -P "$DATA_DIR"
unzip -q "$DATA_DIR/val2017.zip" -d "$DATA_DIR"
rm "$DATA_DIR/val2017.zip"

echo "Downloading COCO val2017 annotations..."
wget -q --show-progress http://images.cocodataset.org/annotations/annotations_trainval2017.zip -P "$DATA_DIR"
unzip -q "$DATA_DIR/annotations_trainval2017.zip" -d "$DATA_DIR"
rm "$DATA_DIR/annotations_trainval2017.zip"

echo "Generating INT8 calibration set (500 images, seed=42)..."
python -c "
from src.data.calibration_set import generate_calibration_set
import os
generate_calibration_set(
    coco_val_dir=os.environ.get('COCO_DATA_DIR', './data/val2017'),
    output_dir=os.environ.get('CALIBRATION_DIR', './data/calibration'),
    n_images=int(os.environ.get('CALIBRATION_N_IMAGES', 500)),
    seed=int(os.environ.get('CALIBRATION_SEED', 42)),
)
"

echo "Data download complete."
