# Data

Data files are not committed to this repo. Run the following to reproduce the full dataset.

## COCO val2017

```bash
# Images (~1 GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip -d data/

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d data/
```

## INT8 Calibration Set

Generated automatically during setup:

```bash
python scripts/export_model.py --calibration
```

500 images sampled from `data/val2017` using `seed=42`. The calibration manifest (`data/calibration/manifest.json`) is committed so results are reproducible without re-generating the set.

## YOLOv8n Weights

Downloaded automatically by ultralytics on first run, or explicitly:

```bash
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P models/
```
