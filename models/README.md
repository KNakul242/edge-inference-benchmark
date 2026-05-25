# Models

Model files are not committed to this repo — too large and runtime-specific.

## Generating Models

```bash
# 1. Export ONNX from pretrained YOLOv8n
python scripts/export_model.py

# Outputs:
#   models/yolov8n.pt          — pretrained weights (downloaded if not present)
#   models/yolov8n.onnx        — ONNX export, opset 17, static 640x640 input
#   models/yolov8n_int8.onnx   — INT8 quantised ONNX (requires calibration set)
```

## TensorRT Engines

Built inside `notebooks/tensorrt_colab.ipynb` on Colab T4. TRT engine files are not portable across GPU generations or TensorRT versions and are not committed.
