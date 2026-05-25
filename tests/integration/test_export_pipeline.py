"""Integration tests for the ONNX export pipeline.

These tests require:
  - models/yolov8n.pt (downloaded by ultralytics or manually)
  - ultralytics, onnx, onnxruntime installed

Run with: pytest tests/integration/ -m integration
Skip in CI/unit-only runs: pytest tests/ -m "not integration"
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_export_produces_valid_onnx_file(tmp_path):
    """Exported .onnx file must load without error via onnxruntime."""
    pytest.importorskip("ultralytics")
    pytest.importorskip("onnxruntime")

    from src.export.onnx_exporter import OnnxExporter
    import onnxruntime as ort

    exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path), opset=17)
    onnx_path = exporter.export()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    assert session is not None


@pytest.mark.integration
def test_export_output_shape_matches_yolov8n(tmp_path):
    """YOLOv8n ONNX output must be (1, 84, 8400) for 640×640 input."""
    pytest.importorskip("ultralytics")
    pytest.importorskip("onnxruntime")

    import numpy as np
    import onnxruntime as ort
    from src.export.onnx_exporter import OnnxExporter

    exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path))
    onnx_path = exporter.export()

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    output = session.run(None, {input_name: dummy})

    assert output[0].shape == (1, 84, 8400), f"Unexpected output shape: {output[0].shape}"


@pytest.mark.integration
def test_onnx_parity_with_pytorch(tmp_path):
    """ONNX output must match PyTorch output within atol=1e-4."""
    pytest.importorskip("ultralytics")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")

    import numpy as np
    import torch
    import onnxruntime as ort
    from ultralytics import YOLO
    from src.export.onnx_exporter import OnnxExporter, validate_output_parity

    exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path))
    onnx_path = exporter.export()

    dummy = np.random.rand(1, 3, 640, 640).astype(np.float32)

    # PyTorch inference
    pt_model = YOLO("yolov8n.pt").model.eval()
    with torch.no_grad():
        pt_out = pt_model(torch.from_numpy(dummy)).cpu().numpy()

    # ONNX inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: dummy})[0]

    validate_output_parity(pt_out, onnx_out, atol=1e-4)
