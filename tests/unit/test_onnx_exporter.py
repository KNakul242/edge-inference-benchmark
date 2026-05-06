"""Unit tests for the ONNX export pipeline.

Tests verify export parameter correctness and parity validation logic
without loading real model weights (mocked at the ultralytics boundary).
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.export.onnx_exporter import OnnxExporter, validate_output_parity


class TestOnnxExporterParameters:
    """Verify the exporter passes locked parameters to ultralytics."""

    def test_export_uses_opset_17(self, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "yolov8n.onnx")

        with patch("src.export.onnx_exporter.YOLO", return_value=mock_model):
            exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path))
            exporter.export()

        call_kwargs = mock_model.export.call_args.kwargs
        assert call_kwargs["opset"] == 17, "opset must be 17 for TensorRT 8.6.x compatibility"

    def test_export_uses_static_input_shape(self, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "yolov8n.onnx")

        with patch("src.export.onnx_exporter.YOLO", return_value=mock_model):
            exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path))
            exporter.export()

        call_kwargs = mock_model.export.call_args.kwargs
        assert call_kwargs["dynamic"] is False, "dynamic=False required for fixed edge deployment shape"

    def test_export_enables_simplification(self, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "yolov8n.onnx")

        with patch("src.export.onnx_exporter.YOLO", return_value=mock_model):
            exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path))
            exporter.export()

        call_kwargs = mock_model.export.call_args.kwargs
        assert call_kwargs["simplify"] is True

    def test_export_uses_correct_format(self, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "yolov8n.onnx")

        with patch("src.export.onnx_exporter.YOLO", return_value=mock_model):
            exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path))
            exporter.export()

        call_kwargs = mock_model.export.call_args.kwargs
        assert call_kwargs["format"] == "onnx"

    def test_export_uses_640_input_size(self, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "yolov8n.onnx")

        with patch("src.export.onnx_exporter.YOLO", return_value=mock_model):
            exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path))
            exporter.export()

        call_kwargs = mock_model.export.call_args.kwargs
        assert call_kwargs["imgsz"] == 640

    def test_export_returns_onnx_path(self, tmp_path: Path) -> None:
        expected_path = str(tmp_path / "yolov8n.onnx")
        mock_model = MagicMock()
        mock_model.export.return_value = expected_path

        with patch("src.export.onnx_exporter.YOLO", return_value=mock_model):
            exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path))
            result = exporter.export()

        assert result == expected_path


class TestOutputParityValidation:
    """Validate that parity checking logic enforces atol=1e-4 correctly."""

    def test_parity_passes_within_tolerance(self) -> None:
        pytorch_output = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        onnx_output = pytorch_output + 5e-5  # within atol=1e-4

        # Should not raise
        validate_output_parity(pytorch_output, onnx_output, atol=1e-4)

    def test_parity_fails_beyond_tolerance(self) -> None:
        pytorch_output = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        onnx_output = pytorch_output + 5e-4  # exceeds atol=1e-4

        with pytest.raises(ValueError, match="parity"):
            validate_output_parity(pytorch_output, onnx_output, atol=1e-4)

    def test_parity_uses_default_atol_1e4(self) -> None:
        """Default tolerance must be 1e-4 per locked spec."""
        sig = inspect.signature(validate_output_parity)
        default_atol = sig.parameters["atol"].default
        assert default_atol == 1e-4

    def test_parity_exact_match_passes(self) -> None:
        output = np.zeros((1, 84, 8400), dtype=np.float32)
        validate_output_parity(output, output.copy(), atol=1e-4)

    def test_parity_checks_full_array(self) -> None:
        """A single outlier element beyond tolerance must fail the check."""
        pytorch_output = np.zeros((1, 84, 8400), dtype=np.float32)
        onnx_output = pytorch_output.copy()
        onnx_output[0, 0, 0] = 1e-3  # one element exceeds tolerance

        with pytest.raises(ValueError, match="parity"):
            validate_output_parity(pytorch_output, onnx_output, atol=1e-4)


class TestNoMagicNumbers:
    """Confirm the exporter reads opset and imgsz from config, not hardcoded."""

    def test_opset_configurable(self, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "yolov8n.onnx")

        with patch("src.export.onnx_exporter.YOLO", return_value=mock_model):
            exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path), opset=16)
            exporter.export()

        assert mock_model.export.call_args.kwargs["opset"] == 16

    def test_imgsz_configurable(self, tmp_path: Path) -> None:
        mock_model = MagicMock()
        mock_model.export.return_value = str(tmp_path / "yolov8n.onnx")

        with patch("src.export.onnx_exporter.YOLO", return_value=mock_model):
            exporter = OnnxExporter(model_path="yolov8n.pt", output_dir=str(tmp_path), imgsz=320)
            exporter.export()

        assert mock_model.export.call_args.kwargs["imgsz"] == 320
