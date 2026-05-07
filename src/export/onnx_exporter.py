"""ONNX export pipeline for YOLOv8n.

Exports a pretrained YOLOv8n model to ONNX with opset 17 and validates
output parity against the PyTorch baseline within the required tolerance.
"""

import logging
from pathlib import Path

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Locked per spec: TensorRT 8.6.x requires opset 17
_DEFAULT_OPSET = 17
_DEFAULT_IMGSZ = 640
_DEFAULT_PARITY_ATOL = 1e-4


class OnnxExporter:
    """Exports a YOLOv8 model to ONNX for cross-runtime inference benchmarking.

    Args:
        model_path: Path to the .pt weights file (downloaded by ultralytics if absent).
        output_dir: Directory where the .onnx file will be written.
        opset: ONNX opset version. Defaults to 17 (TensorRT 8.6.x compatibility).
        imgsz: Square input resolution. Defaults to 640 (locked for edge deployment).
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str,
        opset: int = _DEFAULT_OPSET,
        imgsz: int = _DEFAULT_IMGSZ,
    ) -> None:
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.opset = opset
        self.imgsz = imgsz

    def export(self) -> str:
        """Run the ONNX export pipeline.

        Returns:
            Absolute path to the exported .onnx file.

        Raises:
            RuntimeError: If ultralytics export fails.
        """
        if YOLO is None:  # pragma: no cover
            raise ImportError("ultralytics is required for ONNX export. Run: pip install ultralytics")

        logger.info("Loading model from %s", self.model_path)
        model = YOLO(self.model_path)

        logger.info(
            "Exporting to ONNX — opset=%d, imgsz=%d, dynamic=False, simplify=True",
            self.opset,
            self.imgsz,
        )
        exported_path = model.export(
            format="onnx",
            opset=self.opset,
            dynamic=False,
            simplify=True,
            imgsz=self.imgsz,
        )
        logger.info("Export complete: %s", exported_path)
        return str(exported_path)


def validate_output_parity(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    atol: float = _DEFAULT_PARITY_ATOL,
) -> None:
    """Assert that ONNX output matches PyTorch output within absolute tolerance.

    Runs element-wise comparison across the full output tensor. A single element
    exceeding ``atol`` constitutes a parity failure, indicating the export
    introduced a numerical deviation beyond the acceptable threshold.

    Args:
        pytorch_output: Reference inference output from the PyTorch runtime.
        onnx_output: Inference output from the exported ONNX model.
        atol: Absolute tolerance for element-wise comparison. Defaults to 1e-4.

    Raises:
        ValueError: If any element differs by more than ``atol``.
    """
    if pytorch_output.shape != onnx_output.shape:
        raise ValueError(
            f"Shape mismatch before parity check: PyTorch={pytorch_output.shape} "
            f"vs ONNX={onnx_output.shape}. Outputs must have identical shape."
        )
    max_diff = float(np.max(np.abs(pytorch_output - onnx_output)))
    if max_diff > atol:
        raise ValueError(
            f"ONNX export parity check failed: max absolute deviation {max_diff:.2e} "
            f"exceeds atol={atol:.2e}. The export introduced a numerical discrepancy."
        )
    logger.info("Parity check passed — max deviation: %.2e (atol=%.2e)", max_diff, atol)
