"""Entry point for YOLOv8n → ONNX export.

Reads configuration from benchmark_config.yaml and environment variables.
Run from the repository root:

    python scripts/export_model.py
"""

import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.export.onnx_exporter import OnnxExporter, validate_output_parity
from src.runtimes.onnx_runtime import OnnxRuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/benchmark_config.yaml") -> dict:
    """Load benchmark configuration from YAML."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()

    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model"]["name"]
    imgsz = config["model"]["input_size"]

    exporter = OnnxExporter(
        model_path=str(model_dir / f"{model_name}.pt"),
        output_dir=str(model_dir),
        opset=17,
        imgsz=imgsz,
    )

    onnx_path = exporter.export()
    logger.info("ONNX model written to: %s", onnx_path)

    # Parity validation — required per benchmark protocol (atol=1e-4 vs PyTorch).
    # Runs a single inference pass through both runtimes and asserts that no
    # element deviates by more than atol. Fail-fast here prevents downstream
    # benchmark runs from comparing runtimes with a numerically drifted ONNX model.
    logger.info("Running parity validation: PyTorch vs ONNX (atol=1e-4)...")
    _parity_input = np.random.default_rng(42).random((1, 3, 640, 640)).astype(np.float32)

    pt_runtime = PyTorchRuntime(device="cpu", precision="fp32")
    pt_runtime.load(exporter.model_path)
    pytorch_output = pt_runtime.infer(_parity_input)

    onnx_runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
    onnx_runtime.load(onnx_path)
    onnx_output = onnx_runtime.infer(_parity_input)

    validate_output_parity(pytorch_output, onnx_output)
    logger.info("Parity validation passed — ONNX export is numerically equivalent to PyTorch.")


if __name__ == "__main__":
    main()
