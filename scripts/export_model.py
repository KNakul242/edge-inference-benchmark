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

from src.export.onnx_exporter import OnnxExporter, validate_output_parity

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
        model_path=f"{model_name}.pt",
        output_dir=str(model_dir),
        opset=17,
        imgsz=imgsz,
    )

    onnx_path = exporter.export()
    logger.info("ONNX model written to: %s", onnx_path)


if __name__ == "__main__":
    main()
