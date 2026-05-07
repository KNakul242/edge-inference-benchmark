"""PyTorch inference runtime for the benchmark pipeline.

Active on Fedora: CPU device, FP32 precision only.

# MAC_REQUIRED: MPS device (Apple Silicon Neural Engine) and FP16 via
# torch.autocast('mps') are stubbed below. Implement in feature/mac-runtime
# when Mac M4 is available.
"""

import logging

import numpy as np

from src.runtimes.base_runtime import BaseRuntime

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from ultralytics import YOLO as _ultralytics_YOLO
except ImportError:  # pragma: no cover
    _ultralytics_YOLO = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


class PyTorchRuntime(BaseRuntime):
    """PyTorch inference runtime for CPU and MPS targets.

    Args:
        device: Target device — ``"cpu"`` or ``"mps"`` (Mac M4 Neural Engine).
        precision: Numerical precision — ``"fp32"`` or ``"fp16"`` (MPS only).
    """

    def __init__(self, device: str = "cpu", precision: str = "fp32") -> None:
        self._device = device
        self._precision = precision
        self._model = None

    @property
    def name(self) -> str:
        """Runtime identifier encoding device and precision."""
        return f"pytorch_{self._device}_{self._precision}"

    def load(self, model_path: str) -> None:
        """Load a YOLOv8 checkpoint via ultralytics and set the model to inference mode.

        Ultralytics `.pt` files are Python-pickled dicts, not bare ``nn.Module``
        objects. Loading them with ``torch.load()`` directly returns a dict and
        ``.eval()`` would raise ``AttributeError``. The ultralytics YOLO class
        handles checkpoint parsing and exposes the underlying ``DetectionModel``
        via ``.model``, which is the raw ``nn.Module`` expected by ``infer()``.

        Args:
            model_path: Path to the YOLOv8 .pt weights file.

        Raises:
            ImportError: If torch or ultralytics is not installed.
            FileNotFoundError: If the weights file does not exist.
            RuntimeError: If the device is unavailable.
        """
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required. Run: pip install torch==2.3.1")
        if _ultralytics_YOLO is None:  # pragma: no cover
            raise ImportError(
                "ultralytics is required to load YOLOv8 checkpoints. "
                "Run: pip install ultralytics==8.2.0"
            )

        logger.info("Loading YOLOv8 checkpoint from %s onto %s", model_path, self._device)
        yolo = _ultralytics_YOLO(model_path)
        model = yolo.model.to(self._device)
        model.eval()
        self._model = model
        logger.info("Model loaded — device=%s precision=%s", self._device, self._precision)

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run a single inference pass.

        Args:
            input_tensor: NCHW float32 array, shape (1, 3, 640, 640).

        Returns:
            Model output as a numpy array, shape (1, 84, 8400).

        Raises:
            RuntimeError: If ``load`` has not been called.
        """
        if self._model is None:
            raise RuntimeError(
                f"Runtime '{self.name}' has no model loaded. Call load() before infer()."
            )
        if torch is None:  # pragma: no cover
            raise ImportError("torch is required.")

        tensor = torch.from_numpy(input_tensor).to(self._device)

        # MAC_REQUIRED: FP16 via MPS autocast — implement in feature/mac-runtime
        # if self._precision == "fp16" and self._device == "mps":
        #     with torch.autocast("mps"):
        #         output = self._model(tensor)
        #     return output.cpu().numpy()

        if self._precision == "fp16" and self._device != "mps":
            raise NotImplementedError(
                "FP16 on CPU is not a valid benchmark target. "
                "FP16 requires MPS (Mac M4) — parked until device is available."
            )

        with torch.no_grad():
            output = self._model(tensor)

        result = output.cpu().numpy()
        assert result.shape == (1, 84, 8400), (
            f"PyTorch runtime returned unexpected output shape {result.shape}. "
            "Expected (1, 84, 8400). Ensure yolov8n.pt is loaded via ultralytics "
            "and the model is in eval mode without NMS post-processing."
        )
        return result

    def warmup(self, input_tensor: np.ndarray, n_runs: int) -> None:
        """Execute warmup inference passes before timing begins.

        Args:
            input_tensor: NCHW float32 array, shape (1, 3, 640, 640).
            n_runs: Number of warmup passes. Results are discarded.
        """
        logger.info("Warming up %s — %d passes", self.name, n_runs)
        for _ in range(n_runs):
            self.infer(input_tensor)
