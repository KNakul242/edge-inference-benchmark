"""ONNX Runtime inference wrapper for the benchmark pipeline.

Active on Fedora: CPUExecutionProvider, FP32 only.
Active on Mac M5: CoreMLExecutionProvider, FP32 only. FP16/INT8 via CoreML EP
are not built — no static-quantization pipeline exists yet, a gap that
predates and is independent of Mac hardware availability. Empirical testing
(docs/benchmark-run-1-mac-findings.md, Issue 5) found no reproducible latency
benefit from requesting GPU/Neural Engine compute units over CoreML EP's own
CPU-only path for this model — treat CoreML EP here as "a fast, CPU-optimized
ONNX Runtime path on Apple Silicon," not as confirmed Neural Engine inference.
"""

import logging

import numpy as np

from src.runtimes.base_runtime import BaseRuntime

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_PROVIDER_SHORTNAMES = {
    "CoreMLExecutionProvider": "coreml",
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "cuda",
}


class OnnxRuntime(BaseRuntime):
    """ONNX Runtime inference wrapper with configurable execution provider.

    On Fedora: CPUExecutionProvider, FP32 only.
    On Mac M5: CoreMLExecutionProvider (with CPUExecutionProvider as fallback
    for unsupported graph nodes), FP32 only — see module docstring for the
    FP16/INT8 gap and the GPU/ANE-engagement caveat.

    Args:
        execution_provider: Primary EP — ``"CPUExecutionProvider"`` or
            ``"CoreMLExecutionProvider"`` (both active).
        precision: Numerical precision of the loaded model file — ``"fp32"``
            (active); ``"fp16"`` or ``"int8"`` require a CoreML EP
            static-quantization pipeline that is not yet built.
    """

    def __init__(self, execution_provider: str = "CPUExecutionProvider", precision: str = "fp32") -> None:
        self._execution_provider = execution_provider
        self._precision = precision
        self._session = None
        self._input_name: str | None = None

    @property
    def name(self) -> str:
        """Runtime identifier encoding the EP short name and precision."""
        ep_short = _PROVIDER_SHORTNAMES.get(
            self._execution_provider, self._execution_provider.lower()
        )
        return f"onnx_{ep_short}_{self._precision}"

    def load(self, model_path: str) -> None:
        """Create an ONNX Runtime InferenceSession for the given model.

        Builds a provider list with the requested EP first and CPU EP as
        fallback. On Linux, CoreMLExecutionProvider is not available and
        ONNX Runtime will automatically use CPU EP.

        Args:
            model_path: Path to the .onnx model file.

        Raises:
            ImportError: If onnxruntime is not installed.
        """
        if ort is None:  # pragma: no cover
            raise ImportError("onnxruntime is required. Run: pip install onnxruntime==1.18.1")

        # CoreMLExecutionProvider is placed first when requested, so ONNX
        # Runtime prefers it; session.get_providers() below confirms which
        # EP is actually active and logs a warning on fallback.
        providers = [self._execution_provider]
        if self._execution_provider != "CPUExecutionProvider":
            providers.append("CPUExecutionProvider")

        logger.info("Loading ONNX model: %s  providers=%s", model_path, providers)
        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        active = self._session.get_providers()
        if self._execution_provider not in active:
            logger.warning(
                "%s not available on this platform — active EP: %s",
                self._execution_provider,
                active[0],
            )
        else:
            logger.info("Active execution provider: %s", active[0])

    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run a single inference pass via ONNX Runtime.

        Args:
            input_tensor: NCHW float32 array, shape (1, 3, 640, 640).

        Returns:
            Model output array, shape (1, 84, 8400).

        Raises:
            RuntimeError: If ``load`` has not been called.
        """
        if self._session is None:
            raise RuntimeError(
                f"Runtime '{self.name}' has no session loaded. Call load() before infer()."
            )
        if input_tensor.dtype != np.float32:
            raise ValueError(
                f"Input tensor must be float32, got {input_tensor.dtype}. "
                "Call input_tensor.astype(np.float32) before infer()."
            )
        outputs = self._session.run(None, {self._input_name: input_tensor})
        result = outputs[0]
        if result.shape != (1, 84, 8400):
            raise RuntimeError(
                f"ONNX Runtime returned unexpected output shape {result.shape}. "
                "Expected (1, 84, 8400). Ensure yolov8n.onnx was exported with "
                "opset=17, dynamic=False, simplify=True, without NMS post-processing."
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
