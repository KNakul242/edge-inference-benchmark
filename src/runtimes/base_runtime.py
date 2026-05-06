"""Abstract base class enforcing the runtime interface contract.

All concrete runtimes (PyTorch, ONNX Runtime, TensorRT) must implement this
interface so the benchmark pipeline can treat them interchangeably.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseRuntime(ABC):
    """Interface contract for all inference runtimes in the benchmark pipeline.

    Concrete implementations must provide ``name``, ``load``, ``infer``, and
    ``warmup``. This ensures the latency profiler and accuracy evaluator can
    operate on any runtime without modification.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique runtime identifier used in result schema and logging.

        Convention: ``{framework}_{device}_{precision}``
        e.g. ``pytorch_cpu_fp32``, ``onnx_coreml_fp16``.
        """
        ...

    @abstractmethod
    def load(self, model_path: str) -> None:
        """Load and initialise the model for inference.

        Must be called once before ``infer`` or ``warmup``.

        Args:
            model_path: Path to the model file (.pt, .onnx, or .engine).
        """
        ...

    @abstractmethod
    def infer(self, input_tensor: np.ndarray) -> np.ndarray:
        """Execute a single inference pass.

        Args:
            input_tensor: Preprocessed input, shape (1, 3, 640, 640), float32,
                values in [0, 1], NCHW layout.

        Returns:
            Raw model output as a numpy array. For YOLOv8n: shape (1, 84, 8400).
        """
        ...

    @abstractmethod
    def warmup(self, input_tensor: np.ndarray, n_runs: int) -> None:
        """Run inference passes to warm up the runtime before timing begins.

        Output is discarded. Used to amortise JIT compilation, memory
        allocation, and driver initialisation costs before the benchmark window.

        Args:
            input_tensor: Same input used for benchmarking.
            n_runs: Number of warmup passes to execute.
        """
        ...
