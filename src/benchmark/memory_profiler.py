"""Memory footprint profiler for inference pipeline benchmarking.

Measures peak resident memory allocated during a single inference pass using
tracemalloc (CPU runtimes). For GPU runtimes, the caller is responsible for
using torch.cuda.max_memory_allocated() or nvidia-smi and populating the
result directly.
"""

import logging
import tracemalloc
from dataclasses import dataclass

import numpy as np

from src.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024


@dataclass
class MemoryResult:
    """Peak memory footprint for one runtime × inference pass.

    Attributes:
        peak_mb: Peak resident memory in MB during the inference pass.
        runtime: Runtime identifier (matches ``BaseRuntime.name``).
    """

    peak_mb: float
    runtime: str


def profile_memory(
    runtime: BaseRuntime,
    input_tensor: np.ndarray,
) -> MemoryResult:
    """Profile peak memory allocated during a single inference pass.

    Uses tracemalloc to capture peak allocation during ``runtime.infer()``.
    Suitable for CPU-bound runtimes. For GPU runtimes, memory must be measured
    separately via CUDA APIs and the result constructed directly.

    Args:
        runtime: Initialised runtime implementing the BaseRuntime interface.
        input_tensor: Preprocessed input, shape (1, 3, 640, 640), float32.

    Returns:
        MemoryResult with peak memory in MB and the runtime identifier.
    """
    logger.info("Profiling memory footprint for %s", runtime.name)

    tracemalloc.start()
    try:
        runtime.infer(input_tensor)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    peak_mb = peak_bytes / _BYTES_PER_MB
    logger.info("%s peak memory: %.1f MB", runtime.name, peak_mb)

    return MemoryResult(peak_mb=peak_mb, runtime=runtime.name)
