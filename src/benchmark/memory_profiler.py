"""Memory footprint profiler for inference pipeline benchmarking.

Measures process RSS (resident set size) during inference via psutil — the only
method that captures native C++ allocations from ONNX Runtime and PyTorch.
tracemalloc is retained as a fallback when psutil is unavailable, but it only
measures Python heap and will massively underreport native allocations.
"""

import logging
import tracemalloc
from dataclasses import dataclass

import numpy as np

from src.runtimes.base_runtime import BaseRuntime

try:
    import psutil as _psutil
except ImportError:  # pragma: no cover
    _psutil = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_BYTES_PER_MB = 1024 * 1024


@dataclass
class MemoryResult:
    """Peak memory footprint for one runtime × inference pass.

    Attributes:
        peak_mb: Process RSS in MB sampled immediately after ``infer()`` returns
            (psutil path) or peak Python-heap in MB (tracemalloc fallback).
            Captures model weights and persistent native buffers but NOT
            transient allocations freed inside ``infer()`` — treat as a
            conservative lower bound on true peak-during-inference memory.
            Prefer psutil over tracemalloc — it captures C++ allocations.
        runtime: Runtime identifier (matches ``BaseRuntime.name``).
    """

    peak_mb: float
    runtime: str


def profile_memory(
    runtime: BaseRuntime,
    input_tensor: np.ndarray,
) -> MemoryResult:
    """Profile process memory footprint during a single inference pass.

    Uses psutil to capture process RSS after an inference call, which includes
    model weights and native operator buffers allocated at the C++ layer. Falls
    back to tracemalloc when psutil is unavailable, but logs a warning because
    tracemalloc only captures Python heap and will underreport by orders of
    magnitude for ONNX Runtime and PyTorch sessions.

    Args:
        runtime: Initialised runtime implementing the BaseRuntime interface.
        input_tensor: Preprocessed input, shape (1, 3, 640, 640), float32.

    Returns:
        MemoryResult with peak memory in MB and the runtime identifier.
    """
    logger.info("Profiling memory footprint for %s", runtime.name)

    if _psutil is not None:
        # Sample RSS before and after inference. The post-call snapshot captures
        # model weights and persistent operator buffers resident in memory.
        # Note: transient allocations freed *within* infer() (e.g. intermediate
        # activation tensors, kernel workspace) are not captured — the reported
        # value is a conservative lower bound on peak-during-inference memory.
        proc = _psutil.Process()
        rss_before = proc.memory_info().rss
        runtime.infer(input_tensor)
        rss_after = proc.memory_info().rss
        peak_mb = rss_after / _BYTES_PER_MB
        delta_mb = (rss_after - rss_before) / _BYTES_PER_MB
        logger.info(
            "%s RSS after inference: %.1f MB (delta from pre-infer: %+.1f MB)",
            runtime.name, peak_mb, delta_mb,
        )
    else:
        # Fallback: Python heap only — significantly underreports native allocations.
        # Install psutil for accurate memory measurement: pip install psutil
        logger.warning(
            "%s: psutil unavailable — measuring Python heap via tracemalloc. "
            "This will underreport actual memory by orders of magnitude for "
            "ONNX Runtime and PyTorch sessions.", runtime.name
        )
        tracemalloc.start()
        try:
            runtime.infer(input_tensor)
            _, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        peak_mb = peak_bytes / _BYTES_PER_MB

    return MemoryResult(peak_mb=peak_mb, runtime=runtime.name)
