"""Latency profiler for inference pipeline benchmarking.

Implements the locked benchmark protocol: n_warmup discarded passes followed
by n_runs timed passes. Reports mean, stddev, p95, min, and max latency in
milliseconds using time.perf_counter for sub-millisecond resolution.
"""

import logging
import statistics
import time
from dataclasses import dataclass

import numpy as np

from src.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)


@dataclass
class LatencyResult:
    """Per-run latency statistics from a benchmark session.

    All latency values are in milliseconds. p95 is the deployment-relevant
    metric — it captures the tail of the distribution that a real system would
    experience under normal load variation.

    Attributes:
        mean_ms: Arithmetic mean of timed inference latencies.
        stddev_ms: Sample standard deviation — signals runtime consistency.
        p95_ms: 95th-percentile latency — the deployment-relevant upper bound.
        min_ms: Fastest observed inference pass.
        max_ms: Slowest observed inference pass.
        n_runs: Number of timed passes (warmup excluded).
        n_warmup: Number of discarded warmup passes.
    """

    mean_ms: float
    stddev_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    n_runs: int
    n_warmup: int


def profile_latency(
    runtime: BaseRuntime,
    input_tensor: np.ndarray,
    n_runs: int = 100,
    n_warmup: int = 10,
) -> LatencyResult:
    """Profile inference latency for a given runtime under fixed input conditions.

    Executes ``n_warmup`` discarded passes to amortise JIT compilation and
    driver initialisation costs, then ``n_runs`` timed passes using
    ``time.perf_counter`` for sub-millisecond resolution.

    Args:
        runtime: Initialised runtime implementing the BaseRuntime interface.
        input_tensor: Preprocessed input, shape (1, 3, 640, 640), float32.
        n_runs: Number of timed inference passes. Defaults to 100.
        n_warmup: Number of warmup passes before timing begins. Defaults to 10.

    Returns:
        LatencyResult containing mean, stddev, p95, min, max, and run counts.
    """
    logger.info("Warming up %s — %d passes (discarded)", runtime.name, n_warmup)
    for _ in range(n_warmup):
        runtime.infer(input_tensor)

    logger.info("Profiling %s — %d timed passes", runtime.name, n_runs)
    latencies_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        runtime.infer(input_tensor)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    result = LatencyResult(
        mean_ms=statistics.mean(latencies_ms),
        stddev_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0.0,
        p95_ms=float(np.percentile(latencies_ms, 95)),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        n_runs=n_runs,
        n_warmup=n_warmup,
    )

    logger.info(
        "%s latency — mean=%.2f ms stddev=%.2f ms p95=%.2f ms",
        runtime.name,
        result.mean_ms,
        result.stddev_ms,
        result.p95_ms,
    )
    return result
