"""Typed result schema for benchmark output.

BenchmarkResult is the canonical record written per runtime × precision run.
Every field is required to ensure downstream analysis in results_analysis.ipynb
can operate on a uniform schema without defensive null-checks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class BenchmarkResult:
    """Complete benchmark record for one runtime × precision combination.

    One instance is produced per completed benchmark run and serialised to both
    a JSON file (one per combination) and a row in the summary CSV.

    Attributes:
        runtime: Runtime identifier, e.g. ``"pytorch_cpu_fp32"``.
        precision: Numerical precision — ``"fp32"``, ``"fp16"``, or ``"int8"``.
        hardware: Hardware target — ``"fedora_cpu"``, ``"mac_m4"``, ``"colab_t4"``.
        mean_latency_ms: Mean inference latency across n_runs.
        stddev_latency_ms: Sample standard deviation of latency.
        p95_latency_ms: 95th-percentile latency — deployment-relevant upper bound.
        min_latency_ms: Fastest observed inference pass.
        max_latency_ms: Slowest observed inference pass.
        fps: Throughput — derived as ``1000 / mean_latency_ms``.
        map_50_95: mAP@0.5:0.95 on COCO val2017 (primary accuracy metric).
        map_50: mAP@0.5 (secondary).
        map_delta_vs_fp32: ``map_50_95 − fp32_baseline`` for this runtime.
            Negative indicates accuracy degradation under quantisation.
        peak_memory_mb: Peak resident memory during inference, in MB.
        n_runs: Number of timed inference passes (warmup excluded).
        n_warmup: Number of discarded warmup passes.
        timestamp: ISO 8601 UTC timestamp when the result was recorded.
        onnxruntime_version: Version string for reproducibility.
        torch_version: Version string for reproducibility.
        hardware_info: Dict capturing CPU/GPU model, OS, driver versions.
    """

    runtime: str
    precision: str
    hardware: str

    mean_latency_ms: float
    stddev_latency_ms: float
    p95_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    fps: float

    map_50_95: float
    map_50: float
    map_delta_vs_fp32: float

    peak_memory_mb: float

    n_runs: int
    n_warmup: int

    onnxruntime_version: str
    torch_version: str
    hardware_info: dict

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
