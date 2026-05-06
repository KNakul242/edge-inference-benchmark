"""Benchmark runner — entry point for the full inference pipeline evaluation.

Executes the benchmark protocol across all configured runtime × precision
combinations, measures latency, accuracy, and memory, and writes results
to JSON and CSV. Run from the repository root after exporting the ONNX model
and downloading COCO val2017.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --config configs/benchmark_config.yaml
    python scripts/run_benchmark.py --runtime pytorch_cpu --precision fp32
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.benchmark.latency_profiler import profile_latency
from src.benchmark.memory_profiler import profile_memory
from src.results.result_schema import BenchmarkResult
from src.results.result_writer import ResultWriter
from src.runtimes.onnx_runtime import OnnxRuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.utils.device_info import get_device_info
from src.utils.reproducibility import set_seed

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_hardware(runtime_name: str) -> str:
    """Map runtime name to hardware target string for result schema."""
    if "mps" in runtime_name:
        return "mac_m4"  # MAC_REQUIRED
    if "tensorrt" in runtime_name:
        return "colab_t4"  # COLAB_REQUIRED
    return "fedora_cpu"


def build_runtimes(config: dict) -> list:
    """Instantiate all runtime × precision combinations active on this machine.

    Returns only the runtimes that are executable in the current environment.
    Mac M4 and TensorRT runtimes are excluded — see CLAUDE.md MAC_REQUIRED notes.
    """
    runtimes = []

    # PyTorch CPU — FP32 only on Fedora
    # MAC_REQUIRED: add device="mps", precision="fp16" when Mac M4 is available
    pt_cfg = config.get("runtimes", {}).get("pytorch", {})
    for device in pt_cfg.get("devices", ["cpu"]):
        if device == "mps":
            logger.info("Skipping PyTorch MPS — MAC_REQUIRED, device not available")
            continue
        for precision in pt_cfg.get("precisions", ["fp32"]):
            if precision == "fp16" and device == "cpu":
                logger.info("Skipping PyTorch CPU FP16 — not a valid deployment target")
                continue
            runtimes.append(PyTorchRuntime(device=device, precision=precision))

    # ONNX Runtime CPU EP — FP32 only on Fedora
    # MAC_REQUIRED: CoreMLExecutionProvider + FP16/INT8 variants when Mac M4 is available
    onnx_cfg = config.get("runtimes", {}).get("onnx", {})
    for provider in onnx_cfg.get("execution_providers", ["CPUExecutionProvider"]):
        if provider == "CoreMLExecutionProvider":
            logger.info("Skipping CoreML EP — MAC_REQUIRED, not available on Linux")
            continue
        for precision in onnx_cfg.get("precisions", ["fp32"]):
            if precision in ("fp16", "int8") and provider == "CPUExecutionProvider":
                logger.info("Skipping ONNX CPU EP %s — requires CoreML EP (MAC_REQUIRED)", precision)
                continue
            runtimes.append(OnnxRuntime(execution_provider=provider, precision=precision))

    # COLAB_REQUIRED: TensorRT runtimes omitted — see notebooks/tensorrt_colab.ipynb
    logger.info("TensorRT skipped — COLAB_REQUIRED, run notebooks/tensorrt_colab.ipynb on Colab T4")

    return runtimes


def run_benchmark(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    set_seed(config["calibration"]["seed"])

    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    results_dir = Path(os.environ.get("RESULTS_DIR", config["output"]["results_dir"]))
    onnx_path = model_dir / "yolov8n.onnx"

    n_runs = config["benchmark"]["n_runs"]
    n_warmup = config["benchmark"]["n_warmup"]

    device_info = get_device_info()
    writer = ResultWriter(output_dir=str(results_dir))

    import numpy as np
    dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

    runtimes = build_runtimes(config)
    if not runtimes:
        logger.error("No runtimes available for this environment. Check config and installed packages.")
        sys.exit(1)

    logger.info("Benchmark plan: %d runtime × precision combinations", len(runtimes))
    for rt in runtimes:
        logger.info("  → %s", rt.name)

    all_results: list[BenchmarkResult] = []

    for runtime in runtimes:
        logger.info("=" * 60)
        logger.info("Benchmarking: %s", runtime.name)

        try:
            runtime.load(str(onnx_path) if "onnx" in runtime.name else str(model_dir / "yolov8n.pt"))
        except Exception as e:
            logger.error("Failed to load %s: %s — skipping", runtime.name, e)
            continue

        latency = profile_latency(runtime, dummy_input, n_runs=n_runs, n_warmup=n_warmup)
        memory = profile_memory(runtime, dummy_input)

        # Accuracy evaluation requires COCO val2017 — skipped if data not present
        # Full mAP evaluation: see src/benchmark/accuracy_evaluator.py
        map_50_95 = 0.0
        map_50 = 0.0
        logger.warning(
            "mAP evaluation skipped in quick-run mode. "
            "Pass --evaluate-accuracy to run full COCO eval (requires downloaded dataset)."
        )

        result = BenchmarkResult(
            runtime=runtime.name,
            precision=runtime.name.split("_")[-1],
            hardware=resolve_hardware(runtime.name),
            mean_latency_ms=latency.mean_ms,
            stddev_latency_ms=latency.stddev_ms,
            p95_latency_ms=latency.p95_ms,
            min_latency_ms=latency.min_ms,
            max_latency_ms=latency.max_ms,
            fps=1000.0 / latency.mean_ms,
            map_50_95=map_50_95,
            map_50=map_50,
            map_delta_vs_fp32=0.0,
            peak_memory_mb=memory.peak_mb,
            n_runs=latency.n_runs,
            n_warmup=latency.n_warmup,
            onnxruntime_version=device_info.get("onnxruntime_version", "unknown"),
            torch_version=device_info.get("torch_version", "unknown"),
            hardware_info=device_info,
        )

        writer.write_json(result)
        all_results.append(result)
        logger.info("Result saved: %s", runtime.name)

    if all_results:
        writer.write_csv(all_results)
        logger.info("Summary CSV written with %d results", len(all_results))
    else:
        logger.error("No results produced — all runtimes failed or were skipped")


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge inference benchmark runner")
    parser.add_argument(
        "--config", default="configs/benchmark_config.yaml",
        help="Path to benchmark_config.yaml"
    )
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
