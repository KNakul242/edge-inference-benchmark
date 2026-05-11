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

from src.benchmark.accuracy_evaluator import AccuracyResult, compute_map_delta, evaluate_map
from src.benchmark.latency_profiler import profile_latency
from src.benchmark.memory_profiler import profile_memory
from src.data.coco_loader import CocoLoader
from src.data.calibration_set import generate_calibration_set
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

    Returns runtimes sorted so FP32 always runs before FP16/INT8 within each
    family — required for delta computation to have a baseline available.
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

    # Sort FP32 first within each runtime family so baselines are computed first
    precision_order = {"fp32": 0, "fp16": 1, "int8": 2}
    runtimes.sort(key=lambda rt: precision_order.get(rt.name.rsplit("_", 1)[-1], 99))

    return runtimes


def maybe_generate_calibration(config: dict) -> None:
    """Generate the INT8 calibration set if any INT8 runtime is configured.

    Skips silently when COCO data is not present — calibration is required
    for INT8 quantisation accuracy but not for latency-only runs.
    """
    coco_val_dir = os.environ.get("COCO_DATA_DIR", config["data"]["coco_val_dir"])
    calibration_dir = config["data"]["calibration_dir"]
    manifest_path = Path(calibration_dir) / "manifest.json"

    if manifest_path.exists():
        logger.info("Calibration manifest exists — skipping generation: %s", manifest_path)
        return

    if not Path(coco_val_dir).is_dir():
        logger.warning(
            "COCO val2017 not found at %s — skipping calibration set generation. "
            "Set COCO_DATA_DIR env var to enable INT8 calibration.", coco_val_dir
        )
        return

    n_images = config["calibration"]["n_images"]
    seed = config["calibration"]["seed"]
    logger.info("Generating calibration set: %d images, seed=%d", n_images, seed)
    generate_calibration_set(
        coco_val_dir=coco_val_dir,
        output_dir=calibration_dir,
        n_images=n_images,
        seed=seed,
    )


def run_benchmark(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    set_seed(config["calibration"]["seed"])

    model_dir = Path(os.environ.get("MODEL_DIR", "./models"))
    coco_val_dir = os.environ.get("COCO_DATA_DIR", config["data"]["coco_val_dir"])
    annotations_file = os.environ.get(
        "COCO_ANNOTATIONS", config["data"]["annotations_file"]
    )
    results_dir = Path(os.environ.get("RESULTS_DIR", config["output"]["results_dir"]))
    onnx_path = model_dir / "yolov8n.onnx"
    conf_threshold = config["model"]["conf_threshold"]
    iou_threshold = config["model"]["iou_threshold"]
    eval_conf_threshold = config["model"]["eval_conf_threshold"]
    eval_iou_threshold = config["model"]["eval_iou_threshold"]

    n_runs = config["benchmark"]["n_runs"]
    n_warmup = config["benchmark"]["n_warmup"]

    maybe_generate_calibration(config)

    device_info = get_device_info()
    writer = ResultWriter(output_dir=str(results_dir))

    # Build loader once — all runtimes share the same COCO val images
    coco_available = Path(coco_val_dir).is_dir() and Path(annotations_file).is_file()
    if coco_available:
        loader = CocoLoader(images_dir=coco_val_dir, annotations_file=annotations_file)
        logger.info("COCO val2017: %d images loaded from %s", len(loader), coco_val_dir)
    else:
        loader = None
        logger.warning(
            "COCO val2017 not found — accuracy evaluation disabled. "
            "Set COCO_DATA_DIR and COCO_ANNOTATIONS env vars to enable."
        )

    # For latency profiling use a representative dummy input (same shape as real images).
    # Random values (not zeros) prevent optimisations that may shortcut computation paths.
    import numpy as np
    dummy_input = np.random.default_rng(42).random((1, 3, 640, 640)).astype(np.float32)

    runtimes = build_runtimes(config)
    if not runtimes:
        logger.error("No runtimes available for this environment. Check config and installed packages.")
        sys.exit(1)

    logger.info("Benchmark plan: %d runtime × precision combinations", len(runtimes))
    for rt in runtimes:
        logger.info("  → %s", rt.name)

    all_results: list[BenchmarkResult] = []
    # Keyed by runtime family (e.g. "pytorch_cpu", "onnx_cpu") → fp32 AccuracyResult
    fp32_baselines: dict[str, AccuracyResult] = {}

    for runtime in runtimes:
        logger.info("=" * 60)
        logger.info("Benchmarking: %s", runtime.name)

        # COLAB_REQUIRED: TensorRT needs a .engine file, not .pt or .onnx.
        # When implementing TensorRT, add: elif "tensorrt" in runtime.name: model_path = str(model_dir / f"yolov8n_{precision}.engine")
        model_path = str(onnx_path) if "onnx" in runtime.name else str(model_dir / "yolov8n.pt")
        try:
            runtime.load(model_path)
        except Exception as e:
            logger.error("Failed to load %s: %s — skipping", runtime.name, e)
            continue

        # Latency first — profile_latency runs n_warmup discarded passes internally,
        # leaving the runtime in a warmed-up steady state before memory is sampled.
        latency = profile_latency(runtime, dummy_input, n_runs=n_runs, n_warmup=n_warmup)

        # Memory after warmup — RSS snapshot reflects steady-state footprint, not the
        # first-inference lazy allocation that pre-warmup measurement captures.
        memory = profile_memory(runtime, dummy_input)

        # Accuracy evaluation uses eval thresholds, not deployment thresholds.
        # eval_conf_threshold=0.001 exposes the full PR curve to COCOeval.
        # eval_iou_threshold=0.7 matches the ultralytics reference validator.
        precision = runtime.name.rsplit("_", 1)[-1]
        family = runtime.name.rsplit("_", 1)[0]

        if coco_available and loader is not None:
            try:
                accuracy = evaluate_map(
                    runtime, loader, annotations_file,
                    conf_threshold=eval_conf_threshold,
                    iou_threshold=eval_iou_threshold,
                )
            except ImportError as exc:
                logger.warning("pycocotools not available — mAP set to 0.0: %s", exc)
                accuracy = AccuracyResult(map_50_95=0.0, map_50=0.0, precision=precision, runtime=runtime.name)
        else:
            accuracy = AccuracyResult(map_50_95=0.0, map_50=0.0, precision=precision, runtime=runtime.name)

        # Track FP32 baseline; compute delta for subsequent precisions
        if precision == "fp32":
            fp32_baselines[family] = accuracy
            map_delta = 0.0
        elif family in fp32_baselines:
            map_delta = compute_map_delta(fp32_baselines[family], accuracy)
        else:
            logger.warning(
                "No FP32 baseline for %s — delta set to None; run FP32 first.", runtime.name
            )
            map_delta = None

        result = BenchmarkResult(
            runtime=runtime.name,
            precision=precision,
            hardware=resolve_hardware(runtime.name),
            mean_latency_ms=latency.mean_ms,
            stddev_latency_ms=latency.stddev_ms,
            p95_latency_ms=latency.p95_ms,
            min_latency_ms=latency.min_ms,
            max_latency_ms=latency.max_ms,
            fps=1000.0 / latency.mean_ms,
            map_50_95=accuracy.map_50_95,
            map_50=accuracy.map_50,
            # None serialises as JSON null (not NaN) — distinguishable from 0.0
            # which means "no accuracy degradation vs FP32 baseline".
            map_delta_vs_fp32=map_delta,
            peak_memory_mb=memory.peak_mb,
            peak_memory_delta_mb=memory.delta_mb,
            n_runs=latency.n_runs,
            n_warmup=latency.n_warmup,
            onnxruntime_version=device_info.get("onnxruntime_version", "unknown"),
            torch_version=device_info.get("torch_version", "unknown"),
            hardware_info=device_info,
        )

        writer.write_json(result)
        all_results.append(result)
        logger.info("Result saved: %s  mAP=%.4f  latency=%.2fms", runtime.name, accuracy.map_50_95, latency.mean_ms)

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
