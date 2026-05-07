"""Integration tests for the end-to-end benchmark pipeline.

Requires: models/yolov8n.onnx exported, all packages installed.
Run with: pytest tests/integration/ -m integration
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_latency_profiler_with_real_onnx_runtime(tmp_path):
    """Real OnnxRuntime session must produce valid LatencyResult."""
    ort = pytest.importorskip("onnxruntime")
    from src.runtimes.onnx_runtime import OnnxRuntime
    from src.benchmark.latency_profiler import profile_latency

    runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
    runtime.load("models/yolov8n.onnx")

    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    result = profile_latency(runtime, dummy, n_runs=10, n_warmup=2)

    assert result.mean_ms > 0
    assert result.p95_ms >= result.mean_ms - 1e-9
    assert result.n_runs == 10


@pytest.mark.integration
def test_accuracy_pipeline_wiring(tmp_path):
    """CocoLoader → OnnxRuntime.infer → evaluate_map must produce a valid AccuracyResult.

    Validates the full accuracy pipeline wire-up on a small synthetic dataset
    (3 blank images) without requiring the real COCO annotation file.
    """
    ort = pytest.importorskip("onnxruntime")
    pycocotools = pytest.importorskip("pycocotools")

    import json
    import shutil
    from unittest.mock import MagicMock, patch
    import numpy as np

    from src.benchmark.accuracy_evaluator import AccuracyResult, evaluate_map
    from src.data.coco_loader import CocoLoader, LetterboxMeta

    # Stub loader yielding 3-tuples (tensor, image_id, LetterboxMeta) — matches
    # CocoLoader.__iter__ contract required by evaluate_map.
    loader = MagicMock()
    loader.__len__ = MagicMock(return_value=3)
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    meta = LetterboxMeta(scale=1.0, pad_left=0, pad_top=0, orig_h=640, orig_w=640)
    loader.__iter__ = MagicMock(return_value=iter([(dummy, i + 1, meta) for i in range(3)]))

    # Stub runtime
    runtime = MagicMock()
    runtime.name = "onnx_cpu_fp32"
    runtime.infer.return_value = np.zeros((1, 84, 8400), dtype=np.float32)

    # Stub COCO evaluation plumbing
    mock_coco_gt = MagicMock()
    mock_coco_dt = MagicMock()
    mock_coco_gt.loadRes.return_value = mock_coco_dt
    mock_eval = MagicMock()
    mock_eval.stats = [0.372, 0.530] + [0.0] * 10

    with patch("src.benchmark.accuracy_evaluator.COCO", return_value=mock_coco_gt), \
         patch("src.benchmark.accuracy_evaluator.COCOeval", return_value=mock_eval):
        result = evaluate_map(runtime, loader, str(tmp_path / "ann.json"), conf_threshold=0.99)

    assert isinstance(result, AccuracyResult)
    assert result.map_50_95 == 0.0  # all-zero output → no detections above 0.99 threshold
    assert result.map_50 == 0.0
    assert result.runtime == "onnx_cpu_fp32"
    assert result.precision == "fp32"
    assert result.map_delta_vs_fp32 is None


@pytest.mark.integration
def test_result_writer_roundtrip(tmp_path):
    """BenchmarkResult survives JSON serialisation and deserialisation."""
    import json
    from src.results.result_schema import BenchmarkResult
    from src.results.result_writer import ResultWriter

    result = BenchmarkResult(
        runtime="onnx_cpu_fp32", precision="fp32", hardware="fedora_cpu",
        mean_latency_ms=50.0, stddev_latency_ms=2.0, p95_latency_ms=54.0,
        min_latency_ms=46.0, max_latency_ms=60.0, fps=20.0,
        map_50_95=0.372, map_50=0.530, map_delta_vs_fp32=0.0,
        peak_memory_mb=128.0, n_runs=100, n_warmup=10,
        onnxruntime_version="1.18.1", torch_version="2.3.1",
        hardware_info={"cpu": "test"},
    )
    writer = ResultWriter(output_dir=str(tmp_path))
    path = writer.write_json(result)

    with open(path) as f:
        loaded = json.load(f)

    assert loaded["runtime"] == "onnx_cpu_fp32"
    assert abs(loaded["mean_latency_ms"] - 50.0) < 1e-9
