"""Unit tests for the result schema and writer.

Requires 90%+ coverage. Bugs here produce silently corrupt benchmark outputs.
Critical invariants: schema integrity, no data loss, ISO timestamps, correct
JSON and CSV serialisation.
"""

import csv
import json
import re
from pathlib import Path

import pytest

from src.results.result_schema import BenchmarkResult
from src.results.result_writer import ResultWriter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_result(**overrides) -> BenchmarkResult:
    defaults = dict(
        runtime="pytorch_cpu_fp32",
        precision="fp32",
        hardware="fedora_cpu",
        mean_latency_ms=45.3,
        stddev_latency_ms=2.1,
        p95_latency_ms=49.8,
        min_latency_ms=42.0,
        max_latency_ms=55.0,
        fps=22.1,
        map_50_95=0.372,
        map_50=0.530,
        map_delta_vs_fp32=0.0,
        peak_memory_mb=128.5,
        n_runs=100,
        n_warmup=10,
        onnxruntime_version="1.18.1",
        torch_version="2.3.1",
        hardware_info={"cpu": "Intel i7", "os": "Fedora 39"},
    )
    defaults.update(overrides)
    return BenchmarkResult(**defaults)


# ---------------------------------------------------------------------------
# BenchmarkResult schema
# ---------------------------------------------------------------------------

class TestBenchmarkResultSchema:
    def test_all_required_fields_present(self) -> None:
        result = _make_result()
        assert result.runtime == "pytorch_cpu_fp32"
        assert result.precision == "fp32"
        assert result.hardware == "fedora_cpu"

    def test_latency_fields(self) -> None:
        result = _make_result(mean_latency_ms=10.0, p95_latency_ms=12.0)
        assert result.mean_latency_ms == 10.0
        assert result.p95_latency_ms == 12.0

    def test_accuracy_fields(self) -> None:
        result = _make_result(map_50_95=0.360, map_delta_vs_fp32=-0.012)
        assert result.map_50_95 == 0.360
        assert abs(result.map_delta_vs_fp32 + 0.012) < 1e-9

    def test_hardware_info_is_dict(self) -> None:
        result = _make_result(hardware_info={"cpu": "M4", "os": "macOS"})
        assert isinstance(result.hardware_info, dict)

    def test_fps_field_present(self) -> None:
        result = _make_result(fps=33.3)
        assert result.fps == 33.3

    def test_timestamp_is_auto_populated(self) -> None:
        result = _make_result()
        # timestamp is set by the dataclass post-init
        assert hasattr(result, "timestamp")
        assert result.timestamp is not None


# ---------------------------------------------------------------------------
# JSON writing
# ---------------------------------------------------------------------------

class TestJsonWriter:
    def test_json_file_created(self, tmp_path: Path) -> None:
        result = _make_result()
        writer = ResultWriter(output_dir=str(tmp_path))
        writer.write_json(result)

        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) == 1

    def test_json_filename_encodes_runtime_and_precision(self, tmp_path: Path) -> None:
        result = _make_result(runtime="onnx_coreml_fp16", precision="fp16")
        writer = ResultWriter(output_dir=str(tmp_path))
        path = writer.write_json(result)

        assert "onnx_coreml_fp16" in Path(path).name
        assert "fp16" in Path(path).name

    def test_json_schema_integrity_no_data_loss(self, tmp_path: Path) -> None:
        """Every field in BenchmarkResult must survive serialisation."""
        result = _make_result()
        writer = ResultWriter(output_dir=str(tmp_path))
        path = writer.write_json(result)

        with open(path) as f:
            data = json.load(f)

        assert data["runtime"] == result.runtime
        assert data["precision"] == result.precision
        assert data["hardware"] == result.hardware
        assert abs(data["mean_latency_ms"] - result.mean_latency_ms) < 1e-9
        assert abs(data["p95_latency_ms"] - result.p95_latency_ms) < 1e-9
        assert abs(data["map_50_95"] - result.map_50_95) < 1e-9
        assert abs(data["peak_memory_mb"] - result.peak_memory_mb) < 1e-9
        assert data["n_runs"] == result.n_runs
        assert data["hardware_info"] == result.hardware_info

    def test_timestamp_is_iso_format(self, tmp_path: Path) -> None:
        """Timestamp must be ISO 8601 so results can be sorted and compared."""
        result = _make_result()
        writer = ResultWriter(output_dir=str(tmp_path))
        path = writer.write_json(result)

        with open(path) as f:
            data = json.load(f)

        # ISO 8601 pattern: YYYY-MM-DDTHH:MM:SS
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.match(iso_pattern, data["timestamp"]), (
            f"timestamp '{data['timestamp']}' is not ISO 8601 format"
        )


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

class TestCsvWriter:
    def test_csv_file_created(self, tmp_path: Path) -> None:
        results = [_make_result(), _make_result(runtime="onnx_coreml_fp32", precision="fp32")]
        writer = ResultWriter(output_dir=str(tmp_path))
        writer.write_csv(results)

        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == 1

    def test_csv_has_header_row(self, tmp_path: Path) -> None:
        results = [_make_result()]
        writer = ResultWriter(output_dir=str(tmp_path))
        path = writer.write_csv(results)

        with open(path) as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert len(reader.fieldnames) > 0

    def test_csv_contains_all_results(self, tmp_path: Path) -> None:
        results = [
            _make_result(runtime="pytorch_cpu_fp32"),
            _make_result(runtime="onnx_coreml_fp16"),
            _make_result(runtime="tensorrt_fp32"),
        ]
        writer = ResultWriter(output_dir=str(tmp_path))
        path = writer.write_csv(results)

        with open(path) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 3

    def test_csv_runtime_column_no_data_loss(self, tmp_path: Path) -> None:
        results = [_make_result(runtime="onnx_coreml_int8")]
        writer = ResultWriter(output_dir=str(tmp_path))
        path = writer.write_csv(results)

        with open(path) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["runtime"] == "onnx_coreml_int8"

    def test_csv_numeric_fields_preserved(self, tmp_path: Path) -> None:
        results = [_make_result(mean_latency_ms=12.345, map_50_95=0.372)]
        writer = ResultWriter(output_dir=str(tmp_path))
        path = writer.write_csv(results)

        with open(path) as f:
            rows = list(csv.DictReader(f))

        assert abs(float(rows[0]["mean_latency_ms"]) - 12.345) < 1e-6
        assert abs(float(rows[0]["map_50_95"]) - 0.372) < 1e-6

    def test_csv_header_includes_key_columns(self, tmp_path: Path) -> None:
        results = [_make_result()]
        writer = ResultWriter(output_dir=str(tmp_path))
        path = writer.write_csv(results)

        with open(path) as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])

        required = {"runtime", "precision", "hardware", "mean_latency_ms", "p95_latency_ms",
                    "map_50_95", "map_delta_vs_fp32", "peak_memory_mb", "fps"}
        assert required.issubset(headers), f"Missing CSV columns: {required - headers}"
