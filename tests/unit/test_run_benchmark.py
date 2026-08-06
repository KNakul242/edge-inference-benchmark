"""Unit tests for the benchmark runner entry point.

Covers only the --runtime/--precision CLI filtering logic. scripts/* is
excluded from the coverage gate (pyproject.toml), but this filter is the
safety mechanism that lets a Mac benchmark run target only the new
pytorch_mps/onnx_coreml runtimes without re-executing (and silently
overwriting) the canonical Fedora CPU result files, so it is tested
directly.
"""

from dataclasses import dataclass

import scripts.run_benchmark as run_benchmark


@dataclass
class _FakeRuntime:
    name: str


def _runtimes(*names: str) -> list[_FakeRuntime]:
    return [_FakeRuntime(name=n) for n in names]


class TestFilterRuntimes:
    def test_no_filters_returns_all_runtimes_unchanged(self) -> None:
        runtimes = _runtimes("pytorch_cpu_fp32", "onnx_cpu_fp32", "pytorch_mps_fp32")
        result = run_benchmark.filter_runtimes(runtimes, runtime_filter=None, precision_filter=None)
        assert result == runtimes

    def test_runtime_filter_keeps_only_matching_substring(self) -> None:
        runtimes = _runtimes("pytorch_cpu_fp32", "pytorch_mps_fp32", "onnx_coreml_fp32")
        result = run_benchmark.filter_runtimes(runtimes, runtime_filter=["pytorch_mps"], precision_filter=None)
        assert [rt.name for rt in result] == ["pytorch_mps_fp32"]

    def test_runtime_filter_excludes_canonical_fedora_cpu_runtimes(self) -> None:
        """Regression guard: a Mac-targeted filter must never let cpu/CPUExecutionProvider
        runtimes through, since they would silently overwrite the canonical
        pytorch_cpu_fp32.json / onnx_cpu_fp32.json Fedora baseline files.
        """
        runtimes = _runtimes(
            "pytorch_cpu_fp32",
            "onnx_cpu_fp32",
            "pytorch_mps_fp32",
            "pytorch_mps_fp16",
            "onnx_coreml_fp32",
        )
        result = run_benchmark.filter_runtimes(
            runtimes, runtime_filter=["pytorch_mps", "onnx_coreml"], precision_filter=None
        )
        names = [rt.name for rt in result]
        assert "pytorch_cpu_fp32" not in names
        assert "onnx_cpu_fp32" not in names
        assert set(names) == {"pytorch_mps_fp32", "pytorch_mps_fp16", "onnx_coreml_fp32"}

    def test_precision_filter_keeps_only_matching_precision_suffix(self) -> None:
        runtimes = _runtimes("pytorch_mps_fp32", "pytorch_mps_fp16")
        result = run_benchmark.filter_runtimes(runtimes, runtime_filter=None, precision_filter=["fp32"])
        assert [rt.name for rt in result] == ["pytorch_mps_fp32"]

    def test_runtime_and_precision_filters_combine(self) -> None:
        runtimes = _runtimes(
            "pytorch_cpu_fp32", "pytorch_mps_fp32", "pytorch_mps_fp16", "onnx_coreml_fp32"
        )
        result = run_benchmark.filter_runtimes(
            runtimes, runtime_filter=["pytorch_mps"], precision_filter=["fp16"]
        )
        assert [rt.name for rt in result] == ["pytorch_mps_fp16"]

    def test_preserves_original_order(self) -> None:
        runtimes = _runtimes("onnx_coreml_fp32", "pytorch_mps_fp32", "pytorch_mps_fp16")
        result = run_benchmark.filter_runtimes(
            runtimes, runtime_filter=["pytorch_mps", "onnx_coreml"], precision_filter=None
        )
        assert [rt.name for rt in result] == [
            "onnx_coreml_fp32",
            "pytorch_mps_fp32",
            "pytorch_mps_fp16",
        ]


def _full_matrix_config() -> dict:
    return {
        "runtimes": {
            "pytorch": {"devices": ["cpu", "mps"], "precisions": ["fp32", "fp16"]},
            "onnx": {
                "execution_providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
                "precisions": ["fp32", "fp16", "int8"],
            },
        }
    }


class TestBuildRuntimes:
    def test_includes_pytorch_mps_fp32_and_fp16(self) -> None:
        names = [rt.name for rt in run_benchmark.build_runtimes(_full_matrix_config())]
        assert "pytorch_mps_fp32" in names
        assert "pytorch_mps_fp16" in names

    def test_includes_onnx_coreml_fp32(self) -> None:
        names = [rt.name for rt in run_benchmark.build_runtimes(_full_matrix_config())]
        assert "onnx_coreml_fp32" in names

    def test_excludes_onnx_coreml_fp16_and_int8(self) -> None:
        """Quantization pipeline doesn't exist yet — running these would silently
        mislabel FP32 output as fp16/int8. Must be skipped, not attempted.
        """
        names = [rt.name for rt in run_benchmark.build_runtimes(_full_matrix_config())]
        assert "onnx_coreml_fp16" not in names
        assert "onnx_coreml_int8" not in names

    def test_still_includes_fedora_cpu_baselines(self) -> None:
        names = [rt.name for rt in run_benchmark.build_runtimes(_full_matrix_config())]
        assert "pytorch_cpu_fp32" in names
        assert "onnx_cpu_fp32" in names

    def test_still_excludes_pytorch_cpu_fp16(self) -> None:
        """FP16 is not a valid CPU deployment target regardless of Mac availability."""
        names = [rt.name for rt in run_benchmark.build_runtimes(_full_matrix_config())]
        assert "pytorch_cpu_fp16" not in names


class TestResolveHardware:
    def test_mps_maps_to_mac_m5(self) -> None:
        assert run_benchmark.resolve_hardware("pytorch_mps_fp32") == "mac_m5"

    def test_tensorrt_maps_to_colab_t4(self) -> None:
        assert run_benchmark.resolve_hardware("tensorrt_fp32") == "colab_t4"

    def test_cpu_maps_to_fedora_cpu(self) -> None:
        assert run_benchmark.resolve_hardware("pytorch_cpu_fp32") == "fedora_cpu"
