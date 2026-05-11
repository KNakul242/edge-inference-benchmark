"""Unit tests for the memory profiler.

Memory is measured via psutil process RSS (when available) — the only approach
that captures native C++ allocations from ONNX Runtime and PyTorch, which
tracemalloc cannot see.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.benchmark.memory_profiler import MemoryResult, profile_memory


class TestMemoryResult:
    def test_has_peak_mb_field(self) -> None:
        result = MemoryResult(peak_mb=128.5, delta_mb=0.0, runtime="pytorch_cpu_fp32")
        assert result.peak_mb == 128.5

    def test_has_runtime_field(self) -> None:
        result = MemoryResult(peak_mb=64.0, delta_mb=0.0, runtime="onnx_coreml_fp16")
        assert result.runtime == "onnx_coreml_fp16"

    def test_peak_mb_is_float(self) -> None:
        result = MemoryResult(peak_mb=256.0, delta_mb=0.0, runtime="test")
        assert isinstance(result.peak_mb, float)

    def test_has_delta_mb_field(self) -> None:
        """H2 — MemoryResult must store delta_mb (inference-induced RSS growth), not just peak."""
        result = MemoryResult(peak_mb=256.0, delta_mb=12.5, runtime="test")
        assert result.delta_mb == 12.5

    def test_delta_mb_is_float(self) -> None:
        result = MemoryResult(peak_mb=256.0, delta_mb=0.0, runtime="test")
        assert isinstance(result.delta_mb, float)


class TestProfileMemory:
    def _mock_psutil(self, rss_bytes: int):
        """Return a mock _psutil module that reports the given RSS."""
        mock_psutil = MagicMock()
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = rss_bytes
        mock_psutil.Process.return_value = mock_proc
        return mock_psutil

    def test_returns_memory_result(self, dummy_input: np.ndarray) -> None:
        runtime = MagicMock()
        runtime.name = "mock_runtime"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        with patch("src.benchmark.memory_profiler._psutil", self._mock_psutil(256 * 1024 * 1024)):
            result = profile_memory(runtime, dummy_input)

        assert isinstance(result, MemoryResult)

    def test_peak_mb_converted_from_rss_bytes(self, dummy_input: np.ndarray) -> None:
        """Peak memory must be in MB derived from process RSS in bytes."""
        runtime = MagicMock()
        runtime.name = "mock_runtime"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        rss_bytes = 1024 * 1024 * 256  # 256 MB

        with patch("src.benchmark.memory_profiler._psutil", self._mock_psutil(rss_bytes)):
            result = profile_memory(runtime, dummy_input)

        assert abs(result.peak_mb - 256.0) < 0.1

    def test_runtime_name_in_result(self, dummy_input: np.ndarray) -> None:
        runtime = MagicMock()
        runtime.name = "pytorch_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        with patch("src.benchmark.memory_profiler._psutil", self._mock_psutil(64 * 1024 * 1024)):
            result = profile_memory(runtime, dummy_input)

        assert result.runtime == "pytorch_cpu_fp32"

    def test_infer_called_during_profiling(self, dummy_input: np.ndarray) -> None:
        """infer must be called at least once to populate model buffers before measurement."""
        runtime = MagicMock()
        runtime.name = "test"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        with patch("src.benchmark.memory_profiler._psutil", self._mock_psutil(128 * 1024 * 1024)):
            profile_memory(runtime, dummy_input)

        assert runtime.infer.call_count >= 1
        runtime.infer.assert_called_with(dummy_input)

    def test_falls_back_to_tracemalloc_when_psutil_unavailable(self, dummy_input: np.ndarray) -> None:
        """When psutil is None, must fall back to tracemalloc without crashing."""
        runtime = MagicMock()
        runtime.name = "test"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        with patch("src.benchmark.memory_profiler._psutil", None), \
             patch("src.benchmark.memory_profiler.tracemalloc") as mock_tm:
            mock_tm.get_traced_memory.return_value = (0, 1024 * 1024 * 64)
            result = profile_memory(runtime, dummy_input)

        assert isinstance(result, MemoryResult)
        mock_tm.start.assert_called_once()
        mock_tm.stop.assert_called_once()

    def test_tracemalloc_fallback_converts_bytes_to_mb(self, dummy_input: np.ndarray) -> None:
        """Tracemalloc fallback must report peak_mb in MB, not bytes."""
        runtime = MagicMock()
        runtime.name = "test"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        peak_bytes = 1024 * 1024 * 64  # 64 MB

        with patch("src.benchmark.memory_profiler._psutil", None), \
             patch("src.benchmark.memory_profiler.tracemalloc") as mock_tm:
            mock_tm.get_traced_memory.return_value = (0, peak_bytes)
            result = profile_memory(runtime, dummy_input)

        assert abs(result.peak_mb - 64.0) < 0.1

    def test_delta_mb_computed_from_rss_difference(self, dummy_input: np.ndarray) -> None:
        """H2 — delta_mb must be (rss_after - rss_before) / MB, not just rss_after."""
        runtime = MagicMock()
        runtime.name = "mock_runtime"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        rss_before = 300 * 1024 * 1024  # 300 MB before inference
        rss_after = 345 * 1024 * 1024   # 345 MB after inference

        mock_psutil = MagicMock()
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = rss_before
        mock_psutil.Process.return_value = mock_proc

        def rss_side_effect():
            class Info:
                pass
            info = Info()
            # Alternate: before=300MB, after=345MB
            if mock_proc.memory_info.call_count <= 1:
                info.rss = rss_before
            else:
                info.rss = rss_after
            return info

        mock_proc.memory_info.side_effect = rss_side_effect

        with patch("src.benchmark.memory_profiler._psutil", mock_psutil):
            result = profile_memory(runtime, dummy_input)

        assert abs(result.delta_mb - 45.0) < 0.1

    def test_delta_mb_stored_not_discarded(self, dummy_input: np.ndarray) -> None:
        """H2 — delta_mb must be stored in the returned MemoryResult, not just logged."""
        runtime = MagicMock()
        runtime.name = "pytorch_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        rss_values = iter([200 * 1024 * 1024, 250 * 1024 * 1024])

        mock_psutil = MagicMock()
        mock_proc = MagicMock()
        mock_proc.memory_info.side_effect = lambda: type("I", (), {"rss": next(rss_values)})()
        mock_psutil.Process.return_value = mock_proc

        with patch("src.benchmark.memory_profiler._psutil", mock_psutil):
            result = profile_memory(runtime, dummy_input)

        assert hasattr(result, "delta_mb"), "MemoryResult must have a delta_mb field"
        assert abs(result.delta_mb - 50.0) < 0.1  # 250MB - 200MB = 50MB
