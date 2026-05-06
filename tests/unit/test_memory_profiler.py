"""Unit tests for the memory profiler."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.benchmark.memory_profiler import MemoryResult, profile_memory


class TestMemoryResult:
    def test_has_peak_mb_field(self) -> None:
        result = MemoryResult(peak_mb=128.5, runtime="pytorch_cpu_fp32")
        assert result.peak_mb == 128.5

    def test_has_runtime_field(self) -> None:
        result = MemoryResult(peak_mb=64.0, runtime="onnx_coreml_fp16")
        assert result.runtime == "onnx_coreml_fp16"

    def test_peak_mb_is_float(self) -> None:
        result = MemoryResult(peak_mb=256.0, runtime="test")
        assert isinstance(result.peak_mb, float)


class TestProfileMemory:
    def test_returns_memory_result(self, dummy_input: np.ndarray) -> None:
        runtime = MagicMock()
        runtime.name = "mock_runtime"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        with patch("src.benchmark.memory_profiler.tracemalloc") as mock_tm:
            mock_tm.get_traced_memory.return_value = (1024 * 1024 * 50, 1024 * 1024 * 128)
            result = profile_memory(runtime, dummy_input)

        assert isinstance(result, MemoryResult)

    def test_peak_mb_converted_from_bytes(self, dummy_input: np.ndarray) -> None:
        """Peak memory must be reported in MB, not bytes."""
        runtime = MagicMock()
        runtime.name = "mock_runtime"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        peak_bytes = 1024 * 1024 * 256  # 256 MB

        with patch("src.benchmark.memory_profiler.tracemalloc") as mock_tm:
            mock_tm.get_traced_memory.return_value = (0, peak_bytes)
            result = profile_memory(runtime, dummy_input)

        assert abs(result.peak_mb - 256.0) < 0.1

    def test_runtime_name_in_result(self, dummy_input: np.ndarray) -> None:
        runtime = MagicMock()
        runtime.name = "pytorch_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        with patch("src.benchmark.memory_profiler.tracemalloc") as mock_tm:
            mock_tm.get_traced_memory.return_value = (0, 1024 * 1024 * 64)
            result = profile_memory(runtime, dummy_input)

        assert result.runtime == "pytorch_cpu_fp32"

    def test_tracemalloc_started_and_stopped(self, dummy_input: np.ndarray) -> None:
        runtime = MagicMock()
        runtime.name = "test"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        with patch("src.benchmark.memory_profiler.tracemalloc") as mock_tm:
            mock_tm.get_traced_memory.return_value = (0, 1024)
            profile_memory(runtime, dummy_input)

        mock_tm.start.assert_called_once()
        mock_tm.stop.assert_called_once()

    def test_infer_called_during_profiling(self, dummy_input: np.ndarray) -> None:
        runtime = MagicMock()
        runtime.name = "test"
        runtime.infer.return_value = np.zeros((1, 84, 8400))

        with patch("src.benchmark.memory_profiler.tracemalloc") as mock_tm:
            mock_tm.get_traced_memory.return_value = (0, 1024)
            profile_memory(runtime, dummy_input)

        runtime.infer.assert_called_once_with(dummy_input)
