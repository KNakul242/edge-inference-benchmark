"""Unit tests for the ONNX Runtime inference wrapper.

Tests cover provider selection logic (CoreML EP → CPU EP fallback), name
encoding, and BaseRuntime contract. Hardware-specific EP tests are integration
tests requiring Mac M4 hardware.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.runtimes.base_runtime import BaseRuntime
from src.runtimes.onnx_runtime import OnnxRuntime


class TestOnnxRuntimeInheritsBase:
    def test_is_subclass_of_base_runtime(self) -> None:
        assert issubclass(OnnxRuntime, BaseRuntime)


class TestOnnxRuntimeName:
    def test_name_encodes_provider_and_precision(self) -> None:
        runtime = OnnxRuntime(execution_provider="CoreMLExecutionProvider", precision="fp32")
        assert "coreml" in runtime.name.lower()
        assert "fp32" in runtime.name

    def test_cpu_provider_name(self) -> None:
        runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
        assert "cpu" in runtime.name.lower()
        assert "fp32" in runtime.name

    def test_fp16_in_name(self) -> None:
        runtime = OnnxRuntime(execution_provider="CoreMLExecutionProvider", precision="fp16")
        assert "fp16" in runtime.name

    def test_int8_in_name(self) -> None:
        runtime = OnnxRuntime(execution_provider="CoreMLExecutionProvider", precision="int8")
        assert "int8" in runtime.name


class TestOnnxRuntimeLoad:
    def test_load_creates_inference_session(self, tmp_path) -> None:
        mock_session = MagicMock()

        with patch("src.runtimes.onnx_runtime.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session
            runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
            runtime.load(str(tmp_path / "model.onnx"))

        mock_ort.InferenceSession.assert_called_once()

    def test_load_passes_provider_list(self, tmp_path) -> None:
        mock_session = MagicMock()

        with patch("src.runtimes.onnx_runtime.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session
            runtime = OnnxRuntime(
                execution_provider="CoreMLExecutionProvider", precision="fp32"
            )
            runtime.load(str(tmp_path / "model.onnx"))

        call_kwargs = mock_ort.InferenceSession.call_args
        providers = call_kwargs.kwargs.get("providers") or call_kwargs.args[1]
        assert "CoreMLExecutionProvider" in providers

    def test_load_includes_cpu_ep_as_fallback(self, tmp_path) -> None:
        mock_session = MagicMock()

        with patch("src.runtimes.onnx_runtime.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session
            runtime = OnnxRuntime(
                execution_provider="CoreMLExecutionProvider", precision="fp32"
            )
            runtime.load(str(tmp_path / "model.onnx"))

        call_kwargs = mock_ort.InferenceSession.call_args
        providers = call_kwargs.kwargs.get("providers") or call_kwargs.args[1]
        assert "CPUExecutionProvider" in providers

    def test_load_raises_if_not_loaded(self, dummy_input) -> None:
        runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
        with pytest.raises(RuntimeError, match="load"):
            runtime.infer(dummy_input)


class TestOnnxRuntimeInfer:
    def test_infer_returns_numpy_array(self, dummy_input: np.ndarray) -> None:
        mock_session = MagicMock()
        mock_session.run.return_value = [np.zeros((1, 84, 8400), dtype=np.float32)]
        mock_session.get_inputs.return_value = [MagicMock(name="images")]

        with patch("src.runtimes.onnx_runtime.ort") as mock_ort:
            mock_ort.InferenceSession.return_value = mock_session
            runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
            runtime._session = mock_session
            result = runtime.infer(dummy_input)

        assert isinstance(result, np.ndarray)


class TestOnnxRuntimeWarmup:
    def test_warmup_calls_infer_n_times(self, dummy_input: np.ndarray) -> None:
        runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
        runtime.infer = MagicMock(return_value=np.zeros((1, 84, 8400)))

        runtime.warmup(dummy_input, n_runs=4)

        assert runtime.infer.call_count == 4
