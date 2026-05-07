"""Unit tests for the PyTorch runtime and BaseRuntime contract.

All tests mock torch to avoid requiring GPU hardware or installed weights.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.runtimes.base_runtime import BaseRuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime


class TestBaseRuntimeContract:
    """BaseRuntime must be abstract — concrete subclasses must implement all methods."""

    def test_base_runtime_cannot_be_instantiated(self) -> None:
        with pytest.raises(TypeError):
            BaseRuntime()  # type: ignore[abstract]

    def test_subclass_missing_name_property_raises(self) -> None:
        class Incomplete(BaseRuntime):
            def load(self, model_path: str) -> None: ...
            def infer(self, input_tensor: np.ndarray) -> np.ndarray: ...
            def warmup(self, input_tensor: np.ndarray, n_runs: int) -> None: ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_missing_load_raises(self) -> None:
        class Incomplete(BaseRuntime):
            @property
            def name(self) -> str: return "x"
            def infer(self, input_tensor: np.ndarray) -> np.ndarray: ...
            def warmup(self, input_tensor: np.ndarray, n_runs: int) -> None: ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_missing_infer_raises(self) -> None:
        class Incomplete(BaseRuntime):
            @property
            def name(self) -> str: return "x"
            def load(self, model_path: str) -> None: ...
            def warmup(self, input_tensor: np.ndarray, n_runs: int) -> None: ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_subclass_missing_warmup_raises(self) -> None:
        class Incomplete(BaseRuntime):
            @property
            def name(self) -> str: return "x"
            def load(self, model_path: str) -> None: ...
            def infer(self, input_tensor: np.ndarray) -> np.ndarray: ...

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        class Complete(BaseRuntime):
            @property
            def name(self) -> str: return "complete"
            def load(self, model_path: str) -> None: ...
            def infer(self, input_tensor: np.ndarray) -> np.ndarray:
                return np.zeros((1, 84, 8400))
            def warmup(self, input_tensor: np.ndarray, n_runs: int) -> None: ...

        runtime = Complete()
        assert runtime.name == "complete"


class TestPyTorchRuntimeName:
    def test_cpu_name(self) -> None:
        runtime = PyTorchRuntime(device="cpu", precision="fp32")
        assert runtime.name == "pytorch_cpu_fp32"

    def test_mps_name(self) -> None:
        runtime = PyTorchRuntime(device="mps", precision="fp16")
        assert runtime.name == "pytorch_mps_fp16"

    def test_name_encodes_device_and_precision(self) -> None:
        runtime = PyTorchRuntime(device="cpu", precision="fp32")
        assert "cpu" in runtime.name
        assert "fp32" in runtime.name


class TestPyTorchRuntimeLoad:
    def test_load_uses_ultralytics_yolo(self, tmp_path) -> None:
        mock_model = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        with patch("src.runtimes.pytorch_runtime.torch"), \
             patch("src.runtimes.pytorch_runtime._ultralytics_YOLO", return_value=mock_yolo_instance):
            runtime = PyTorchRuntime(device="cpu", precision="fp32")
            runtime.load(str(tmp_path / "model.pt"))

        mock_yolo_instance.model.to.assert_called_once_with("cpu")

    def test_load_sets_model_to_eval_mode(self, tmp_path) -> None:
        mock_model = MagicMock()
        mock_yolo_instance = MagicMock()
        mock_yolo_instance.model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        with patch("src.runtimes.pytorch_runtime.torch"), \
             patch("src.runtimes.pytorch_runtime._ultralytics_YOLO", return_value=mock_yolo_instance):
            runtime = PyTorchRuntime(device="cpu", precision="fp32")
            runtime.load(str(tmp_path / "model.pt"))

        mock_model.eval.assert_called_once()

    def test_load_raises_if_file_missing(self) -> None:
        with patch("src.runtimes.pytorch_runtime.torch"), \
             patch("src.runtimes.pytorch_runtime._ultralytics_YOLO") as mock_yolo_cls:
            mock_yolo_cls.side_effect = FileNotFoundError("no such file")
            runtime = PyTorchRuntime(device="cpu", precision="fp32")
            with pytest.raises(FileNotFoundError):
                runtime.load("/nonexistent/model.pt")


class TestPyTorchRuntimeInfer:
    def _make_loaded_runtime(self, device: str = "cpu", precision: str = "fp32") -> PyTorchRuntime:
        runtime = PyTorchRuntime(device=device, precision=precision)
        mock_output = MagicMock()
        mock_output.cpu.return_value.numpy.return_value = np.zeros((1, 84, 8400), dtype=np.float32)
        runtime._model = MagicMock(return_value=mock_output)
        return runtime

    def test_infer_raises_if_not_loaded(self, dummy_input: np.ndarray) -> None:
        runtime = PyTorchRuntime(device="cpu", precision="fp32")
        with pytest.raises(RuntimeError, match="load"):
            runtime.infer(dummy_input)

    def test_infer_returns_numpy_array(self, dummy_input: np.ndarray) -> None:
        runtime = self._make_loaded_runtime()
        with patch("src.runtimes.pytorch_runtime.torch") as mock_torch:
            mock_torch.no_grad.return_value.__enter__ = MagicMock(return_value=None)
            mock_torch.no_grad.return_value.__exit__ = MagicMock(return_value=False)
            mock_tensor = MagicMock()
            mock_output = MagicMock()
            mock_output.cpu.return_value.numpy.return_value = np.zeros((1, 84, 8400), dtype=np.float32)
            mock_torch.from_numpy.return_value.to.return_value = mock_tensor
            runtime._model = MagicMock(return_value=mock_output)
            result = runtime.infer(dummy_input)

        assert isinstance(result, np.ndarray)


class TestPyTorchRuntimeWarmup:
    def test_warmup_calls_infer_n_times(self, dummy_input: np.ndarray) -> None:
        runtime = PyTorchRuntime(device="cpu", precision="fp32")
        runtime.infer = MagicMock(return_value=np.zeros((1, 84, 8400), dtype=np.float32))

        runtime.warmup(dummy_input, n_runs=5)

        assert runtime.infer.call_count == 5

    def test_warmup_zero_runs_does_not_call_infer(self, dummy_input: np.ndarray) -> None:
        runtime = PyTorchRuntime(device="cpu", precision="fp32")
        runtime.infer = MagicMock(return_value=np.zeros((1, 84, 8400), dtype=np.float32))

        runtime.warmup(dummy_input, n_runs=0)

        runtime.infer.assert_not_called()


class TestPyTorchRuntimeInheritsBase:
    def test_is_subclass_of_base_runtime(self) -> None:
        assert issubclass(PyTorchRuntime, BaseRuntime)
