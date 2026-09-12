"""Integration tests for cross-runtime output consistency.

Validates that PyTorch and ONNX Runtime produce outputs within the parity
tolerance on identical inputs. Requires real model files.
Run with: pytest tests/integration/ -m integration
"""

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.mark.integration
def test_pytorch_onnx_output_parity():
    """PyTorch CPU and ONNX CPU EP must agree within atol=1e-3 on same input.

    1e-3 (not 1e-4) is the correct tolerance — see
    ``docs/specs/IMPLEMENTATION_SPEC.md`` and
    ``src/export/onnx_exporter.py``'s ``_DEFAULT_PARITY_ATOL``.
    """
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")

    from src.runtimes.pytorch_runtime import PyTorchRuntime
    from src.runtimes.onnx_runtime import OnnxRuntime
    from src.export.onnx_exporter import validate_output_parity

    dummy = np.random.default_rng(0).random((1, 3, 640, 640)).astype(np.float32)

    pt_runtime = PyTorchRuntime(device="cpu", precision="fp32")
    pt_runtime.load("models/yolov8n.pt")
    pt_out = pt_runtime.infer(dummy)

    onnx_runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
    onnx_runtime.load("models/yolov8n.onnx")
    onnx_out = onnx_runtime.infer(dummy)

    validate_output_parity(pt_out, onnx_out, atol=1e-3)
