"""Shared fixtures for unit and integration tests."""

import numpy as np
import pytest


@pytest.fixture
def dummy_input() -> np.ndarray:
    """Standard NCHW float32 input: (1, 3, 640, 640), values in [0, 1]."""
    rng = np.random.default_rng(0)
    return rng.random((1, 3, 640, 640)).astype(np.float32)


@pytest.fixture
def dummy_output() -> np.ndarray:
    """Standard YOLOv8n ONNX output tensor: (1, 84, 8400)."""
    rng = np.random.default_rng(0)
    return rng.random((1, 84, 8400)).astype(np.float32)
