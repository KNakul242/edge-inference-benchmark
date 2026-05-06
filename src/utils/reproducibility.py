"""Reproducibility utilities — seed setting and determinism helpers.

All randomness in the benchmark pipeline flows through a single seed (42)
configured here. Calibration set sampling, any stochastic preprocessing,
and numpy operations must call set_seed() before execution.
"""

import logging
import random

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 42


def set_seed(seed: int = _DEFAULT_SEED) -> None:
    """Set random seeds for Python, numpy, and (if available) torch.

    Call once at pipeline entry before any stochastic operation. The seed
    value is logged so it appears in every run's output for traceability.

    Args:
        seed: Integer seed. Defaults to 42 (locked per project spec).
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not installed in this environment

    logger.info("Reproducibility seed set: %d", seed)
