"""Unit tests for reproducibility utilities."""

import random

import numpy as np
import pytest

from src.utils.reproducibility import set_seed


class TestSetSeed:
    def test_numpy_seed_is_deterministic(self) -> None:
        set_seed(42)
        a = np.random.rand(10)
        set_seed(42)
        b = np.random.rand(10)
        np.testing.assert_array_equal(a, b)

    def test_python_random_is_deterministic(self) -> None:
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_different_seeds_give_different_results(self) -> None:
        set_seed(42)
        a = np.random.rand(10)
        set_seed(99)
        b = np.random.rand(10)
        assert not np.array_equal(a, b)

    def test_default_seed_is_42(self) -> None:
        set_seed()
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)
