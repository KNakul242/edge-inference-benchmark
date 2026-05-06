"""Unit tests for the latency profiler.

This module requires 90%+ coverage — bugs here produce silently wrong
benchmark numbers, which invalidates the entire study.

Critical invariants verified:
- Warmup runs are discarded from timing
- p95 is the 95th percentile, not mean or p90
- time.perf_counter is used, not time.time
- Returned count equals n_runs, not n_runs + n_warmup
- Mean and stddev are numerically correct on known input
"""

import inspect
import statistics
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from src.benchmark.latency_profiler import LatencyResult, profile_latency


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(return_value: np.ndarray | None = None) -> MagicMock:
    runtime = MagicMock()
    runtime.infer.return_value = return_value or np.zeros((1, 84, 8400), dtype=np.float32)
    return runtime


def _controlled_perf_counter(latencies_ms: list[float], n_warmup: int = 0) -> list[float]:
    """Build a perf_counter side-effect sequence for known latencies.

    Each inference call consumes two perf_counter values (before/after).
    Warmup calls get 0ms latency so they don't pollute the sequence.
    """
    values: list[float] = []
    t = 0.0
    for _ in range(n_warmup):
        values.extend([t, t])  # 0ms warmup
        t += 0.001
    for lat_ms in latencies_ms:
        values.append(t)
        values.append(t + lat_ms / 1000.0)
        t += lat_ms / 1000.0 + 0.001
    return values


# ---------------------------------------------------------------------------
# Warmup isolation
# ---------------------------------------------------------------------------

class TestWarmupIsolation:
    def test_warmup_calls_are_discarded_from_timing(self) -> None:
        """Warmup run count must not appear in n_runs or the measurement list."""
        runtime = _make_runtime()
        latencies = [10.0] * 20
        counter_values = _controlled_perf_counter(latencies, n_warmup=5)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=20, n_warmup=5)

        assert result.n_runs == 20
        assert result.n_warmup == 5
        # Total infer calls = warmup + timed
        assert runtime.infer.call_count == 25

    def test_warmup_zero_still_profiles_correctly(self) -> None:
        runtime = _make_runtime()
        latencies = [10.0] * 10
        counter_values = _controlled_perf_counter(latencies, n_warmup=0)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=10, n_warmup=0)

        assert result.n_runs == 10
        assert runtime.infer.call_count == 10

    def test_result_count_is_n_runs_not_total(self) -> None:
        runtime = _make_runtime()
        latencies = [5.0] * 15
        counter_values = _controlled_perf_counter(latencies, n_warmup=10)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=15, n_warmup=10)

        assert result.n_runs == 15  # not 25


# ---------------------------------------------------------------------------
# p95 correctness
# ---------------------------------------------------------------------------

class TestP95Correctness:
    def test_p95_is_95th_percentile_not_mean(self) -> None:
        """90 fast + 10 slow: np.percentile p95 lands unambiguously in the slow bucket.

        With [10]*90 + [100]*10 (100 total), the 95th percentile position is
        0.95*99=94.05 → interpolates between index 94 (100ms) and 95 (100ms) → 100ms.
        Using [10]*95 + [100]*5 instead would straddle the bucket boundary.
        """
        runtime = _make_runtime()
        latencies = [10.0] * 90 + [100.0] * 10
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=100, n_warmup=0)

        assert result.p95_ms >= 90.0, f"p95={result.p95_ms} should be in the slow bucket (~100ms)"
        assert result.mean_ms < 20.0, f"mean={result.mean_ms} should be close to 10ms"

    def test_p95_not_p90(self) -> None:
        """Verify p95 differs from p90 on a distribution where they separate."""
        runtime = _make_runtime()
        # 90 at 10ms, 5 at 50ms, 5 at 100ms → p90=10ms, p95=50ms
        latencies = [10.0] * 90 + [50.0] * 5 + [100.0] * 5
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=100, n_warmup=0)

        p90 = float(np.percentile([10.0] * 90 + [50.0] * 5 + [100.0] * 5, 90))
        assert result.p95_ms > p90, "p95 must exceed p90 on this distribution"

    def test_p95_uniform_distribution(self) -> None:
        """Uniform latency: p95 must equal mean approximately."""
        runtime = _make_runtime()
        latencies = [20.0] * 100
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=100, n_warmup=0)

        assert abs(result.p95_ms - 20.0) < 1.0


# ---------------------------------------------------------------------------
# Timer correctness
# ---------------------------------------------------------------------------

class TestTimerCorrectness:
    def test_uses_perf_counter_not_time_time(self) -> None:
        """Verify perf_counter is called, not time.time, for sub-ms resolution."""
        runtime = _make_runtime()
        latencies = [5.0] * 5
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=5, n_warmup=0)

        assert mock_time.perf_counter.called
        mock_time.time.assert_not_called()

    def test_perf_counter_called_twice_per_run(self) -> None:
        """Each timed run requires one perf_counter call before and one after."""
        runtime = _make_runtime()
        n_runs = 7
        latencies = [5.0] * n_runs
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=n_runs, n_warmup=0)

        assert mock_time.perf_counter.call_count == n_runs * 2

    def test_source_contains_perf_counter(self) -> None:
        """Source-level guard: perf_counter must appear in implementation."""
        from src.benchmark import latency_profiler
        source = inspect.getsource(latency_profiler)
        assert "perf_counter" in source


# ---------------------------------------------------------------------------
# Statistical correctness
# ---------------------------------------------------------------------------

class TestStatisticalCorrectness:
    def test_mean_correct_on_known_latencies(self) -> None:
        runtime = _make_runtime()
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=5, n_warmup=0)

        expected_mean = statistics.mean(latencies)
        assert abs(result.mean_ms - expected_mean) < 0.1

    def test_stddev_correct_on_known_latencies(self) -> None:
        runtime = _make_runtime()
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=5, n_warmup=0)

        expected_stddev = statistics.stdev(latencies)
        assert abs(result.stddev_ms - expected_stddev) < 0.1

    def test_min_max_bounds(self) -> None:
        runtime = _make_runtime()
        latencies = [5.0, 10.0, 15.0, 20.0, 25.0]
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=5, n_warmup=0)

        assert abs(result.min_ms - 5.0) < 0.1
        assert abs(result.max_ms - 25.0) < 0.1

    def test_p95_geq_mean(self) -> None:
        runtime = _make_runtime()
        latencies = [10.0] * 90 + [100.0] * 10
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=100, n_warmup=0)

        assert result.p95_ms >= result.mean_ms - 1e-9  # float tolerance at distribution boundary

    def test_min_leq_mean_leq_max(self) -> None:
        runtime = _make_runtime()
        latencies = [10.0] * 90 + [100.0] * 10
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=100, n_warmup=0)

        assert result.min_ms <= result.mean_ms <= result.max_ms


# ---------------------------------------------------------------------------
# Return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_latency_result_dataclass(self) -> None:
        runtime = _make_runtime()
        latencies = [10.0] * 5
        counter_values = _controlled_perf_counter(latencies)

        with patch("src.benchmark.latency_profiler.time") as mock_time:
            mock_time.perf_counter.side_effect = counter_values
            result = profile_latency(runtime, np.zeros((1, 3, 640, 640)), n_runs=5, n_warmup=0)

        assert isinstance(result, LatencyResult)
        assert isinstance(result.mean_ms, float)
        assert isinstance(result.stddev_ms, float)
        assert isinstance(result.p95_ms, float)
        assert isinstance(result.min_ms, float)
        assert isinstance(result.max_ms, float)
        assert isinstance(result.n_runs, int)
        assert isinstance(result.n_warmup, int)
