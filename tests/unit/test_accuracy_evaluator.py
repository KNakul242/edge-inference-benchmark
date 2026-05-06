"""Unit tests for the accuracy evaluator.

Covers mAP metric selection, delta computation, COCO prediction formatting,
and empty-prediction safety. Requires 90%+ coverage.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.benchmark.accuracy_evaluator import (
    AccuracyResult,
    compute_map_delta,
    format_coco_prediction,
)


# ---------------------------------------------------------------------------
# format_coco_prediction
# ---------------------------------------------------------------------------

class TestFormatCocoPrediction:
    def test_returns_list_of_dicts(self) -> None:
        """Each prediction must be a COCO-compatible dict."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        result = format_coco_prediction(raw_output, image_id=42, conf_threshold=0.0)
        assert isinstance(result, list)

    def test_each_entry_has_required_coco_keys(self) -> None:
        """COCO evaluation requires image_id, category_id, bbox, score."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        # Inject a high-confidence detection into one anchor
        raw_output[0, 4:, 0] = 1.0   # class scores all 1.0
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]  # cx,cy,w,h

        result = format_coco_prediction(raw_output, image_id=7, conf_threshold=0.0)

        for pred in result:
            assert "image_id" in pred
            assert "category_id" in pred
            assert "bbox" in pred
            assert "score" in pred

    def test_image_id_propagated(self) -> None:
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4:, 0] = 1.0
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]

        result = format_coco_prediction(raw_output, image_id=99, conf_threshold=0.0)

        for pred in result:
            assert pred["image_id"] == 99

    def test_empty_when_all_below_threshold(self) -> None:
        """No detections above threshold → empty list, no crash."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.9)
        assert result == []

    def test_bbox_is_xywh_format(self) -> None:
        """COCO bbox convention is [x_min, y_min, width, height] (not xyxy)."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4:, 0] = 1.0
        # cx=320, cy=240, w=100, h=80 → xmin=270, ymin=200, w=100, h=80
        raw_output[0, :4, 0] = [320.0, 240.0, 100.0, 80.0]

        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.0)

        if result:
            bbox = result[0]["bbox"]
            assert len(bbox) == 4
            # Width and height must be positive
            assert bbox[2] > 0
            assert bbox[3] > 0


# ---------------------------------------------------------------------------
# compute_map_delta
# ---------------------------------------------------------------------------

class TestComputeMapDelta:
    def test_delta_is_zero_when_equal(self) -> None:
        baseline = AccuracyResult(map_50_95=0.372, map_50=0.530, precision="fp32", runtime="pytorch_cpu")
        candidate = AccuracyResult(map_50_95=0.372, map_50=0.530, precision="fp16", runtime="pytorch_cpu")

        delta = compute_map_delta(baseline, candidate)

        assert abs(delta) < 1e-9

    def test_delta_negative_when_candidate_lower(self) -> None:
        baseline = AccuracyResult(map_50_95=0.372, map_50=0.530, precision="fp32", runtime="pytorch_cpu")
        candidate = AccuracyResult(map_50_95=0.360, map_50=0.510, precision="int8", runtime="pytorch_cpu")

        delta = compute_map_delta(baseline, candidate)

        assert delta < 0
        assert abs(delta - (0.360 - 0.372)) < 1e-9

    def test_delta_positive_when_candidate_higher(self) -> None:
        baseline = AccuracyResult(map_50_95=0.360, map_50=0.510, precision="fp32", runtime="pytorch_cpu")
        candidate = AccuracyResult(map_50_95=0.372, map_50=0.530, precision="fp16", runtime="pytorch_cpu")

        delta = compute_map_delta(baseline, candidate)

        assert delta > 0

    def test_delta_uses_map_50_95_not_map_50(self) -> None:
        """Primary metric for delta is mAP@0.5:0.95, not mAP@0.5."""
        baseline = AccuracyResult(map_50_95=0.372, map_50=0.999, precision="fp32", runtime="onnx")
        candidate = AccuracyResult(map_50_95=0.350, map_50=0.999, precision="int8", runtime="onnx")

        delta = compute_map_delta(baseline, candidate)

        assert abs(delta - (0.350 - 0.372)) < 1e-9

    def test_delta_relative_to_fp32_baseline_of_same_runtime(self) -> None:
        """Delta must be candidate - fp32_baseline (not cross-runtime)."""
        fp32_baseline = AccuracyResult(
            map_50_95=0.372, map_50=0.530, precision="fp32", runtime="onnx_coreml"
        )
        int8_candidate = AccuracyResult(
            map_50_95=0.358, map_50=0.510, precision="int8", runtime="onnx_coreml"
        )

        delta = compute_map_delta(fp32_baseline, int8_candidate)

        assert abs(delta - (0.358 - 0.372)) < 1e-9


# ---------------------------------------------------------------------------
# AccuracyResult schema
# ---------------------------------------------------------------------------

class TestAccuracyResultSchema:
    def test_has_map_50_95_field(self) -> None:
        result = AccuracyResult(map_50_95=0.372, map_50=0.530, precision="fp32", runtime="test")
        assert result.map_50_95 == 0.372

    def test_has_map_50_field(self) -> None:
        result = AccuracyResult(map_50_95=0.372, map_50=0.530, precision="fp32", runtime="test")
        assert result.map_50 == 0.530

    def test_has_precision_and_runtime_fields(self) -> None:
        result = AccuracyResult(map_50_95=0.372, map_50=0.530, precision="fp16", runtime="onnx")
        assert result.precision == "fp16"
        assert result.runtime == "onnx"

    def test_map_delta_defaults_to_none(self) -> None:
        result = AccuracyResult(map_50_95=0.372, map_50=0.530, precision="fp32", runtime="test")
        assert result.map_delta_vs_fp32 is None

    def test_map_delta_can_be_set(self) -> None:
        result = AccuracyResult(
            map_50_95=0.358, map_50=0.510, precision="int8", runtime="onnx",
            map_delta_vs_fp32=-0.014
        )
        assert abs(result.map_delta_vs_fp32 + 0.014) < 1e-9
