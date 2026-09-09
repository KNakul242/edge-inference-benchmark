"""Unit tests for the accuracy evaluator.

Covers mAP metric selection, delta computation, COCO prediction formatting
(including letterbox coordinate rescaling), and empty-prediction safety.
Requires 90%+ coverage.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.coco_loader import LetterboxMeta
from src.benchmark.accuracy_evaluator import (
    AccuracyResult,
    compute_map_delta,
    evaluate_map,
    format_coco_prediction,
)


# ---------------------------------------------------------------------------
# format_coco_prediction
# ---------------------------------------------------------------------------

class TestFormatCocoPredictionOutputValidation:
    """H3 — output range and shape validation catches wrong ONNX export format."""

    def test_raises_on_wrong_output_shape(self) -> None:
        """Shape other than (1, 84, 8400) must raise immediately — not silently misparse."""
        wrong_shape = np.zeros((1, 80, 8400), dtype=np.float32)
        with pytest.raises(ValueError, match="shape"):
            format_coco_prediction(wrong_shape, image_id=1, conf_threshold=0.5)

    def test_raises_when_class_scores_are_logits(self) -> None:
        """Unbounded logit-range scores must raise — indicates missing sigmoid in export."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4:, 0] = 5.0   # logit value well above 1.0
        with pytest.raises(ValueError, match="probability range"):
            format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5)

    def test_raises_when_class_scores_are_negative(self) -> None:
        """Negative class scores indicate logit outputs, not post-sigmoid probabilities."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4:, 0] = -3.0
        with pytest.raises(ValueError, match="probability range"):
            format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5)

    def test_valid_probability_scores_do_not_raise(self) -> None:
        """Scores in [0, 1] must pass validation without error."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4, 0] = 0.9
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]
        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5)
        assert isinstance(result, list)


class TestFormatCocoPrediction:
    def test_returns_list_of_dicts(self) -> None:
        """Each prediction must be a COCO-compatible dict."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        result = format_coco_prediction(raw_output, image_id=42, conf_threshold=0.0)
        assert isinstance(result, list)

    def test_each_entry_has_required_coco_keys(self) -> None:
        """COCO evaluation requires image_id, category_id, bbox, score."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4:, 0] = 1.0
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]

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

    def test_bbox_is_xywh_format_without_meta(self) -> None:
        """Without letterbox_meta, COCO bbox is [x_min, y_min, w, h] in model space."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4:, 0] = 1.0
        # cx=320, cy=240, w=100, h=80 → xmin=270, ymin=200, w=100, h=80
        raw_output[0, :4, 0] = [320.0, 240.0, 100.0, 80.0]

        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.0)

        if result:
            bbox = result[0]["bbox"]
            assert len(bbox) == 4
            assert bbox[2] > 0
            assert bbox[3] > 0

    def test_coordinate_rescaling_with_letterbox_meta(self) -> None:
        """With letterbox_meta, bbox must be rescaled to original image coordinates."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        # Inject one high-confidence detection; use conf_threshold=0.5 to isolate it
        raw_output[0, 4:, 0] = 1.0   # class scores = 1.0 for anchor 0
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 80.0]  # cx=320, cy=320, w=100, h=80

        # Letterbox: 480×640 image → scale=1.0, pad_left=0, pad_top=80
        meta = LetterboxMeta(scale=1.0, pad_left=0, pad_top=80, orig_h=480, orig_w=640)
        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5, letterbox_meta=meta)

        assert len(result) == 1
        bbox = result[0]["bbox"]
        # x_min: (cx - w/2 - pad_left) / scale = (320 - 50 - 0) / 1.0 = 270
        # y_min: (cy - h/2 - pad_top) / scale = (320 - 40 - 80) / 1.0 = 200
        assert abs(bbox[0] - 270.0) < 1e-4   # x_min
        assert abs(bbox[1] - 200.0) < 1e-4   # y_min
        assert abs(bbox[2] - 100.0) < 1e-4   # w
        assert abs(bbox[3] - 80.0) < 1e-4    # h

    def test_coordinate_rescaling_applies_scale_factor(self) -> None:
        """When scale < 1.0 (large image), output boxes must be divided by scale."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4:, 0] = 1.0   # high-confidence detection at anchor 0
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]

        # scale=0.5 means original image was 2× bigger (1280×1280)
        meta = LetterboxMeta(scale=0.5, pad_left=0, pad_top=0, orig_h=1280, orig_w=1280)
        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5, letterbox_meta=meta)

        bbox = result[0]["bbox"]
        # x_min = (320 - 50 - 0) / 0.5 = 540
        # y_min = (320 - 50 - 0) / 0.5 = 540
        # w = 100 / 0.5 = 200
        assert abs(bbox[0] - 540.0) < 1e-4
        assert abs(bbox[1] - 540.0) < 1e-4
        assert abs(bbox[2] - 200.0) < 1e-4
        assert abs(bbox[3] - 200.0) < 1e-4

    def test_without_meta_bbox_unchanged_from_model_space(self) -> None:
        """When letterbox_meta is None, boxes are left in model coordinate space."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4:, 0] = 1.0
        raw_output[0, :4, 0] = [320.0, 240.0, 100.0, 80.0]

        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.0, letterbox_meta=None)
        bbox = result[0]["bbox"]

        assert abs(bbox[0] - 270.0) < 1e-4  # cx - w/2 = 270
        assert abs(bbox[1] - 200.0) < 1e-4  # cy - h/2 = 200

    def test_out_of_bounds_bbox_left_edge_clamped_to_zero(self) -> None:
        """M4 — box extending left of image edge must be clamped; bbox_w adjusted to preserve right edge.

        Anchor near the left edge: cx=10, cy=400, w=40, h=40.
        With scale=1.0, pad_left=0, pad_top=80, orig_h=480, orig_w=640:
          x_min = (10-20-0)/1.0 = -10 → clamped to 0
          x2_orig = -10 + 40 = 30 → bbox_w = 30 - 0 = 30 (not 40)
        """
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4, 0] = 0.9
        raw_output[0, :4, 0] = [10.0, 400.0, 40.0, 40.0]  # cx, cy, w, h

        meta = LetterboxMeta(scale=1.0, pad_left=0, pad_top=80, orig_h=480, orig_w=640)
        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.8, letterbox_meta=meta)

        assert len(result) == 1
        x_min, y_min, bbox_w, bbox_h = result[0]["bbox"]
        assert x_min == pytest.approx(0.0), f"x_min={x_min} must be clamped to 0"
        assert bbox_w == pytest.approx(30.0), f"bbox_w={bbox_w} must be 30 (right edge preserved at 30)"

    def test_out_of_bounds_bbox_right_edge_clamped_to_image_width(self) -> None:
        """M4 — box extending right of image edge must be clamped to orig_w.

        Anchor near the right edge: cx=630, cy=400, w=40, h=40.
        With scale=1.0, pad_left=0, orig_w=640:
          x_min = (630-20-0)/1.0 = 610
          x2_orig = 610 + 40 = 650 → clamped to 640 → bbox_w = 640-610 = 30
        """
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4, 0] = 0.9
        raw_output[0, :4, 0] = [630.0, 400.0, 40.0, 40.0]

        meta = LetterboxMeta(scale=1.0, pad_left=0, pad_top=80, orig_h=480, orig_w=640)
        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.8, letterbox_meta=meta)

        assert len(result) == 1
        x_min, y_min, bbox_w, bbox_h = result[0]["bbox"]
        assert x_min == pytest.approx(610.0)
        assert bbox_w == pytest.approx(30.0), f"bbox_w={bbox_w} must be 30 (clamped at orig_w=640)"

    def test_in_bounds_bbox_not_modified_by_clamping(self) -> None:
        """M4 — a box fully within image bounds must not be altered by clamping."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4, 0] = 0.9
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 80.0]  # cx=320,cy=320 — fully in bounds

        meta = LetterboxMeta(scale=1.0, pad_left=0, pad_top=80, orig_h=480, orig_w=640)
        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.8, letterbox_meta=meta)

        assert len(result) == 1
        x_min, y_min, bbox_w, bbox_h = result[0]["bbox"]
        assert x_min == pytest.approx(270.0)   # (320-50-0)/1.0
        assert y_min == pytest.approx(200.0)   # (320-40-80)/1.0
        assert bbox_w == pytest.approx(100.0)  # unchanged
        assert bbox_h == pytest.approx(80.0)   # unchanged

    def test_category_id_class_0_maps_to_coco_id_1(self) -> None:
        """Class index 0 (person) must map to COCO category_id 1, not 0."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4, 0] = 0.9   # class 0 = person; row 4+0=4
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]

        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5)

        assert len(result) == 1
        assert result[0]["category_id"] == 1

    def test_category_id_class_11_maps_to_coco_id_13_not_12(self) -> None:
        """Class index 11 must map to COCO category_id 13, not 12.

        COCO 2017 skips ID 12. The naive class_idx + 1 = 12 is wrong;
        the correct lookup via _COCO_CATEGORY_IDS[11] gives 13.
        """
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        raw_output[0, 4 + 11, 0] = 0.9   # class 11 at row 4+11=15
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]

        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5)

        assert len(result) == 1
        assert result[0]["category_id"] == 13
        assert result[0]["category_id"] != 12

    def test_nms_suppresses_overlapping_boxes_of_same_class(self) -> None:
        """Two heavily overlapping boxes of the same class → only highest score kept."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        # Anchor 0: class 0, score=0.9 — higher-confidence box
        raw_output[0, 4, 0] = 0.9
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]
        # Anchor 1: class 0, score=0.7 — nearly identical location (IoU ≈ 0.92 > 0.45)
        raw_output[0, 4, 1] = 0.7
        raw_output[0, :4, 1] = [322.0, 322.0, 100.0, 100.0]

        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5, iou_threshold=0.45)

        assert len(result) == 1
        assert abs(result[0]["score"] - 0.9) < 1e-4

    def test_nms_keeps_non_overlapping_boxes(self) -> None:
        """Two spatially separate boxes of the same class → both survive NMS."""
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        # Anchor 0: class 0, score=0.9 — top-left quadrant
        raw_output[0, 4, 0] = 0.9
        raw_output[0, :4, 0] = [100.0, 100.0, 50.0, 50.0]
        # Anchor 1: class 0, score=0.8 — bottom-right quadrant (no spatial overlap)
        raw_output[0, 4, 1] = 0.8
        raw_output[0, :4, 1] = [500.0, 500.0, 50.0, 50.0]

        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5, iou_threshold=0.45)

        assert len(result) == 2

    def test_agnostic_nms_suppresses_overlapping_different_class_boxes(self) -> None:
        """Agnostic NMS: two overlapping boxes of different classes → only highest score kept.

        Per-class NMS (the YOLOv8 reference default — verified against
        ultralytics==8.2.103's non_max_suppression(), agnostic=False by
        default) would keep both, since different classes mean no
        cross-class suppression. This pipeline's agnostic NMS is a
        deliberate divergence from that reference, not a match — it
        suppresses the lower-score box regardless of class label.
        """
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        # Anchor 0: class 0 (person), score=0.9 — higher confidence
        raw_output[0, 4, 0] = 0.9
        raw_output[0, :4, 0] = [320.0, 320.0, 100.0, 100.0]
        # Anchor 1: class 1 (bicycle), score=0.7 — overlapping (IoU ≈ 0.92), different class
        raw_output[0, 5, 1] = 0.7
        raw_output[0, :4, 1] = [322.0, 322.0, 100.0, 100.0]

        result = format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5, iou_threshold=0.45)

        # Agnostic NMS: lower-score box suppressed regardless of class
        assert len(result) == 1
        assert abs(result[0]["score"] - 0.9) < 1e-4

    def test_nms_degenerate_zero_area_boxes_do_not_raise_runtime_warning(self, recwarn) -> None:
        """Two identical zero-area (point) boxes → union=0 for that pair.

        np.where(union > 0, inter / union, 0.0) evaluates inter/union
        unconditionally for every pair, including union==0, producing a
        spurious 0/0 RuntimeWarning before np.where discards it in favour
        of 0.0. The computed result is unaffected either way — this test
        guards the log-noise fix (L1), not a correctness fix.
        """
        raw_output = np.zeros((1, 84, 8400), dtype=np.float32)
        # Two identical zero-width/zero-height boxes, same class, same location.
        raw_output[0, 4, 0] = 0.9
        raw_output[0, :4, 0] = [320.0, 320.0, 0.0, 0.0]
        raw_output[0, 4, 1] = 0.7
        raw_output[0, :4, 1] = [320.0, 320.0, 0.0, 0.0]

        format_coco_prediction(raw_output, image_id=1, conf_threshold=0.5, iou_threshold=0.45)

        runtime_warnings = [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
        assert not runtime_warnings, f"Expected no RuntimeWarning, got: {[str(w.message) for w in runtime_warnings]}"


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

    def test_raises_when_baseline_precision_is_not_fp32(self) -> None:
        """Baseline that is not FP32 must be rejected — delta is always vs FP32."""
        baseline = AccuracyResult(
            map_50_95=0.360, map_50=0.510, precision="fp16", runtime="pytorch_cpu"
        )
        candidate = AccuracyResult(
            map_50_95=0.340, map_50=0.490, precision="int8", runtime="pytorch_cpu"
        )

        with pytest.raises(ValueError, match="fp32"):
            compute_map_delta(baseline, candidate)

    def test_raises_on_cross_runtime_comparison(self) -> None:
        """Passing a PyTorch baseline against an ONNX candidate must be rejected."""
        pytorch_fp32 = AccuracyResult(
            map_50_95=0.372, map_50=0.530, precision="fp32", runtime="pytorch_cpu_fp32"
        )
        onnx_int8 = AccuracyResult(
            map_50_95=0.355, map_50=0.510, precision="int8", runtime="onnx_cpu_int8"
        )

        with pytest.raises(ValueError, match="runtime"):
            compute_map_delta(pytorch_fp32, onnx_int8)

    def test_same_runtime_full_name_passes_guard(self) -> None:
        """Full runtime names with precision suffix must pass the family check."""
        baseline = AccuracyResult(
            map_50_95=0.372, map_50=0.530, precision="fp32", runtime="onnx_cpu_fp32"
        )
        candidate = AccuracyResult(
            map_50_95=0.360, map_50=0.515, precision="int8", runtime="onnx_cpu_int8"
        )

        delta = compute_map_delta(baseline, candidate)

        assert abs(delta - (0.360 - 0.372)) < 1e-9

    def test_family_extraction_handles_precision_embedded_in_name(self) -> None:
        """L1 — str.removesuffix prevents double-replacement when precision appears in family name.

        A runtime named 'fp32_detector_fp32' has '_fp32' both in the middle and at the end.
        str.replace removes ALL occurrences → 'detector_' (wrong).
        str.removesuffix only removes the trailing occurrence → 'fp32_detector' (correct).
        The two runtimes 'fp32_detector_fp32' and 'fp32_detector_int8' share the family
        'fp32_detector', so compute_map_delta must NOT raise on this comparison.
        """
        baseline = AccuracyResult(
            map_50_95=0.372, map_50=0.530, precision="fp32", runtime="fp32_detector_fp32"
        )
        candidate = AccuracyResult(
            map_50_95=0.360, map_50=0.515, precision="int8", runtime="fp32_detector_int8"
        )
        delta = compute_map_delta(baseline, candidate)
        assert abs(delta - (0.360 - 0.372)) < 1e-9


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


# ---------------------------------------------------------------------------
# evaluate_map — full COCO evaluation pipeline
# ---------------------------------------------------------------------------

def _make_mock_loader(n_images: int = 3):
    """Return a mock CocoLoader yielding (tensor, image_id, meta) 3-tuples."""
    loader = MagicMock()
    loader.__len__ = MagicMock(return_value=n_images)
    dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
    # LetterboxMeta for a square image: scale=1, no padding
    meta = LetterboxMeta(scale=1.0, pad_left=0, pad_top=0, orig_h=640, orig_w=640)
    loader.__iter__ = MagicMock(
        return_value=iter([(dummy, i + 1, meta) for i in range(n_images)])
    )
    return loader


class TestEvaluateMap:
    def _patch_coco(self, mock_stats: list[float]):
        """Context-manager helper: patch COCO and COCOeval with known stats."""
        from unittest.mock import patch, MagicMock

        mock_coco_gt = MagicMock()
        mock_coco_dt = MagicMock()
        mock_coco_gt.loadRes.return_value = mock_coco_dt

        mock_eval = MagicMock()
        mock_eval.stats = mock_stats

        coco_patch = patch("src.benchmark.accuracy_evaluator.COCO", return_value=mock_coco_gt)
        eval_patch = patch("src.benchmark.accuracy_evaluator.COCOeval", return_value=mock_eval)
        return coco_patch, eval_patch

    def test_returns_accuracy_result(self, tmp_path) -> None:
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400), dtype=np.float32)
        loader = _make_mock_loader()

        stats = [0.372, 0.530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            result = evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

        assert isinstance(result, AccuracyResult)

    def _detection_output(self) -> np.ndarray:
        """Return model output with one high-confidence detection above threshold."""
        out = np.zeros((1, 84, 8400), dtype=np.float32)
        out[0, 4, 0] = 0.9   # class 0 score = 0.9 (above default conf_threshold=0.5)
        out[0, :4, 0] = [320.0, 240.0, 100.0, 80.0]  # cx, cy, w, h
        return out

    def test_map_50_95_taken_from_coco_stats_index_0(self, tmp_path) -> None:
        """Primary metric mAP@0.5:0.95 must be stats[0], not stats[1]."""
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = self._detection_output()
        loader = _make_mock_loader()

        stats = [0.372, 0.999, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            result = evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

        assert abs(result.map_50_95 - 0.372) < 1e-9
        assert abs(result.map_50 - 0.999) < 1e-9

    def test_map_50_taken_from_coco_stats_index_1(self, tmp_path) -> None:
        """Secondary metric mAP@0.5 must be stats[1], not stats[0]."""
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = self._detection_output()
        loader = _make_mock_loader()

        stats = [0.111, 0.530, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            result = evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

        assert abs(result.map_50 - 0.530) < 1e-9

    def test_runtime_name_in_result(self, tmp_path) -> None:
        runtime = MagicMock()
        runtime.name = "pytorch_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400), dtype=np.float32)
        loader = _make_mock_loader()

        stats = [0.372, 0.530] + [0.0] * 10
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            result = evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

        assert result.runtime == "pytorch_cpu_fp32"

    def test_precision_extracted_from_runtime_name(self, tmp_path) -> None:
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400), dtype=np.float32)
        loader = _make_mock_loader()

        stats = [0.372, 0.530] + [0.0] * 10
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            result = evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

        assert result.precision == "fp32"

    def test_infer_called_once_per_image(self, tmp_path) -> None:
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400), dtype=np.float32)
        loader = _make_mock_loader(n_images=5)

        stats = [0.372, 0.530] + [0.0] * 10
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

        assert runtime.infer.call_count == 5

    def test_empty_predictions_returns_zero_map(self, tmp_path) -> None:
        """All detections below threshold → no crash, mAP returns 0.0."""
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400), dtype=np.float32)
        loader = _make_mock_loader(n_images=2)

        stats = [0.372, 0.530] + [0.0] * 10
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            result = evaluate_map(
                runtime, loader, str(tmp_path / "ann.json"), conf_threshold=0.99
            )

        assert result.map_50_95 == 0.0
        assert result.map_50 == 0.0

    def test_map_delta_defaults_to_none(self, tmp_path) -> None:
        """evaluate_map sets map_delta_vs_fp32=None; caller computes it later."""
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400), dtype=np.float32)
        loader = _make_mock_loader()

        stats = [0.372, 0.530] + [0.0] * 10
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            result = evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

        assert result.map_delta_vs_fp32 is None

    def test_raises_import_error_without_pycocotools(self, tmp_path) -> None:
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        loader = _make_mock_loader()

        with patch("src.benchmark.accuracy_evaluator.COCO", None), \
             patch("src.benchmark.accuracy_evaluator.COCOeval", None):
            with pytest.raises(ImportError, match="pycocotools"):
                evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

    def test_letterbox_meta_passed_to_format_coco_prediction(self, tmp_path) -> None:
        """evaluate_map must pass letterbox_meta from loader to format_coco_prediction."""
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = self._detection_output()
        loader = _make_mock_loader(n_images=1)

        stats = [0.372, 0.530] + [0.0] * 10
        cp, ep = self._patch_coco(stats)
        with cp, ep, patch(
            "src.benchmark.accuracy_evaluator.format_coco_prediction",
            wraps=__import__(
                "src.benchmark.accuracy_evaluator", fromlist=["format_coco_prediction"]
            ).format_coco_prediction,
        ) as mock_fmt:
            evaluate_map(runtime, loader, str(tmp_path / "ann.json"))

        # Each call must include the letterbox_meta keyword argument
        for call in mock_fmt.call_args_list:
            assert "letterbox_meta" in call.kwargs or len(call.args) >= 4

    def test_raises_on_partial_image_set(self, tmp_path) -> None:
        """M2 — if loader yields fewer images than its __len__, raise RuntimeError.

        A partial evaluation submits predictions for fewer images than the full
        5000-image GT set, causing COCOeval to compute mAP over a fraction of GT
        annotations and systematically suppressing recall — a silent wrong result.
        """
        runtime = MagicMock()
        runtime.name = "onnx_cpu_fp32"
        runtime.infer.return_value = np.zeros((1, 84, 8400), dtype=np.float32)

        # Loader claims 5 images but only yields 3 (simulates partial dataset)
        partial_loader = MagicMock()
        partial_loader.__len__ = MagicMock(return_value=5)
        meta = LetterboxMeta(scale=1.0, pad_left=0, pad_top=0, orig_h=640, orig_w=640)
        partial_loader.__iter__ = MagicMock(
            return_value=iter([
                (np.zeros((1, 3, 640, 640), dtype=np.float32), i + 1, meta)
                for i in range(3)
            ])
        )

        stats = [0.372, 0.530] + [0.0] * 10
        cp, ep = self._patch_coco(stats)
        with cp, ep:
            with pytest.raises(RuntimeError, match="Partial"):
                evaluate_map(runtime, partial_loader, str(tmp_path / "ann.json"))

    def test_eval_defaults_use_low_conf_and_high_iou(self, tmp_path) -> None:
        """Default conf_threshold=0.001 and iou_threshold=0.7 for mAP evaluation.

        These defaults expose the full PR curve to COCOeval and match the
        ultralytics reference validator, unlike the deployment defaults (0.5/0.45).
        """
        import inspect
        sig = inspect.signature(evaluate_map)
        assert sig.parameters["conf_threshold"].default == 0.001, (
            "evaluate_map conf_threshold default must be 0.001 for mAP evaluation"
        )
        assert sig.parameters["iou_threshold"].default == 0.7, (
            "evaluate_map iou_threshold default must be 0.7 to match ultralytics reference"
        )
