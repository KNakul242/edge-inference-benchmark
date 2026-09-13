"""Unit tests for the webcam demo's pure detection-decoding logic.

scripts/* is excluded from the coverage gate (pyproject.toml), and the
capture/display loop in webcam_demo.main() is not unit-testable without a
real camera. But _nms and decode_detections are pure numeric functions with
real failure modes (a missed suppression, a mis-scaled box, a degenerate
zero-area box) that would silently produce wrong on-screen detections, so
they're tested directly here -- same rationale as test_run_benchmark.py's
CLI-filter tests.
"""

import numpy as np

from scripts.webcam_demo import (
    COCO_NAMES,
    _fit_top_anchored,
    _label_anchor,
    _nms,
    _summarize_stage_timings,
    decode_detections,
)
from src.data.coco_loader import LetterboxMeta


class TestNms:
    """_nms is class-aware (D4, 2026-09-13): only boxes sharing the same
    class_id can suppress each other. Diverges deliberately from
    accuracy_evaluator._apply_nms's agnostic design -- see _nms's docstring.
    """

    def test_empty_input_returns_empty(self):
        keep = _nms(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64), 0.45)
        assert keep.tolist() == []

    def test_single_box_survives(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        class_ids = np.array([0], dtype=np.int64)
        keep = _nms(boxes, scores, class_ids, 0.45)
        assert keep.tolist() == [0]

    def test_non_overlapping_boxes_both_survive(self):
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 0], dtype=np.int64)
        keep = _nms(boxes, scores, class_ids, 0.45)
        assert set(keep.tolist()) == {0, 1}

    def test_heavily_overlapping_same_class_suppresses_lower_score(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 0], dtype=np.int64)  # same class -- suppression applies
        keep = _nms(boxes, scores, class_ids, 0.45)
        assert keep.tolist() == [0]

    def test_heavily_overlapping_different_class_both_survive(self):
        # D4: the actual behavior change. Same geometry as the same-class
        # suppression test above, but different classes -- e.g. a phone
        # detection overlapping a person detection. Agnostic NMS would drop
        # the lower-scoring box here; class-aware NMS must not.
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int64)  # different classes
        keep = _nms(boxes, scores, class_ids, 0.45)
        assert set(keep.tolist()) == {0, 1}

    def test_keeps_higher_score_regardless_of_input_order(self):
        boxes = np.array([[0, 0, 10, 9], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.6, 0.95], dtype=np.float32)
        class_ids = np.array([0, 0], dtype=np.int64)
        keep = _nms(boxes, scores, class_ids, 0.45)
        assert keep.tolist() == [1]

    def test_degenerate_zero_area_box_does_not_raise_or_warn(self):
        # A zero-width box makes union == 0 for any pairing against it --
        # regression guard for the np.divide(..., where=union > 0) fix
        # (ported from accuracy_evaluator._apply_nms's L1 fix).
        boxes = np.array([[5, 5, 5, 20], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.5], dtype=np.float32)
        class_ids = np.array([0, 0], dtype=np.int64)
        with np.errstate(divide="raise", invalid="raise"):
            keep = _nms(boxes, scores, class_ids, 0.45)
        assert set(keep.tolist()) == {0, 1}


class TestDecodeDetections:
    def _make_raw_output(self, cx, cy, w, h, class_idx, score, n_classes=80, n_anchors=1):
        """Build a (1, 4+n_classes, n_anchors) raw model output with one hot anchor."""
        out = np.zeros((1, 4 + n_classes, n_anchors), dtype=np.float32)
        out[0, 0, 0] = cx
        out[0, 1, 0] = cy
        out[0, 2, 0] = w
        out[0, 3, 0] = h
        out[0, 4 + class_idx, 0] = score
        return out

    def _identity_meta(self):
        return LetterboxMeta(scale=1.0, pad_left=0, pad_top=0, orig_h=640, orig_w=640)

    def test_below_confidence_threshold_returns_empty(self):
        raw = self._make_raw_output(cx=320, cy=320, w=40, h=40, class_idx=0, score=0.2)
        result = decode_detections(raw, self._identity_meta(), conf=0.5)
        assert result == []

    def test_above_threshold_returns_correct_class_name_and_score(self):
        person_idx = COCO_NAMES.index("person")
        raw = self._make_raw_output(cx=320, cy=320, w=40, h=40, class_idx=person_idx, score=0.9)
        result = decode_detections(raw, self._identity_meta(), conf=0.5)
        assert len(result) == 1
        x1, y1, x2, y2, score, name = result[0]
        assert name == "person"
        assert score == np.float32(0.9)
        assert (x1, y1, x2, y2) == (300, 300, 340, 340)

    def test_letterbox_padding_and_scale_are_reversed(self):
        # Model-space box at (100,100)-(200,200) in a 640x640 letterboxed frame
        # that came from a 320x320 original image scaled by 0.5 with 160px
        # top/bottom padding removed (pure horizontal letterbox for simplicity
        # here: pad_top=160, pad_left=0, scale=0.5).
        cx, cy, w, h = 150, 180, 100, 40  # -> box (100,160)-(200,200) in model space
        idx = 5
        raw = self._make_raw_output(cx=cx, cy=cy, w=w, h=h, class_idx=idx, score=0.8)
        meta = LetterboxMeta(scale=0.5, pad_left=0, pad_top=160, orig_h=100, orig_w=320)
        result = decode_detections(raw, meta, conf=0.5)
        assert len(result) == 1
        x1, y1, x2, y2, score, name = result[0]
        # (100-0)/0.5=200, (160-160)/0.5=0, (200-0)/0.5=400 clamped to orig_w=320, (200-160)/0.5=80
        assert (x1, y1, x2, y2) == (200, 0, 320, 80)
        assert name == COCO_NAMES[idx]

    def test_no_anchor_above_threshold_returns_empty_without_raising(self):
        raw = np.zeros((1, 84, 100), dtype=np.float32)
        result = decode_detections(raw, self._identity_meta(), conf=0.5)
        assert result == []


class TestCocoNames:
    def test_has_exactly_80_unique_entries(self):
        assert len(COCO_NAMES) == 80
        assert len(set(COCO_NAMES)) == 80


class TestLabelAnchor:
    """_label_anchor(x1, y1, label_w, label_h, frame_w, frame_h, bottom_margin).

    frame_h/bottom_margin describe the single HUD bar reserved at the
    bottom of the frame (see _BOTTOM_BAR_HEIGHT) -- draw_frame paints that
    bar *after* detections, so any label anchored inside it gets silently
    painted over. The anchor must never place a label there.
    """

    def test_box_well_inside_frame_anchors_at_box_left_edge(self):
        lx, ly = _label_anchor(x1=100, y1=100, label_w=60, label_h=14, frame_w=640, frame_h=480, bottom_margin=56)
        assert lx == 100
        assert ly == 98  # y1 - 2

    def test_box_near_right_edge_clamps_label_within_frame(self):
        lx, ly = _label_anchor(x1=620, y1=100, label_w=60, label_h=14, frame_w=640, frame_h=480, bottom_margin=56)
        assert lx == 640 - 60 - 4
        assert lx + 60 + 4 <= 640

    def test_box_at_left_edge_never_produces_negative_anchor(self):
        lx, _ = _label_anchor(x1=0, y1=100, label_w=60, label_h=14, frame_w=640, frame_h=480, bottom_margin=56)
        assert lx == 0

    def test_narrow_frame_narrower_than_label_still_clamps_non_negative(self):
        # Pathological but must not crash or go negative: label wider than the frame.
        lx, _ = _label_anchor(x1=10, y1=100, label_w=700, label_h=14, frame_w=640, frame_h=480, bottom_margin=56)
        assert lx == 0

    def test_box_near_top_edge_keeps_label_below_frame_top(self):
        lx, ly = _label_anchor(x1=100, y1=2, label_w=60, label_h=14, frame_w=640, frame_h=480, bottom_margin=56)
        assert ly == 18  # label_h + 4
        assert ly - 14 - 3 >= 0  # label background top edge stays on-screen

    def test_box_near_bottom_of_frame_keeps_label_above_reserved_hud_bar(self):
        # A box whose top edge (y1) sits inside the reserved bottom HUD strip --
        # e.g. a detection near the very bottom of the frame. The default
        # "place the label 2px above y1" rule would anchor it inside the bar,
        # where draw_frame's HUD rectangle (painted after detections) would
        # silently cover it. Must be pulled up above the reserved zone instead.
        frame_h, bottom_margin = 480, 56
        lx, ly = _label_anchor(x1=100, y1=470, label_w=60, label_h=14, frame_w=640,
                                frame_h=frame_h, bottom_margin=bottom_margin)
        assert ly + 1 <= frame_h - bottom_margin

    def test_box_at_frame_bottom_edge_label_fully_above_reserved_zone(self):
        frame_h, bottom_margin = 480, 56
        lx, ly = _label_anchor(x1=100, y1=478, label_w=60, label_h=14, frame_w=640,
                                frame_h=frame_h, bottom_margin=bottom_margin)
        # entire label background (ly-label_h-3 .. ly+1) must sit above the bar
        assert ly + 1 <= frame_h - bottom_margin
        assert ly - 14 - 3 >= 0


class TestFitTopAnchored:
    """_fit_top_anchored(frame_w, frame_h, window_w, window_h) -> (resized_w, resized_h, x_offset).

    When the OS window (e.g. native macOS fullscreen) is a different size/
    aspect ratio than the captured frame, cv2's Cocoa backend does NOT
    stretch the image to fill it -- it leaves empty space and, observed
    directly (screenshot), anchors the image toward the bottom of the
    window, leaving blank padding at the TOP instead. main() uses this to
    build its own black canvas sized to the window and paste the resized
    frame in flush at the top instead, so any leftover padding (now a
    deliberate black bar, not the backend's gray default) ends up at the
    bottom, matching Nakul's ask directly ("i meant to shift the grey
    space", not the HUD text).
    """

    def test_window_matches_frame_exactly_no_scaling_no_offset(self):
        w, h, x_off = _fit_top_anchored(frame_w=640, frame_h=480, window_w=640, window_h=480)
        assert (w, h, x_off) == (640, 480, 0)

    def test_window_taller_than_needed_scales_by_width_no_horizontal_offset(self):
        # 640x480 (4:3) frame in a 640x900 window -- width-constrained,
        # leftover vertical space goes unused here (caller top-anchors it).
        w, h, x_off = _fit_top_anchored(frame_w=640, frame_h=480, window_w=640, window_h=900)
        assert (w, h) == (640, 480)
        assert x_off == 0

    def test_window_wider_than_needed_centers_horizontally(self):
        w, h, x_off = _fit_top_anchored(frame_w=640, frame_h=480, window_w=1000, window_h=480)
        assert (w, h) == (640, 480)
        assert x_off == (1000 - 640) // 2

    def test_window_smaller_than_frame_downscales_preserving_aspect_ratio(self):
        w, h, x_off = _fit_top_anchored(frame_w=1280, frame_h=960, window_w=640, window_h=480)
        assert (w, h) == (640, 480)
        assert x_off == 0

    def test_never_returns_zero_or_negative_dimensions(self):
        w, h, x_off = _fit_top_anchored(frame_w=640, frame_h=480, window_w=1, window_h=1)
        assert w >= 1 and h >= 1 and x_off >= 0


class TestSummarizeStageTimings:
    """_summarize_stage_timings(samples) -> per-stage mean/stdev/min/max/p95/n.

    Used by main()'s --profile-frames investigation (FPS-gap breakdown) to
    turn raw per-frame stage timings into a reportable summary.
    """

    def test_known_values_computed_correctly(self):
        samples = {"stage_a": [10.0, 20.0, 30.0, 40.0, 50.0]}
        summary = _summarize_stage_timings(samples)
        s = summary["stage_a"]
        assert s["mean_ms"] == 30.0
        assert s["min_ms"] == 10.0
        assert s["max_ms"] == 50.0
        assert s["n"] == 5
        assert s["stdev_ms"] > 0.0

    def test_single_sample_stdev_is_zero_not_a_crash(self):
        summary = _summarize_stage_timings({"stage_a": [15.0]})
        assert summary["stage_a"]["stdev_ms"] == 0.0
        assert summary["stage_a"]["mean_ms"] == 15.0
        assert summary["stage_a"]["n"] == 1

    def test_empty_samples_dict_returns_empty_summary(self):
        assert _summarize_stage_timings({}) == {}

    def test_stage_with_empty_list_is_omitted(self):
        summary = _summarize_stage_timings({"stage_a": [1.0, 2.0], "stage_b": []})
        assert "stage_a" in summary
        assert "stage_b" not in summary

    def test_multiple_stages_summarized_independently(self):
        summary = _summarize_stage_timings({
            "capture_ms": [5.0, 5.0, 5.0],
            "inference_ms": [10.0, 12.0, 14.0],
        })
        assert summary["capture_ms"]["mean_ms"] == 5.0
        assert summary["capture_ms"]["stdev_ms"] == 0.0
        assert summary["inference_ms"]["mean_ms"] == 12.0

    def test_p95_is_included_and_within_range(self):
        samples = {"stage_a": list(range(1, 101))}  # 1..100
        summary = _summarize_stage_timings(samples)
        assert 94.0 <= summary["stage_a"]["p95_ms"] <= 96.0
