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

from scripts.webcam_demo import COCO_NAMES, _label_anchor, _nms, decode_detections
from src.data.coco_loader import LetterboxMeta


class TestNms:
    def test_empty_input_returns_empty(self):
        keep = _nms(np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), 0.45)
        assert keep.tolist() == []

    def test_single_box_survives(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        keep = _nms(boxes, scores, 0.45)
        assert keep.tolist() == [0]

    def test_non_overlapping_boxes_both_survive(self):
        boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, 0.45)
        assert set(keep.tolist()) == {0, 1}

    def test_heavily_overlapping_boxes_suppresses_lower_score(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 9]], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        keep = _nms(boxes, scores, 0.45)
        assert keep.tolist() == [0]

    def test_keeps_higher_score_regardless_of_input_order(self):
        boxes = np.array([[0, 0, 10, 9], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.6, 0.95], dtype=np.float32)
        keep = _nms(boxes, scores, 0.45)
        assert keep.tolist() == [1]

    def test_degenerate_zero_area_box_does_not_raise_or_warn(self):
        # A zero-width box makes union == 0 for any pairing against it --
        # regression guard for the np.divide(..., where=union > 0) fix
        # (ported from accuracy_evaluator._apply_nms's L1 fix).
        boxes = np.array([[5, 5, 5, 20], [0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.5], dtype=np.float32)
        with np.errstate(divide="raise", invalid="raise"):
            keep = _nms(boxes, scores, 0.45)
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
    def test_box_well_inside_frame_anchors_at_box_left_edge(self):
        lx, ly = _label_anchor(x1=100, y1=100, label_w=60, label_h=14, frame_w=640)
        assert lx == 100
        assert ly == 98  # y1 - 2

    def test_box_near_right_edge_clamps_label_within_frame(self):
        lx, ly = _label_anchor(x1=620, y1=100, label_w=60, label_h=14, frame_w=640)
        assert lx == 640 - 60 - 4
        assert lx + 60 + 4 <= 640

    def test_box_at_left_edge_never_produces_negative_anchor(self):
        lx, _ = _label_anchor(x1=0, y1=100, label_w=60, label_h=14, frame_w=640)
        assert lx == 0

    def test_narrow_frame_narrower_than_label_still_clamps_non_negative(self):
        # Pathological but must not crash or go negative: label wider than the frame.
        lx, _ = _label_anchor(x1=10, y1=100, label_w=700, label_h=14, frame_w=640)
        assert lx == 0

    def test_box_near_top_edge_keeps_label_below_frame_top(self):
        lx, ly = _label_anchor(x1=100, y1=2, label_w=60, label_h=14, frame_w=640)
        assert ly == 18  # label_h + 4
        assert ly - 14 - 3 >= 0  # label background top edge stays on-screen
