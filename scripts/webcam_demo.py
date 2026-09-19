"""Live YOLOv8n object detection via ONNX Runtime + CoreML EP (Mac M5).

Illustrates edge inference throughput at 640×640 FP32 — the same
runtime/precision combination benchmarked in the study (ONNX Runtime +
CoreML EP, Apple M5, ~9.3 ms mean / ~101 FPS). The overhead panel shows
per-frame inference latency and rolling FPS so the constraint is
visible throughout.

Ported 2026-09-12 from an earlier MVP built on Fedora (ONNX Runtime CPU
EP, ~72 ms / ~10 FPS) that predated Mac M5 hardware by two months —
see `feature/webcam-demo` history. CoreML EP FP16/INT8 quantization was
never built (see `docs/specs/VISION.md` Decisions Locked), so this
demo runs FP32 — the overlay says so explicitly, not "FP16" per the
original Phase 2 spec draft, which assumed a precision path that
doesn't exist. Same reasoning for omitting any "Neural Engine
Accelerated" annotation the original spec also called for: this
study's own repeated `MLComputeUnits` measurement (Kruskal-Wallis
p=0.57, see `docs/benchmark-run-1-mac-findings.md`) found no
reproducible evidence CoreML EP is engaging the Neural Engine for this
model — asserting it on screen would contradict this project's own
finding.

Usage (from project root):
    python scripts/webcam_demo.py
    python scripts/webcam_demo.py --model models/yolov8n.onnx --camera 0
    python scripts/webcam_demo.py --conf 0.4   # lower threshold, more boxes
    python scripts/webcam_demo.py --scale 2.0  # larger display window
    python scripts/webcam_demo.py --provider CPUExecutionProvider  # fallback

Press Q to quit, F to toggle fullscreen (macOS's own green
traffic-light button also works). Known limitation, accepted rather
than chased further: cv2's fullscreen support has a long-standing,
still-open upstream bug on macOS's Cocoa GUI backend
(opencv/opencv#23118, opencv/opencv-python#804/#769) that can leave
empty space around the frame instead of filling the screen cleanly --
see docs/issue-log/2026-09-13-webcam-demo-fullscreen-abandoned.md for
what was tried. The display still letterboxes as best it can against
whatever size cv2 reports each frame (padding, if any, stays at the
bottom, never covers the HUD) -- it just can't always get a correct
size to work with when fullscreen is involved.
"""

import argparse
import collections
import json
import logging
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow `src.*` imports when the script is run directly from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.coco_loader import letterbox_preprocess  # noqa: E402
from src.runtimes.onnx_runtime import OnnxRuntime      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# YOLOv8n COCO 80-class names in model class-index order.
# Index maps directly to the class dimension of the (84, 8400) output.
COCO_NAMES: list[str] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

_CONF_DEFAULT = 0.5
_IOU_DEFAULT = 0.45
_WARMUP_FRAMES = 5
_FPS_WINDOW = 30  # rolling average window (frames)
_WINDOW_TITLE = "YOLOv8n  -  ONNX Runtime + CoreML EP"
_PROVIDER_DEFAULT = "CoreMLExecutionProvider"
# Single HUD bar reserved at the frame's bottom -- deliberately not split
# top/bottom. A top bar competed with the camera feed's own vertical span
# (worse in fullscreen, where window/feed aspect mismatches push content
# around) and, since it was painted after detections, silently covered any
# label anchored underneath it. One bottom-anchored bar avoids both.
_BOTTOM_BAR_HEIGHT = 56

# Live inference-time chart + stage-timing table -- fixed panel drawn
# directly on the frame (same coordinate space as boxes/bottom HUD), top-right
# corner. Deliberately NOT tied to the display-letterbox canvas (the
# fullscreen investigation's grey-band code): that canvas is recomputed live
# per frame from cv2.getWindowImageRect and only exists when window aspect
# != camera aspect, so anything drawn there would appear in fullscreen and
# vanish in default windowed mode -- wrong for a chart meant to be a
# permanent, always-present, screenshot-able part of the demo.
_CHART_DEQUE_LEN = 150        # ~10s at ~15fps -- long enough to see jitter shape
_LIVE_STATS_WINDOW = 30       # rolling window for the table's mean/% figures
_CHART_Y_MIN_MS = 0.0
_CHART_Y_MAX_MS = 30.0        # fixed, not auto-scaled -- see draw_chart_and_table
_PANEL_MARGIN = 10            # from the frame's own top/right edges
_PANEL_PAD = 6                # inner padding
_PANEL_W = 220
_CHART_H = 40
_TABLE_ROW_H = 12
_TABLE_STAGES = (
    "capture_ms", "preprocess_ms", "inference_ms", "decode_ms",
    "annotate_ms", "canvas_rebuild_ms", "draw_display_ms",
)
_TABLE_LABELS = {
    "capture_ms": "capture", "preprocess_ms": "preproc",
    "inference_ms": "infer", "decode_ms": "decode",
    "annotate_ms": "annotate", "canvas_rebuild_ms": "canvas",
    "draw_display_ms": "display",
}

# Per-class BGR colours, cycled by class index
_PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 128), (128, 0, 255),
    (0, 255, 128), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 0),
]


def _fit_top_anchored(
    frame_w: int, frame_h: int, window_w: int, window_h: int,
) -> tuple[int, int, int]:
    """Compute a letterbox fit for frame_w x frame_h into window_w x window_h.

    cv2's Cocoa backend (macOS) does not stretch imshow's content to fill a
    window/screen bigger than the frame -- observed directly, it leaves
    blank space and anchors the image toward the bottom, so the padding
    lands at the *top* (e.g. under a fullscreen menu bar). main() uses this
    function's result to build its own canvas and paste the resized frame
    in flush at the top instead, so any leftover space is pushed to the
    bottom -- deliberately, not stretched/distorted to fill the window.

    Args:
        frame_w: Captured frame width, pixels.
        frame_h: Captured frame height, pixels.
        window_w: Actual window/display width, pixels.
        window_h: Actual window/display height, pixels.

    Returns:
        (resized_w, resized_h, x_offset): dimensions to resize the frame to
        (aspect ratio preserved) and the horizontal offset to center it at.
        Vertical offset is always 0 -- the caller pastes flush at the top.
    """
    if window_w <= 0 or window_h <= 0:
        return frame_w, frame_h, 0
    scale = min(window_w / frame_w, window_h / frame_h)
    resized_w = max(1, int(frame_w * scale))
    resized_h = max(1, int(frame_h * scale))
    x_offset = max(0, (window_w - resized_w) // 2)
    return resized_w, resized_h, x_offset


def _summarize_stage_timings(
    samples: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Summarize per-frame pipeline-stage timings collected by --profile-frames.

    Investigates the gap between the demo's wall-clock FPS (~15) and its
    inference-only latency (~13 ms, which alone would support ~75 FPS) by
    breaking down where the rest of each frame's time actually goes:
    capture, preprocess, inference, decode, annotate (draw_frame -- boxes,
    labels, HUD bar), the per-frame display-canvas rebuild (isolated
    separately from actual camera/AVFoundation driver latency, since it's
    code this project controls), chart_render (the live inference-time
    chart + stage-timing table's own render cost -- measured, not assumed
    cheap, before any decision on whether it needs threading), and
    draw+display (imshow/waitKey). A small residual (frame mirroring, FPS bookkeeping,
    the perf_counter() calls themselves) is intentionally left untimed as
    negligible -- summed stage totals won't exactly equal total_ms.

    Args:
        samples: stage name -> list of per-frame durations in milliseconds.

    Returns:
        stage name -> {"mean_ms", "stdev_ms", "min_ms", "max_ms", "p95_ms",
        "n"}. A stage with zero samples is omitted entirely. A stage with
        exactly one sample gets stdev_ms=0.0 (statistics.stdev requires at
        least two points, and a single-sample spread is meaningless anyway).
    """
    summary: dict[str, dict[str, float]] = {}
    for stage, values in samples.items():
        if not values:
            continue
        summary[stage] = {
            "mean_ms": statistics.mean(values),
            "stdev_ms": statistics.stdev(values) if len(values) >= 2 else 0.0,
            "min_ms": min(values),
            "max_ms": max(values),
            "p95_ms": float(np.percentile(values, 95)),
            "n": len(values),
        }
    return summary


def _map_latency_series_to_polyline(
    values_ms: list[float], chart_w: int, chart_h: int, y_min: float, y_max: float,
) -> np.ndarray:
    """Map a latency series (oldest first) to pixel points for cv2.polylines.

    Args:
        values_ms: Latency samples in ms, oldest first, newest last.
        chart_w: Chart panel width, pixels. Points span x=0 (oldest) to
            x=chart_w-1 (newest), evenly spaced.
        chart_h: Chart panel height, pixels.
        y_min: Value mapped to the chart's bottom edge (y=chart_h).
        y_max: Value mapped to the chart's top edge (y=0) -- higher latency
            draws higher on screen, the conventional "spike = bad" reading.
            Values outside [y_min, y_max] are clamped to stay on the panel
            rather than drawing off it (an outlier shouldn't break the axes).

    Returns:
        (N, 1, 2) int32 array -- already the shape cv2.polylines expects
        (pass as ``[this_array]``). Empty input returns a (0, 1, 2) array.
    """
    if not values_ms:
        return np.zeros((0, 1, 2), dtype=np.int32)

    n = len(values_ms)
    span = y_max - y_min
    points = np.zeros((n, 1, 2), dtype=np.int32)
    for i, v in enumerate(values_ms):
        clamped = min(max(v, y_min), y_max)
        x = 0 if n == 1 else round(i * (chart_w - 1) / (n - 1))
        y = round(chart_h - (clamped - y_min) / span * chart_h) if span > 0 else chart_h
        points[i, 0, 0] = x
        points[i, 0, 1] = y
    return points


def _compute_stage_percentages(
    stage_means: dict[str, float], total_mean: float,
) -> dict[str, float]:
    """Compute each stage's share of total frame time, for the live table.

    Args:
        stage_means: stage name -> rolling mean latency, ms.
        total_mean: rolling mean of the whole frame's wall-clock time, ms.

    Returns:
        stage name -> percentage of total_mean. All zero if total_mean is
        not positive (startup, before enough samples exist) rather than
        raising a division error.
    """
    if total_mean <= 0:
        return {k: 0.0 for k in stage_means}
    return {k: (v / total_mean) * 100.0 for k, v in stage_means.items()}


def _nms(
    boxes_xyxy: np.ndarray, scores: np.ndarray, class_ids: np.ndarray, iou_threshold: float,
) -> np.ndarray:
    """Class-aware greedy non-maximum suppression.

    Only boxes sharing the same class_id can suppress each other.
    Deliberately diverges from accuracy_evaluator._apply_nms's *agnostic*
    (class-independent) design (D4, 2026-09-13, following up on the
    2026-09-12 P1 review): agnostic NMS is correct there because it matches
    the reference protocol this project's canonical mAP numbers are
    measured against — an aggregate metric over 5,000 images, where
    occasional co-located cross-class suppression is a bounded, disclosed
    cost. This demo's goal is different: show every real object actually
    present in a live desk scene (IMPLEMENTATION_SPEC's "person, laptop,
    phone, cup" — often close together), where agnostic suppression could
    silently drop a real object of a different class. This function has no
    consumers outside this script (confirmed during the D4 review), so it
    changes directly rather than needing an opt-in flag the way
    accuracy_evaluator._apply_nms's would (that one stays agnostic --
    every canonical Phase 1 mAP number depends on it).

    Args:
        boxes_xyxy: (K, 4) float32, [x1, y1, x2, y2].
        scores: (K,) float32.
        class_ids: (K,) int, predicted class index per box. Suppression is
            scoped to boxes sharing the same class_id.
        iou_threshold: Suppress same-class boxes with IoU above this value.

    Returns:
        Surviving box indices, grouped by class (each class's survivors in
        score-descending order; overall order not globally score-sorted).
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int64)

    x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1]
    x2, y2 = boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    keep: list[int] = []
    for cls in np.unique(class_ids):
        idxs = np.where(class_ids == cls)[0]
        order = idxs[scores[idxs].argsort()[::-1]]
        while len(order) > 0:
            i = int(order[0])
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            inter = (
                np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
                * np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
            )
            union = areas[i] + areas[rest] - inter
            # np.divide with where= skips the division for union<=0 pairs (degenerate
            # zero-area boxes) instead of computing it and discarding the result --
            # np.where evaluates both branches unconditionally and would otherwise
            # emit a spurious "invalid value encountered in divide" RuntimeWarning.
            # Ported from accuracy_evaluator._apply_nms's L1 fix (2026-09-09) --
            # this forked copy hadn't inherited it until now (2026-09-12).
            iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
            order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int64)


def decode_detections(
    raw_output: np.ndarray,
    meta,
    conf: float = _CONF_DEFAULT,
    iou: float = _IOU_DEFAULT,
) -> list[tuple[int, int, int, int, float, str]]:
    """Decode raw YOLOv8n output to (x1, y1, x2, y2, score, class_name) in original image coords.

    YOLOv8n output is (1, 84, 8400): 4 box rows (cx, cy, w, h in 640×640
    model space) + 80 class-score rows (post-sigmoid [0,1]).

    Args:
        raw_output: Model output, shape (1, 84, 8400).
        meta: LetterboxMeta from letterbox_preprocess — used to rescale
            box coordinates from 640×640 model space to original image space.
        conf: Minimum confidence score to retain a detection.
        iou: IoU threshold for NMS.

    Returns:
        List of (x1, y1, x2, y2, score, class_name) tuples in pixel
        coordinates of the original (unletterboxed) frame.
    """
    output = raw_output[0]          # (84, 8400)
    boxes_cwh = output[:4, :]       # cx, cy, w, h  in 640×640 space
    class_scores = output[4:, :]    # (80, 8400)  post-sigmoid

    # Validate class scores are post-sigmoid probabilities. Logit outputs
    # (unbounded) indicate the ONNX graph is missing sigmoid activation --
    # would silently threshold in the wrong value space. Same check as
    # accuracy_evaluator.format_coco_prediction (D2); ported here since
    # --model is a user-facing CLI flag, not a hardcoded pinned path. Exits
    # rather than raises -- this is an interactive live demo, not a batch job.
    if class_scores.size > 0 and (
        float(class_scores.min()) < -1e-3 or float(class_scores.max()) > 1 + 1e-3
    ):
        logger.error(
            "Class score output outside valid probability range [0, 1]: "
            "min=%.4f, max=%.4f. The ONNX model at --model should include "
            "sigmoid activation (opset=17, nms=False export).",
            float(class_scores.min()), float(class_scores.max()),
        )
        sys.exit(1)

    max_scores = class_scores.max(axis=0)   # (8400,)
    class_ids = class_scores.argmax(axis=0) # (8400,)

    mask = max_scores >= conf
    if not mask.any():
        return []

    cx, cy = boxes_cwh[0, mask], boxes_cwh[1, mask]
    w, h   = boxes_cwh[2, mask], boxes_cwh[3, mask]
    scores = max_scores[mask]
    ids    = class_ids[mask]

    # cxcywh → xyxy in model (640×640 letterboxed) space
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    keep = _nms(boxes, scores, ids, iou)
    if len(keep) == 0:
        return []

    results: list[tuple[int, int, int, int, float, str]] = []
    for idx in keep:
        bx1, by1, bx2, by2 = boxes[idx]
        # Reverse letterbox: remove padding, undo scale. Compute the
        # unclamped high corner first and clamp the low corner independently
        # -- then derive the high corner from the low corner plus a
        # non-negative extent, rather than clamping both corners
        # independently. An anchor whose box lies inside the letterbox
        # padding band (routine for a 16:9 webcam frame into a square input)
        # would otherwise produce x2 < x1 / y2 < y1 (D1). Same pattern as
        # accuracy_evaluator.format_coco_prediction's width/height clamp.
        x1 = (bx1 - meta.pad_left) / meta.scale
        y1 = (by1 - meta.pad_top) / meta.scale
        x2 = (bx2 - meta.pad_left) / meta.scale
        y2 = (by2 - meta.pad_top) / meta.scale
        rx1 = max(0.0, x1)
        ry1 = max(0.0, y1)
        rx2 = rx1 + max(0.0, min(x2, float(meta.orig_w)) - rx1)
        ry2 = ry1 + max(0.0, min(y2, float(meta.orig_h)) - ry1)
        results.append((
            int(rx1), int(ry1), int(rx2), int(ry2),
            float(scores[idx]),
            COCO_NAMES[int(ids[idx])],
        ))
    return results


def _label_anchor(
    x1: int, y1: int, label_w: int, label_h: int,
    frame_w: int, frame_h: int, bottom_margin: int,
) -> tuple[int, int]:
    """Compute the top-left anchor for a detection's label, clamped to the frame.

    Without clamping, a box near the right edge draws its label background
    and text partly off-screen (x1 + label width > frame width). The box
    itself never needs this (decode_detections already clamps box
    coordinates to the frame), but the label extends further right of x1
    than the box does, so it needs its own clamp.

    ``bottom_margin`` is the height of the HUD bar reserved at the bottom of
    the frame (see ``_BOTTOM_BAR_HEIGHT``). ``draw_frame`` paints that bar
    *after* detections, so a label anchored inside it would be silently
    painted over -- a box near the bottom edge must have its label pulled up
    above the bar, not just above the frame's literal bottom pixel.

    Args:
        x1: Detection box's left edge, pixels (already clamped to the frame).
        y1: Detection box's top edge, pixels.
        label_w: Text label width, from cv2.getTextSize.
        label_h: Text label height, from cv2.getTextSize.
        frame_w: Frame width, pixels.
        frame_h: Frame height, pixels.
        bottom_margin: Height of the reserved HUD strip at the frame's bottom.

    Returns:
        (lx, ly): lx is clamped so the label's right edge (lx + label_w + 4)
        never exceeds frame_w and never goes negative. ly keeps the label
        just above the box, clamped so its background never renders above
        the frame's top or inside the bottom HUD strip.
    """
    lx = max(0, min(x1, frame_w - label_w - 4))
    ly = max(y1 - 2, label_h + 4)
    ly = min(ly, frame_h - bottom_margin - 1)
    return lx, ly


def draw_frame(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int, float, str]],
    latency_ms: float,
    fps: float,
) -> None:
    """Draw bounding boxes and HUD overlay onto frame in-place.

    Args:
        frame: BGR frame from VideoCapture (modified in-place).
        detections: Output of decode_detections.
        latency_ms: Inference-only latency for this frame.
        fps: Rolling FPS (wall-clock, includes capture + pre/postprocessing).
    """
    fh, fw = frame.shape[:2]

    # --- Bounding boxes ---
    for (x1, y1, x2, y2, score, name) in detections:
        colour = _PALETTE[COCO_NAMES.index(name) % len(_PALETTE)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{name}  {score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx, ly = _label_anchor(x1, y1, tw, th, fw, fh, _BOTTOM_BAR_HEIGHT)
        cv2.rectangle(frame, (lx, ly - th - 3), (lx + tw + 4, ly + 1), colour, -1)
        cv2.putText(frame, label, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    # --- HUD (single bottom bar; see _BOTTOM_BAR_HEIGHT) ---
    # Precision is stated as FP32 because that's what actually runs --
    # CoreML EP FP16 was never built (see VISION.md Decisions Locked).
    # Deliberately no "Neural Engine Accelerated" annotation here: this
    # study's own repeated MLComputeUnits measurement (Kruskal-Wallis
    # p=0.57) found no reproducible evidence of Neural Engine engagement
    # for this model -- asserting it on screen would be a claim this
    # project's own data doesn't support. See module docstring.
    bar_top = fh - _BOTTOM_BAR_HEIGHT
    cv2.rectangle(frame, (0, bar_top), (fw, fh), (15, 15, 15), -1)
    cv2.putText(frame, "ONNX Runtime + CoreML EP  |  FP32",
                (10, bar_top + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame,
                f"Inference: {latency_ms:6.1f} ms   FPS: {fps:5.1f}   Apple M5  --  F: fullscreen  Q: quit",
                (10, bar_top + 43), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)


def draw_chart_and_table(
    frame: np.ndarray,
    latency_history: collections.deque[float],
    live_stage_times: dict[str, collections.deque[float]],
) -> None:
    """Draw the live inference-time chart + stage-timing table, top-right.

    Fixed panel drawn directly on ``frame`` (see the module-level comment by
    ``_PANEL_W`` for why this isn't tied to the display-letterbox canvas) --
    always present, screenshot-able at any window size or fullscreen state.
    Semi-transparent background so the camera feed underneath stays visible.

    Args:
        frame: BGR frame (modified in-place).
        latency_history: rolling per-frame inference_ms values, oldest first
            (chart content).
        live_stage_times: stage name -> rolling deque of per-frame ms values
            for that stage, same categories as --profile-frames, plus
            "total_ms" (table content -- rolling mean and % of frame time).
    """
    fh, fw = frame.shape[:2]
    plot_w = _PANEL_W - 2 * _PANEL_PAD

    # Layout via a running y-cursor (panel_y) rather than separate
    # width/height formulas that could drift out of sync with what's
    # actually drawn below.
    panel_x0 = fw - _PANEL_W - _PANEL_MARGIN
    panel_y0 = _PANEL_MARGIN
    x = panel_x0 + _PANEL_PAD
    y = panel_y0 + _PANEL_PAD

    values = list(latency_history)
    stats_line_h = 12
    y += stats_line_h
    chart_top = y
    y += _CHART_H
    y += 4  # gap before table
    table_header_h = 12
    y += table_header_h
    n_table_rows = -(-len(_TABLE_STAGES) // 2)  # ceil for a 2-column layout
    y += n_table_rows * _TABLE_ROW_H
    panel_h = (y - panel_y0) + _PANEL_PAD
    panel_x1 = fw - _PANEL_MARGIN
    panel_y1 = panel_y0 + panel_h

    # Semi-transparent dark background so the panel stays legible without
    # fully hiding whatever's in the camera feed behind it.
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x0, panel_y0), (panel_x1, panel_y1), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, dst=frame)
    cv2.rectangle(frame, (panel_x0, panel_y0), (panel_x1, panel_y1), (90, 90, 90), 1)

    # --- Chart ---
    if values:
        cv2.putText(
            frame,
            f"infer ms  mean={statistics.mean(values):4.1f} min={min(values):4.1f} max={max(values):4.1f}",
            (x, panel_y0 + _PANEL_PAD + stats_line_h - 3),
            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200, 200, 200), 1, cv2.LINE_AA,
        )
    # Fixed gridline at 10ms -- a stable visual reference point across
    # frames, matching the fixed (not auto-scaled) y-axis range.
    grid_y = chart_top + round(_CHART_H - (10.0 - _CHART_Y_MIN_MS) / (_CHART_Y_MAX_MS - _CHART_Y_MIN_MS) * _CHART_H)
    cv2.line(frame, (x, grid_y), (x + plot_w, grid_y), (60, 60, 60), 1)
    pts = _map_latency_series_to_polyline(values, plot_w, _CHART_H, _CHART_Y_MIN_MS, _CHART_Y_MAX_MS)
    if len(pts) >= 2:
        pts_shifted = pts + np.array([[x, chart_top]], dtype=np.int32)
        cv2.polylines(frame, [pts_shifted], isClosed=False, color=(0, 220, 255), thickness=1, lineType=cv2.LINE_AA)

    # --- Stage-timing table (2 columns, same categories as --profile-frames) ---
    table_y0 = chart_top + _CHART_H + 4
    header_y = table_y0 + table_header_h
    cv2.putText(frame, "stage (ms/%)", (x, header_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 170, 170), 1, cv2.LINE_AA)

    stage_means = {
        stage: (statistics.mean(live_stage_times[stage]) if live_stage_times.get(stage) else 0.0)
        for stage in _TABLE_STAGES
    }
    total_mean = statistics.mean(live_stage_times["total_ms"]) if live_stage_times.get("total_ms") else 0.0
    pct = _compute_stage_percentages(stage_means, total_mean)

    col_w = plot_w // 2
    rows_start = header_y + _TABLE_ROW_H  # first row's baseline, one row below the header's
    for i, stage in enumerate(_TABLE_STAGES):
        col, row = divmod(i, n_table_rows)
        tx = x + col * col_w
        ty = rows_start + row * _TABLE_ROW_H
        label = _TABLE_LABELS[stage]
        text = f"{label:8s}{stage_means[stage]:4.1f} {pct[stage]:3.0f}%"
        cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (210, 210, 210), 1, cv2.LINE_AA)


def main() -> None:
    """Entry point — parse args, load model, run capture loop."""
    parser = argparse.ArgumentParser(
        description="YOLOv8n webcam demo — ONNX Runtime + CoreML EP (Mac M5)"
    )
    parser.add_argument(
        "--model", default="models/yolov8n.onnx",
        help="Path to yolov8n.onnx (default: models/yolov8n.onnx)",
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--conf", type=float, default=_CONF_DEFAULT,
        help=f"Detection confidence threshold (default: {_CONF_DEFAULT})",
    )
    parser.add_argument(
        "--scale", type=float, default=1.5,
        help="Display scale factor applied before imshow (default: 1.5)",
    )
    parser.add_argument(
        "--provider", default=_PROVIDER_DEFAULT,
        help=f"ONNX Runtime execution provider (default: {_PROVIDER_DEFAULT}). "
             "OnnxRuntime automatically falls back to CPUExecutionProvider if the "
             "requested provider isn't available.",
    )
    parser.add_argument(
        "--profile-frames", type=int, default=0,
        help="If > 0, time each pipeline stage (capture, preprocess, "
             "inference, decode, annotate, the per-frame display-canvas "
             "rebuild, the live chart+table's own render cost, draw+display) "
             "per frame, stop automatically after this many frames, log a "
             "summary, and save raw per-frame samples to --profile-out. "
             "0 (default): off, normal demo behaviour.",
    )
    parser.add_argument(
        "--profile-out", default="docs/webcam-demo-profile.json",
        help="Where to save raw per-frame profiling samples when "
             "--profile-frames > 0 (default: docs/webcam-demo-profile.json).",
    )
    args = parser.parse_args()

    # --- Load model ---
    runtime = OnnxRuntime(execution_provider=args.provider, precision="fp32")
    runtime.load(args.model)

    # --- Open camera ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error(
            "Cannot open camera %d. Check --camera index, and on macOS check "
            "System Settings -> Privacy & Security -> Camera has granted access "
            "to the terminal/app this is running from (first-run camera access "
            "on macOS requires an explicit grant, unlike Linux).",
            args.camera,
        )
        sys.exit(1)

    # --- Create display window up-front with an explicit size ---
    # WINDOW_NORMAL (not the AUTOSIZE default): pins the viewport to the
    # intended display size rather than relying on the backend's default
    # autosize behaviour, which is inconsistent across platforms/backends.
    win_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
    win_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)
    cv2.namedWindow(_WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(_WINDOW_TITLE, win_w, win_h)

    # --- Warmup with real frames so JIT, memory allocation, and EP init are amortised ---
    logger.info("Warming up with %d real frames...", _WARMUP_FRAMES)
    warmed = 0
    while warmed < _WARMUP_FRAMES:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Could not read frame during warmup — retrying")
            continue
        frame = cv2.flip(frame, 1)  # mirror -- see main loop for why
        tensor, _ = letterbox_preprocess(frame)
        runtime.infer(tensor)
        warmed += 1
    logger.info("Warmup complete. Starting live loop — press F for fullscreen, Q to quit.")

    fps_times: collections.deque[float] = collections.deque(maxlen=_FPS_WINDOW)
    t_prev = time.perf_counter()
    is_fullscreen = False

    # Live inference-time chart + stage-timing table state -- always
    # populated (unlike profile_samples below, which only exists for the
    # one-shot --profile-frames CLI investigation). This is the permanent,
    # always-on-screen panel.
    latency_chart: collections.deque[float] = collections.deque(maxlen=_CHART_DEQUE_LEN)
    live_stage_times: dict[str, collections.deque[float]] = {
        stage: collections.deque(maxlen=_LIVE_STATS_WINDOW) for stage in _TABLE_STAGES
    }
    live_stage_times["total_ms"] = collections.deque(maxlen=_LIVE_STATS_WINDOW)

    profiling = args.profile_frames > 0
    profile_samples: dict[str, list[float]] = {
        "capture_ms": [], "preprocess_ms": [], "inference_ms": [],
        "decode_ms": [], "annotate_ms": [], "canvas_rebuild_ms": [],
        "draw_display_ms": [], "chart_render_ms": [], "total_ms": [],
    }
    frames_profiled = 0
    if profiling:
        logger.info("Profiling enabled: stopping automatically after %d frames.", args.profile_frames)

    while True:
        t_loop_start = time.perf_counter()

        t_cap0 = time.perf_counter()
        ret, frame = cap.read()
        t_cap1 = time.perf_counter()
        if not ret:
            logger.warning("Frame capture failed — camera may have disconnected.")
            break

        # Mirror horizontally so on-screen movement matches the viewer's own
        # left/right, the natural webcam convention (most consumer camera
        # apps do this; a raw cv2.VideoCapture feed does not by default).
        # Applied before preprocessing/inference so decode_detections' box
        # coordinates land correctly on the frame actually being displayed --
        # flipping only the display frame afterward would misalign boxes.
        frame = cv2.flip(frame, 1)

        # --- Preprocess ---
        t_pre0 = time.perf_counter()
        tensor, meta = letterbox_preprocess(frame)
        t_pre1 = time.perf_counter()

        # --- Inference (timed) ---
        t0 = time.perf_counter()
        raw = runtime.infer(tensor)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # --- Decode ---
        t_dec0 = time.perf_counter()
        detections = decode_detections(raw, meta, conf=args.conf)
        t_dec1 = time.perf_counter()

        # --- Rolling FPS (wall-clock includes capture, pre/postprocessing, draw) ---
        t_now = time.perf_counter()
        fps_times.append(t_now - t_prev)
        t_prev = t_now
        fps = len(fps_times) / sum(fps_times) if fps_times else 0.0

        # --- Draw and display ---
        t_annotate0 = time.perf_counter()
        draw_frame(frame, detections, latency_ms, fps)
        t_annotate1 = time.perf_counter()

        # Live inference-time chart + stage-timing table. Always updated
        # (not gated behind --profile-frames) -- this is the permanent panel,
        # not the CLI investigation tool. Fed from the same timer variables
        # already computed above/below, not a second measurement pass.
        latency_chart.append(latency_ms)
        live_stage_times["capture_ms"].append((t_cap1 - t_cap0) * 1000.0)
        live_stage_times["preprocess_ms"].append((t_pre1 - t_pre0) * 1000.0)
        live_stage_times["inference_ms"].append(latency_ms)
        live_stage_times["decode_ms"].append((t_dec1 - t_dec0) * 1000.0)
        live_stage_times["annotate_ms"].append((t_annotate1 - t_annotate0) * 1000.0)

        t_chart0 = time.perf_counter()
        draw_chart_and_table(frame, latency_chart, live_stage_times)
        t_chart1 = time.perf_counter()

        # Query the window's *actual* current size every frame rather than
        # trusting the --scale-derived size fixed at startup: the OS window
        # can be resized or (macOS) put into native fullscreen afterward,
        # and cv2's Cocoa backend does not stretch imshow's content to match
        # -- it leaves blank space and anchors the frame toward the bottom
        # of the window, which visually reads as the HUD/feed being pushed
        # down with empty grey padding above it. Building our own canvas at
        # the window's real size and pasting the (aspect-preserved, not
        # stretched) frame in flush at the top keeps any leftover padding
        # at the bottom instead. See _fit_top_anchored.
        t_canvas0 = time.perf_counter()
        try:
            _, _, win_w, win_h = cv2.getWindowImageRect(_WINDOW_TITLE)
        except cv2.error:
            win_w, win_h = 0, 0

        if win_w > 0 and win_h > 0:
            rw, rh, x_off = _fit_top_anchored(frame.shape[1], frame.shape[0], win_w, win_h)
            resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
            canvas[0:rh, x_off:x_off + rw] = resized
            display = canvas
        elif args.scale != 1.0:
            dh = int(frame.shape[0] * args.scale)
            dw = int(frame.shape[1] * args.scale)
            display = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            display = frame
        t_canvas1 = time.perf_counter()

        cv2.imshow(_WINDOW_TITLE, display)
        key = cv2.waitKey(1) & 0xFF
        t_draw1 = time.perf_counter()

        live_stage_times["canvas_rebuild_ms"].append((t_canvas1 - t_canvas0) * 1000.0)
        live_stage_times["draw_display_ms"].append((t_draw1 - t_canvas1) * 1000.0)
        live_stage_times["total_ms"].append((t_draw1 - t_loop_start) * 1000.0)

        if profiling:
            profile_samples["capture_ms"].append((t_cap1 - t_cap0) * 1000.0)
            profile_samples["preprocess_ms"].append((t_pre1 - t_pre0) * 1000.0)
            profile_samples["inference_ms"].append(latency_ms)
            profile_samples["decode_ms"].append((t_dec1 - t_dec0) * 1000.0)
            profile_samples["annotate_ms"].append((t_annotate1 - t_annotate0) * 1000.0)
            profile_samples["canvas_rebuild_ms"].append((t_canvas1 - t_canvas0) * 1000.0)
            profile_samples["draw_display_ms"].append((t_draw1 - t_canvas1) * 1000.0)
            profile_samples["chart_render_ms"].append((t_chart1 - t_chart0) * 1000.0)
            profile_samples["total_ms"].append((t_draw1 - t_loop_start) * 1000.0)
            frames_profiled += 1
            if frames_profiled >= args.profile_frames:
                logger.info("Profiling complete: %d frames captured.", frames_profiled)
                break

        if key == ord("q"):
            logger.info("Q pressed — stopping.")
            break
        if key == ord("f"):
            # Simplest possible toggle -- same native-fullscreen transition
            # as macOS's own green traffic-light button, so it carries the
            # same known limitation (see module docstring / issue log): can
            # leave empty space around the frame instead of filling the
            # screen, due to a still-open upstream OpenCV/Cocoa bug this
            # project can't fix. Accepted rather than engineered around
            # further -- a one-key toggle beats manual drag-resizing even
            # with that imperfection.
            is_fullscreen = not is_fullscreen
            cv2.setWindowProperty(
                _WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL,
            )

    if profiling and frames_profiled > 0:
        summary = _summarize_stage_timings(profile_samples)
        logger.info("--- Stage timing summary (%d frames) ---", frames_profiled)
        for stage in ("capture_ms", "preprocess_ms", "inference_ms", "decode_ms",
                      "annotate_ms", "canvas_rebuild_ms", "draw_display_ms",
                      "chart_render_ms", "total_ms"):
            s = summary.get(stage)
            if s is None:
                continue
            logger.info(
                "%-18s mean=%6.2f ms  stdev=%5.2f  min=%6.2f  max=%7.2f  p95=%6.2f  (n=%d)",
                stage, s["mean_ms"], s["stdev_ms"], s["min_ms"], s["max_ms"], s["p95_ms"], s["n"],
            )
        out_path = Path(args.profile_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "raw_samples_ms": profile_samples}, f, indent=2)
        logger.info("Raw per-frame samples saved to %s", out_path)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
