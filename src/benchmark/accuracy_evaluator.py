"""mAP accuracy evaluator for inference pipeline benchmarking.

Evaluates mAP@0.5:0.95 on the full COCO val2017 set (5000 images) per the
COCO standard. Delta is always relative to the FP32 baseline of the same
runtime — never cross-runtime — to isolate precision cost from runtime cost.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

from src.data.coco_loader import LetterboxMeta

if TYPE_CHECKING:
    from src.data.coco_loader import CocoLoader
    from src.runtimes.base_runtime import BaseRuntime

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:  # pragma: no cover
    COCO = None  # type: ignore[assignment,misc]
    COCOeval = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# YOLOv8n output layout: (1, 84, N) where 84 = 4 box coords + 80 class scores
_BOX_DIM = 4
_N_CLASSES = 80

# COCO 2017 category IDs for the 80 object classes in YOLOv8 class-index order.
# COCO IDs are NOT consecutive 1–80: 11 IDs are absent (12, 26, 29, 30, 45, 66,
# 68, 69, 71, 83, and the sequence has further gaps at higher values).
# Using `class_idx + 1` is wrong for any class beyond index 10 (fire hydrant).
_COCO_CATEGORY_IDS: list[int] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
    41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
]


def _apply_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Greedy agnostic (class-independent) non-maximum suppression.

    Sorts all boxes by score descending and suppresses any box with IoU above
    ``iou_threshold`` against a higher-scoring box, regardless of class label.
    This matches YOLOv8's reference post-processing convention and produces
    mAP numbers comparable to the published ultralytics baseline (~0.372
    mAP@0.5:0.95 for YOLOv8n on COCO val2017).

    Per-class NMS (previous implementation) was methodologically correct but
    diverged from the YOLOv8 reference by retaining same-region multi-class
    predictions that agnostic NMS suppresses, depressing absolute mAP by
    approximately 0.002–0.01.

    Args:
        boxes_xyxy: (K, 4) float32, bounding boxes in [x1, y1, x2, y2] format.
        scores: (K,) float32, confidence scores per box.
        iou_threshold: Boxes with IoU above this value are suppressed.

    Returns:
        Array of indices (into 0..K-1) of boxes that survive NMS, in
        score-descending order.
    """
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int64)

    x1, y1 = boxes_xyxy[:, 0], boxes_xyxy[:, 1]
    x2, y2 = boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    order = scores.argsort()[::-1]
    keep: list[int] = []

    while len(order) > 0:
        i = int(order[0])
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int64)


@dataclass
class AccuracyResult:
    """mAP accuracy metrics for one runtime × precision combination.

    Attributes:
        map_50_95: Primary metric — mAP@0.5:0.95 (COCO standard).
        map_50: Secondary metric — mAP@0.5.
        precision: Numerical precision used (``"fp32"``, ``"fp16"``, ``"int8"``).
        runtime: Runtime identifier (matches ``BaseRuntime.name``).
        map_delta_vs_fp32: map_50_95 − fp32_baseline for this runtime.
            Negative means accuracy degradation. None until delta is computed.
    """

    map_50_95: float
    map_50: float
    precision: str
    runtime: str
    map_delta_vs_fp32: Optional[float] = field(default=None)


def format_coco_prediction(
    raw_output: np.ndarray,
    image_id: int,
    conf_threshold: float,
    iou_threshold: float = 0.45,
    letterbox_meta: Optional[LetterboxMeta] = None,
) -> list[dict]:
    """Convert YOLOv8n raw output to COCO-compatible prediction format.

    YOLOv8n output shape is (1, 84, 8400) where the first 4 rows are
    (cx, cy, w, h) in absolute pixel coordinates of the model's 640×640 input
    space, and rows 4–83 are per-class confidence scores (post-sigmoid, range [0,1]).

    Pipeline:
    1. Validate output shape and class score range (catches wrong export format).
    2. Filter anchors whose max class score ≥ ``conf_threshold``.
    3. Apply agnostic NMS at ``iou_threshold`` to suppress duplicate boxes.
    4. Map surviving class indices to real COCO category IDs via
       ``_COCO_CATEGORY_IDS`` (COCO IDs are not consecutive 1–80; there are 11
       gaps). Using ``class_idx + 1`` is wrong for any class beyond index 10.
    5. Optionally rescale box coordinates from 640×640 letterboxed model space
       back to original image coordinates using ``letterbox_meta``.

    Args:
        raw_output: Model output array, shape (1, 84, 8400).
        image_id: COCO image ID, embedded in each prediction dict.
        conf_threshold: Minimum confidence score to retain a detection. No default —
            caller must pass explicitly. For mAP evaluation use 0.001 (exposes full
            PR curve). For deployment latency use 0.5.
        iou_threshold: IoU threshold for agnostic NMS. Default 0.45 (deployment
            convention). For mAP evaluation pass 0.7 (ultralytics reference).
        letterbox_meta: Letterbox parameters from ``CocoLoader.__iter__``.
            When provided, coordinates are rescaled to original image space.
            When None, coordinates are left in model input (640×640) space.

    Returns:
        List of COCO-formatted prediction dicts, each containing
        ``image_id``, ``category_id``, ``bbox`` ([x_min, y_min, w, h]), and ``score``.

    Raises:
        ValueError: If ``raw_output`` has an unexpected shape or class scores are
            outside the valid probability range [0, 1], indicating a wrong export
            format (e.g. logit outputs or NMS baked into the ONNX graph).
    """
    if raw_output.shape != (1, _N_CLASSES + _BOX_DIM, 8400):
        raise ValueError(
            f"Unexpected model output shape {raw_output.shape}. "
            f"Expected (1, 84, 8400) for YOLOv8n at 640×640 with "
            "opset=17, dynamic=False, nms=False."
        )

    predictions: list[dict] = []
    output = raw_output[0]  # (84, N)

    boxes_cwh = output[:_BOX_DIM, :]     # (4, N) — cx, cy, w, h
    class_scores = output[_BOX_DIM:, :]  # (80, N)

    # Validate that class scores are post-sigmoid probabilities. Logit outputs
    # (unbounded) indicate the ONNX graph is missing sigmoid activation — silent
    # wrong mAP would result, since the threshold comparison operates in the wrong
    # value space.
    if class_scores.size > 0 and (
        float(class_scores.min()) < -1e-3 or float(class_scores.max()) > 1 + 1e-3
    ):
        raise ValueError(
            f"Class score output outside valid probability range [0, 1]: "
            f"min={float(class_scores.min()):.4f}, max={float(class_scores.max()):.4f}. "
            "YOLOv8n ONNX export should include sigmoid activation. "
            "Verify export parameters: format='onnx', nms=False."
        )

    max_scores = class_scores.max(axis=0)    # (N,)
    class_ids = class_scores.argmax(axis=0)  # (N,)

    above_threshold = np.where(max_scores >= conf_threshold)[0]
    if len(above_threshold) == 0:
        return predictions

    # Convert cx,cy,w,h → xyxy for NMS; all in model (640×640) coordinate space
    cx = boxes_cwh[0, above_threshold]
    cy = boxes_cwh[1, above_threshold]
    w  = boxes_cwh[2, above_threshold]
    h  = boxes_cwh[3, above_threshold]
    boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    kept = _apply_nms(
        boxes_xyxy,
        max_scores[above_threshold],
        iou_threshold,
    )
    # Map local NMS indices back to indices into the full 8400-anchor array
    surviving = above_threshold[kept]

    for idx in surviving:
        cx_i, cy_i, w_i, h_i = boxes_cwh[:, idx]
        cls_idx = int(class_ids[idx])

        if cls_idx < 0 or cls_idx >= len(_COCO_CATEGORY_IDS):
            logger.warning(
                "Class index %d out of range [0, 79] — skipping detection (image_id=%d)",
                cls_idx, image_id,
            )
            continue

        if letterbox_meta is not None:
            # Rescale from 640×640 letterboxed model-input space to original image space.
            # Subtracting padding offsets removes the gray border; dividing by scale
            # maps back to the original image dimensions.
            scale = letterbox_meta.scale
            x_min = float((cx_i - w_i / 2 - letterbox_meta.pad_left) / scale)
            y_min = float((cy_i - h_i / 2 - letterbox_meta.pad_top) / scale)
            bbox_w = float(w_i / scale)
            bbox_h = float(h_i / scale)
            # Clamp to image bounds. Anchors at edges produce boxes that extend into
            # the letterbox padding, yielding negative x_min / y_min or right/bottom
            # edges beyond orig_w / orig_h. Preserve the original right/bottom edge
            # before clamping the left/top — otherwise the width/height grows wrongly.
            x2 = x_min + bbox_w
            y2 = y_min + bbox_h
            x_min = max(0.0, x_min)
            y_min = max(0.0, y_min)
            bbox_w = max(0.0, min(x2, float(letterbox_meta.orig_w)) - x_min)
            bbox_h = max(0.0, min(y2, float(letterbox_meta.orig_h)) - y_min)
        else:
            x_min = float(cx_i - w_i / 2)
            y_min = float(cy_i - h_i / 2)
            bbox_w = float(w_i)
            bbox_h = float(h_i)

        predictions.append({
            "image_id": image_id,
            "category_id": _COCO_CATEGORY_IDS[cls_idx],
            "bbox": [x_min, y_min, bbox_w, bbox_h],
            "score": float(max_scores[idx]),
        })

    return predictions


def evaluate_map(
    runtime: "BaseRuntime",
    loader: "CocoLoader",
    annotations_file: str,
    conf_threshold: float = 0.001,
    iou_threshold: float = 0.7,
) -> "AccuracyResult":
    """Evaluate mAP@0.5:0.95 on COCO val2017 using pycocotools COCOeval.

    Iterates over every image in ``loader``, runs inference, collects COCO-format
    predictions (confidence-filtered, NMS-suppressed, and coordinate-rescaled via
    the letterbox metadata from each ``(tensor, image_id, meta)`` 3-tuple), then
    evaluates against ground-truth annotations.

    ``map_delta_vs_fp32`` is left as ``None``; the caller must compute it via
    ``compute_map_delta()`` after all precision variants for a runtime have run.

    Args:
        runtime: Loaded runtime implementing ``BaseRuntime``.
        loader: ``CocoLoader`` whose ``__iter__`` yields
            ``(tensor, image_id, LetterboxMeta)`` 3-tuples.
        annotations_file: Path to ``instances_val2017.json``.
        conf_threshold: Minimum detection confidence for mAP evaluation. Defaults
            to 0.001 — exposes the full precision-recall curve to COCOeval.
            Do NOT use the deployment threshold (0.5) here; it truncates the PR
            curve and suppresses mAP by ~29% (benchmark-run-1-findings.md, Issue 1).
        iou_threshold: NMS IoU threshold for suppressing duplicate boxes. Defaults
            to 0.7 — matches ultralytics' reference validator, which preserves more
            candidate boxes at the high detection density produced by conf=0.001.

    Returns:
        ``AccuracyResult`` with ``map_50_95`` (primary) and ``map_50`` (secondary)
        populated. ``map_delta_vs_fp32`` is ``None`` until set by the caller.

    Raises:
        ImportError: If ``pycocotools`` is not installed.
        RuntimeError: If fewer images are evaluated than the loader reports, which
            indicates a partial or corrupted dataset.
    """
    if COCO is None or COCOeval is None:
        raise ImportError(
            "pycocotools is required for mAP evaluation. "
            "Run: pip install pycocotools==2.0.7"
        )

    precision = runtime.name.rsplit("_", 1)[-1]
    logger.info(
        "Evaluating mAP@0.5:0.95 for %s across %d images", runtime.name, len(loader)
    )

    coco_gt = COCO(annotations_file)
    all_predictions: list[dict] = []
    n_evaluated = 0

    for image_tensor, image_id, letterbox_meta in loader:
        n_evaluated += 1
        if n_evaluated % 500 == 0:
            logger.info("%s: evaluated %d/%d images", runtime.name, n_evaluated, len(loader))
        raw_output = runtime.infer(image_tensor)
        preds = format_coco_prediction(
            raw_output,
            image_id=image_id,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            letterbox_meta=letterbox_meta,
        )
        all_predictions.extend(preds)

    if n_evaluated != len(loader):
        raise RuntimeError(
            f"Partial image set: {runtime.name} evaluated {n_evaluated} of "
            f"{len(loader)} images ({len(loader) - n_evaluated} skipped). "
            "Check for corrupted files or non-numeric COCO filenames. "
            "A partial evaluation submits predictions for fewer images than the "
            "full GT annotation set, systematically suppressing mAP recall."
        )

    if not all_predictions:
        logger.warning(
            "%s produced no detections above conf_threshold=%.2f — mAP reported as 0.0",
            runtime.name, conf_threshold,
        )
        return AccuracyResult(
            map_50_95=0.0, map_50=0.0, precision=precision, runtime=runtime.name
        )

    coco_dt = coco_gt.loadRes(all_predictions)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    result = AccuracyResult(
        map_50_95=float(coco_eval.stats[0]),  # mAP@0.5:0.95 — primary metric
        map_50=float(coco_eval.stats[1]),      # mAP@0.5 — secondary
        precision=precision,
        runtime=runtime.name,
    )
    logger.info(
        "%s — mAP@0.5:0.95=%.4f  mAP@0.5=%.4f",
        runtime.name, result.map_50_95, result.map_50,
    )
    return result


def compute_map_delta(
    baseline: AccuracyResult,
    candidate: AccuracyResult,
) -> float:
    """Compute mAP@0.5:0.95 delta between candidate and FP32 baseline.

    Delta sign convention: negative = accuracy degradation relative to baseline.
    The baseline must be the FP32 result for the same runtime as the candidate.

    Args:
        baseline: FP32 reference AccuracyResult for this runtime.
        candidate: AccuracyResult at a lower precision (FP16 or INT8).

    Returns:
        ``candidate.map_50_95 − baseline.map_50_95``. Negative indicates
        degradation below the FP32 deployment baseline.

    Raises:
        ValueError: If ``baseline`` does not have precision ``"fp32"``, or if
            ``baseline`` and ``candidate`` appear to be from different runtime
            families (cross-runtime delta is methodologically invalid).
    """
    if baseline.precision != "fp32":
        raise ValueError(
            f"Baseline must have precision='fp32', got '{baseline.precision}'. "
            "mAP delta is always relative to the FP32 result of the same runtime."
        )

    # Strip known precision suffixes to extract the runtime family for comparison.
    # This catches the most common error: accidentally passing a PyTorch baseline
    # against an ONNX candidate (or vice versa).
    def _family(runtime_name: str) -> str:
        for suffix in ("_fp32", "_fp16", "_int8"):
            runtime_name = runtime_name.removesuffix(suffix)
        return runtime_name

    baseline_family = _family(baseline.runtime)
    candidate_family = _family(candidate.runtime)
    if baseline_family != candidate_family:
        raise ValueError(
            f"Cross-runtime delta detected: baseline runtime '{baseline.runtime}' "
            f"(family '{baseline_family}') does not match candidate runtime "
            f"'{candidate.runtime}' (family '{candidate_family}'). "
            "mAP delta must be computed within the same runtime family."
        )

    delta = candidate.map_50_95 - baseline.map_50_95
    logger.info(
        "mAP delta [%s]: %s vs %s baseline → %.4f",
        candidate.runtime,
        candidate.precision,
        baseline.precision,
        delta,
    )
    return delta
