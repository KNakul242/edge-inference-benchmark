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
    conf_threshold: float = 0.5,
    letterbox_meta: Optional[LetterboxMeta] = None,
) -> list[dict]:
    """Convert YOLOv8n raw output to COCO-compatible prediction format.

    YOLOv8n output shape is (1, 84, 8400) where the first 4 rows are
    (cx, cy, w, h) in absolute pixel coordinates of the model's 640×640 input
    space, and rows 4–83 are per-class confidence scores.

    When ``letterbox_meta`` is provided, box coordinates are rescaled from the
    model's letterboxed input space back to the original image coordinate
    system — required for correct COCO evaluation because ground-truth
    annotations are in original image coordinates.

    Args:
        raw_output: Model output array, shape (1, 84, N).
        image_id: COCO image ID, embedded in each prediction dict.
        conf_threshold: Minimum confidence to include a detection.
        letterbox_meta: Letterbox parameters from ``CocoLoader.__iter__``.
            When provided, coordinates are rescaled to original image space.
            When None, coordinates are left in model input (640×640) space.

    Returns:
        List of COCO-formatted prediction dicts, each containing
        ``image_id``, ``category_id``, ``bbox``, and ``score``.
    """
    predictions: list[dict] = []
    output = raw_output[0]  # (84, N)

    boxes = output[:_BOX_DIM, :]      # (4, N) — cx, cy, w, h
    class_scores = output[_BOX_DIM:, :]  # (80, N)

    max_scores = class_scores.max(axis=0)       # (N,)
    class_ids = class_scores.argmax(axis=0)     # (N,)

    above_threshold = np.where(max_scores >= conf_threshold)[0]
    if len(above_threshold) == 0:
        return predictions

    for idx in above_threshold:
        cx, cy, w, h = boxes[:, idx]

        if letterbox_meta is not None:
            # Rescale from 640×640 letterboxed model-input space to original image space.
            # Subtracting padding offsets removes the gray border; dividing by scale
            # maps back to the original image dimensions.
            scale = letterbox_meta.scale
            x_min = float((cx - w / 2 - letterbox_meta.pad_left) / scale)
            y_min = float((cy - h / 2 - letterbox_meta.pad_top) / scale)
            bbox_w = float(w / scale)
            bbox_h = float(h / scale)
        else:
            x_min = float(cx - w / 2)
            y_min = float(cy - h / 2)
            bbox_w = float(w)
            bbox_h = float(h)

        predictions.append(
            {
                "image_id": image_id,
                "category_id": int(class_ids[idx]) + 1,  # COCO categories are 1-indexed
                "bbox": [x_min, y_min, bbox_w, bbox_h],
                "score": float(max_scores[idx]),
            }
        )

    return predictions


def evaluate_map(
    runtime: "BaseRuntime",
    loader: "CocoLoader",
    annotations_file: str,
    conf_threshold: float = 0.5,
) -> "AccuracyResult":
    """Evaluate mAP@0.5:0.95 on COCO val2017 using pycocotools COCOeval.

    Iterates over every image in ``loader``, runs inference, collects COCO-format
    predictions (with coordinates rescaled to original image space via the
    letterbox metadata from each ``(tensor, image_id, meta)`` tuple), then
    evaluates against ground-truth annotations.

    ``map_delta_vs_fp32`` is left as ``None``; the caller must compute it via
    ``compute_map_delta()`` after all precision variants for a runtime have run.

    Args:
        runtime: Loaded runtime implementing ``BaseRuntime``.
        loader: ``CocoLoader`` whose ``__iter__`` yields
            ``(tensor, image_id, LetterboxMeta)`` 3-tuples.
        annotations_file: Path to ``instances_val2017.json``.
        conf_threshold: Minimum detection confidence. Defaults to 0.5.

    Returns:
        ``AccuracyResult`` with ``map_50_95`` (primary) and ``map_50`` (secondary)
        populated. ``map_delta_vs_fp32`` is ``None`` until set by the caller.

    Raises:
        ImportError: If ``pycocotools`` is not installed.
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

    for image_tensor, image_id, letterbox_meta in loader:
        raw_output = runtime.infer(image_tensor)
        preds = format_coco_prediction(
            raw_output,
            image_id=image_id,
            conf_threshold=conf_threshold,
            letterbox_meta=letterbox_meta,
        )
        all_predictions.extend(preds)

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
    """
    delta = candidate.map_50_95 - baseline.map_50_95
    logger.info(
        "mAP delta [%s]: %s vs %s baseline → %.4f",
        candidate.runtime,
        candidate.precision,
        baseline.precision,
        delta,
    )
    return delta
