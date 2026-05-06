"""mAP accuracy evaluator for inference pipeline benchmarking.

Evaluates mAP@0.5:0.95 on the full COCO val2017 set (5000 images) per the
COCO standard. Delta is always relative to the FP32 baseline of the same
runtime — never cross-runtime — to isolate precision cost from runtime cost.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

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
) -> list[dict]:
    """Convert YOLOv8n raw output to COCO-compatible prediction format.

    YOLOv8n output shape is (1, 84, 8400) where the first 4 rows are
    (cx, cy, w, h) in absolute pixel coordinates and rows 4–83 are per-class
    confidence scores. This function applies a confidence threshold and
    converts to COCO's [x_min, y_min, width, height] convention.

    Args:
        raw_output: Model output array, shape (1, 84, N).
        image_id: COCO image ID, embedded in each prediction dict.
        conf_threshold: Minimum confidence to include a detection.

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
        x_min = float(cx - w / 2)
        y_min = float(cy - h / 2)

        predictions.append(
            {
                "image_id": image_id,
                "category_id": int(class_ids[idx]) + 1,  # COCO categories are 1-indexed
                "bbox": [x_min, y_min, float(w), float(h)],
                "score": float(max_scores[idx]),
            }
        )

    return predictions


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
