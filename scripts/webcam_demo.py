"""Live YOLOv8n object detection via ONNX Runtime CPU EP.

Illustrates CPU-class inference throughput at 640×640 FP32 — the same
hardware configuration benchmarked in the study (ONNX Runtime CPU EP,
Intel Core Ultra 5 125H, ~72 ms mean / ~10 FPS). The overhead panel
shows per-frame inference latency and rolling FPS so the constraint is
visible throughout.

Usage (from project root):
    python scripts/webcam_demo.py
    python scripts/webcam_demo.py --model models/yolov8n.onnx --camera 0
    python scripts/webcam_demo.py --conf 0.4   # lower threshold, more boxes
    python scripts/webcam_demo.py --scale 2.0  # larger display window

Press Q to quit.
"""

import argparse
import collections
import logging
import os
import sys
import time
from pathlib import Path

# Force X11 backend — the py3_11 conda env lacks the Qt Wayland plugin,
# which causes cv2.imshow to silently fail on Wayland sessions.
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

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
_WINDOW_TITLE = "YOLOv8n  -  CPU inference demo"

# Per-class BGR colours, cycled by class index
_PALETTE = [
    (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 128), (128, 0, 255),
    (0, 255, 128), (255, 255, 0), (0, 255, 255), (255, 0, 255), (128, 255, 0),
]


def _nms(boxes_xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Agnostic (class-independent) greedy non-maximum suppression.

    Identical algorithm to accuracy_evaluator._apply_nms — kept here so
    the demo script has no dependency on private benchmark internals.

    Args:
        boxes_xyxy: (K, 4) float32, [x1, y1, x2, y2].
        scores: (K,) float32.
        iou_threshold: Suppress boxes with IoU above this value.

    Returns:
        Surviving box indices in score-descending order.
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
        inter = (
            np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
            * np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
        )
        union = areas[i] + areas[rest] - inter
        order = rest[np.where(union > 0, inter / union, 0.0) <= iou_threshold]

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

    keep = _nms(boxes, scores, iou)
    if len(keep) == 0:
        return []

    results: list[tuple[int, int, int, int, float, str]] = []
    for idx in keep:
        bx1, by1, bx2, by2 = boxes[idx]
        # Reverse letterbox: remove padding, undo scale
        rx1 = max(0.0, (bx1 - meta.pad_left) / meta.scale)
        ry1 = max(0.0, (by1 - meta.pad_top) / meta.scale)
        rx2 = min(float(meta.orig_w), (bx2 - meta.pad_left) / meta.scale)
        ry2 = min(float(meta.orig_h), (by2 - meta.pad_top) / meta.scale)
        results.append((
            int(rx1), int(ry1), int(rx2), int(ry2),
            float(scores[idx]),
            COCO_NAMES[int(ids[idx])],
        ))
    return results


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
        lx, ly = x1, max(y1 - 2, th + 4)
        cv2.rectangle(frame, (lx, ly - th - 3), (lx + tw + 4, ly + 1), colour, -1)
        cv2.putText(frame, label, (lx + 2, ly - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)

    # --- Top HUD ---
    cv2.rectangle(frame, (0, 0), (fw, 56), (15, 15, 15), -1)
    cv2.putText(frame, "ONNX Runtime  |  CPU EP  |  FP32",
                (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Inference: {latency_ms:6.1f} ms     FPS (wall): {fps:5.1f}",
                (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 220, 255), 1, cv2.LINE_AA)

    # --- Bottom HUD ---
    cv2.rectangle(frame, (0, fh - 24), (fw, fh), (15, 15, 15), -1)
    cv2.putText(frame,
                "YOLOv8n  640x640  batch=1  Intel Core Ultra 5 125H  --  press Q to quit",
                (10, fh - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)


def main() -> None:
    """Entry point — parse args, load model, run capture loop."""
    parser = argparse.ArgumentParser(
        description="YOLOv8n webcam demo — ONNX Runtime CPU EP"
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
    args = parser.parse_args()

    # --- Load model ---
    runtime = OnnxRuntime(execution_provider="CPUExecutionProvider", precision="fp32")
    runtime.load(args.model)

    # --- Open camera ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera %d. Check --camera index.", args.camera)
        sys.exit(1)

    # --- Create display window up-front with an explicit size ---
    # WINDOW_NORMAL (not the AUTOSIZE default): the bundled Qt backend can
    # open autosized windows at a near-zero zoom level on Wayland/XWayland,
    # showing only a sliver of the frame. An explicit resizeWindow pins the
    # viewport to the intended display size.
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
        tensor, _ = letterbox_preprocess(frame)
        runtime.infer(tensor)
        warmed += 1
    logger.info("Warmup complete. Starting live loop — press Q to quit.")

    fps_times: collections.deque[float] = collections.deque(maxlen=_FPS_WINDOW)
    t_prev = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Frame capture failed — camera may have disconnected.")
            break

        # --- Preprocess ---
        tensor, meta = letterbox_preprocess(frame)

        # --- Inference (timed) ---
        t0 = time.perf_counter()
        raw = runtime.infer(tensor)
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # --- Decode ---
        detections = decode_detections(raw, meta, conf=args.conf)

        # --- Rolling FPS (wall-clock includes capture, pre/postprocessing, draw) ---
        t_now = time.perf_counter()
        fps_times.append(t_now - t_prev)
        t_prev = t_now
        fps = len(fps_times) / sum(fps_times) if fps_times else 0.0

        # --- Draw and display ---
        draw_frame(frame, detections, latency_ms, fps)
        if args.scale != 1.0:
            dh = int(frame.shape[0] * args.scale)
            dw = int(frame.shape[1] * args.scale)
            display = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_LINEAR)
        else:
            display = frame
        cv2.imshow(_WINDOW_TITLE, display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            logger.info("Q pressed — stopping.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
