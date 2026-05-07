"""COCO val2017 data loader and preprocessor.

Provides deterministic image loading and preprocessing to the fixed NCHW
input format required by all runtimes in the benchmark pipeline. Uses
letterbox resizing (preserve aspect ratio, pad to square) to match the
preprocessing convention used during YOLOv8n training — mandatory for
producing deployment-comparable mAP numbers.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_INPUT_SIZE = 640
# YOLOv8 training uses neutral gray padding (114/255 normalised)
_LETTERBOX_PAD_VALUE = 114


@dataclass
class LetterboxMeta:
    """Metadata produced by letterbox preprocessing.

    Required to map model-output bounding boxes (in 640×640 letterboxed space)
    back to the original image coordinate system for correct COCO evaluation.

    Attributes:
        scale: The uniform scale factor applied to both dimensions.
        pad_left: Pixels of gray padding added to the left edge.
        pad_top: Pixels of gray padding added to the top edge.
        orig_h: Original image height in pixels.
        orig_w: Original image width in pixels.
    """
    scale: float
    pad_left: int
    pad_top: int
    orig_h: int
    orig_w: int


def letterbox_preprocess(
    bgr: np.ndarray, input_size: int = _INPUT_SIZE
) -> tuple[np.ndarray, LetterboxMeta]:
    """Preprocess a BGR image using letterboxing to preserve aspect ratio.

    Resizes the image so its longest edge fits ``input_size``, then pads the
    shorter edge with neutral gray (114/255) to produce a square. This matches
    YOLOv8's training preprocessing — plain ``cv2.resize`` to a square distorts
    aspect ratio and causes systematic mAP degradation on non-square images.

    Returns both the preprocessed tensor and the metadata needed to reverse-map
    model-output bounding boxes to the original image coordinate space.

    Args:
        bgr: Input image in BGR format, shape (H, W, 3), uint8.
        input_size: Target square resolution. Defaults to 640.

    Returns:
        Tuple of (tensor, meta) where tensor is (1, 3, input_size, input_size)
        float32 and meta carries the scale and padding for coordinate rescaling.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required. Run: pip install opencv-python")

    orig_h, orig_w = bgr.shape[:2]
    scale = min(input_size / orig_h, input_size / orig_w)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(bgr, (new_w, new_h))

    pad_left = (input_size - new_w) // 2
    pad_top = (input_size - new_h) // 2

    # Pad with neutral gray — the value used during YOLOv8 training
    padded = np.full((input_size, input_size, 3), _LETTERBOX_PAD_VALUE, dtype=np.uint8)
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalised = rgb.astype(np.float32) / 255.0
    nchw = normalised.transpose(2, 0, 1)[np.newaxis, :]

    meta = LetterboxMeta(
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        orig_h=orig_h,
        orig_w=orig_w,
    )
    return nchw, meta


def preprocess_image(bgr: np.ndarray, input_size: int = _INPUT_SIZE) -> np.ndarray:
    """Letterbox-preprocess a BGR image and return the tensor only.

    Convenience wrapper over ``letterbox_preprocess`` for callers that do not
    need the coordinate-rescaling metadata (e.g. latency profiling with dummy
    inputs). For accuracy evaluation, use ``letterbox_preprocess`` directly to
    obtain the ``LetterboxMeta`` required by ``format_coco_prediction``.

    Args:
        bgr: Input image in BGR format, shape (H, W, 3), uint8.
        input_size: Target square resolution. Defaults to 640.

    Returns:
        Preprocessed tensor, shape (1, 3, input_size, input_size), float32.
    """
    tensor, _ = letterbox_preprocess(bgr, input_size)
    return tensor


class CocoLoader:
    """Loads image paths from a COCO val2017 directory.

    Args:
        images_dir: Path to the directory containing COCO .jpg images.
        annotations_file: Path to instances_val2017.json, or None if not needed.
    """

    def __init__(self, images_dir: str, annotations_file: str | None) -> None:
        self._images_dir = Path(images_dir)
        self._annotations_file = annotations_file
        self.image_paths: list[str] = sorted(
            str(p) for p in self._images_dir.glob("*.jpg")
        )
        logger.info("CocoLoader: found %d images in %s", len(self.image_paths), images_dir)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self):
        """Yield (tensor, image_id, meta) for each image in sorted order.

        ``image_id`` is parsed from the COCO filename stem (e.g.
        ``000000001234.jpg`` → ``1234``). ``meta`` is a ``LetterboxMeta``
        instance required by ``format_coco_prediction`` to rescale model-output
        boxes from 640×640 letterboxed space to original image coordinates.

        Raises:
            ImportError: If ``opencv-python`` is not installed.
        """
        if cv2 is None:  # pragma: no cover
            raise ImportError("opencv-python is required. Run: pip install opencv-python")
        for path in self.image_paths:
            try:
                image_id = int(Path(path).stem)
            except ValueError:
                logger.warning(
                    "Non-numeric filename '%s' — skipping (expected COCO 12-digit format)", path
                )
                continue
            bgr = cv2.imread(path)
            if bgr is None:
                logger.warning(
                    "cv2.imread returned None for %s — file may be corrupted or "
                    "unreadable. Skipping image_id=%d.", path, image_id
                )
                continue
            tensor, meta = letterbox_preprocess(bgr)
            yield tensor, image_id, meta
