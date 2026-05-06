"""COCO val2017 data loader and preprocessor.

Provides deterministic image loading and preprocessing to the fixed NCHW
input format required by all runtimes in the benchmark pipeline.
"""

import logging
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_INPUT_SIZE = 640


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


def preprocess_image(bgr: np.ndarray, input_size: int = _INPUT_SIZE) -> np.ndarray:
    """Preprocess a BGR image into the NCHW float32 format used by all runtimes.

    Resizes to ``input_size × input_size``, converts BGR → RGB, normalises
    pixel values to [0, 1], and transposes to NCHW with a batch dimension.

    Args:
        bgr: Input image in BGR format, shape (H, W, 3), uint8.
        input_size: Target square resolution. Defaults to 640.

    Returns:
        Preprocessed tensor, shape (1, 3, input_size, input_size), float32.
    """
    if cv2 is None:  # pragma: no cover
        raise ImportError("opencv-python is required. Run: pip install opencv-python")

    resized = cv2.resize(bgr, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalised = rgb.astype(np.float32) / 255.0
    nchw = normalised.transpose(2, 0, 1)[np.newaxis, :]  # HWC → NCHW with batch
    return nchw
