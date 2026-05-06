"""INT8 calibration set sampler.

Samples a fixed subset of COCO val2017 images using a deterministic seed for
reproducible INT8 quantisation calibration across ONNX Runtime and TensorRT.
The calibration manifest is committed to the repo so results can be reproduced
even if the full COCO dataset is re-downloaded.
"""

import json
import logging
import random
import shutil
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def generate_calibration_set(
    coco_val_dir: str,
    output_dir: str,
    n_images: int = 500,
    seed: int = 42,
) -> str:
    """Sample a reproducible INT8 calibration set from COCO val2017.

    Copies ``n_images`` randomly selected .jpg files from ``coco_val_dir`` into
    ``output_dir`` and writes a manifest recording the seed and selected image
    names. The same seed always produces the same selection.

    Args:
        coco_val_dir: Path to the COCO val2017 images directory.
        output_dir: Destination directory for the calibration subset.
        n_images: Number of images to select. Defaults to 500.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        Path to the written manifest.json file.

    Raises:
        ValueError: If ``coco_val_dir`` contains fewer images than ``n_images``.
    """
    src = Path(coco_val_dir)
    dst = Path(output_dir)
    dst.mkdir(parents=True, exist_ok=True)

    all_images = sorted(src.glob("*.jpg"))
    if len(all_images) < n_images:
        raise ValueError(
            f"Requested {n_images} calibration images but only {len(all_images)} "
            f"available in {coco_val_dir}"
        )

    rng = random.Random(seed)
    np.random.seed(seed)
    selected = rng.sample(all_images, n_images)

    for img_path in selected:
        shutil.copy2(img_path, dst / img_path.name)

    manifest = {
        "seed": seed,
        "n_images": n_images,
        "source_dir": str(src),
        "images": sorted(img.name for img in selected),
    }
    manifest_path = dst / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    logger.info(
        "Calibration set: %d images written to %s (seed=%d)", n_images, dst, seed
    )
    return str(manifest_path)
