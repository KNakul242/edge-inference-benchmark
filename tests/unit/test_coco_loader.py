"""Unit tests for the COCO val2017 data loader and preprocessor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.coco_loader import preprocess_image, CocoLoader


class TestPreprocessImage:
    def test_output_shape_is_nchw(self, tmp_path: Path) -> None:
        """Preprocessed image must be (1, 3, 640, 640) NCHW."""
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            result = preprocess_image(dummy_bgr)

        assert result.shape == (1, 3, 640, 640)

    def test_output_dtype_is_float32(self) -> None:
        dummy_bgr = np.zeros((640, 640, 3), dtype=np.uint8)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            result = preprocess_image(dummy_bgr)

        assert result.dtype == np.float32

    def test_output_values_normalised_to_0_1(self) -> None:
        """Pixel values must be in [0, 1] after normalisation."""
        dummy_bgr = np.full((640, 640, 3), 255, dtype=np.uint8)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.full((640, 640, 3), 255, dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.full((640, 640, 3), 255, dtype=np.uint8)
            result = preprocess_image(dummy_bgr)

        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.0 - 1e-6

    def test_resize_called_with_640(self) -> None:
        dummy_bgr = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            preprocess_image(dummy_bgr)

        # cv2.resize(src, dsize) — dsize is the second positional arg
        call_args = mock_cv2.resize.call_args
        dsize = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("dsize")
        assert dsize == (640, 640)


class TestCocoLoader:
    def test_len_returns_number_of_images(self, tmp_path: Path) -> None:
        for i in range(5):
            (tmp_path / f"{i:012d}.jpg").touch()

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        assert len(loader) == 5

    def test_empty_directory_returns_zero(self, tmp_path: Path) -> None:
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        assert len(loader) == 0

    def test_only_jpg_files_counted(self, tmp_path: Path) -> None:
        (tmp_path / "image.jpg").touch()
        (tmp_path / "image.png").touch()
        (tmp_path / "readme.txt").touch()

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        assert len(loader) == 1

    def test_image_paths_sorted_deterministically(self, tmp_path: Path) -> None:
        for name in ["b.jpg", "a.jpg", "c.jpg"]:
            (tmp_path / name).write_bytes(b"")

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        names = [Path(p).name for p in loader.image_paths]
        assert names == sorted(names)

    def test_iter_yields_tensor_image_id_tuples(self, tmp_path: Path) -> None:
        """__iter__ must yield (tensor, image_id) pairs — one per image."""
        for name in ["000000000042.jpg", "000000000099.jpg"]:
            (tmp_path / name).write_bytes(b"")

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        items = []
        dummy_nchw = np.zeros((1, 3, 640, 640), dtype=np.float32)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            items = list(loader)

        assert len(items) == 2

    def test_iter_tensor_shape(self, tmp_path: Path) -> None:
        """Each yielded tensor must have shape (1, 3, 640, 640)."""
        (tmp_path / "000000000001.jpg").write_bytes(b"")

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            tensor, _ = next(iter(loader))

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_iter_image_id_extracted_from_filename(self, tmp_path: Path) -> None:
        """image_id must be the integer parsed from the COCO filename stem."""
        (tmp_path / "000000001234.jpg").write_bytes(b"")

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, image_id = next(iter(loader))

        assert image_id == 1234

    def test_iter_order_matches_sorted_paths(self, tmp_path: Path) -> None:
        """Image IDs must be yielded in sorted filename order."""
        for name in ["000000000099.jpg", "000000000042.jpg"]:
            (tmp_path / name).write_bytes(b"")

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        ids = []

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            ids = [image_id for _, image_id in loader]

        assert ids == [42, 99]
