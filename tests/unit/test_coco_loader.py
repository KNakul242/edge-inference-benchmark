"""Unit tests for the COCO val2017 data loader and preprocessor."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.data.coco_loader import CocoLoader, LetterboxMeta, letterbox_preprocess, preprocess_image


# ---------------------------------------------------------------------------
# letterbox_preprocess — new function replacing plain resize
# ---------------------------------------------------------------------------

class TestLetterboxPreprocess:
    def test_output_tensor_shape_is_nchw(self) -> None:
        """Output must be (1, 3, 640, 640) regardless of input aspect ratio."""
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            tensor, _ = letterbox_preprocess(bgr)
        assert tensor.shape == (1, 3, 640, 640)

    def test_output_dtype_is_float32(self) -> None:
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            tensor, _ = letterbox_preprocess(bgr)
        assert tensor.dtype == np.float32

    def test_returns_letterbox_meta(self) -> None:
        """Must return a LetterboxMeta alongside the tensor."""
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, meta = letterbox_preprocess(bgr)
        assert isinstance(meta, LetterboxMeta)

    def test_scale_fits_longer_edge(self) -> None:
        """Scale must map the longest edge to input_size without overflow."""
        # 480×640 image: longer edge is width (640) → scale = 1.0, new_h = 480
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, meta = letterbox_preprocess(bgr, input_size=640)
        assert abs(meta.scale - 1.0) < 1e-6
        assert meta.orig_h == 480
        assert meta.orig_w == 640

    def test_scale_does_not_upscale_small_images(self) -> None:
        """Images smaller than input_size in both dims must not be upscaled.

        Matches ultralytics==8.2.103's reference validation preprocessing:
        YOLODataset.build_transforms() uses LetterBox(scaleup=False) whenever
        self.augment is False (i.e. validation) — LetterBox.__call__ then
        clamps ``r = min(r, 1.0)`` ("only scale down, do not scale up (for
        better val mAP)"). Without the same clamp, ~1 in 5 COCO val2017
        images (both dims < 640) get upscaled here but would not be by the
        reference, silently diverging from the baseline this study compares
        against.
        """
        # 320×320 image: naive min(640/320, 640/320) = 2.0 would upscale.
        # Clamped to 1.0, scale must stay at 1.0 and new dims must stay 320.
        bgr = np.zeros((320, 320, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((320, 320, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, meta = letterbox_preprocess(bgr, input_size=640)

        assert abs(meta.scale - 1.0) < 1e-6
        resize_call_args = mock_cv2.resize.call_args
        new_w, new_h = resize_call_args[0][1]
        assert (new_w, new_h) == (320, 320)

    def test_padding_offsets_are_symmetric(self) -> None:
        """A landscape image (h < w) is padded vertically; pad_left = 0."""
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, meta = letterbox_preprocess(bgr, input_size=640)
        # New height = 480, pad_top = (640-480)//2 = 80
        assert meta.pad_top == 80
        assert meta.pad_left == 0

    def test_square_image_has_zero_padding(self) -> None:
        """A 640×640 input should need no padding — scale=1.0, pads=0."""
        bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, meta = letterbox_preprocess(bgr, input_size=640)
        assert meta.pad_top == 0
        assert meta.pad_left == 0
        assert abs(meta.scale - 1.0) < 1e-6

    def test_resize_called_with_aspect_preserving_dims(self) -> None:
        """cv2.resize must be called with (new_w, new_h), NOT (640, 640)."""
        # 480×640 → scale=1.0 → resize to (640, 480) = (new_w, new_h)
        bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            letterbox_preprocess(bgr, input_size=640)

        call_args = mock_cv2.resize.call_args
        dsize = call_args.args[1] if len(call_args.args) > 1 else call_args.kwargs.get("dsize")
        # dsize is (new_w, new_h) = (640, 480) — NOT (640, 640)
        assert dsize == (640, 480)

    def test_normalised_values_in_0_1_range(self) -> None:
        """Pixel values must be in [0, 1] after normalisation."""
        bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.full((640, 640, 3), 255, dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.full((640, 640, 3), 255, dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            tensor, _ = letterbox_preprocess(bgr, input_size=640)
        assert tensor.max() <= 1.0 + 1e-6
        assert tensor.min() >= 0.0 - 1e-6


# ---------------------------------------------------------------------------
# preprocess_image — kept as backward-compat wrapper over letterbox_preprocess
# ---------------------------------------------------------------------------

class TestPreprocessImage:
    def test_output_shape_is_nchw(self) -> None:
        """Preprocessed image must be (1, 3, 640, 640) NCHW."""
        bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            result = preprocess_image(bgr)
        assert result.shape == (1, 3, 640, 640)

    def test_output_dtype_is_float32(self) -> None:
        bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            result = preprocess_image(bgr)
        assert result.dtype == np.float32

    def test_output_values_normalised_to_0_1(self) -> None:
        bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.full((640, 640, 3), 255, dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.full((640, 640, 3), 255, dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            result = preprocess_image(bgr)
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.0 - 1e-6

    def test_returns_tensor_only_not_meta(self) -> None:
        """preprocess_image must return np.ndarray, not a tuple."""
        bgr = np.zeros((640, 640, 3), dtype=np.uint8)
        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.resize.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            result = preprocess_image(bgr)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# CocoLoader
# ---------------------------------------------------------------------------

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
        (tmp_path / "000000000001.jpg").touch()  # numeric-stem COCO file
        (tmp_path / "image.png").touch()
        (tmp_path / "readme.txt").touch()

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        assert len(loader) == 1

    def test_image_paths_sorted_deterministically(self, tmp_path: Path) -> None:
        for name in ["000000000002.jpg", "000000000001.jpg", "000000000003.jpg"]:
            (tmp_path / name).write_bytes(b"")

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        names = [Path(p).name for p in loader.image_paths]
        assert names == sorted(names)

    def test_len_excludes_non_numeric_stem_jpg_files(self, tmp_path: Path) -> None:
        """M1 — __len__ must count only numeric-stem .jpg files to match what __iter__ yields.

        Spurious .jpg files (thumbnails, previews, etc.) in the images directory are
        skipped by __iter__ (non-numeric stem → ValueError). If __len__ counts them,
        n_evaluated < len(loader) triggers a RuntimeError on every such directory.
        """
        (tmp_path / "000000000001.jpg").touch()   # valid COCO filename → counted
        (tmp_path / "thumbnail.jpg").touch()       # non-numeric stem → excluded
        (tmp_path / "preview.jpg").touch()          # non-numeric stem → excluded

        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)
        assert len(loader) == 1  # only the numeric-stem file

    # ------------------------------------------------------------------
    # __iter__ — yields (tensor, image_id, LetterboxMeta) 3-tuples
    # ------------------------------------------------------------------

    def test_iter_yields_three_tuples(self, tmp_path: Path) -> None:
        """__iter__ must yield (tensor, image_id, meta) — 3-tuples."""
        (tmp_path / "000000000042.jpg").write_bytes(b"")
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            items = list(loader)

        assert len(items) == 1
        assert len(items[0]) == 3  # (tensor, image_id, meta)

    def test_iter_tensor_shape(self, tmp_path: Path) -> None:
        """Each yielded tensor must have shape (1, 3, 640, 640)."""
        (tmp_path / "000000000001.jpg").write_bytes(b"")
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            tensor, _, _ = next(iter(loader))

        assert tensor.shape == (1, 3, 640, 640)
        assert tensor.dtype == np.float32

    def test_iter_image_id_extracted_from_filename(self, tmp_path: Path) -> None:
        """image_id must be the integer parsed from the COCO filename stem."""
        (tmp_path / "000000001234.jpg").write_bytes(b"")
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, image_id, _ = next(iter(loader))

        assert image_id == 1234

    def test_iter_meta_is_letterbox_meta(self, tmp_path: Path) -> None:
        """Third element of each yielded tuple must be a LetterboxMeta instance."""
        (tmp_path / "000000000001.jpg").write_bytes(b"")
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, _, meta = next(iter(loader))

        assert isinstance(meta, LetterboxMeta)

    def test_iter_meta_carries_original_dimensions(self, tmp_path: Path) -> None:
        """LetterboxMeta must record the original image dimensions."""
        (tmp_path / "000000000001.jpg").write_bytes(b"")
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            _, _, meta = next(iter(loader))

        assert meta.orig_h == 480
        assert meta.orig_w == 640

    def test_iter_order_matches_sorted_paths(self, tmp_path: Path) -> None:
        """Image IDs must be yielded in sorted filename order."""
        for name in ["000000000099.jpg", "000000000042.jpg"]:
            (tmp_path / name).write_bytes(b"")
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            ids = [image_id for _, image_id, _ in loader]

        assert ids == [42, 99]

    def test_iter_skips_unreadable_images(self, tmp_path: Path) -> None:
        """cv2.imread returning None must be skipped, not crash the iteration."""
        for name in ["000000000001.jpg", "000000000002.jpg"]:
            (tmp_path / name).write_bytes(b"")
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        def imread_side_effect(path):
            if "000000000001" in path:
                return None  # simulate corrupted/unreadable image
            return np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.side_effect = imread_side_effect
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            items = list(loader)

        # Image 001 is skipped; image 002 is yielded
        assert len(items) == 1
        _, image_id, _ = items[0]
        assert image_id == 2

    def test_iter_skips_non_numeric_filenames_without_crash(self, tmp_path: Path) -> None:
        """Non-numeric JPEG filename must be skipped gracefully, not raise ValueError."""
        (tmp_path / "thumbnail.jpg").write_bytes(b"")        # non-numeric stem
        (tmp_path / "000000000042.jpg").write_bytes(b"")     # valid COCO filename
        loader = CocoLoader(images_dir=str(tmp_path), annotations_file=None)

        with patch("src.data.coco_loader.cv2") as mock_cv2:
            mock_cv2.imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.resize.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            mock_cv2.cvtColor.return_value = np.zeros((640, 640, 3), dtype=np.uint8)
            mock_cv2.COLOR_BGR2RGB = 4
            items = list(loader)  # must not raise

        # Only the numeric-stem file is yielded
        assert len(items) == 1
        _, image_id, _ = items[0]
        assert image_id == 42
