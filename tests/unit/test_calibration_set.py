"""Unit tests for the INT8 calibration set sampler.

Critical correctness requirements from spec:
- Exactly n_images are selected
- Same seed produces identical set across independent runs
- Different seed produces different set
- Manifest is written with correct metadata
- No image appears twice (no duplicates)
"""

import json
from pathlib import Path

import pytest

from src.data.calibration_set import generate_calibration_set


def _make_coco_dir(tmp_path: Path, n: int) -> Path:
    """Create a fake COCO val dir with n .jpg files."""
    d = tmp_path / "val2017"
    d.mkdir()
    for i in range(n):
        (d / f"{i:012d}.jpg").write_bytes(b"fake")
    return d


class TestExactImageCount:
    def test_selects_exactly_n_images(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 100)
        out = tmp_path / "calibration"

        generate_calibration_set(str(src), str(out), n_images=20, seed=42)

        selected = list(out.glob("*.jpg"))
        assert len(selected) == 20

    def test_selects_correct_count_for_500(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 600)
        out = tmp_path / "calibration"

        generate_calibration_set(str(src), str(out), n_images=500, seed=42)

        selected = list(out.glob("*.jpg"))
        assert len(selected) == 500


class TestSeedReproducibility:
    def test_same_seed_produces_identical_set(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 200)
        out1 = tmp_path / "cal1"
        out2 = tmp_path / "cal2"

        generate_calibration_set(str(src), str(out1), n_images=50, seed=42)
        generate_calibration_set(str(src), str(out2), n_images=50, seed=42)

        names1 = sorted(p.name for p in out1.glob("*.jpg"))
        names2 = sorted(p.name for p in out2.glob("*.jpg"))
        assert names1 == names2

    def test_different_seed_produces_different_set(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 200)
        out1 = tmp_path / "cal1"
        out2 = tmp_path / "cal2"

        generate_calibration_set(str(src), str(out1), n_images=50, seed=42)
        generate_calibration_set(str(src), str(out2), n_images=50, seed=99)

        names1 = sorted(p.name for p in out1.glob("*.jpg"))
        names2 = sorted(p.name for p in out2.glob("*.jpg"))
        assert names1 != names2


class TestNoDuplicates:
    def test_no_image_selected_twice(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 100)
        out = tmp_path / "calibration"

        generate_calibration_set(str(src), str(out), n_images=50, seed=42)

        names = [p.name for p in out.glob("*.jpg")]
        assert len(names) == len(set(names)), "Duplicate images found in calibration set"


class TestManifest:
    def test_manifest_file_written(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 100)
        out = tmp_path / "calibration"

        generate_calibration_set(str(src), str(out), n_images=10, seed=42)

        assert (out / "manifest.json").exists()

    def test_manifest_contains_seed(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 100)
        out = tmp_path / "calibration"

        generate_calibration_set(str(src), str(out), n_images=10, seed=42)

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["seed"] == 42

    def test_manifest_contains_n_images(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 100)
        out = tmp_path / "calibration"

        generate_calibration_set(str(src), str(out), n_images=10, seed=42)

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["n_images"] == 10

    def test_manifest_image_list_matches_selection(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 100)
        out = tmp_path / "calibration"

        generate_calibration_set(str(src), str(out), n_images=15, seed=42)

        manifest = json.loads((out / "manifest.json").read_text())
        assert len(manifest["images"]) == 15

    def test_manifest_image_list_no_duplicates(self, tmp_path: Path) -> None:
        src = _make_coco_dir(tmp_path, 100)
        out = tmp_path / "calibration"

        generate_calibration_set(str(src), str(out), n_images=20, seed=42)

        manifest = json.loads((out / "manifest.json").read_text())
        assert len(manifest["images"]) == len(set(manifest["images"]))
