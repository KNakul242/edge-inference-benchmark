"""Unit tests for device info capture."""

from src.utils.device_info import get_device_info


class TestGetDeviceInfo:
    def test_returns_dict(self) -> None:
        info = get_device_info()
        assert isinstance(info, dict)

    def test_has_platform_key(self) -> None:
        info = get_device_info()
        assert "platform" in info
        assert isinstance(info["platform"], str)
        assert len(info["platform"]) > 0

    def test_has_python_version(self) -> None:
        info = get_device_info()
        assert "python_version" in info

    def test_has_numpy_version(self) -> None:
        info = get_device_info()
        assert "numpy_version" in info
        # numpy is installed in this environment
        assert info["numpy_version"] != "not installed"

    def test_has_cpu_count(self) -> None:
        info = get_device_info()
        assert "cpu_count" in info

    def test_has_architecture(self) -> None:
        info = get_device_info()
        assert "architecture" in info

    def test_gpu_key_present(self) -> None:
        info = get_device_info()
        assert "gpu" in info  # value may be "not available" on CPU machines
