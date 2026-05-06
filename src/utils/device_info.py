"""Hardware and runtime environment capture.

Embeds device info into every BenchmarkResult so results are self-describing
and reproducible annotations survive long after the run completes.
"""

import logging
import platform
import subprocess
import sys

logger = logging.getLogger(__name__)


def get_device_info() -> dict:
    """Capture hardware and runtime environment metadata.

    Embedded in every BenchmarkResult.hardware_info field so benchmark
    numbers are always traceable to the exact environment that produced them.

    Returns:
        Dict containing OS, CPU, Python version, and available runtime versions.
    """
    info: dict = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python_version": sys.version,
        "architecture": platform.machine(),
    }

    # Runtime versions — guarded so missing packages don't crash
    for pkg, attr in [
        ("torch", "__version__"),
        ("onnxruntime", "__version__"),
        ("numpy", "__version__"),
        ("ultralytics", "__version__"),
    ]:
        try:
            mod = __import__(pkg)
            info[f"{pkg}_version"] = getattr(mod, attr, "unknown")
        except ImportError:
            info[f"{pkg}_version"] = "not installed"

    # CPU core count
    try:
        import os
        info["cpu_count"] = os.cpu_count()
    except Exception:
        info["cpu_count"] = "unknown"

    # GPU info via nvidia-smi if available (Colab T4)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            info["gpu"] = result.stdout.strip()
    except Exception:
        info["gpu"] = "not available"

    logger.info("Device info captured: %s", info.get("platform"))
    return info
