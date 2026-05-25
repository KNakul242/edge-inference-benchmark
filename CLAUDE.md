# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A cross-runtime inference benchmarking study for edge-constrained object detection. Exports a pretrained YOLOv8n model to ONNX and evaluates it across three runtimes (PyTorch, ONNX Runtime + CoreML EP, TensorRT) at FP32/FP16/INT8 precision on real hardware. The output is a deployment decision framework, not a model training or accuracy improvement project — this is a **systems engineering study**.

## Commands

```bash
# Environment setup
bash scripts/setup_environment.sh
bash scripts/download_data.sh

# Export YOLOv8n → ONNX
python scripts/export_model.py

# Run full benchmark suite
python scripts/run_benchmark.py

# Tests (TDD — write tests before implementation)
pytest tests/unit/                            # Run before every commit
pytest tests/integration/                     # Run before every PR
pytest tests/ --cov=src --cov-fail-under=80   # Coverage gate (80% min; 90%+ for benchmark modules)

# Run a single test file
pytest tests/unit/test_latency_profiler.py -v
```

## Architecture

**Two-phase pipeline:**

**Phase 1 (primary)** — Benchmark Pipeline:
- Export: `src/export/onnx_exporter.py` — YOLOv8 → ONNX with opset=17, parity validation (atol=1e-4 vs PyTorch)
- Data: `src/data/` — COCO val2017 loading, preprocessing, INT8 calibration set (500 images, seed=42)
- Runtimes: `src/runtimes/` — `BaseRuntime` ABC + three concrete implementations (PyTorch, ONNX Runtime, TensorRT)
- Benchmark: `src/benchmark/` — Latency profiler (100 runs, 10 warmup, mean/stddev/p95), mAP evaluator, memory profiler
- Results: `src/results/` — Typed result dataclass, JSON + CSV writer
- Utils: `src/utils/` — Device info capture, reproducibility (seed=42)

**Phase 2 (if time permits)** — Real-time webcam demo on Mac M4 via ONNX Runtime + CoreML EP.

**Configuration:** All tunable parameters live in `configs/benchmark_config.yaml` and environment variables (`COCO_DATA_DIR`, `MODEL_DIR`, `RESULTS_DIR`, etc.). No hardcoded paths or magic numbers in `src/`.

## Mac M4 — Excluded from Phase 1

**Hardware was unavailable throughout the study window (2026-05-01 to 2026-05-25). This is a permanent gap in Phase 1 results, not a deferral.** Logged as a locked decision in `docs/specs/VISION.md`. Phase 2 (webcam demo) is also not executed.

Mac-specific code stubs remain in place, marked `# MAC_REQUIRED:`. Do not activate or extend them without explicit instruction.

| Component | Location | Status |
|---|---|---|
| CoreML Execution Provider | `src/runtimes/onnx_runtime.py` | Stubbed — not executed |
| PyTorch MPS device | `src/runtimes/pytorch_runtime.py` | Stubbed — not executed |
| FP16 via MPS autocast | `src/runtimes/pytorch_runtime.py` | Stubbed — not executed |
| coremltools==7.2 | `requirements.txt` | Commented out |
| Phase 2 webcam demo | `scripts/webcam_demo.py` | Not started — not in Phase 1 scope |

**Phase 1 complete runtime targets:**
- PyTorch CPU — FP32 (Fedora, three sessions)
- ONNX Runtime CPU EP — FP32 (Fedora, three sessions)
- TensorRT — FP32, FP16, INT8 (Colab T4, two sessions)

**Notebooks:** `notebooks/tensorrt_colab.ipynb` is self-contained for Colab T4. `notebooks/results_analysis.ipynb` runs after all benchmarks complete.

## Locked Technical Decisions

| Parameter | Value | Reason |
|---|---|---|
| Base model | YOLOv8n | Edge model class |
| Dataset | COCO val2017, all 80 classes | Reproducible, standard baseline |
| mAP metric | mAP@0.5:0.95 | COCO standard |
| Benchmark runs | 100 runs, 10 warmup | p95 is deployment-relevant |
| INT8 calibration | 500 COCO images, seed=42 | Reproducible, sufficient coverage |
| ONNX opset | 17 | TensorRT 8.6.x compatibility |
| Input shape | 640×640×3 NCHW, batch=1 | Fixed edge inference |
| Timing | `time.perf_counter()` | Sub-ms resolution |

**Pinned versions:** Python 3.11.x, torch==2.3.1, torchvision==0.18.1, ultralytics==8.2.x, onnxruntime==1.18.x, coremltools==7.2, tensorrt==8.6.x, pytest==8.x, pytest-cov==5.x. Fedora and Mac environments must be identical.

## Development Rules

**Git workflow (strict Gitflow):**
- Never branch from `main` — always branch from `develop`
- Feature branches: `feature/<name>`, release branch: `release/phase-1`
- Commit format: `type(scope): short description` (types: feat, fix, test, refactor, docs, chore, perf)

**TDD protocol:** RED → GREEN → REFACTOR. Write the failing test first.

**Critical tests that must exist:**
- Latency profiler: warmup discarded, p95 correct, uses `perf_counter`, correct count
- Accuracy evaluator: mAP@0.5:0.95, delta relative to FP32 baseline of same runtime
- ONNX exporter: output parity within atol=1e-4, correct opset
- Calibration set: exactly n_images, seed reproducibility, no duplicates
- Result writer: schema integrity, no data loss, ISO timestamps

**Code standards:**
- Every public function/class requires type hints and docstrings
- No `print()` in `src/` — use `logging` (`logger.info/warning/error`)
- All runtimes must inherit from `BaseRuntime` (name, load, infer, warmup)

**Framing guard:** All code, comments, docstrings, and writing must use systems engineering language — "deployment constraints", "inference pipeline", "latency vs accuracy tradeoff", "runtime selection". Never "model speed test", "deep learning project", or "improved accuracy".

## Ground-Truth Documentation

The locked specifications live in `docs/specs/` (gitignored — local only):
- `docs/specs/VISION.md` — Project thread, mental models, phased structure, locked decisions
- `docs/specs/IMPLEMENTATION_SPEC.md` — Full runtime matrix, benchmark protocol, ONNX export pipeline, Colab notebook spec
- `docs/specs/DEVELOPMENT_RULES.md` — Gitflow, TDD protocol, PR checklist, phase gates

When in doubt, these docs are authoritative.
