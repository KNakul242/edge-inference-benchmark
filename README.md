# edge-inference-benchmark

Cross-runtime inference benchmarking for edge-constrained object detection — YOLOv8n exported to ONNX and evaluated across PyTorch CPU, ONNX Runtime CPU EP, and TensorRT at FP32 / FP16 / INT8 precision, with full COCO val2017 mAP evaluation and a deployment decision framework derived from the measured data.

---

## What This Is

A systems engineering study of what happens between a trained perception model and a running system on constrained hardware. The same pretrained YOLOv8n model is exported once to ONNX (opset 17, static 640×640 input) and then evaluated across three inference runtimes — PyTorch CPU baseline, ONNX Runtime with CPU Execution Provider, and TensorRT on a Colab-hosted Tesla T4 — at FP32, FP16, and INT8 precision levels where the hardware supports them. Each runtime × precision combination runs under a fixed benchmark protocol: 100 timed inference passes, 10 warmup passes discarded, with mean, stddev, and p95 latency reported. Accuracy is measured as mAP@0.5:0.95 on the full 5,000-image COCO val2017 set, with delta computed relative to the FP32 baseline of the same runtime.

This is not a model training or accuracy improvement project. The YOLOv8n model is pretrained and fixed — the study begins after the model exists. What is measured is the deployment-relevant behaviour of the same model across different runtime and precision configurations: how latency changes, how accuracy changes, whether the latency distribution is stable enough to plan an SLA around, and what the memory footprint implies for hardware selection. These are the questions that determine whether a perception model is actually deployable on specific hardware, as opposed to merely functional.

---

## Results

> **Hardware coverage:** CPU benchmarks on Fedora Linux 42 (Intel Core Ultra 5 125H). TensorRT benchmarks on Colab-hosted Tesla T4 (TRT 10.16.1.11, CUDA 12.8). Mac M4 / ONNX Runtime + CoreML EP results are absent — hardware was unavailable during the study window. See [Runtime and Hardware Scope](#runtime-and-hardware-scope) for the implication.

> **mAP note:** All mAP values are from this pipeline's evaluation (class-agnostic NMS, eval_conf_threshold=0.001). Published YOLOv8n baseline: 0.372 mAP@0.5:0.95 (Ultralytics, class-aware NMS). The 3.4% gap is structural and constant across all runtimes — it does not affect relative comparisons within this study. See [Benchmark Methodology](#benchmark-methodology).

### Latency and Accuracy

| Runtime | Hardware | Precision | Mean latency | p95 latency | FPS (p95) | mAP@0.5:0.95 | mAP Δ vs FP32 |
|---|---|---|---:|---:|---:|---:|---:|
| PyTorch CPU | Fedora CPU | FP32 | 77.5 ms ⁵ | 95.4 ms ⁵ | 10.5 | 0.3595 | baseline |
| ONNX Runtime CPU EP | Fedora CPU | FP32 | 72.1 ms ⁵ | 81.4 ms ⁵ | 12.3 | 0.3595 | baseline |
| TensorRT | Colab T4 | FP32 | 5.91 ms | 8.48 ms | 118 | 0.3595 | baseline |
| TensorRT | Colab T4 | FP16 | 3.66 ms | 4.25 ms | 235 | 0.3594 | −0.0001 ¹ |
| TensorRT | Colab T4 | INT8 | 3.54 ms | 4.99 ms | 200 | 0.3174 | **−0.042** |

¹ TRT FP16 mAP delta (−0.0001) is below COCOeval measurement noise — sign reverses between the two Colab runs. FP32 and FP16 are 0.3595 for all practical purposes.

⁵ CPU numbers are from Run 3 (2026-05-11), a thermally throttled session. Mean latency ranged 1.7× across three sessions on identical hardware: PyTorch 51–88 ms, ONNX Runtime 43–72 ms. No single session is authoritative without CPU governor pinning; see [Emergent Findings](#emergent-findings) for the thermal characterisation. FPS derived from p95 latency.

### Memory Footprint

| Runtime | Precision | Peak RSS |
|---|---|---|
| PyTorch CPU | FP32 | 376 MB ² |
| ONNX Runtime CPU EP | FP32 | ~460 MB ³ |
| TensorRT | FP32 / FP16 / INT8 | 12.7 MB (I/O buffers only) ⁴ |

² PyTorch CPU RSS confirmed stable to ±1.2% across three independent benchmark sessions (367–376 MB). Inference allocates +3.5 MB marginal memory above the loaded model; no allocation growth per call.

³ ONNX Runtime true steady-state footprint measured in isolation (Run 1, 460 MB). Sequential pipeline runs show 818 MB due to un-reclaimed heap from prior mAP evaluation — `peak_memory_delta_mb = 0.0` in those runs confirms zero marginal inference cost; the inflated RSS is entirely pre-existing allocation. True session footprint: ~460 MB.

⁴ The 12.7 MB reflects PyTorch-allocated I/O tensors (`d_input` + `d_output`). TRT allocates engine weights and activation workspace through its own cudaMalloc pools, which are invisible to `torch.cuda.max_memory_allocated()`. Estimated true VRAM: ~80–130 MB (FP16), ~150–250 MB (FP32), ~50–80 MB (INT8). This is a known measurement gap documented in `docs/benchmark-run-trt-2-findings.md`.

### Latency–Accuracy Tradeoff

![Latency vs mAP tradeoff](results/figures/tradeoff_scatter.png)

*Each point is one runtime × precision combination. X-axis: mean inference latency. Y-axis: mAP delta from FP32 baseline of the same runtime. The ideal operating region is bottom-left (fast and lossless). TRT FP16 sits at the efficient frontier. TRT INT8 offers 3.3% lower mean latency than FP16 but costs 4.2 pp mAP and delivers 17% worse p95 (4.99 ms vs 4.25 ms). For any SLA-constrained deployment, FP16 is the operating point.*

![Inference Latency Under Precision Constraints](results/figures/latency_comparison.png)

---

## Benchmark Methodology

**Protocol:** 100 timed inference passes per runtime × precision combination, preceded by 10 discarded warmup passes. Warmup amortises JIT compilation, CUDA driver initialisation, and first-inference memory allocation — costs that do not recur on subsequent calls and would inflate the first measurement. Reporting mean + stddev + p95: mean characterises typical throughput; p95 characterises the latency percentile a deployed system must budget for under normal call variation. A deployment SLA is written against p95, not mean.

**Timing:** `time.perf_counter()` — not `time.time()`. `perf_counter` provides sub-millisecond monotonic resolution, independent of system clock adjustments. At the 2–8 ms latencies measured for TRT, a 1 ms step in `time.time()` would corrupt individual measurements.

**Batch size:** 1 throughout. Batch size > 1 amortises per-call kernel launch overhead across frames and does not represent the latency experienced by any individual inference call in a single-frame edge pipeline. All reported numbers reflect the deployment-relevant single-frame case.

**Accuracy evaluation:** mAP measured on the full COCO val2017 set (5,000 images). Detection confidence threshold for mAP evaluation: 0.001 (not the deployment threshold of 0.5). COCOeval constructs a precision-recall curve across all submitted confidence scores — submitting only high-confidence detections permanently truncates the curve and suppresses mAP by approximately 27% relative to the corrected measurement (0.263 vs 0.359). This was identified as the primary accuracy measurement error in Run 1 and corrected before subsequent runs. The `eval_conf_threshold` (0.001) and deployment `conf_threshold` (0.5) are separate parameters in `configs/benchmark_config.yaml`.

**mAP vs published baseline:** This pipeline measures 0.3595 mAP@0.5:0.95; Ultralytics reports 0.372 for YOLOv8n on COCO val2017. The 3.4% gap is fully explained by NMS methodology: this pipeline uses class-agnostic NMS (any two overlapping boxes are candidates for suppression regardless of class), while Ultralytics' validator uses per-class NMS. Class-agnostic NMS suppresses some co-localised true positives in crowded multi-object scenes that per-class NMS would retain. The gap is constant across all runtimes — mAP delta comparisons within this study are valid.

**mAP delta:** Always computed relative to the FP32 baseline of the same runtime. TRT INT8 delta is relative to TRT FP32, not PyTorch CPU FP32. Cross-runtime mAP comparison conflates precision cost with runtime cost and is not reported.

**INT8 calibration:** 500 images sampled from COCO val2017 with fixed seed 42. The calibration set was designed to serve both ONNX Runtime (Mac M4, pending) and TensorRT INT8 engines. In the current study, only TRT INT8 calibration was executed. The calibration manifest is committed at `data/calibration/manifest.json` — the exact image set is reproducible without re-running the sampler. Note: using val2017 images for calibration introduces an estimated ~0.001–0.002 mAP optimism for INT8 (correct practice is COCO train2017, which requires an additional 18 GB download). This is documented in the manifest and does not change any qualitative conclusion in this study.

**Iterative methodology:** The benchmark pipeline was run five times total — three CPU sessions on Fedora (May 8, May 8, May 11) and two TRT sessions on Colab (May 13, May 18). Each session produced a findings document in `docs/` recording measurement artefacts identified, fixes applied, and what the incremental run revealed that the previous one did not. The canonical results use Run 3 for CPU and Run 2 for TRT. The per-session findings documents are in docs/ — a local-only directory by project convention, containing working engineering notes that informed subsequent fixes. The canonical result files and the pipeline itself reflect the outcomes of that process.

---

## Runtime and Hardware Scope

| Runtime | Hardware | FP32 | FP16 | INT8 | Status |
|---|---|:---:|:---:|:---:|---|
| PyTorch CPU | Intel Core Ultra 5 125H (Fedora) | ✓ | — | — | Complete |
| ONNX Runtime CPU EP | Intel Core Ultra 5 125H (Fedora) | ✓ | — | — | Complete |
| TensorRT | Tesla T4 (Colab) | ✓ | ✓ | ✓ | Complete |
| ONNX Runtime + CoreML EP | Mac M4 Neural Engine | ✓ | ✓ | ✓ | Not executed — hardware unavailable |
| PyTorch MPS | Mac M4 GPU | ✓ | ✓ | — | Not executed — hardware unavailable |

**Mac M4 / CoreML EP:** The study was designed with three hardware targets. The Mac M4 was not available during the benchmark window. Mac-specific code is implemented in `src/runtimes/onnx_runtime.py` and `src/runtimes/pytorch_runtime.py` (marked `# MAC_REQUIRED:`) and has not been executed. The Mac M4 Neural Engine via CoreML Execution Provider and TensorRT on a Colab T4 represent the same class of problem — hardware-accelerated inference on constrained silicon — with different vendor stacks. TensorRT on Jetson and CoreML EP on Apple Silicon are architecturally equivalent deployment scenarios. The CoreML EP surface is a known gap in the current results, not a design omission.

**TensorRT on Apple Silicon:** TensorRT does not run on Apple Silicon. This is an architecture boundary, not a project limitation. The study benchmarks across hardware classes intentionally: CPU-only baseline, NVIDIA GPU (T4), and Apple Neural Engine targets. The absence of TensorRT on Mac is consistent with this design.

---

## Emergent Findings

These findings were not known before the benchmark ran. Each is supported by specific numbers across multiple sessions.

**Hardware acceleration is not an optimisation at this resolution — it is a prerequisite.** TRT FP16 on a Colab T4 delivers 3.66 ms mean / 4.25 ms p95 at 235 FPS (at p95). ONNX Runtime CPU EP on a Core Ultra 5 125H delivers 72 ms mean / 81 ms p95 at 12.3 FPS (at p95). The gap is 12–20× at mean and 11–19× at p95, depending on CPU thermal state (cold-start Turbo Boost to sustained throttle). For 640×640 YOLOv8n inference, the relevant deployment question is not which CPU runtime to prefer — both fail the 30 FPS real-time threshold at every observed thermal state — but whether CPU is the right hardware class at all.

**Naive mAP evaluation produces plausible-looking wrong numbers and does not crash.** Run 1 measured mAP@0.5:0.95 = 0.263 across all runtimes. The correct value is 0.359. The cause: using the deployment confidence threshold (conf=0.5) for COCOeval detection submission. COCOeval constructs a precision-recall curve from all submitted detections; pre-filtering to conf=0.5 permanently discards the lower half of that curve before COCOeval sees it. The result is a plausible number — not NaN, not zero, in a reasonable range for a detection model — that is completely wrong. The fix required separating `eval_conf_threshold` (0.001) from `conf_threshold` (0.5) at the config level. Any pipeline that uses a deployment-level threshold for mAP evaluation will reproduce this error silently.

**INT8 tail latency is session-dependent on shared GPU infrastructure — and the instability direction is opposite to what the hardware model predicts.** INT8 coefficient of variation jumped from 5.7% (Colab Run 1) to 15.0% (Run 2), while FP16 improved from 14.3% to 6.8% — a complete inversion of the variance hierarchy. The mechanism: INT8's deeply pipelined Tensor Core path achieves its throughput advantage precisely because it is more deeply pipelined. Under a stable boost clock, the pipeline runs without stalls. Under clock frequency variation, deeper pipeline stages incur proportionally larger stall penalties per call — making INT8 the precision most sensitive to GPU clock instability, not the least. The practical consequence: INT8 p95 was 3.02 ms (faster than FP16's 3.56 ms) in Run 1 and 4.99 ms (17% slower than FP16's 4.25 ms) in Run 2. The ordering of these two precisions at the deployment-relevant percentile reversed between sessions. Combined with a −0.042 mAP@0.5:0.95 penalty stable to five significant figures across both runs, INT8 is not the dominant choice at any decision-relevant metric on this hardware and model.

**CPU benchmark numbers from a single session on unpinned hardware are not authoritative.** PyTorch CPU FP32 mean latency across three sessions on identical hardware and code: 51 ms, 77.5 ms, 88 ms — a 1.7× range with no code change between sessions. ONNX Runtime CPU EP ranged 43–72 ms mean across the same sessions. The variation is machine thermal state, not randomness: cold-start Turbo Boost, partial throttle under prior load, and sustained throttle during a long inference run occupy distinct regions of the distribution. The three-session envelope is the honest characterisation. Reporting any single session's number — whether the fastest or the slowest — as representative would be misleading in either direction.

**FP32 → FP16 on TensorRT has no measurable accuracy cost.** TRT FP16 mAP = 0.3594 vs FP32 = 0.3595; the −0.0001 delta reverses sign between the two Colab runs and is below COCOeval summation noise. Latency drops 38% at mean, 50% at p95. FP32 is a dominated option wherever Tensor Core FP16 execution is available. Note: TRT FP32 also exhibits above-threshold variance (CV = 16.6% in Run 2, 25.6% in Run 1), consistent with CUDA-core clock sensitivity; the reported p95 = 8.48 ms carries this uncertainty.

**The ORT vs. PyTorch speed advantage is 7–14% — the 2× figure from Run 1 was a thermally-contaminated measurement.** Run 1 measured ORT at 42.7 ms vs PyTorch at 87.7 ms (2.05×), but PyTorch had hit the thermal throttle floor while ORT ran into a brief recovery window between tests. Under matched thermal conditions the ratio is 1.07–1.14×.

**ONNX export equivalence held to 7 significant figures under a demanding test.** At eval_conf_threshold=0.001, near-marginal anchor outputs (confidence 0.001–0.010) are in play where the 9.77×10⁻⁴ max tensor deviation is on the order of the threshold itself — any systematic activation bias would submit a different detection set. mAP agreed to the seventh decimal across 42 million evaluated anchor outputs.

---

## Deployment Decision Framework

These framing questions emerge from the benchmark data. They do not prescribe a deployment choice for any specific system — they structure what evidence is available and what gaps remain.

**For latency-constrained GPU inference (e.g., robotics, edge compute with NVIDIA hardware):**
TRT FP16 is the reference operating point. Mean latency 3.66 ms, p95 4.25 ms, 235 FPS (at p95), mAP@0.5:0.95 = 0.3594 — statistically indistinguishable from FP32 accuracy. INT8 offers 3.3% additional mean speedup at a cost of −4.2 pp mAP and a less predictable tail (p95 4.99 ms on the second Colab run, 17% slower than FP16 at the same percentile). NVIDIA's own TRT deployment guidance classifies INT8 as appropriate when the accuracy cost is below 1 pp absolute; at −4.2 pp, this study's empirical result exceeds that threshold by 4.2×. For YOLOv8 nano-class models, INT8 quantisation carries more activation sensitivity per parameter than larger models — the accuracy cost is not hypothetical. Unless the target application can tolerate a −11.7% relative mAP reduction and GPU clock frequency can be stabilised, FP16 is the dominant choice.

**For CPU-only edge deployment (no discrete GPU, passively cooled):**
Neither PyTorch CPU nor ONNX Runtime CPU EP delivers real-time inference at 640×640 FP32 on a modern laptop-class CPU under sustained load. ONNX Runtime CPU EP reduces latency 7–14% relative to PyTorch (state-dependent) at the cost of ~85 MB additional steady-state RSS (~460 MB vs ~376 MB). On a device with 512 MB available RAM, ONNX Runtime CPU EP is viable with narrow headroom; on 256 MB, neither runtime is viable in FP32 at this resolution. For real-time CPU inference at 640×640, quantisation (INT8) or resolution reduction (416×416 or lower) would be required — neither is benchmarked in this study at the CPU level. The CPU results establish the baseline; hardware acceleration (Neural Engine, GPU) is required for real-time deployment at this resolution.

**For systems with hard accuracy constraints:**
The INT8 mAP@0.5:0.95 penalty of −0.042 absolute (−11.7% relative) on TRT for YOLOv8n corresponds to a meaningful reduction in detection recall across COCO's 80 classes. The degradation is concentrated at smaller objects and lower-confidence predictions — categories where a nano-class model already operates near its detection floor. In any system where a missed detection has physical consequences, INT8 deployment of YOLOv8n should be evaluated against per-class recall requirements, not assumed safe on the basis of supported precision modes. The mAP number is stable across two independent runs (0.31742 in both) — the accuracy cost is a property of the model and calibration methodology, not noise.

**For memory-budgeted deployment planning:**
CPU memory footprints are characterised: PyTorch CPU FP32 = 376 MB (stable to ±1.2% across three sessions), ONNX Runtime CPU EP = ~460 MB (isolated measurement). TRT VRAM is not characterised — the 12.7 MB figure in the result schema reflects PyTorch I/O tensor allocation only. Before using these results to size GPU memory on embedded targets (Jetson Orin, Xavier), nvidia-smi VRAM measurement before/after engine load must be added to the TRT benchmark cell. The fix is documented and approximately 10 lines of code.

---

## Reproduction

```bash
# Environment — creates conda env with pinned dependencies
bash scripts/setup_environment.sh

# COCO val2017 images and annotations (~1 GB)
bash scripts/download_data.sh

# Export YOLOv8n → ONNX (opset 17, dynamic=False, simplify=True, parity validated)
python scripts/export_model.py

# CPU benchmark suite (PyTorch CPU + ONNX Runtime CPU EP)
python scripts/run_benchmark.py

# TensorRT benchmark — self-contained, run on Colab T4
# notebooks/tensorrt_colab.ipynb

# Post-benchmark analysis and figure generation
jupyter notebook notebooks/results_analysis.ipynb
```

**Pinned versions — Fedora:** Python 3.11.11, PyTorch 2.3.1+cpu, ONNX Runtime 1.18.1, numpy 1.26.4, ultralytics 8.2.103.

**Pinned versions — Colab T4 (Run 2 environment):** TensorRT 10.16.1.11, CUDA 12.8, PyTorch 2.10.0+cu128, ONNX Runtime 1.26.0, numpy 2.0.2, Python 3.12.13.

**INT8 calibration:** Manifest committed at `data/calibration/manifest.json` — exact 500-image set (seed 42, COCO val2017) is reproducible without re-running the sampler.

**Tests:**

```bash
pytest tests/unit/                             # 194 tests — run before every commit
pytest tests/integration/                      # End-to-end pipeline validation
pytest tests/ --cov=src --cov-fail-under=80    # 80% floor; 90%+ on benchmark modules
```

---

## Project Structure

```
edge-inference-benchmark/
├── configs/
│   └── benchmark_config.yaml         # All tunable parameters — eval and deployment
│                                     # thresholds are separate; no magic numbers in src/
├── src/
│   ├── runtimes/
│   │   ├── base_runtime.py           # Abstract interface — name, load, infer, warmup
│   │   ├── pytorch_runtime.py        # CPU (active) + MPS stub (MAC_REQUIRED)
│   │   ├── onnx_runtime.py           # CPU EP (active) + CoreML EP stub (MAC_REQUIRED)
│   │   └── tensorrt_runtime.py       # TRT engine execution (Colab only)
│   ├── benchmark/
│   │   ├── latency_profiler.py       # 100 runs, 10 warmup, perf_counter, p95
│   │   ├── accuracy_evaluator.py     # mAP@0.5:0.95 on full COCO val2017
│   │   └── memory_profiler.py        # RSS + delta_mb; VRAM via torch.cuda
│   ├── data/
│   │   ├── coco_loader.py            # Letterbox preprocessing, COCO ID mapping
│   │   └── calibration_set.py        # 500-image INT8 calibration set, seed=42
│   ├── export/
│   │   └── onnx_exporter.py          # YOLOv8 → ONNX with parity validation
│   └── results/
│       ├── result_schema.py          # BenchmarkResult dataclass — full field set
│       └── result_writer.py          # JSON (one per run) + summary CSV
├── notebooks/
│   ├── tensorrt_colab.ipynb          # Self-contained TRT FP32/FP16/INT8 on Colab T4
│   └── results_analysis.ipynb        # Loads canonical JSON files, produces figures
├── results/
│   └── figures/                      # Committed — latency_comparison, memory_footprint,
│                                     # tradeoff_scatter (PNG)
└── tests/
    ├── unit/                         # One test file per source module (194 tests)
    └── integration/                  # Pipeline validation from input to result schema
```

---

*Benchmark environment: Fedora Linux 42 (Intel Core Ultra 5 125H, CPU-only) for PyTorch and ONNX Runtime; Google Colab Tesla T4 (TRT 10.16.1.11, CUDA 12.8) for TensorRT. Model: YOLOv8n (Ultralytics, pretrained on COCO, 3.2M parameters). Dataset: COCO val2017, 5,000 images, 80 classes.*
