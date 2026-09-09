# edge-inference-benchmark

Cross-runtime inference benchmarking for edge-constrained object detection — YOLOv8n exported to ONNX and evaluated across PyTorch CPU, ONNX Runtime CPU EP, and TensorRT at FP32 / FP16 / INT8 precision, with full COCO val2017 mAP evaluation and a deployment decision framework derived from the measured data.

---

## What This Is

A systems engineering study of what happens between a trained perception model and a running system on constrained hardware. The same pretrained YOLOv8n model is exported once to ONNX (opset 17, static 640×640 input) and then evaluated across three inference runtimes — PyTorch CPU baseline, ONNX Runtime with CPU Execution Provider, and TensorRT on a Colab-hosted Tesla T4 — at FP32, FP16, and INT8 precision levels where the hardware supports them. Each runtime × precision combination runs under a fixed benchmark protocol: 100 timed inference passes, 10 warmup passes discarded, with mean, stddev, and p95 latency reported. Accuracy is measured as mAP@0.5:0.95 on the full 5,000-image COCO val2017 set, with delta computed relative to the FP32 baseline of the same runtime.

This is not a model training or accuracy improvement project. The YOLOv8n model is pretrained and fixed — the study begins after the model exists. What is measured is the deployment-relevant behaviour of the same model across different runtime and precision configurations: how latency changes, how accuracy changes, whether the latency distribution is stable enough to plan an SLA around, and what the memory footprint implies for hardware selection. These are the questions that determine whether a perception model is actually deployable on specific hardware, as opposed to merely functional.

---

## Results

> **Hardware coverage:** CPU benchmarks on Fedora Linux 42 (Intel Core Ultra 5 125H). TensorRT benchmarks on Colab-hosted Tesla T4 (TRT 10.16.1.11, CUDA 12.8). Mac M5 (Apple M5 MacBook Air) benchmarks completed 2026-08-08 for PyTorch MPS (FP32, FP16) and ONNX Runtime + CoreML EP (FP32) — see [Runtime and Hardware Scope](#runtime-and-hardware-scope) for what remains unbuilt (CoreML EP FP16/INT8).

> **mAP note:** All mAP values are from this pipeline's evaluation (class-agnostic NMS, eval_conf_threshold=0.001). Published YOLOv8n baseline: 0.372 mAP@0.5:0.95 (Ultralytics, class-aware NMS). The 3.4% gap is structural and constant across all runtimes — it does not affect relative comparisons within this study. See [Benchmark Methodology](#benchmark-methodology).

### Latency and Accuracy

| Runtime | Hardware | Precision | Mean latency | p95 latency | FPS (p95) | mAP@0.5:0.95 | mAP Δ vs FP32 |
|---|---|---|---:|---:|---:|---:|---:|
| PyTorch CPU | Fedora CPU | FP32 | 77.5 ms ⁵ | 95.4 ms ⁵ | 10.5 | 0.3595 | baseline |
| ONNX Runtime CPU EP | Fedora CPU | FP32 | 72.1 ms ⁵ | 81.4 ms ⁵ | 12.3 | 0.3595 | baseline |
| TensorRT | Colab T4 | FP32 | 5.91 ms | 8.48 ms | 118 | 0.3595 | baseline |
| TensorRT | Colab T4 | FP16 | 3.66 ms | 4.25 ms | 235 | 0.3594 | −0.0001 ¹ |
| TensorRT | Colab T4 | INT8 | 3.54 ms | 4.99 ms | 200 | 0.3174 | **−0.042** |
| PyTorch MPS | Mac M5 | FP32 | 7.41 ms | 7.78 ms | 129 | 0.3595 | baseline |
| PyTorch MPS | Mac M5 | FP16 | 7.53 ms | 7.97 ms | 126 | 0.3595 | −0.0000 ⁶ |
| ONNX Runtime + CoreML EP | Mac M5 | FP32 | 8.24 ms | 9.35 ms | 107 | 0.3594 ⁸ | baseline |

¹ TRT FP16 mAP delta (−0.0001) is below COCOeval measurement noise — sign reverses between the two Colab runs. FP32 and FP16 are 0.3595 for all practical purposes.

⁵ CPU numbers are from Run 3 (2026-05-11), a thermally throttled session. Mean latency ranged 1.7× across three sessions on identical hardware: PyTorch 51–88 ms, ONNX Runtime 43–72 ms. No single session is authoritative without CPU governor pinning; see [Emergent Findings](#emergent-findings) for the thermal characterisation. FPS derived from p95 latency.

⁶ Mac MPS FP16 mAP delta (−0.0000384) is below COCOeval measurement noise — smaller than TRT's FP16 delta and consistent with FP16 having no measurable accuracy cost on this model at either GPU-class hardware target. FP16 here is a true full-precision cast (`.half()` on model and input), not autocast mixed precision — see `docs/issue-log/2026-08-07-mps-autocast-unsupported.md`.

⁸ Unlike every other FP32 result in this study, `onnx_coreml_fp32`'s mAP (0.35942) sits 7.7×10⁻⁵ away from the cross-runtime FP32 cluster (~0.35950) — two orders of magnitude beyond this study's own established floating-point noise band (10⁻⁶–10⁻⁸, established across the other four FP32 results). Too small to change any conclusion in this study, but flagged rather than silently rounded away: possibly attributable to CoreML's default model conversion format, not directly confirmed. See `docs/benchmark-run-1-mac-findings.md`, Issue 3.

### Memory Footprint

| Runtime | Precision | Peak RSS |
|---|---|---|
| PyTorch CPU | FP32 | 376 MB ² |
| ONNX Runtime CPU EP | FP32 | ~460 MB ³ |
| TensorRT | FP32 / FP16 / INT8 | 12.7 MB (I/O buffers only) ⁴ |
| PyTorch MPS | FP32 | 378 MB |
| PyTorch MPS | FP16 | 432 MB |
| ONNX Runtime + CoreML EP | FP32 | 799 MB ⁷ |

² PyTorch CPU RSS confirmed stable to ±1.2% across three independent benchmark sessions (367–376 MB). Inference allocates +3.5 MB marginal memory above the loaded model; no allocation growth per call.

³ ONNX Runtime true steady-state footprint measured in isolation (Run 1, 460 MB). Sequential pipeline runs show 818 MB due to un-reclaimed heap from prior mAP evaluation — `peak_memory_delta_mb = 0.0` in those runs confirms zero marginal inference cost; the inflated RSS is entirely pre-existing allocation. True session footprint: ~460 MB.

⁴ The 12.7 MB reflects PyTorch-allocated I/O tensors (`d_input` + `d_output`). TRT allocates engine weights and activation workspace through its own cudaMalloc pools, which are invisible to `torch.cuda.max_memory_allocated()`. Estimated true VRAM: ~80–130 MB (FP16), ~150–250 MB (FP32), ~50–80 MB (INT8). This is a known measurement gap documented in `docs/benchmark-run-trt-2-findings.md`.

⁷ ONNX Runtime + CoreML EP's 799 MB is notably higher than the CPU EP's ~460 MB for the same FP32 model. CoreML EP partitions the graph at load time (confirmed in logs: 11 partitions covering 221 of 233 nodes, with the remaining 12 nodes falling back to CPU EP) — the partition/fallback stitching and CoreML's own compiled-model representation both plausibly add overhead the single-EP CPU path doesn't incur, though this hasn't been decomposed further (see `docs/benchmark-run-1-mac-findings.md`, Observation C for the fuller picture, including a possible reduced-precision internal representation that hasn't been directly confirmed).

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

**mAP vs published baseline:** This pipeline measures 0.3595 mAP@0.5:0.95; Ultralytics reports 0.372 for YOLOv8n on COCO val2017. Part of the 3.4% gap is explained by NMS methodology: this pipeline uses class-agnostic NMS (any two overlapping boxes are candidates for suppression regardless of class), while Ultralytics' validator uses per-class NMS by default — class-agnostic NMS suppresses some co-localised true positives in crowded multi-object scenes that per-class NMS would retain. **Correction (2026-09-09):** the gap was previously described as "fully explained" by NMS alone — that was incomplete. A DS review found `letterbox_preprocess()` (`src/data/coco_loader.py`) had no upscale clamp, diverging from the reference's `LetterBox(scaleup=False)` for the ~1-in-5 COCO val2017 images smaller than 640px in both dimensions; this has been fixed (test-first), but every canonical result in this table was generated before the fix and needs re-measurement to isolate how much of the 3.4% gap is NMS versus preprocessing. The gap is still constant across all runtimes measured with the *same* (pre-fix) pipeline, so relative comparisons within this study remain valid — the open question is the absolute gap's composition, not the internal consistency of the comparisons.

**mAP delta:** Always computed relative to the FP32 baseline of the same runtime. TRT INT8 delta is relative to TRT FP32, not PyTorch CPU FP32. Cross-runtime mAP comparison conflates precision cost with runtime cost and is not reported.

**INT8 calibration:** 500 images sampled from COCO val2017 with fixed seed 42. The calibration set sampler is generic and was designed to serve both ONNX Runtime and TensorRT INT8 engines; only TRT INT8 calibration was executed. ONNX Runtime INT8 requires a static-quantization step that was never built — a gap independent of Mac hardware availability, tracked as a follow-up (see [Runtime and Hardware Scope](#runtime-and-hardware-scope)). The calibration manifest is committed at `data/calibration/manifest.json` — the exact image set is reproducible without re-running the sampler. Note: using val2017 images for calibration introduces an estimated ~0.001–0.002 mAP optimism for INT8 (correct practice is COCO train2017, which requires an additional 18 GB download). This is documented in the manifest and does not change any qualitative conclusion in this study.

**Iterative methodology:** The benchmark pipeline was run five times total — three CPU sessions on Fedora (May 8, May 8, May 11) and two TRT sessions on Colab (May 13, May 18). Each session produced a findings document in `docs/` recording measurement artefacts identified, fixes applied, and what the incremental run revealed that the previous one did not. The canonical results use Run 3 for CPU and Run 2 for TRT. The per-session findings documents are in docs/ — a local-only directory by project convention, containing working engineering notes that informed subsequent fixes. The canonical result files and the pipeline itself reflect the outcomes of that process.

---

## Runtime and Hardware Scope

| Runtime | Hardware | FP32 | FP16 | INT8 | Status |
|---|---|:---:|:---:|:---:|---|
| PyTorch CPU | Intel Core Ultra 5 125H (Fedora) | ✓ | — | — | Complete |
| ONNX Runtime CPU EP | Intel Core Ultra 5 125H (Fedora) | ✓ | — | — | Complete |
| TensorRT | Tesla T4 (Colab) | ✓ | ✓ | ✓ | Complete |
| PyTorch MPS | Apple M5 (Mac) | ✓ | ✓ | — | Complete (2026-08-08) |
| ONNX Runtime + CoreML EP | Apple M5 (Mac) | ✓ | — | — | FP32 complete (2026-08-08); FP16/INT8 need a static-quantization pipeline that was never built — a gap independent of Mac availability, tracked as a follow-up |

**Mac M5 / CoreML EP:** The study was designed with three hardware targets. The Mac was not available during the original benchmark window (2026-05-01–2026-05-25) and became available 2026-08-05 (Apple M5 MacBook Air). PyTorch MPS (FP32, FP16) and ONNX Runtime + CoreML EP (FP32) are now benchmarked. Mac-specific code in `src/runtimes/onnx_runtime.py` and `src/runtimes/pytorch_runtime.py` (formerly marked `# MAC_REQUIRED:`) is active. TensorRT on Jetson and CoreML EP on Apple Silicon are architecturally comparable deployment scenarios in principle — both are vendor-specific accelerator paths — but a follow-up audit (`docs/benchmark-run-1-mac-findings.md`, Issue 5) found that this study's measurements do not confirm CoreML EP is engaging Apple's GPU or Neural Engine for this model: a repeated, multi-round `MLComputeUnits` comparison found no reproducible latency difference between requesting `CPUOnly`, `CPUAndGPU`, `CPUAndNeuralEngine`, or `ALL`, even though CoreML EP as a whole is genuinely ~4.4× faster than ONNX Runtime's generic `CPUExecutionProvider` (attributable to Apple's optimized Accelerate/BNNS CPU kernels, confirmed via a dedicated control). Read as "a fast, CPU-optimized ONNX Runtime path on Apple Silicon," not as confirmed Neural Engine inference, until that's verified directly (e.g. `powermetrics` hardware counters — not done in this study). FP16 on MPS uses explicit `.half()` casting rather than `torch.autocast` — the latter does not support `device_type="mps"` on the pinned `torch==2.3.1` (see `docs/issue-log/2026-08-07-mps-autocast-unsupported.md`).

**TensorRT on Apple Silicon:** TensorRT does not run on Apple Silicon. This is an architecture boundary, not a project limitation. The study benchmarks across hardware classes intentionally: CPU-only baseline, NVIDIA GPU (T4), and Apple Silicon (M5). The absence of TensorRT on Mac is consistent with this design.

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

**Apple Silicon is competitive with a discrete NVIDIA GPU for this model class — and FP16 provides no latency benefit on MPS, unlike TensorRT.** PyTorch MPS FP32 delivers 7.41 ms mean / 7.78 ms p95 (129 FPS at p95) — 10.5× faster than CPU FP32 (77.5 ms) and within 1.25× of TRT FP32 on a discrete T4 GPU (5.91 ms). ONNX Runtime + CoreML EP FP32 is slightly slower at 8.24 ms mean. A follow-up audit (`docs/benchmark-run-1-mac-findings.md`) investigated why a dedicated accelerator path would be slower than a generic GPU-compute path (MPS) and found the graph-partitioning explanation incomplete: a repeated, multi-round test comparing CoreML EP's `MLComputeUnits` settings (`CPUOnly` vs `CPUAndGPU` vs `CPUAndNeuralEngine` vs `ALL`) found no reproducible latency difference between any of them, even though CoreML EP as a whole is genuinely ~4.4× faster than ONNX Runtime's generic CPU execution provider. The likely explanation is that CoreML EP's advantage here comes from Apple's optimized Accelerate/BNNS CPU kernels, not from Neural Engine or GPU engagement — meaning the latency gap versus raw MPS (which does use the GPU) may reflect "optimized CPU path vs. GPU path" rather than "CoreML acceleration overhead." Not confirmed via hardware counters (would need `powermetrics`/Instruments); see the findings doc for the full methodology and what would resolve it definitively. Unlike TensorRT, where FP16 cuts mean latency 38% relative to FP32, MPS FP16 shows no latency improvement (7.53 ms vs 7.41 ms mean — within measurement noise, if anything marginally slower) despite an identical accuracy profile (mAP delta −0.0000384, noise-level, smaller than TRT's own FP16 delta). For a nano-class model at 640×640, Apple Silicon's Metal shader path does not appear to be the same kind of throughput bottleneck NVIDIA's Tensor Cores relieve — FP16 casting overhead may offset any compute-side gain at this scale. Not further decomposed in this study (no Metal-level profiling); a candidate follow-up.

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

**For Apple Silicon / on-device edge deployment (products built on Apple hardware — iOS/macOS-embedded perception, robotics with an Apple SoC):**
PyTorch MPS FP32 (7.41 ms mean, 129 FPS at p95) and ONNX Runtime + CoreML EP FP32 (8.24 ms mean, 107 FPS at p95) both clear real-time thresholds comfortably at this resolution, with mAP@0.5:0.95 = 0.3595 / 0.3594 — statistically the same as the Fedora CPU and TRT FP32 baselines. Unlike the NVIDIA path, FP16 provides no measured latency benefit on MPS in this study (7.53 ms mean, effectively identical to FP32 within noise) — FP32 is the simpler operating point on Apple Silicon at this model size unless a specific memory or power constraint favours FP16's smaller weight footprint. INT8 on this hardware path remains unbuilt — no ONNX Runtime static-quantization pipeline exists in this codebase yet, a gap independent of Mac hardware availability (see Runtime and Hardware Scope) — so a deployment requiring INT8-class throughput on Apple Silicon cannot be evaluated from this study's data. CoreML EP's higher steady-state memory (799 MB vs MPS's 378 MB for the identical FP32 model) should factor into device memory budgeting alongside the raw latency numbers; a deployment choosing between PyTorch MPS and ONNX Runtime + CoreML EP on Apple Silicon should currently default to PyTorch MPS unless CoreML-specific tooling (Core ML model deployment on iOS, for instance) is otherwise required — this study's own measurements do not confirm CoreML EP is deriving a Neural-Engine-specific benefit over MPS's direct GPU path (see Runtime and Hardware Scope). Scope note: this is specific to YOLOv8n's small, few-GFLOPs compute profile at batch=1 and to this ONNX export's graph (12 of 233 nodes fall back to CPU regardless of compute-unit setting) — it should not be generalized to larger models or fully-supported graphs, where fixed accelerator dispatch overhead is a smaller fraction of total compute time and the calculus may differ.

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

# Mac benchmark suite (PyTorch MPS + ONNX Runtime CoreML EP) — run on Apple Silicon
# --runtime/--precision filters target only these, without re-running (and
# overwriting) the canonical Fedora CPU result files under the same filenames
python scripts/run_benchmark.py --runtime pytorch_mps --runtime onnx_coreml

# TensorRT benchmark — self-contained, run on Colab T4
# notebooks/tensorrt_colab.ipynb

# Post-benchmark analysis and figure generation
jupyter notebook notebooks/results_analysis.ipynb
```

**Pinned versions — Fedora:** Python 3.11.11, PyTorch 2.3.1+cpu, ONNX Runtime 1.18.1, numpy 1.26.4, ultralytics 8.2.103.

**Pinned versions — Colab T4 (Run 2 environment):** TensorRT 10.16.1.11, CUDA 12.8, PyTorch 2.10.0+cu128, ONNX Runtime 1.26.0, numpy 2.0.2, Python 3.12.13.

**Pinned versions — Mac M5 (actual benchmark environment, 2026-08-08):** Python 3.11.15, PyTorch 2.3.1, ONNX Runtime 1.18.1, numpy 1.26.4, ultralytics 8.2.103, coremltools 7.2. Same torch/onnxruntime/numpy/ultralytics pins as Fedora, per the project's cross-environment version-parity requirement. Note: `coremltools==7.2` has not been officially tested against `torch==2.3.1` upstream (most recently tested against 2.2.0) — no issues observed in this study, flagged for awareness.

**INT8 calibration:** Manifest committed at `data/calibration/manifest.json` — exact 500-image set (seed 42, COCO val2017) is reproducible without re-running the sampler.

**Tests:**

```bash
pytest tests/unit/                             # 225 tests — run before every commit
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
│   │   ├── pytorch_runtime.py        # CPU + MPS (FP32/FP16, explicit .half() cast) — active
│   │   ├── onnx_runtime.py           # CPU EP + CoreML EP (FP32) — active; FP16/INT8 not built
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
    ├── unit/                         # One test file per source module (225 tests)
    └── integration/                  # Pipeline validation from input to result schema
```

---

*Benchmark environment: Fedora Linux 42 (Intel Core Ultra 5 125H, CPU-only) for PyTorch and ONNX Runtime; Google Colab Tesla T4 (TRT 10.16.1.11, CUDA 12.8) for TensorRT; Apple M5 MacBook Air for PyTorch MPS and ONNX Runtime + CoreML EP. Model: YOLOv8n (Ultralytics, pretrained on COCO, 3.2M parameters). Dataset: COCO val2017, 5,000 images, 80 classes.*
