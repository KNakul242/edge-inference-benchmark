# Resume Insights — Edge Inference Benchmark

*Structured input document for a resume agent. Not the resume bullets themselves — the exact, defensible claims and framing context the agent needs to write two sharp bullets.*

---

## Section 1 — Project Summary

A cross-runtime inference benchmarking study of YOLOv8n (pretrained, fixed) exported to ONNX and evaluated across PyTorch CPU, ONNX Runtime CPU Execution Provider, and TensorRT at FP32 / FP16 / INT8 precision. Benchmark protocol: 100 timed inference passes per runtime × precision combination, 10 warmup passes discarded, reporting mean + stddev + p95 latency via `time.perf_counter()`. Accuracy measured as mAP@0.5:0.95 on the full 5,000-image COCO val2017 set, delta computed relative to the FP32 baseline of the same runtime. Hardware: Intel Core Ultra 5 125H on Fedora Linux (CPU benchmarks), Tesla T4 on Colab (TensorRT). The study ran five sessions total — three CPU and two Colab — each producing a findings document recording identified artefacts, fixes applied, and incremental findings. Canonical results: PyTorch CPU FP32 77.5 ms mean / 95.4 ms p95 (thermally throttled; range: 51–88 ms mean across three sessions), ONNX Runtime CPU FP32 72.1 ms mean / 81.4 ms p95 (range: 43–72 ms mean), TRT FP16 3.66 ms mean / 4.25 ms p95 at mAP@0.5:0.95 = 0.3594. Mac M4 / CoreML EP surface was designed into the project but not executed — hardware was unavailable during the study window.

---

## Section 2 — Technical Substance

**Benchmark protocol decisions (each deliberate):**
- 100 timed runs, 10 warmup discarded. Warmup amortises JIT compilation, CUDA driver init, and first-inference memory allocation. Reporting p95 — not mean — because a deployment SLA is designed around the tail of the latency distribution, not the average.
- `time.perf_counter()` not `time.time()`. At TRT latencies of 2–8 ms, a 1 ms system clock adjustment step corrupts individual measurements.
- Batch size 1 throughout. Batch > 1 amortises per-call overhead and does not represent single-frame edge inference latency.
- `eval_conf_threshold = 0.001` separated from `deployment conf_threshold = 0.5`. Discovered in Run 1 that using conf=0.5 for mAP evaluation permanently truncates the precision-recall curve — COCOeval constructs the curve from all submitted detections and cannot recover what was never submitted. Effect: mAP suppressed from correct 0.359 to 0.263 (−27% relative to the corrected pipeline result). Fix required understanding what COCOeval actually computes, not just running it.

**Hardware acceleration gap (headline cross-runtime comparison):**
- TRT FP16 (Colab T4): 3.66 ms mean, 4.25 ms p95, 235 FPS (at p95)
- ONNX Runtime CPU EP (Core Ultra 5 125H): 72.1 ms mean, 81.4 ms p95, 12.3 FPS (at p95)
- Speedup: 12–20× at mean, 11–19× at p95 (range across CPU thermal states — throttled to cold-start). The relevant framing for robotics deployment: this is not a tuning question, it is a hardware class question.

**Exact benchmark results (all from canonical result files):**
- TRT FP32: 5.91 ms mean, 8.48 ms p95, 118 FPS (at p95), mAP@0.5:0.95 = 0.3595; CV = 16.6% (Run 2), 25.6% (Run 1) — p95 carries meaningful uncertainty
- TRT FP16: 3.66 ms mean, 4.25 ms p95, 235 FPS (at p95), mAP = 0.3594, Δ = −0.0001 (below measurement noise — indistinguishable from FP32 in both runs)
- TRT INT8: 3.54 ms mean, 4.99 ms p95, 200 FPS (at p95), mAP = 0.3174, Δ = −0.042 (confirmed stable across two independent Colab runs)
- PyTorch CPU FP32: 77.5 ms mean, 95.4 ms p95, 10.5 FPS (at p95), mAP = 0.3595 (Run 3, thermally throttled; range across three sessions: 51–88 ms mean)
- ONNX Runtime CPU FP32: 72.1 ms mean, 81.4 ms p95, 12.3 FPS (at p95), mAP = 0.3595 (Run 3 thermally throttled; range: 43–72 ms mean)

**INT8 mAP delta at each runtime:**
- TensorRT INT8: Δ = −0.042 mAP@0.5:0.95 (−11.7% relative vs TRT FP32 baseline), stable to five significant figures across two independent Colab runs (0.31742 in both). ONNX Runtime CPU INT8 and PyTorch CPU INT8 were not benchmarked — CPU precision ablations ran FP32 only.

**Memory footprint (characterised):**
- PyTorch CPU FP32: 376 MB RSS, confirmed stable to ±1.2% across three sessions. Inference allocates +3.5 MB marginal memory; no per-call growth.
- ONNX Runtime CPU FP32: ~460 MB true steady-state (isolated Run 1 measurement). Sequential pipeline runs contaminated to 818 MB by un-reclaimed mAP evaluation heap from prior runtime; `peak_memory_delta_mb = 0.0` in those runs makes the contamination self-evident in the result JSON.
- TRT VRAM: Not characterised. Measured 12.7 MB (I/O tensors only, PyTorch allocator); engine VRAM (~80–250 MB) allocated through TRT's internal cudaMalloc pools, invisible to `torch.cuda.max_memory_allocated()`. Known gap, documented with fix.

**INT8 calibration methodology:**
- 500 images sampled from COCO val2017, fixed seed 42. Calibration manifest committed to repo (`data/calibration/manifest.json`) — exact image set reproducible without the full COCO dataset on hand. Same 500-image set used for TensorRT INT8 calibration; also designed for a future ONNX Runtime INT8 calibration run (Mac M4, pending). Only TRT INT8 calibration was executed in this study. Calibration draws from the evaluation set (val2017 not train2017), introducing an estimated ~0.001–0.002 mAP optimism for INT8 — documented in the manifest; does not affect qualitative conclusions.

**Runtime abstraction:**
- `BaseRuntime` ABC enforces a four-method interface (`name`, `load`, `infer`, `warmup`) across PyTorch, ONNX Runtime, and TensorRT implementations. The latency profiler and accuracy evaluator operate on `BaseRuntime` — adding a new runtime requires implementing the ABC, not modifying the benchmark pipeline. CoreML EP and PyTorch MPS stubs are implemented and marked `# MAC_REQUIRED:`.

**Test coverage:**
- 194 unit tests (after Run 3 pipeline), 80% coverage floor, 90%+ on `latency_profiler.py`, `accuracy_evaluator.py`, and `result_writer.py`. The 90%+ requirement on benchmark-critical modules is explicit in the project rules: bugs in these modules produce silently wrong numbers, which is the worst failure mode in a benchmarking study. Unit tests verify warmup pass discarding, p95 calculation correctness, `perf_counter` usage, run count accuracy, mAP delta cross-runtime guard, and result schema field completeness.

**Iterative benchmark methodology:**
- Five benchmark sessions total (three CPU on Fedora, two TRT on Colab). Each session produced a structured findings document recording: what was wrong, what was fixed, what the incremental run revealed that the prior run did not. This is not three re-runs of the same thing — each session built on fixes from a DS review cycle applied between sessions. The Run 1 mAP artefact (conf_threshold misuse), the Run 2 ONNX Runtime memory contamination artefact (un-reclaimed eval heap), and the Run 3 thermal characterisation were each visible only because multiple sessions were run and compared.

---

## Section 3 — Emergent Insights

The most significant finding — and the one that required two Colab sessions to confirm — is that TRT INT8's variance behaviour is opposite to what the hardware execution model predicts under the wrong conditions. Run 1 on Colab showed INT8 as the most stable precision: CV = 5.7% vs FP16's 14.3%, p95/mean ratio 1.08×. This aligned with the expected hardware model — deeply pipelined INT8 Tensor Core operations should have the lowest per-call variance. Run 2, on a different Colab instance, showed INT8 as the most *unstable* precision: CV = 15.0% vs FP16's 6.8%, p95/mean ratio 1.41× vs FP16's 1.16×. The variance hierarchy fully inverted. The explanation is that deeper pipelining creates more pipeline stages that can stall under GPU clock frequency variation — INT8's throughput advantage over FP16 depends on a stable sustained boost clock. On shared Colab infrastructure without clock pinning, the INT8 path is more sensitive to instance thermal state, not less. The practical consequence is direct: INT8's p95 is 4.99 ms vs FP16's 4.25 ms in the second session. A system with a 5 ms latency SLA passes FP16 and fails INT8 — and this was not predictable from Run 1 alone.

The CPU thermal characterisation also required multiple sessions to emerge. Run 1 (PyTorch mean 87.7 ms) and Run 2 (PyTorch mean 51.0 ms) were produced on identical hardware, identical code, with no configuration change. The 41% difference is machine thermal state — cold-start Turbo Boost vs thermally throttled sustained load. A single benchmark session on a laptop-class CPU without governor pinning produces a number that could be anywhere in a 1.7× range; reporting it as "the latency" is misleading in either direction. The study characterises the performance envelope across three sessions — cold-start (51 ms), partially throttled (77.5 ms), sustained throttle (87.7 ms) — rather than presenting a single number as authoritative. Neither extreme clears 30 FPS. This also establishes what the measured hardware acceleration gap (12–20×, state-dependent) means in practice: the CPU surface isn't slow in the sense of needing optimisation — it's slow in the sense of being the wrong hardware class for this inference workload at this resolution. The gap is architectural, not addressable by runtime selection alone.

The accuracy measurement correction (conf_threshold separation) was foundational before any precision comparison could be trusted. Run 1 reported mAP = 0.263 across all runtimes using conf=0.5 for COCOeval submission. This appeared to confirm that both runtimes were identically wrong — which they were, but for the wrong reason. The issue was that COCOeval expects to receive detections at all confidence levels so it can construct a full precision-recall curve internally; pre-filtering to conf=0.5 discards the lower half of the curve before COCOeval sees it. The fix required separating eval_conf_threshold (0.001) from deployment conf_threshold (0.5) at the config level and understanding what COCOeval actually computes. After the fix, mAP recovered to 0.3594–0.3595 across all runtimes — and the agreement across runtimes (to 7 significant figures at conf=0.001, where 42 million anchor outputs are evaluated) provided a strong numerical validation that the ONNX export is equivalent to PyTorch inference.

---

## Section 4 — Framing Context for Resume Agent

**Target role context:** ML engineer transitioning into robotics and autonomy systems. The value signal this project carries is not "can run a model" — it is "understands what happens between a trained model and a running system." The project demonstrates deployment-level reasoning: runtime selection under hardware constraints, precision tradeoff quantification, measurement discipline, and the ability to identify and correct methodology errors before they silently corrupt results.

**The two claims this project best supports:**

(1) Can instrument and reason about inference performance across heterogeneous hardware and runtime stacks. The evidence is the benchmark itself: three runtimes, three precisions, two hardware classes, five benchmark sessions, with latency measurement discipline (p95, perf_counter, warmup, batch=1) and accuracy measurement that required identifying and correcting a non-obvious evaluation methodology error. The finding that INT8 tail latency is session-dependent on shared GPU infrastructure — confirmed by two Colab runs showing variance hierarchy inversion — is the kind of result that requires knowing what to measure and why, not just running a model.

(2) Applies engineering discipline to ML systems work — not just to application code. The evidence: TDD with 194 unit tests and 90%+ coverage on benchmark-critical modules (explicit policy, not incidental); iterative benchmark methodology with structured findings documents and DS review cycles between sessions; typed result schema (BenchmarkResult dataclass) with all fields required; config-driven parameters with no magic numbers in src/; BaseRuntime ABC enforcing interface consistency across all runtimes; and a calibration manifest committed to the repo so INT8 results are reproducible. These are the practices of someone who knows that ML systems bugs are often silent — the code runs and produces plausible-looking numbers that are wrong.

**What to avoid in the bullets:**
- Any framing involving percentage improvement in model accuracy — this study does not improve model accuracy
- "Optimised model performance" or similar — the model is pretrained and fixed
- Data science portfolio language: "analysed results", "trained a YOLOv8 model", "achieved X% accuracy"
- Inflated speedup claims — the 2.05× ORT-vs-PyTorch figure from Run 1 was an artefact; the correct number is 7–14%
- "Built a fast inference pipeline" — the pipeline is a benchmarking tool, not a production serving system

**Where the strongest signal is:**
The benchmark methodology decisions — eval_conf_threshold separation (the non-obvious one), p95 over mean, perf_counter, batch=1, warmup protocol — and the emergent findings derived from comparing multiple sessions. These separate someone who ran a model from someone who thought carefully about how to produce trustworthy numbers and what the numbers mean for deployment decisions.

**On the Mac M4 gap:**
The absence of CoreML EP results does not weaken the project's signal. The CPU-to-TRT characterisation is complete and rigorous. The Mac M4 surface is designed and stubbed, which shows the architecture was built for it. The gap is an honest hardware availability constraint, not a capability gap. Do not omit it from the bullet or description — state it plainly as "Fedora CPU and Colab T4 benchmarked; Mac M4 / CoreML EP surface pending hardware."

---

*Generated: 2026-05-25. Source: five benchmark sessions (Fedora 2026-05-08/08/11, Colab T4 2026-05-13/18) with per-session findings documents in docs/. Canonical results from results/summary.csv.*
