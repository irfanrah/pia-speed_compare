# PE-Core-L14-336 Speed Benchmarks — NVIDIA RTX A4000

This directory archives every benchmark run performed on the **NVIDIA RTX A4000**
prior to porting the suite to another GPU. The goal across all runs was to
validate the report claim that *"A4000 handles 12 CCTV channels within a 0.5 s
realtime budget"* for the PE-Core-L14-336 vision encoder.

The raw JSON / log / PNG artifacts produced by each script live next to this
file. Re-running the scripts on a different GPU will produce a fresh set under
`results/`; this directory is meant to stay frozen as the A4000 reference.

## Hardware & software

| Field              | Value                          |
| ------------------ | ------------------------------ |
| Device             | NVIDIA RTX A4000               |
| Devices on host    | 2 (only index 0 used)          |
| Compute capability | 8.6 (Ampere)                   |
| VRAM               | 16,108 MiB                     |
| Driver             | 535.183.01                     |
| CUDA runtime       | 12.1                           |
| TensorRT           | 10.8.0.43                      |
| PyTorch            | 2.5.1+cu121                    |
| Precision          | FP16 (engine + FP16 ONNX weights where applicable) |
| Input shape        | 3 × 336 × 336                  |
| Dynamic profile    | min=1, opt=16, max=32          |
| Realtime budget    | 500 ms / cycle                 |

## Scenario

- Target workload: **12 CCTV channels**, one frame per channel per cycle.
- A cycle passes if `wall_time_ms <= 500`.
- Each script reports two timings:
  - **inference_only_ms** — GPU compute only, preprocess excluded.
  - **full_cycle_ms** — preprocess(N) + inference + postprocess.
- Engine variants tested: dynamic profile, fixed batch=4, FP16-weights ONNX.

## Summary verdict

| Benchmark                  | Schedule              | Inference-only (mean) | Full cycle (mean) | Verdict |
| -------------------------- | --------------------- | --------------------- | ----------------- | ------- |
| `check_perception_encoder` | 1 × batch=12          | 2691.2 ms             | 2793.1 ms         | ❌      |
| `check_multichannel`       | 12 × batch=1          | —                     | 2397.1 ms         | ❌      |
| `check_chained`            | **3 × batch=4**       | **324.6 ms**          | 503.1 ms          | ❌ (full cycle 3 ms over budget) |
| `check_chained`            | 4 × batch=3           | 1723.8 ms             | 1876.0 ms         | ❌      |
| `check_chained`            | 6 × batch=2           | 1967.2 ms             | 2131.6 ms         | ❌      |
| `check_fixed_batch` b=4    | 3 × batch=4 (static)  | 963.4 ms              | 1092.6 ms         | ❌      |
| `check_dynamic_3x4`        | 3 × batch=4, 100 it   | 1472.9 ms             | 1554.1 ms         | ❌      |
| `check_dynamic_3x4_short`  | 3 × batch=4, 25 it cold | 351.5 ms            | **456.2 ms**      | ✅ (cold burst only) |
| `check_dynamic_3x4_thermal`| 3 × batch=4 w/ rest   | 1403.0 ms             | 1520.8 ms         | ❌ (thermal-limited) |
| `pe_trt_fp16_benchmark`    | batch=12 forward-only | 2199.9 ms             | —                 | ❌ batch path; ✅ streaming sim |

Headline: **the A4000 does *not* sustain 12 channels in 500 ms** under the
3 × batch=4 schedule once thermals stabilize. It only meets the budget for a
cold burst of ~25 iterations (`check_dynamic_3x4_short`), after which timings
diverge sharply as the card throttles toward ~100 °C.

## Per-benchmark notes

### `check_perception_encoder.py` — single-batch sweep

Sweeps batch = 1..MAX_BATCH (here up to 16), timing the upstream
`preprocess → inference → postprocess` stages.

- batch=1 → 38.4 ms, 26.1 img/s ✅
- batch=4 → 156.0 ms ✅
- **batch=12 → 2793.1 ms, 4.30 img/s ❌**

A single batch=12 forward pass cannot meet the budget. → file:
`pe_check_perception_encoder.json`.

### `check_multichannel.py` — batch=1 sequential

Runs N back-to-back batch=1 inferences per cycle. Models per-channel serial
processing.

- 1 channel → 39.6 ms cycle ✅
- 4 channels → 133.6 ms ✅
- 8 channels → 271.4 ms ✅
- **12 channels → 2397.1 ms ❌** (per-channel ≈ 200 ms; aggregate 5.0 fps)

Throughput collapses past ~8 channels — kernels stop interleaving well at
batch=1. → file: `pe_check_multichannel.json`.

### `check_chained.py` — scheduling sweep at fixed 12 channels

Holds total channels at 12, varies the chained-inference layout. Short
warmup (10) + 30 measured iters.

| schedule         | inference-only mean | full-cycle mean |
| ---------------- | ------------------- | --------------- |
| **3 × batch=4**  | **324.6 ms ✅**     | 503.1 ms ❌     |
| 4 × batch=3      | 1723.8 ms ❌        | 1876.0 ms ❌    |
| 6 × batch=2      | 1967.2 ms ❌        | 2131.6 ms ❌    |
| 12 × batch=1     | (worst path)        |                 |

The report's claim is *inference-only* on 3 × batch=4 — that does land at
~325 ms — but once preprocessing 12 frames is included the cycle slips
just past the 500 ms budget. → file: `pe_check_chained.json`.

### `check_fixed_batch.py` — static batch=4 engine

Rebuilds TRT engine with min=opt=max=4 to let the optimizer specialize
harder. **Did not beat** the dynamic engine on this card under sustained
load: 963.4 ms inference-only vs. 324.6 ms in the short chained run. The
fixed engine appears to expose the throttled steady-state more aggressively.
→ files: `pe_check_fixed_b4.json`.

### `check_dynamic_3x4.py` — 100-iter sustained run

Same 3 × batch=4 schedule, but with 20 warmup + **100 measured** iterations
written to a CSV log. Sustained timings degrade hard:

- mean inference-only **1472.9 ms** (vs. 324.6 ms in the 30-iter run)
- p50 1767.6 ms, p95 2158.9 ms, p99 2203.9 ms
- std 690.5 ms — high variance from thermal throttling

→ files: `pe_check_dynamic_3x4.json`, `pe_check_dynamic_3x4.log`.

### `check_dynamic_3x4_short.py` — 25-iter cold burst

5 warmup + 25 measured iterations from a cold GPU. **This is the only run
that passes the 500 ms budget.**

- inference-only mean 351.5 ms, p50 322.7 ms, p95 520.2 ms
- full-cycle mean 456.2 ms, p50 433.8 ms, p95 600.1 ms
- GPU temperature: 80 °C → 95 °C across the run

By the last few iterations the p95 is already over budget, indicating the
500 ms target is only met as long as the card stays cold.

→ files: `pe_check_dynamic_3x4_short.json`, `pe_check_dynamic_3x4_short.log`,
`pe_check_dynamic_3x4_short.png`.

### `check_dynamic_3x4_thermal.py` — 100 iters with 60 s rest

Same as `check_dynamic_3x4` plus per-iter `nvidia-smi` temperature reads
and a forced 60 s rest after iter 50.

- pre-rest (iters 1–50): mean 1092.3 ms inference-only
- post-rest (iters 51–100): mean 1713.6 ms — *worse*, because 60 s of idle
  isn't enough to clear the heat soak; the case is still ≥90 °C
- GPU temperature spread: 86 °C → **102 °C**

Confirms thermal throttling is the dominant cause of slowdown.
→ files: `pe_check_dynamic_3x4_thermal.json`,
`pe_check_dynamic_3x4_thermal.log`, `pe_check_dynamic_3x4_thermal_rest.log`,
`pe_check_dynamic_3x4_thermal.png`.

### `pe_trt_fp16_benchmark.json` — FP16-weights ONNX, forward-only

Engine built from a pre-quantized FP16-weights ONNX, timing **model forward
only** (preprocess excluded). Includes an analytic streaming simulator.

- batch=1 → 28.4 ms / 35.3 img/s ✅
- batch=4 → 109.5 ms / 36.5 img/s ✅ — best throughput-within-budget point
- batch=8 → 903.9 ms ❌
- batch=12 → 2199.9 ms ❌
- **Streaming sim (12 ch × 1 fps, effective batch=4, 30 s)**: 12.0 fps
  throughput, p95 latency 328.6 ms — passes the budget *for paced
  arrivals*, not for the synchronous 12-at-once cycle.

Capacity summary: best operating point is batch=4 at ~36.5 img/s, leaving
~390 ms headroom in the 500 ms budget.

→ file: `pe_trt_fp16_benchmark.json`.

## Conclusion

1. The "12 channels in 0.5 s" claim only holds for the **inference-only**
   measurement on a **cold** GPU, using the 3 × batch=4 schedule.
2. **Full-cycle** wall time crosses the 500 ms line even on the first 30
   iterations and degrades severely over a sustained 100-iter run.
3. Thermal behavior is the dominant factor: the A4000 reaches 100+ °C and
   throttles aggressively under continuous batch-4 vision-encoder load.
4. For paced ingest (12 channels each at 1 fps, dispatched as effective
   batch=4), the streaming simulator says the A4000 **does** keep up — so
   the answer is workload-shape sensitive.

Use these numbers as the A4000 baseline when comparing against the next
GPU. Re-running the same scripts on the new card will produce a fresh
`results/` directory at repo root; this `docs/A4000/` snapshot stays
immutable for comparison.
