# FT_PE_INT8 — speed comparison bundle (T=3 canonical)

Drop-in INT8 twin of `src/speed_calculate_FTPE.py`. Same four-stage harness
(`full_cycle` / `three_quarters_cycle` / `half_cycle` / `inference`), same
`FTPEService` production-tick semantics — the only thing that changes is the
TRT engine plugged into `service._inference_stage`.

The canonical FT-T3 INT8+CRL recipe (per
[FINAL_REPORT_20260519.md](docs/FINAL_REPORT_20260519.md)) is the recommended
**FT video low-latency** production path:

> `dynamic INT8+CRL σ_k=2.5`, cos `0.989-0.992` across B=1..32, **1.27-1.44×**
> faster than BF16 on A6000.

This bundle re-runs the deploy locally and bench-marks the resulting engine
through the production FTPEService pipeline so the numbers are directly
comparable to `src/speed_calculate_FTPE.py` (BF16) and `speed_calculate_PE.py`.

## Layout

```
src/FTPE_INT8/
├── README.md                            (this file)
├── docs/
│   ├── A4000_DEPLOY.md                  ← deploy checklist (copied verbatim)
│   └── FINAL_REPORT_20260519.md         ← INT8 deploy report (copied verbatim)
├── pe_int8/                             ← deploy lib, copied from
│   │                                       /mnt/nas200_kurnianto/code/TYPE8_PE_research/
│   │                                       src/claude_exp8_finish1/pe_int8/
│   ├── export_onnx.py                   FT_PE .pt → FP32 ONNX (trace at BT)
│   ├── build_engine_py.py               ONNX → TRT engine
│   ├── quantize_onnx.py                 modelopt PTQ (INT8)
│   ├── surgery.py                       strip layout-only Q/DQ
│   ├── crl_pass.py                      CRL clamping pre-pass
│   ├── bench_trt.py                     reference deploy bench (cos / median ms)
│   ├── ft_loader.py                     FT_PE checkpoint loader
│   ├── video_utils.py                   clip ndarray helpers
│   └── __init__.py
├── scripts/
│   └── run_on_a4000.sh                  ← one-shot deploy (15 min on A4000)
└── speed_calculate_FTPE_INT8.py         ← four-stage speed bench
```

`scripts/speed_calculate_FTPE_INT8.sh` lives at the repo root under
`scripts/`, alongside the other bench wrappers.

## Prerequisites

The deploy step needs three things that are NOT in this repo (size and
licence reasons):

| Item | Where it lives | Size | Bring-your-own |
|---|---|---:|---|
| PE vendor (`pia-prompt_optimization`) | `/mnt/nas200_kurnianto/code/pia-prompt_optimization` on the PIA NAS | ~1 GB | symlink or set `PE_VENDOR` |
| FT_PE QAT deploy checkpoint | `output/ftpe_t3/qat/qat_deploy_fp32.pt` on the source NAS | 2.5 GB | copy or symlink |
| Clip pack (val/ + test/) | `assets/clips/` on the source NAS | ~250 MB | copy or symlink |
| Python deps | `modelopt 0.43`, `onnxruntime-gpu`, `transformers 4.51.3`, `peft` | — | `pip install ...` |

See [docs/A4000_DEPLOY.md](docs/A4000_DEPLOY.md) for the canonical install
checklist. The conda env this repo uses (`Product-AI-mono`) has `torch 2.5.1`,
`tensorrt 10.8.0.43` already — you'll need to add `modelopt[onnx]==0.43.0` and
the ORT/peft deps before the deploy can run.

## Workflow

### Step 1 — build the INT8 engine (once)

```bash
PE_VENDOR=/mnt/nas200_kurnianto/code/pia-prompt_optimization \
PT_CKPT=/path/to/qat_deploy_fp32.pt \
DATASET_ROOT=/path/to/clips \
T_FRAMES=3 BATCH=4 SIGMA_K=2.5 GPU=0 \
OUT_DIR=$PWD/assets/QAT \
bash src/FTPE_INT8/scripts/run_on_a4000.sh
```

This produces, under `assets/QAT/` (gitignored):

```
engines/
├── bf16_b4t3.engine                  ← reference engine for cos comparison
├── int8_b4t3.engine                  ← QAT-only INT8
└── int8_b4t3_crl.engine              ← QAT + CRL (the canonical shipping engine)
```

Move / symlink the canonical one to where the speed bench expects it:

```bash
ln -sf "$PWD/assets/QAT/engines/int8_b4t3_crl.engine" \
       "$PWD/assets/QAT/int8_b4t3_crl.engine"
```

Acceptance criteria for the deploy (from `docs/A4000_DEPLOY.md` §5):

| Engine | Expected cos | Expected ms on A4000 |
|---|---:|---:|
| `pt_bf16` (reference) | 1.0 by definition | ~90 |
| `bf16_b4t3.engine` | 0.9982 ± 0.001 | ~55-65 |
| `int8_b4t3.engine` | 0.974 ± 0.002 | ~38-50 |
| `int8_b4t3_crl.engine` | 0.991 ± 0.002 | ~38-50 |

If `bf16_b4t3` cos is **< 0.998**, the build chain is wrong — stop and
investigate; the INT8 numbers will be meaningless.

### Step 2 — run the four-stage speed bench

```bash
BATCH=4 FRAMES=3 ./scripts/speed_calculate_FTPE_INT8.sh
```

The bench wraps the INT8 engine in an adapter that exposes the
`(B, T, 3, H, W) → (B, T, 1024)` interface `FTPEService._inference_stage`
expects (the engine itself is built with a flat `(BT, 3, 336, 336)` input,
because the temporal mean-pool happens host-side; the adapter flattens
input and unflattens output around each engine call).

Output:

```
results/ftpe_int8_<gpu>_b<B>_t3_<ts>.json    ← schema matches speed_calculate_FTPE
```

The JSON has the same `stages` / `iterations` / `throughput` /
`gpu_temperature_c` keys as the BF16 FT_PE bench, so `aggregate_results.sh`
and `plot_results.sh` pick it up without changes.

## Why a flat-BT engine?

`PE-Core-L14-336` is a per-image encoder; the FT-T3 fine-tune doesn't add
any cross-frame operators inside the model. `encode_video` in the upstream
PE module just does `(B, T, C, H, W) → reshape → encode_image → mean(dim=1)`
host-side. So the INT8 deploy ONNX is traced with a fixed `BT = B * T` flat
input — TRT INT8 calibration sees `BT` independent frames, which keeps the
per-tensor activation quant sane and dodges shape-broadcast headaches that
the temporal `(B, T, ...)` graph has (the FT_PE "no_mean_pooling" engine has
a `(B, T, 1024)` output exactly to defer mean-pool to Python).

The adapter (`_Int8EngineAdapter` in `speed_calculate_FTPE_INT8.py`) bridges
the two conventions in one place.

## Cross-checking against the BF16 bench

After both benches have run on the same GPU at the same `B` and `T=3`:

```bash
./scripts/aggregate_results.sh        # writes results/summary.txt with both
```

Subtraction games (same as the BF16 bench):

```
full        − three_quarters     ≈ disk read cost
full        − half               = text-side cost
ftpe_bf16   − ftpe_int8 (inference stage) = INT8 speedup on the encoder
```

The INT8+CRL canonical numbers from the source (A6000) are in
`docs/FINAL_REPORT_20260519.md` §C; expect ~20-30% higher latency on A4000
(smaller card, slower memory) but the **speedup ratio vs BF16 should match
within ±5%**.

## Known issues (carried over from the upstream report)

- **B=8 INT8 static at T=8** OOMs during modelopt PTQ calibration (`ORT BFC
  arena 226 MB single-allocation limit`). Not relevant for T=3 at B=4-16.
- **FT-T8 dynamic bench at B≥16** fails with a shape-broadcast error in
  `bench_dynamic.py`. Not relevant — this bundle ships T=3 only.
- **ZS-T1 dynamic engine cos collapse → 0.4** is unrelated (T=1 path only).
  T=3 dynamic engines are fine across B=1..32.

See `docs/FINAL_REPORT_20260519.md` §"Known issues (open)" for the full list.
