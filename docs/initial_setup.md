# Initial setup — reference for the Claude Code agent

This doc gets a fresh agent (or a fresh machine) from "git-cloned repo" to
"all four speed benches running." It's a checklist plus the gotchas I hit
the hard way the first time through; read it top-to-bottom before doing
the work, and skip steps that already passed.

The four target scripts:

```
scripts/speed_calculate_PE.sh         # BF16  zero-shot PE      (T=1)
scripts/speed_calculate_FTPE.sh       # BF16  fine-tuned FT_PE  (T=1/3/8)
scripts/speed_calculate_PE_INT8.sh    # INT8  zero-shot PE      (T=1)
scripts/speed_calculate_FTPE_INT8.sh  # INT8  fine-tuned FT_PE  (T=3 canonical)
```

The BF16 wrappers auto-download the model from HuggingFace and build a
TensorRT engine on first run. The INT8 wrappers require a prior pass of
`src/FTPE_INT8/scripts/run_int8_pipeline.sh` to produce the INT8 engine
(see "INT8 prerequisites" below).

---

## 1. System prerequisites

- **GPU**: NVIDIA, Ampere or newer (sm 8.0+ recommended; sm 8.6 tested on
  RTX A4000). FT_PE INT8 dynamic builds use ~10–12 GB during TRT
  engine-build; FT_PE INT8 inference at the OPT shape (BT=48) pre-allocates
  ~6 GB of GPU workspace.
- **NVIDIA driver**: ≥ 535 (the test box ran 535.183.01).
- **CUDA toolkit**: not strictly required at the system level — the pip
  `nvidia-*` wheels ship the runtime libs the test harness needs.
- **Disk**: ~20 GB free in `assets/QAT/` for INT8 build artefacts (ONNX
  variants + engines + calibration tensor), plus ~5 GB for HF download
  cache under `src/FTPE_INT8/.hf_cache/`.

Both directories are gitignored.

## 2. Python environment

Tested on Python 3.12 / PyTorch 2.9.1 / TensorRT 10.8. The following pip
deps are the minimum the four scripts need; install them into the active
conda/venv:

```bash
pip install \
    "torch>=2.9" \
    "tensorrt>=10.8" \
    "onnxscript" \
    "nvidia-modelopt[onnx]" \
    "huggingface_hub" \
    "onnx" "onnxruntime-gpu" \
    "numpy" "pillow" "opencv-python"
```

Notes the agent should know:

- **`onnxscript`** is needed by `torch.onnx.export` in torch ≥ 2.9 even
  when we pass `dynamo=False` — the legacy exporter's import chain still
  pulls it in. Without it, `export_onnx.py` dies with
  `ModuleNotFoundError: No module named 'onnxscript'`.
- **`nvidia-modelopt[onnx]`** brings in `onnxruntime-gpu`, `cupy`,
  `polygraphy`, `onnx-graphsurgeon`, etc. — it's the PTQ engine. Without
  it, step 3 of `run_int8_pipeline.sh` fails with
  `No module named 'modelopt'`.
- The script tries to expose pip's `nvidia-*` CUDA libs to onnxruntime
  via `LD_LIBRARY_PATH` so the CUDA EP can find `libcudnn.so.9`. Many of
  these wheels are **namespace packages** (`__file__ = None`); the
  fallback now also globs `<site-packages>/nvidia/<name>/lib`. If you ever
  see `Failed to load library … libcudnn.so.9` and ORT silently drops to
  CPU, that's the symptom — `nvidia-cudnn-cu12` is installed but the
  library path discovery isn't picking it up.

## 3. PE vendor (`pia-prompt_optimization`)

Both the speed bench Python and the INT8 export pipeline need the
upstream Perception Encoder source for `core.vision_encoder.pe`. Clone it
next to this repo (it does NOT live inside this repo):

```bash
git clone https://github.com/PIA-SPACE-LAB/pia-prompt_optimization \
    /home/kurnianto/code/pia-prompt_optimization
# verify the expected layout:
test -d /home/kurnianto/code/pia-prompt_optimization/src/PE/perception_models \
    && echo OK
```

The scripts auto-resolve this path via env var `PE_VENDOR` (set explicitly)
or by walking a few well-known sibling directories. If neither works,
either set `PE_VENDOR=/path/to/pia-prompt_optimization` or place the
checkout at `src/FTPE_INT8/vendor/pia-prompt_optimization/`.

## 4. HuggingFace authentication

The model checkpoints live in **private** PIA-SPACE-LAB HF repos. The
pipeline reads the token from your HF login cache, but only if
`huggingface-cli login` has been run (or `HF_TOKEN` is exported).

```bash
huggingface-cli login            # interactive — paste the user's HF token
huggingface-cli whoami           # sanity-check: should print 'irfanrah'
                                 #               (or whichever user) and
                                 #               PIA-SPACE-LAB org membership
```

**The pipeline overrides `HF_HOME` to a project-local cache.** That's why
the script also reads `~/.cache/huggingface/token` and exports it as
`HF_TOKEN` itself — without that lift, private-repo downloads 401 even
when `whoami` works. If the agent ever sees a `RepositoryNotFoundError /
401` on a download, that's the diagnosis.

The repos / dataset the pipeline pulls:

- `PIA-SPACE-LAB/PE-Core-L14-336`  (zero-shot PE — QAT checkpoint at
  `splitqkv_qat/qat_deploy_fp32.pt`, public model at
  `PE-Core-L14-336.onnx`, text features at `text_features.json`)
- `PIA-SPACE-LAB/FT_PE-Core-L14-336_260318`  (fine-tuned FT_PE — QAT
  checkpoint at `splitqkv_qat_t3/qat_deploy_fp32.pt`, public ONNX at
  `FT_PE-Core-L14-336_260318_vision_no_mean_pooling.onnx`, text features
  at `FT_text_features.json`)
- `PIA-SPACE-LAB/PE_INT8_QAT_CRL` (dataset — val/ + test/ clips for PTQ
  calibration and cos/MSE validation)

## 5. Running the four speed benches

### 5a. BF16 PE (zero-shot)

```bash
BATCH=16 bash scripts/speed_calculate_PE.sh
```

First run downloads `PE-Core-L14-336.onnx` from HF and builds a TRT BF16
engine at `assets/model/PE-Core-L14-336.engine`. Subsequent runs reuse
the engine. Output JSON: `results/pe_<gpu>_b<B>_<timestamp>.json`.

### 5b. BF16 FT_PE (fine-tuned, temporal T)

```bash
BATCH=16 FRAMES=3 bash scripts/speed_calculate_FTPE.sh
```

The first run builds a wide dynamic engine (`MAX_BATCH=16 MAX_FRAMES=8`
by default) so subsequent (B, T) combinations within that profile reuse
the engine. If you ask for a larger B or T than the engine's profile, the
wrapper widens `MAX_BATCH` / `MAX_FRAMES` and rebuilds. Output JSON:
`results/ftpe_<gpu>_b<B>_t<T>_<timestamp>.json`.

### 5c. INT8 PE (zero-shot)

**Requires** the INT8 pipeline to have produced a dynamic-profile PE INT8
engine first — see "INT8 prerequisites" below. With that in place:

```bash
BATCH=16 bash scripts/speed_calculate_PE_INT8.sh
```

The default `PE_INT8_ENGINE` is `assets/QAT/pe/engines/int8_pe_dyn_b1-16_t1.engine`.
Output JSON: `results/pe_<gpu>_b<B>_<timestamp>_int8.json`.

### 5d. INT8 FT_PE (fine-tuned, T=3 canonical)

**Requires** the INT8 pipeline to have produced a dynamic FT_PE engine.
With that in place:

```bash
BATCH=24 FRAMES=3 bash scripts/speed_calculate_FTPE_INT8.sh
```

The default `FTPE_INT8_ENGINE` is
`assets/QAT/ftpe/engines/int8_ftpe_dyn_b4-128_t3_crl.engine`. The bench
bootstraps `FTPEService` with the BF16 production engine at
`assets/model/FT_PE-Core-L14-336_260318_vision_no_mean_pooling.engine`
(materialised by step 5b if you ran it once) and then swaps the
`service.model` for an adapter that wraps the INT8 engine. Output JSON:
`results/ftpe_int8_<gpu>_b<B>_t<T>_<timestamp>_int8.json`.

## 6. INT8 prerequisites — build the engines first

Both INT8 speed scripts need a TRT engine on disk. Build them via:

```bash
# Zero-shot PE INT8 deploy: static B=16 engine at
#   assets/QAT/pe/engines/int8_pe_b16t1.engine
VARIANT=pe bash src/FTPE_INT8/scripts/run_int8_pipeline.sh

# Fine-tuned FT_PE INT8 deploy: dynamic-batch engine at
#   assets/QAT/ftpe/engines/int8_ftpe_dyn_b4-128_t3_crl.engine
VARIANT=ftpe bash src/FTPE_INT8/scripts/run_int8_pipeline.sh
```

Each variant runs the same six-stage pipeline (PT → ONNX → BF16 ref
engine → modelopt PTQ → surgery + INT8 engine → CRL pre-pass + recalibrate
+ INT8+CRL engine), then the optional cos-vs-PT-BF16 eval on the test
split. End-to-end runtime is roughly 25–40 min per variant on a 16 GB
GPU, depending on calibration speed.

### The PE-INT8 zero-mask gotcha (read this!)

The deploy pipeline ships PE as a **static B=16** engine. The speed
bench drives the engine through `PEService`, which at init calls
`_init_default_values()` — and that hands the model a `(1, 3, 336, 336)`
zero tensor to compute a zero-mask. A static B=16 engine rejects that
shape with `IExecutionContext::setInputShape: Static dimension mismatch`.

Workaround: after the pipeline finishes the PE variant, build a
**dynamic** B=1..16 engine from the same clean INT8 ONNX:

```bash
python3 src/FTPE_INT8/pe_int8/build_dynamic_engine.py \
    --onnx assets/QAT/pe/onnx/int8_pe_b16t1_clean.onnx \
    --save_engine assets/QAT/pe/engines/int8_pe_dyn_b1-16_t1.engine \
    --min_shape input:1x3x336x336 \
    --opt_shape input:16x3x336x336 \
    --max_shape input:16x3x336x336 \
    --mode strongly_typed
```

That's the engine `speed_calculate_PE_INT8.sh` looks for by default.
The comment block at the top of the wrapper repeats this command.

FT_PE doesn't have this problem because the pipeline already builds a
dynamic profile (B=4..128, OPT=16) by default — `(1, 3, 336, 336)` isn't
within `[12, 384]` but `FTPEService` doesn't probe at B=1; it boots from
the production BF16 engine and then swaps in the INT8 adapter for
inference only.

## 7. Validating the engines

Before running the speed benches, the agent can sanity-check that the
INT8 engines actually match the PT BF16 reference using the random-image
cos / MSE comparator:

```bash
PE_VENDOR=/home/kurnianto/code/pia-prompt_optimization \
PYTHONPATH=src/FTPE_INT8/pe_int8:$PE_VENDOR/src/PE/perception_models:$PE_VENDOR \
python3 src/FTPE_INT8/scripts/test_int8_random.py \
    --engine-dir assets/QAT/pe/engines \
    --ft_ckpt src/FTPE_INT8/.hf_cache/pe/splitqkv_qat/qat_deploy_fp32.pt \
    --batch_videos 16 --frames_per_video 1
```

Expected: the BF16 engine should land at `cos ≈ 0.9999` / `mse ≈ 1e-7`
vs PT BF16. INT8 lands lower on out-of-distribution random pixels
(typical: PE INT8+CRL ≈ 0.98, FT_PE INT8+CRL ≈ 0.99). If the BF16 engine
itself doesn't match PT to ~1e-7, the pipeline's PT → ONNX → TRT path
is broken — debug that first before benchmarking.

## 8. Thermal behaviour — the bench is hot

On a 16 GB A4000, the back-to-back benches push GPU temp from ~75 °C at
idle to **100 °C** within a single run. The A4000 clock-throttles
somewhere around 95–97 °C — once temperature crosses that, the per-iter
latency widens dramatically (std jumps 10×). Symptom:

```
  full_cycle  mean=2555.700 ms  std=909.016  p95=3169.789 ms
```

`std` of ~36 % of mean and `p95` 1.2× mean is the throttle signature
(versus PE INT8 on the same hardware: `mean=214 ms std=6 ms`).

If the agent's job is to compare engines / configurations and the
numbers look noisy, force a cool-down before the run:

- `sleep 600` between runs is the project convention (the user has been
  using it).
- Or cap the GPU power: `nvidia-smi -pl 130` (default 140 W on A4000)
  reduces sustained heat.
- For unthrottled numbers, look at `min_ms` in the JSON instead of
  `mean_ms`.

When the agent runs benches it owns: re-issue benches with
`SLEEP_BEFORE=600` (or use `until <check>` patterns) but DO NOT poll
loop-and-`sleep`-2-second-style — the harness blocks short repeated
sleeps. Use `run_in_background: true` with a watcher that exits on a
disk file or on `! pgrep` for the bench process.

## 9. Quick diagnostic if something breaks

| Symptom | Diagnosis | Fix |
|---|---|---|
| `ModuleNotFoundError: onnxscript` | pip install missing | `pip install onnxscript` |
| `ModuleNotFoundError: modelopt`   | pip install missing | `pip install "nvidia-modelopt[onnx]"` |
| `TorchExportError: FakeTensor Device Propagation` | torch 2.9+ dynamo path | already fixed: `dynamo=False` in `export_onnx.py` |
| `RepositoryNotFoundError: 401` on HF | private repo + token not lifted | the script now lifts the cached token; verify `huggingface-cli whoami` |
| `Failed to load library … libcudnn.so.9` | ORT can't see cuDNN | fixed: namespace-package fallback in the `LD_LIBRARY_PATH` discovery |
| `Static dimension mismatch while setting input shape ... Expected [16,3,336,336]` | static engine + `PEService` B=1 probe | use the dynamic PE INT8 engine (see § 6) |
| `Cuda Runtime (out of memory)` running multiple engines | dynamic-profile workspace pre-alloc | load engines one at a time; `test_int8_random.py` already does this |
| FT_PE `half_cycle > full_cycle` in some iters | OLD bench convention in `speed_calculate_FTPE_INT8.py` | known; the main `speed_calculate_FTPE.py` was refactored to share-batch but the INT8 variant still has the old loop. Use the BF16 bench for a clean per-iter ordering. |

## 10. What "passing" looks like on an A4000

Reference numbers from this branch's verification runs (fresh GPU, no
throttling):

| Bench | Inference (ms) | Throughput (img/s) |
|---|---|---|
| PE BF16, B=10 (prior run) | ~378 | ~26 |
| **PE INT8, B=16**         | ~112 | **~143** |
| FT_PE BF16, B=12, T=3 (prior run) | ~? (see results/) | ~? |
| **FT_PE INT8, B=24, T=3** (cool start) | ~571 | **~126 frames/s** |
| FT_PE INT8, B=24, T=3 (throttled)      | ~2434 | ~30 frames/s |

Speedup PE INT8 vs PE BF16: ~3.4× on inference; FT_PE similar.
