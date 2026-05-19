# A4000 deploy — single-page checklist

Tested target: NVIDIA RTX A4000 (Ampere sm_86, 16 GB). Should also work on A5000/A6000/A100 unchanged.

## Pre-flight (on the source host, where you cloned this repo)

What you need to copy:

| Item | Source | Size | Required |
|---|---|---|---|
| Bundle | `src/claude_exp8_finish1/` (sans `output/`, `results_runs/`) | ~50 MB | always |
| PE vendor | `/mnt/nas200_kurnianto/code/pia-prompt_optimization/` | ~1 GB | always |
| Deploy ckpt | `src/claude_exp8_finish1/output/ftpe_t3/qat/qat_deploy_fp32.pt` | 2.5 GB | always |
| Clip pack | `assets/clips/` (val/ for calib, test/ for bench) | ~250 MB | always |

Pack one tarball:

```bash
cd /mnt/nas200_kurnianto/code
tar czf /tmp/exp8_a4000.tar.gz \
    --exclude='claude_exp8_finish1/output' \
    --exclude='claude_exp8_finish1/results_runs' \
    --exclude='claude_exp8_finish1/.hf_cache' \
    --exclude='claude_exp8_finish1/calib' \
    TYPE8_PE_research/src/claude_exp8_finish1 \
    TYPE8_PE_research/assets/clips \
    pia-prompt_optimization

# QAT ckpt + .ft_cache separately so the tar above stays small
tar czf /tmp/exp8_a4000_ckpt.tar.gz \
    TYPE8_PE_research/src/claude_exp8_finish1/output/ftpe_t3/qat/qat_deploy_fp32.pt
```

Two tarballs total: ~1.3 GB compressed.

## On the A4000 box

### 1. Conda env

The env on the A6000 was `blue-vlm` (Python 3.11). Recreate:

```bash
conda create -n pe-int8 python=3.11 -y
conda activate pe-int8

pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install tensorrt==10.16.0.27
pip install nvidia-modelopt[onnx]==0.43.0
pip install onnx onnxruntime-gpu transformers==4.51.3
pip install peft   # only needed if loading raw FT_PE-...pt; for qat_deploy_fp32.pt skip
```

Verify versions match what produced the engines:

```bash
python -c "import torch, tensorrt, modelopt; \
  print(f'torch {torch.__version__}, trt {tensorrt.__version__}, modelopt {modelopt.__version__}')"
# expected: torch 2.2.0, trt 10.16.0, modelopt 0.43.0
```

### 2. Unpack

```bash
mkdir -p ~/pe_deploy && cd ~/pe_deploy
tar xzf /path/to/exp8_a4000.tar.gz       # → TYPE8_PE_research/ + pia-prompt_optimization/
tar xzf /path/to/exp8_a4000_ckpt.tar.gz
```

### 3. Smoke-test the bundle (no GPU work yet)

```bash
export PE_VENDOR=$PWD/pia-prompt_optimization
PE_INT8=$PWD/TYPE8_PE_research/src/claude_exp8_finish1/pe_int8

python - <<'PY'
import os, sys
sys.path.insert(0, os.environ.get('PE_INT8', '.'))
import ft_loader
print(f"ft_loader OK, layout detector ready: {ft_loader._detect_qkv_layout}")
print(f"PE_VENDOR resolved: {ft_loader._PE_VENDOR}")
PY
```

If this prints two OK lines, the bundle + vendor are wired correctly.

### 4. Run the deploy

```bash
cd ~/pe_deploy/TYPE8_PE_research/src/claude_exp8_finish1

PE_VENDOR=~/pe_deploy/pia-prompt_optimization \
PT_CKPT=$PWD/output/ftpe_t3/qat/qat_deploy_fp32.pt \
DATASET_ROOT=$PWD/../../assets/clips \
T_FRAMES=3 BATCH=4 SIGMA_K=2.5 GPU=0 \
bash scripts/run/run_on_a4000.sh
```

Expected wall-time on a free A4000: **12-15 min** (export 30s + BF16 build 60s + INT8 PTQ 4-5 min + surgery + INT8 build + CRL pass 2 min + INT8+CRL PTQ + build + bench 1-2 min).

If the A4000 has < 16 GB free, set `SKIP_CRL=1` to skip the CRL variant (saves ~5 min and ~3 GB peak memory).

### 5. Read results

```
a4000_run/results/results.md
```

Look for the `===== B=4  T=3 =====` block. Acceptance criteria:

| Engine | Expected cos | Expected ms |
|---|---:|---:|
| `pt_bf16` (reference) | 1.0 by definition | ~90 |
| `bf16_b4t3.engine` | 0.9982 ± 0.001 | ~55-65 (A4000 is slower than A6000) |
| `int8_b4t3.engine` | 0.974 ± 0.002 | ~38-50 |
| `int8_b4t3_crl.engine` | 0.991 ± 0.002 | ~38-50 |

If `bf16_b4t3.engine` cos is < 0.998, something is wrong in the build chain — STOP, do not trust the INT8 numbers.

## Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: core.vision_encoder.pe` | `PE_VENDOR` env not set or wrong | `export PE_VENDOR=/path/to/pia-prompt_optimization` |
| `RuntimeError: ckpt not found` | `PT_CKPT` path wrong | check `ls $PT_CKPT` |
| `Permission denied: a4000_run/onnx/...` | dir owned by another user | `rm -rf a4000_run && mkdir a4000_run` |
| `Failed to allocate memory for requested buffer` during PTQ | OOM on small GPUs (< 12 GB) | drop `BATCH` to 2 or 1 |
| `BF16 dyn cos = 0.4` | T=1 specific TRT bug (only affects ZS path) | known issue — see `docs/FINAL_REPORT_20260518.md` row B; ship static for ZS |
| All cos = NaN | Empty calibration (val/ has 0 clips) | `find $DATASET_ROOT/val -name '*.mp4' | wc -l` must be > 0 |

## What to send back

To compare against the A6000 reference:

```bash
tar czf /tmp/a4000_results.tar.gz a4000_run/results a4000_run/logs
```

Then compare `results.md` row-by-row against `src/claude_exp8_finish1/docs/FINAL_REPORT_20260518.md` row C (FT-T3 static). Latencies will be ~20-30% higher on A4000 (smaller card, slower memory); **cos numbers should match within ±0.002**.
