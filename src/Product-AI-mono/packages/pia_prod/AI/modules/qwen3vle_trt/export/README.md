# Qwen3-VL-Embedding TensorRT Export

PyTorch (HF) → ONNX → TensorRT conversion pipeline. Produces the engine files consumed by `Qwen3VLETrtService` at runtime.

## ONNX Directory Contents

`QWEN3VLE_TRT_ONNX_DIR_PATH` (default: `assets/model/Qwen3-VL-Embedding-2B-onnx`) holds everything needed to build and run the TRT engines:

| File / Dir | Purpose |
|------------|---------|
| `Vision.onnx` (+ `.onnx.data`) | Vision encoder (fixed image resolution baked in) |
| `Transformer.onnx` (+ `.onnx.data`) | Transformer decoder layers (no KV cache, embedding mode, dynamic `seq_len`) |
| `rotary_params.npz` | mRoPE parameters (`inv_freq`, `mrope_section`), token embedding weights, image/grid config (`image_height/width`, `height_factor/width_factor`, `patch_size`, `merge_size`, `hidden_size`, `head_dim`, etc.) |
| `tokenizer/` | HF `Qwen3VLProcessor` / tokenizer files, used at runtime for text preprocessing |
| `Vision.engine` | Built locally by step c |
| `Transformer.engine` | Built locally by step c |

The `.onnx`, `.npz`, and `tokenizer/` files are portable and can be uploaded to / downloaded from HuggingFace. The `.engine` files are **not** portable — they're specific to the GPU architecture and TRT version, so they must always be built locally.

## Usage

Two entry points depending on what you already have.

### Option A — You already have the ONNX assets (from HF)

If `QWEN3VLE_TRT_ONNX_DIR_PATH` already contains `Vision.onnx`, `Transformer.onnx`, `rotary_params.npz`, and `tokenizer/`, only the TRT engine build is needed:

```bash
python -m pia_prod.AI.modules.qwen3vle_trt.export.c_export_onnx_to_trt
```

### Option B — Regenerate everything from the HF PyTorch model

Use this when you don't have the ONNX assets, or want to re-export after a model/config change:

```bash
# Full pipeline: ONNX export → TRT build → parity test
bash packages/pia_prod/AI/modules/qwen3vle_trt/export/run_all.sh
```

Individual steps:

```bash
python packages/pia_prod/AI/modules/qwen3vle_trt/export/a_export_to_onnx.py
python packages/pia_prod/AI/modules/qwen3vle_trt/export/b_export_onnx_vision.py
python -m pia_prod.AI.modules.qwen3vle_trt.export.c_export_onnx_to_trt
python test/test_compare_qwen3vle_parity.py
```

## Script Roles

- **`a_export_to_onnx.py`** — Exports `Transformer.onnx` and saves `rotary_params.npz`. Also produces an initial `Vision.onnx` via a manual path (norm fusion, GELU replacement), which gets overwritten in step b.
- **`b_export_onnx_vision.py`** — Re-exports `Vision.onnx` by wrapping the HF `Qwen3VLVisionModel` directly. Traces the exact PyTorch code path, so numerics match HF.
- **`c_export_onnx_to_trt.py`** — Builds `.engine` files. FP16 globally, with normalization-sensitive layers forced to FP32 to avoid overflow in RMSNorm's `x * rsqrt(square(x).sum())` pattern. Skips if the engine already exists — delete `*.engine` to rebuild.

## Config

Read from [config.py](../config.py):

| Variable | Default | |
|----------|---------|--|
| `IMG_SIZE` | `(768, 768)` | Fixed input resolution — baked into ONNX (must be a multiple of `patch_size × merge_size`) |
| `TEMPORAL_SIZE` | `1` | Frames per clip |
| `QWEN3VLE_TRT_PT_MODEL_PATH` | env | PT model path for ONNX conversion |
| `QWEN3VLE_TRT_ONNX_DIR_PATH` | env | Output directory |

Changing `IMG_SIZE` or `TEMPORAL_SIZE` requires re-running the full pipeline.

## Note

The export pipeline currently only produces reliable engines for `TEMPORAL_SIZE ≤ 2`. Beyond that, the Torch vs TRT cosine similarity drops below 0.99 and parity can no longer be guaranteed. If you need longer temporal contexts, the export path will need further investigation (likely around the temporal patching / rotary handling).
