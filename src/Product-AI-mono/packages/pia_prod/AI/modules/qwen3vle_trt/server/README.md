# PIA Qwen3-VL-Embedding TRT Server

FastAPI server that wraps `Qwen3VLETrtModel` and serves Qwen3-VL-Embedding
over HTTP. Backed by TensorRT engines mounted at runtime.

The server accepts JPEG/PNG frames at any resolution and resizes each one
to the engine's fixed `IMG_SIZE` before inference. Clients do not need to
pre-size — just decode and POST.

## Quick start

```bash
cd packages/pia_prod/AI/modules/qwen3vle_trt/server
cp .env.example .env    # adjust host paths for your TRT engines
docker compose up -d --build
```

Health check:

```bash
curl http://localhost:9002/health
# → {"status":"ok","model_loaded":true}
```

Smoke test:

```bash
IMG_B64=$(base64 -w0 /path/to/test.jpg)
curl -X POST http://localhost:9002/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d "{\"frames\":[\"$IMG_B64\"]}" | head -c 200
```

## Files

| File | Purpose |
|---|---|
| `server.py` | FastAPI app with `/v1/embeddings` + `/health`. Decodes base64, resizes to engine size, runs TRT inference. Async GPU concurrency limit via semaphore. |
| `Dockerfile` | NVIDIA CUDA runtime base; installs FastAPI, Uvicorn, torch, transformers, TRT. |
| `docker-compose.yml` | Wires GPU passthrough; mounts the host's compiled TRT engines into the container at runtime. |
| `.env.example` | Environment template (engine paths, GPU, port, concurrency limit). |
| `requirements.txt` | Minimal pip set for the server (no monorepo deps needed). |
| `benchmark.py` | Latency / throughput benchmark client. |

## Environment variables

See `.env.example`. Key ones:

| Variable | Purpose |
|---|---|
| `TRT_DIR` | Container path to the directory containing `Vision.engine`, `Transformer.engine`, `rotary_params.npz`, `tokenizer/`, and `text_features_*.json`. |
| `PROCESSOR_PATH` | Container path to the HF tokenizer/processor config files. Defaults to `${TRT_DIR}/tokenizer`. |
| `SERVER_PORT` | Port the FastAPI app listens on (default `9002`). Must match `QWEN3VLE_TRT_API_PORT` on upstream services. |
| `MAX_INFLIGHT_REQUESTS` | Caps queue depth for backpressure / OOM safety. Does **not** raise GPU concurrency — the TRT context is single-threaded and engine work is serialised on a single CUDA stream. Tune this to bound how many requests can sit waiting (and how much memory is in flight), not to scale throughput. Default `2`. |
| `DEVICE` | `cuda` (default) — TRT requires CUDA. |

## HTTP API

### `POST /v1/embeddings`

Request:

```json
{
  "frames": ["<base64 JPEG bytes>", "<base64 JPEG bytes>", ...]
}
```

- `frames` — `T` JPEG (or PNG) frames, base64-encoded. Any resolution; the
  server resizes to the engine's fixed `IMG_SIZE` (`(h=768, w=768)` at the
  time of writing — see `qwen3vle_trt/config.py`).
- `T` should match what the engines were compiled for (typically 1).

Response:

```json
{
  "embedding": [0.012, -0.034, ...],
  "dim": 2048
}
```

One `[1, D]` pooled video embedding for the `T`-frame clip. Returned as a
flat list so callers don't need numpy.

Errors:

- `400 frames must not be empty`
- `500 <exception message>` — see container logs (`docker logs ...`) for a
  full traceback. Common cause: TRT shape mismatch, which now should not
  happen since the server resizes internally.

### `GET /health`

```json
{"status": "ok", "model_loaded": true}
```

`model_loaded` is `false` briefly while engines are warming up after
container start. Wait for it to flip true (or check the container logs
for `Qwen3VLETrtModel ready`) before sending traffic.

## Resize behavior

Every decoded frame is resized to `IMG_SIZE` via `PIL.Image.resize` with
`BILINEAR`. This is a direct resize (no letterbox), matching the
`torchvision.transforms.functional.resize` call in the in-process
`Qwen3VLETrtService._predict` path. So a 1920×1080 frame becomes a
squished 768×768 — the engines, the in-process service, and this server
all agree on that geometry.

If you need letterboxing or a different aspect-ratio policy, do it on the
client before encoding; the server will still pass the result through
its (now no-op for already-correct-sized inputs) resize.

## Deployment — external engine mounts

TensorRT `.engine` files are massive (multi-GB) and tied to the GPU
architecture they were compiled on. Keep them out of the image:

1. **Don't `COPY`** the engines into the `Dockerfile`.
2. **Bind-mount them at runtime** via `docker-compose.yml`:
   ```
   - ${TRT_ENGINE_DIR}:/models/Qwen3VLE-TRT:ro
   ```

Recompiling or swapping engines on the host is then a no-rebuild operation —
just restart the container.

## Upstream service connection

Point upstream services at this server:

```bash
export QWEN3VLE_TRT_API_HOST=localhost   # or the K8s service name
export QWEN3VLE_TRT_API_PORT=9002
```
