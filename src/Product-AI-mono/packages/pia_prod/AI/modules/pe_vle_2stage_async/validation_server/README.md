English | [한국어](README.ko.md)

# PE-VLE Validation Server

Single-container, single-process validation service. **Fully self-
contained** — no `pia_prod` import, no monorepo bind-mount, no shared
state with the AI service host. Mirrors `pe_vqa_2stage/validation_server`'s
deployment shape line-for-line.

```
PeVle2StageAsyncService ──fire+forget HTTP──▶ validation_server (one container, one process)
                                      ├─ vllm.AsyncLLMEngine (in-process, GPU)
                                      ├─ AnomalyClassifier (numpy cosine vs text anchors)
                                      ├─ S3Uploader (thumbnail upload)
                                      ├─ RedisUUIDTracker (pairing dedup)
                                      └─ MessagePublisher
                                            ├─ RabbitMQPublisher (KTT)
                                            └─ KafkaPublisher (ES MACS, fastavro)
                                                       │
                                                       ▼
                                                 RabbitMQ / Kafka
```

## Layout

```
validation_server/
├── Dockerfile           # FROM vllm/vllm-openai
├── README.md
├── config.py            # All env-driven constants
├── docker-compose.yml   # GPU, /models mount, MQ/S3/Redis env
├── download_model.py    # HF Hub fetch (entrypoint hook + manual host script)
├── entrypoint.sh        # Detects empty MODEL_PATH → downloads → exec server.py
├── publishers.py        # MessagePublisher / S3Uploader / RedisUUIDTracker / message builders
├── requirements.txt
└── server.py            # FastAPI app: validate() endpoint, embedder, classifier
```

## Quick start

`MODEL_DIR` is required — the validator binds it to `/models` and stores
the FP8 checkpoint + anchor JSON there. `HF_MODEL_REPO_ID` triggers an
auto-fetch on first launch when that directory is empty.

```bash
cd packages/pia_prod/AI/modules/pe_vle_2stage_async/validation_server

# Pick a host directory the validator can read+write to. The first launch
# downloads the FP8 checkpoint + text-features JSON into here and the
# subsequent launches skip the download (see "Model download" below).
#
# Inside this monorepo (default for in-tree dev):
export MODEL_DIR="$(git rev-parse --show-toplevel)/assets/model"
# Or any absolute path you control (standalone / non-git deploy):
# export MODEL_DIR="/srv/pe-vle/models"
# export MODEL_DIR="$HOME/pe_vle_models"
# Create both the parent and the MODEL_NAME subdir up-front so the
# downloaded files end up under the current user (otherwise Docker
# auto-creates the subdir as root on first up).
mkdir -p "${MODEL_DIR}/${MODEL_NAME:-Qwen3-VL-Embedding-2B-FP8}"

HF_MODEL_REPO_ID="PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8" \
HF_AUTH_TOKEN="$HF_TOKEN" \
  docker compose up -d --build

# Health (anchor JSON loaded; vLLM weights loaded — first start is slow).
curl http://localhost:8200/health
# → {"status":"ok","anchor_buckets":["falldown","fire","smoke"],"model_path":"/models/Qwen3-VL-Embedding-2B-FP8","messaging_backend":"kafka","fail_open":true}
```

vLLM cold-start takes ~60–120s on the first launch (after the HF fetch,
which can add several minutes the first time). The healthcheck's
`start_period` allows for that.

The default messaging backend is **kafka** (for ES MACS). To switch to
RabbitMQ for KTT-style deployments:

```bash
MESSAGING_BACKEND=rabbitmq \
BACKEND_RABBITMQ_IP=rabbitmq.host \
BACKEND_RABBITMQ_USER_NAME=user BACKEND_RABBITMQ_PASSWORD=pass \
  docker compose up -d --build
```

### Override knobs (advanced)

The defaults pull `PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8` into
`${MODEL_DIR}/Qwen3-VL-Embedding-2B-FP8/`. The download source and the
on-disk subdir name are env-driven, so they're overridable without
touching code:

- `HF_MODEL_REPO_ID` — HF repo to fetch (default `PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8`).
- `MODEL_NAME` — subdir name on both host (`${MODEL_DIR}/${MODEL_NAME}`)
  and container (`/models/${MODEL_NAME}`). Default `Qwen3-VL-Embedding-2B-FP8`.
- `HF_TEXT_FEATURES_FILE` — anchor JSON filename if it differs from
  `VLE_FP8_text_features.json` inside the repo.
- `HF_TEXT_FEATURES_REPO_ID` — HF repo for the anchor JSON if it lives
  separately from the model.

Set these before `docker compose up`. The host↔container mapping in
"Model download" below shows how they flow through.

### Alternative: `.env` file workflow

`docker compose` auto-loads a `.env` file from the working directory, so
the inline env above can be moved to a file:

```bash
cp .env.example .env
# Edit .env: set HF_AUTH_TOKEN, override MODEL_DIR / MODEL_NAME / HF_MODEL_REPO_ID as needed
docker compose up -d --build
```

The shipped `.env.example` defaults `MODEL_DIR` to the monorepo's
`assets/model` (relative path); change it to an absolute path for
standalone deploys.

## Model download

The validator needs two artifacts beside the runtime code:

1. **FP8 checkpoint** — a HuggingFace snapshot of the Qwen3-VL-Embedding-2B
   FP8-quantized weights. Lives under `${MODEL_DIR}/${MODEL_NAME}/`.
2. **Text-features JSON** (`VLE_FP8_text_features.json`) — pre-computed
   normal/target text-embedding anchors used by `AnomalyClassifier`. Lives
   beside the checkpoint by default.

### How the host directory maps into the container

```
Host filesystem                              Inside container
${MODEL_DIR}/                                /models/
└── ${MODEL_NAME}/            ──:rw──>       └── ${MODEL_NAME}/            ← MODEL_PATH
    ├── model.safetensors                        ├── model.safetensors
    └── VLE_FP8_text_features.json               └── VLE_FP8_text_features.json
```

- `MODEL_DIR` — host directory you choose (required, no default).
- `MODEL_NAME` — subdir name on both host and container (default `Qwen3-VL-Embedding-2B-FP8`).
- `MODEL_PATH` — derived as `/models/${MODEL_NAME}` inside the container.
- `:rw` — downloaded files persist on the host; subsequent launches skip the fetch.

There are three operator paths to obtain them. Pick whichever fits your
deployment posture; they're not mutually exclusive.

### Path A — auto-fetch via `entrypoint.sh` (default)

The container's entrypoint inspects `${MODEL_PATH}` on boot. If
`model.safetensors` or the text-features JSON is missing, it shells out to
`download_model.py` to pull from HF Hub before starting `server.py`. The
fetched files persist under `${MODEL_DIR}` on the host (the volume mount
is `:rw`), so the second launch and any sibling validators skip the
download entirely.

Required env (set in shell, `.env`, or the compose `environment:` block):

| Variable | Purpose |
|---|---|
| `HF_MODEL_REPO_ID` | The repo ID for the FP8 checkpoint, e.g. `<owner>/Qwen3-VL-Embedding-2B-FP8`. |
| `HF_AUTH_TOKEN` | Required for private/gated repos. Compose only forwards `HF_AUTH_TOKEN` (not `HF_TOKEN`); if your token lives in `$HF_TOKEN` (HF CLI default), map it explicitly: `HF_AUTH_TOKEN="$HF_TOKEN" docker compose up`. The `$HF_TOKEN` fallback in `download_model.py` only applies when running the script directly on the host (Path B). |
| `HF_TEXT_FEATURES_REPO_ID` | Optional — defaults to `HF_MODEL_REPO_ID`. |
| `HF_TEXT_FEATURES_FILE` | Optional — defaults to `VLE_FP8_text_features.json`. |

After the download, `entrypoint.sh` exports `HF_HUB_OFFLINE=1` itself, so
vLLM never reaches out to HF Hub during normal serving. The same compose
file therefore handles both first-run-online and subsequent-offline cases.

### Path B — manual host-side download (for air-gapped GPU nodes)

If the deployment target has no internet, run the download on a separate
host that does, then sync the resulting directory across.

```bash
# On the internet-connected host (e.g. your laptop):
pip install huggingface_hub
# Run from this directory (packages/pia_prod/AI/modules/pe_vle_2stage_async/validation_server/):
HF_TOKEN="$HF_TOKEN" python download_model.py \
    --model-dir /tmp/Qwen3-VL-Embedding-2B-FP8 \
    --repo-id PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8

# Ship to the GPU node:
rsync -aP /tmp/Qwen3-VL-Embedding-2B-FP8 user@gpu-node:/srv/models/

# On the GPU node:
mkdir -p /srv/models
MODEL_DIR=/srv/models docker compose up -d --build
# entrypoint sees the populated dir → skips download → starts server.py.
```

### Path C — pre-populate the mount before `up`

Copy the snapshot into `${MODEL_DIR}/Qwen3-VL-Embedding-2B-FP8/` by hand
(e.g. via `huggingface-cli download`, or rsync from another node). The
entrypoint's existence check skips the download in that case.

```bash
# Prerequisites: huggingface_hub installed + token cached on this host
pip install huggingface_hub
huggingface-cli login   # caches token at ~/.cache/huggingface/token
                        # (or skip and rely on $HF_TOKEN env)

# MODEL_DIR must be set and the directory must exist (see Quick start).
# HF_MODEL_REPO_ID and MODEL_NAME fall back to the PIA-SPACE-LAB defaults
# when unset — override them for your own checkpoint.
huggingface-cli download "${HF_MODEL_REPO_ID:-PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8}" \
    --local-dir "${MODEL_DIR}/${MODEL_NAME:-Qwen3-VL-Embedding-2B-FP8}"
docker compose up -d --build
```

### Failure modes

- `MODEL_DIR is required` from `docker compose up` → the env var isn't set.
  Set it (see "Quick start") and retry.
- `ERROR: HF_MODEL_REPO_ID is unset and ... is incomplete` from the
  entrypoint → either set `HF_MODEL_REPO_ID` (path A) or pre-populate
  the mount (paths B/C).
- `RuntimeError: No text_features loaded from ...` after a successful
  download → verify `HF_TEXT_FEATURES_FILE` matches the actual file name
  inside the HF repo (default expects `VLE_FP8_text_features.json`).

### Offline operation

Once `${MODEL_DIR}` contains both the checkpoint and the anchor JSON, the
validator never needs HF Hub access again — `entrypoint.sh` exports
`HF_HUB_OFFLINE=1` before launching `server.py`. Validate by setting
`HF_AUTH_TOKEN=` (empty) and `HF_MODEL_REPO_ID=` (empty), or by running
the container with `--network none` for a connectivity smoke test.

## Why fully self-contained?

- **Zero cross-host coupling.** No `/opt/mono` bind-mount; no need for
  the `pia_prod` package to be installed on the validator's host. The
  container is portable to any GPU host with the FP8 checkpoint.
- **Direct publishing.** The validator publishes to RabbitMQ (or Kafka)
  itself with `pika.basic_publish` / `confluent_kafka.Producer.produce`.
  No `match_outputs` indirection through `pia_prod.AI.utils.init`'s
  shared producer thread.
- **Independent scaling.** Run multiple validator replicas behind a
  load balancer; each holds its own MQ/S3/Redis connections.

## Wire contract (identical to pe_vqa_2stage)

### `POST /api/v1/validate`

Request:

```json
{
  "thumbnail_b64": "<base64 JPEG, BGR pixels (cv2.imencode output)>",
  "is_start": true,
  "category_name": "fire_pe_vle_ret",
  "stream_id": "0_pia",
  "event_uuid": "abc-123",
  "event_type": "retEvent",
  "user_param": { ... }
}
```

- `is_start=True` requires `thumbnail_b64` → 400 otherwise.

Response:

```json
{"status": "sent",       "uuid": "abc-123"}
{"status": "discarded",  "uuid": "abc-123", "reason": "vle_rejected"}
{"status": "discarded",  "uuid": "abc-123", "reason": "vle_error"}            // only if PE_VLE_FAIL_OPEN=false
{"status": "discarded",  "uuid": "abc-123", "reason": "start_not_validated"} // is_start=False with no matching start
{"status": "error",      "uuid": "abc-123", "reason": "..."}
```

### `GET /health`

```json
{
  "status": "ok",
  "anchor_buckets": ["falldown", "fire", "smoke"],
  "model_path": "/models/Qwen3-VL-Embedding-2B-FP8",
  "messaging_backend": "kafka",
  "fail_open": true
}
```

## Decision flow

| Stage | Input | Outcome | `status` | `reason` |
|---|---|---|---|---|
| Boundary | `is_start=True`, no `thumbnail_b64` | 400 (rejected at boundary) | — | — |
| End-state | `is_start=False`, UUID **registered** in Redis | Publish + delete UUID | `sent` | — |
| End-state | `is_start=False`, UUID **not** registered | Drop (orphan end) | `discarded` | `start_not_validated` |
| vle_id missing | `vle_id` not mapped (e.g. falldown) | Skip vLLM; register UUID; S3 + publish | `sent` | — |
| Anchor classify | `target_max > normal_max` | Register UUID; S3 + publish | `sent` | — |
| Anchor classify | `target_max ≤ normal_max` | Drop | `discarded` | `vle_rejected` |
| vLLM error | `PE_VLE_FAIL_OPEN=true` (default) | Treat as confirmed; register UUID; S3 + publish | `sent` | — |
| vLLM error | `PE_VLE_FAIL_OPEN=false` | Drop | `discarded` | `vle_error` |
| Publish error | `pika`/`confluent_kafka` raises | Server-side error | `error` | exception text |

## Environment

### vLLM

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_PATH` | `/models/Qwen3-VL-Embedding-2B-FP8` | Container path to FP8 checkpoint dir. |
| `MODEL_NAME` | `Qwen3-VL-Embedding-2B-FP8` | Subdir name under host `MODEL_DIR`. |
| `MODEL_DIR` | `…/assets/model` | Host directory mounted to `/models` (read-write — entrypoint downloads checkpoint here on first launch). |
| `DTYPE` | `bfloat16` | vLLM compute dtype. |
| `GPU_MEMORY_UTILIZATION` | `0.3` | Fraction of GPU memory vLLM may reserve (auto-sizes KV cache). |
| `KV_CACHE_MEMORY_BYTES` | `1G` | **Hard cap on KV cache size** (overrides the auto-sizing above). Format: `1G`, `512M`, raw int bytes. Set to empty string to fall back to vLLM's auto-sizing from `GPU_MEMORY_UTILIZATION`. |
| `MAX_NUM_SEQS` | _unset_ (vLLM default) | Max concurrent sequences in the engine batch. Tune via `benchmark.py` if you need to bound batch size. |
| `MAX_NUM_BATCHED_TOKENS` | _unset_ (vLLM default) | Max tokens per scheduler step. |
| `VLLM_MAX_CONCURRENCY` | `10` | FastAPI-layer cap on in-flight `/api/v1/validate` calls (asyncio Semaphore). Backpressure dial, not a parallelism dial. |
| `MAX_MODEL_LEN` | `8192` | Max sequence length. |
| `LIMIT_MM_PER_PROMPT_VIDEO/IMAGE` | `1` | Per-prompt multi-modal caps. |
| `QWEN3VLE_VLLM_TEXT_FEATURES_PATH` | `/models/.../VLE_FP8_text_features.json` | Anchor JSON path. |

### Policy

| Variable | Default | Purpose |
|---|---|---|
| `PE_VLE_FAIL_OPEN` | `true` | If true, vLLM errors → publish anyway. |
| `PE_VLE_TO_VLE_CATEGORY_EVENT_MAP_JSON` | `{"fire_pe_vle_ret":"fire_vle_ret",...}` | PE→VLE category map. |
| `VLE_CATEGORY_EVENT_MAP_JSON` | `{"fire":["fire_vle_ret",...],...}` | vle_id→bucket map. |

### Messaging backend

| Variable | Default | Purpose |
|---|---|---|
| `MESSAGING_BACKEND` | `kafka` | `kafka` (ES MACS, default) or `rabbitmq` (KTT). |

### RabbitMQ (KTT — same keys as pia_ai_package)

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_RABBITMQ_IP/PORT` | `localhost` / `5672` | Broker. |
| `BACKEND_RABBITMQ_USER_NAME/PASSWORD` | `guest` / `guest` | Credentials. |
| `BACKEND_RABBITMQ_EXCHANGE` | `""` | Exchange (default direct). |
| `PYTHON_RABBITMQ_HEARBEAT_INTERVAL` | `60` | Heartbeat interval. |
| `BACKEND_RET_RESULT_RABBITMQ_QUEUE_NAME` | `ret_queue_dev` | Routing key / queue name. |

### Kafka (ES MACS)

| Variable | Default | Purpose |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Broker bootstrap. |
| `KAFKA_TOPIC_EVENT_PROCESS` | `event.process` | Topic. |

### S3 (thumbnail uploads — `BACKEND_S3_SECRET_KEY`)

| Variable | Default | Purpose |
|---|---|---|
| `BACKEND_S3_ACCESS_KEY/SECRET_KEY` | `""` | Credentials. Both empty → uploads are skipped (see note below). |
| `BACKEND_S3_THUMBNAIL_BUCKET_REGION` | `""` | Region. |
| `BACKEND_S3_ENDPOINT` | `""` (see warning below) | Custom endpoint (e.g. MinIO). Leave empty → AWS default. |
| `BACKEND_S3_THUMBNAIL_BUCKET_NAME` | `thumbnail` | Bucket. |

> **Latency note — empty `BACKEND_S3_ENDPOINT` adds ~2s per request unless creds are also empty.**
> S3 upload runs synchronously inside `/api/v1/validate` (see
> `server.py` — `await ... run_in_executor(...)` on `s3_uploader.upload_jpeg`).
> When `BACKEND_S3_ENDPOINT` is empty *and* `BACKEND_S3_ACCESS_KEY` is set,
> boto3 falls back to the real AWS endpoint (`s3.amazonaws.com`) and every
> upload pays DNS + TCP/TLS + SigV4 signing + default retry until it fails
> — measured at ~2000 ms per request. That latency is added directly to the
> validator's response time, dropping concurrent throughput by ~18×.
>
> Two ways to avoid the penalty:
> 1. **Empty credentials** (`BACKEND_S3_ACCESS_KEY=""`, `BACKEND_S3_SECRET_KEY=""`)
>    — `S3Uploader` short-circuits on construction: no boto3 client is built,
>    `upload_jpeg` returns the key without any network call, and the validator
>    logs a single `[S3] disabled (empty credentials)` line on first skip so
>    the disabled state is visible in operations.
> 2. **Fail-fast dummy endpoint** (`BACKEND_S3_ENDPOINT=http://127.0.0.1:1`)
>    — boto3 builds a client but every PUT fails with ConnectionRefused in a
>    few ms, so the validator's per-request latency stays near the vLLM-only
>    baseline. Use this when you want to keep the upload path live for later
>    swap-in of a real endpoint without restarting.
>
> `.env.example` ships option (1) by leaving credentials empty *and* sets the
> fail-fast endpoint anyway as belt-and-braces. Override both when wiring up
> MinIO/AWS for real.
>
> Measured on RTX PRO 6000 Blackwell + `vllm/vllm-openai:v0.14.0-cu130` +
> Qwen3-VL-Embedding-2B-FP8, fire-frame thumbnail (1080×1920 JPEG ~570 KB):
>
> | scenario | S3 step / req | N=10 throughput |
> |---|---:|---:|
> | empty endpoint, creds set (AWS fallback) | ~2000 ms | 3.3 req/s |
> | `http://127.0.0.1:1` (fail-fast endpoint) | 5–40 ms | 61.5 req/s |
> | empty creds (short-circuit, this PR) | ~0 ms | 250+ req/s |
> | MinIO on the same host | 8–67 ms | 65.4 req/s |

### Redis

| Variable | Default | Purpose |
|---|---|---|
| `REDIS_IP/PORT/DB` | `localhost` / `6379` / `0` | Connection. |
| `UUID_TTL_SECONDS` | `3600` | TTL for in-flight start UUIDs. |
| `UUID_KEY_PREFIX` | `pe_vle_2stage:uuid:` | Key namespace (distinct from pe_vqa). |

### Server bind

| Variable | Default | Purpose |
|---|---|---|
| `VALIDATION_SERVER_HOST/PORT` | `0.0.0.0` / `8200` | FastAPI bind. |

## Differences from pe_vqa_2stage's validation_server

| | pe_vqa_2stage | pe_vle (this) |
|---|---|---|
| Verification engine | Chat-completions VLM via vllm-openai (HTTP loopback) | `vllm.AsyncLLMEngine` in-process (no HTTP) |
| Container layout | One container, two processes (entrypoint.sh) | One container, one process |
| Per-category prompts | `prompts.py` text prompts | Anchor JSON loaded at startup |
| Wire shape | `/api/v1/validate`, `{"status":...}` envelope | identical |
| Publishing | Self-contained RabbitMQ/Kafka + S3 + Redis | identical (this commit ports it) |
| Fail-open env | `PE_VQA_2STAGE_FAIL_OPEN` | `PE_VLE_FAIL_OPEN` |
| UUID key prefix | `pe_vqa_2stage:uuid:` | `pe_vle_2stage:uuid:` |
| Default backend | `kafka` (ES MACS first) | identical (`kafka` default) |
