[English](README.md) | 한국어

# PE-VLE Validation Server

Single-container, single-process 검증 서비스. **완전 self-contained** —
`pia_prod` import 없음, monorepo bind-mount 없음, AI 서비스 호스트와
공유 상태 없음. `pe_vqa_2stage/validation_server`의 배포 형태를 라인
단위로 그대로 따릅니다.

```
PeVle2StageAsyncService ──fire+forget HTTP──▶ validation_server (one container, one process)
                                      ├─ vllm.AsyncLLMEngine (in-process, GPU)
                                      ├─ AnomalyClassifier (numpy cosine vs text anchors)
                                      ├─ S3Uploader (썸네일 업로드)
                                      ├─ RedisUUIDTracker (페어링 dedup)
                                      └─ MessagePublisher
                                            ├─ RabbitMQPublisher (KTT)
                                            └─ KafkaPublisher (ES MACS, fastavro)
                                                       │
                                                       ▼
                                                 RabbitMQ / Kafka
```

## 디렉토리 구조

```
validation_server/
├── Dockerfile           # FROM vllm/vllm-openai
├── README.md
├── config.py            # 환경변수 기반 상수 정의
├── docker-compose.yml   # GPU, /models 마운트, MQ/S3/Redis 환경변수
├── download_model.py    # HF Hub fetch (entrypoint 훅 + 호스트 수동 스크립트)
├── entrypoint.sh        # 빈 MODEL_PATH 감지 → 다운로드 → server.py 실행
├── publishers.py        # MessagePublisher / S3Uploader / RedisUUIDTracker / 메시지 빌더
├── requirements.txt
└── server.py            # FastAPI 앱: validate() 엔드포인트, embedder, classifier
```

## Quick start

`MODEL_DIR`은 필수입니다 — validator가 이 경로를 `/models`에 바인드
마운트하고 FP8 체크포인트와 anchor JSON을 저장합니다. `HF_MODEL_REPO_ID`는
해당 디렉토리가 비어있을 때 첫 실행 시 자동 fetch를 트리거합니다.

```bash
cd packages/pia_prod/AI/modules/pe_vle_2stage_async/validation_server

# validator가 read+write 가능한 호스트 디렉토리를 지정합니다. 첫 실행 시
# FP8 체크포인트와 text-features JSON이 여기로 다운로드되며, 이후 실행
# 시에는 다운로드를 건너뜁니다 (아래 "Model download" 참고).
#
# 이 monorepo 안에서 (in-tree 개발 기본):
export MODEL_DIR="$(git rev-parse --show-toplevel)/assets/model"
# 또는 본인이 관리하는 임의의 절대경로 (standalone / 비-git 배포):
# export MODEL_DIR="/srv/pe-vle/models"
# export MODEL_DIR="$HOME/pe_vle_models"
# 부모 디렉토리와 MODEL_NAME 서브디렉토리를 미리 생성하여 다운로드된
# 파일이 현재 사용자 소유로 들어오게 합니다 (그렇지 않으면 Docker가
# 첫 실행 시 root 소유로 자동 생성합니다).
mkdir -p "${MODEL_DIR}/${MODEL_NAME:-Qwen3-VL-Embedding-2B-FP8}"

HF_MODEL_REPO_ID="PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8" \
HF_AUTH_TOKEN="$HF_TOKEN" \
  docker compose up -d --build

# Health check (anchor JSON 로드 + vLLM weights 로드 — 첫 실행은 느림)
curl http://localhost:8200/health
# → {"status":"ok","anchor_buckets":["falldown","fire","smoke"],"model_path":"/models/Qwen3-VL-Embedding-2B-FP8","messaging_backend":"kafka","fail_open":true}
```

vLLM cold-start는 첫 실행 시 약 60–120초 소요됩니다 (HF fetch는 처음에
추가로 수 분 더 걸릴 수 있음). compose의 healthcheck `start_period`가
이를 감안합니다.

기본 메시지 백엔드는 **kafka** (ES MACS용)입니다. KTT 스타일 배포를 위해
RabbitMQ로 전환하려면:

```bash
MESSAGING_BACKEND=rabbitmq \
BACKEND_RABBITMQ_IP=rabbitmq.host \
BACKEND_RABBITMQ_USER_NAME=user BACKEND_RABBITMQ_PASSWORD=pass \
  docker compose up -d --build
```

### Override knobs (advanced)

기본값은 `PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8`을
`${MODEL_DIR}/Qwen3-VL-Embedding-2B-FP8/`에 다운로드합니다. 다운로드
출처와 디스크상의 서브디렉토리 이름은 환경변수 기반이라 코드 수정 없이
override 가능합니다:

- `HF_MODEL_REPO_ID` — fetch할 HF repo (기본 `PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8`).
- `MODEL_NAME` — 호스트(`${MODEL_DIR}/${MODEL_NAME}`)와
  컨테이너(`/models/${MODEL_NAME}`) 양쪽의 서브디렉토리 이름.
  기본 `Qwen3-VL-Embedding-2B-FP8`.
- `HF_TEXT_FEATURES_FILE` — repo 내 anchor JSON 파일명이
  `VLE_FP8_text_features.json`과 다를 경우 지정.
- `HF_TEXT_FEATURES_REPO_ID` — anchor JSON이 모델과 별도 repo에 있을
  경우 지정.

`docker compose up` 전에 이들을 설정하세요. 아래 "Model download"
섹션의 호스트↔컨테이너 매핑이 이 변수들이 어떻게 흐르는지 보여줍니다.

### Alternative: `.env` 파일 워크플로우

`docker compose`는 작업 디렉토리의 `.env` 파일을 자동 로드합니다. 따라서
위 inline 환경변수를 파일 기반으로 옮길 수 있습니다:

```bash
cp .env.example .env
# .env 편집: HF_AUTH_TOKEN 설정, 필요시 MODEL_DIR / MODEL_NAME / HF_MODEL_REPO_ID override
docker compose up -d --build
```

기본 `.env.example`은 `MODEL_DIR`을 monorepo의 `assets/model`(상대경로)로
설정합니다. standalone 배포의 경우 절대경로로 변경하세요.

## Model download

validator는 런타임 코드 외에 두 가지 아티팩트가 필요합니다:

1. **FP8 체크포인트** — Qwen3-VL-Embedding-2B의 FP8-quantized weights
   HuggingFace snapshot. `${MODEL_DIR}/${MODEL_NAME}/`에 위치.
2. **Text-features JSON** (`VLE_FP8_text_features.json`) — `AnomalyClassifier`가
   사용하는 사전 계산된 normal/target 텍스트 임베딩 anchor. 기본적으로
   체크포인트와 같은 위치에 저장됩니다.

### 호스트 디렉토리가 컨테이너에 어떻게 매핑되는가

```
호스트 파일시스템                            컨테이너 내부
${MODEL_DIR}/                                /models/
└── ${MODEL_NAME}/            ──:rw──>       └── ${MODEL_NAME}/            ← MODEL_PATH
    ├── model.safetensors                        ├── model.safetensors
    └── VLE_FP8_text_features.json               └── VLE_FP8_text_features.json
```

- `MODEL_DIR` — 사용자가 선택하는 호스트 디렉토리 (필수, 기본값 없음).
- `MODEL_NAME` — 호스트와 컨테이너 양쪽의 서브디렉토리 이름
  (기본 `Qwen3-VL-Embedding-2B-FP8`).
- `MODEL_PATH` — 컨테이너 내부에서 `/models/${MODEL_NAME}`으로 결정됨.
- `:rw` — 다운로드된 파일이 호스트에 영구 저장됨; 이후 실행 시 fetch를 건너뜀.

이 두 아티팩트를 획득하는 세 가지 운영 경로가 있습니다. 자신의 배포
상황에 맞는 것을 고르세요. 상호 배타적이지 않습니다.

### Path A — `entrypoint.sh`를 통한 자동 fetch (기본)

컨테이너의 entrypoint가 부팅 시 `${MODEL_PATH}`를 검사합니다.
`model.safetensors` 또는 text-features JSON이 없으면 `download_model.py`를
호출하여 HF Hub에서 가져온 후 `server.py`를 시작합니다. 다운로드된
파일은 호스트의 `${MODEL_DIR}` 아래에 영구 저장됩니다 (볼륨 마운트가
`:rw`이므로). 두 번째 실행 및 모든 sibling validator는 다운로드를
건너뜁니다.

필수 환경변수 (shell, `.env`, 또는 compose의 `environment:` 블록에 설정):

| 변수 | 용도 |
|---|---|
| `HF_MODEL_REPO_ID` | FP8 체크포인트의 repo ID (예: `<owner>/Qwen3-VL-Embedding-2B-FP8`). |
| `HF_AUTH_TOKEN` | private/gated repo에 필요. compose는 `HF_AUTH_TOKEN`만 forward합니다 (`HF_TOKEN`은 forward하지 않음); 토큰이 `$HF_TOKEN`(HF CLI 기본값)에 있다면 명시적으로 매핑하세요: `HF_AUTH_TOKEN="$HF_TOKEN" docker compose up`. `download_model.py`의 `$HF_TOKEN` 폴백은 호스트에서 스크립트를 직접 실행할 때(Path B)만 적용됩니다. |
| `HF_TEXT_FEATURES_REPO_ID` | 선택 — 기본값은 `HF_MODEL_REPO_ID`. |
| `HF_TEXT_FEATURES_FILE` | 선택 — 기본값은 `VLE_FP8_text_features.json`. |

다운로드 후 `entrypoint.sh`가 자체적으로 `HF_HUB_OFFLINE=1`을 export
합니다. 따라서 정상 서비스 중에는 vLLM이 HF Hub에 접근하지 않습니다.
동일한 compose 파일이 first-run-online과 subsequent-offline 두 케이스를
모두 처리합니다.

### Path B — 호스트에서 수동 다운로드 (air-gapped GPU 노드용)

배포 대상에 인터넷이 없는 경우, 인터넷이 되는 별도 호스트에서 다운로드한
후 결과 디렉토리를 동기화합니다.

```bash
# 인터넷 가능한 호스트(예: 노트북)에서:
pip install huggingface_hub
# 이 디렉토리에서 실행 (packages/pia_prod/AI/modules/pe_vle_2stage_async/validation_server/):
HF_TOKEN="$HF_TOKEN" python download_model.py \
    --model-dir /tmp/Qwen3-VL-Embedding-2B-FP8 \
    --repo-id PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8

# GPU 노드로 전송:
rsync -aP /tmp/Qwen3-VL-Embedding-2B-FP8 user@gpu-node:/srv/models/

# GPU 노드에서:
mkdir -p /srv/models
MODEL_DIR=/srv/models docker compose up -d --build
# entrypoint이 채워진 디렉토리를 감지 → 다운로드 건너뜀 → server.py 시작
```

### Path C — `up` 전에 마운트 위치를 미리 채우기

스냅샷을 `${MODEL_DIR}/Qwen3-VL-Embedding-2B-FP8/`에 직접 복사합니다
(예: `huggingface-cli download` 사용, 또는 다른 노드에서 rsync).
entrypoint의 존재 여부 검사가 다운로드를 건너뜁니다.

```bash
# Prerequisites: 호스트에 huggingface_hub 설치 + 토큰 캐시
pip install huggingface_hub
huggingface-cli login   # ~/.cache/huggingface/token에 토큰 캐시
                        # (또는 $HF_TOKEN 환경변수 사용)

# MODEL_DIR이 설정되어 있고 디렉토리가 존재해야 합니다 (Quick start 참고).
# HF_MODEL_REPO_ID와 MODEL_NAME은 unset일 때 PIA-SPACE-LAB 기본값으로
# 폴백됩니다 — 자체 체크포인트 사용 시 override하세요.
huggingface-cli download "${HF_MODEL_REPO_ID:-PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8}" \
    --local-dir "${MODEL_DIR}/${MODEL_NAME:-Qwen3-VL-Embedding-2B-FP8}"
docker compose up -d --build
```

### 실패 모드

- `docker compose up`에서 `MODEL_DIR is required` → 환경변수 미설정.
  ("Quick start" 참고하여 설정 후 재시도).
- entrypoint에서 `ERROR: HF_MODEL_REPO_ID is unset and ... is incomplete`
  → `HF_MODEL_REPO_ID`를 설정하거나(Path A), 마운트를 미리 채우세요(Path B/C).
- 다운로드 성공 후 `RuntimeError: No text_features loaded from ...` →
  HF repo 내부의 실제 파일명과 `HF_TEXT_FEATURES_FILE`이 일치하는지 확인
  (기본 기대값은 `VLE_FP8_text_features.json`).

### 오프라인 운영

`${MODEL_DIR}`에 체크포인트와 anchor JSON이 모두 갖춰지면, validator는
더 이상 HF Hub에 접근할 필요가 없습니다 — `entrypoint.sh`가 `server.py`
실행 전에 `HF_HUB_OFFLINE=1`을 export합니다. 검증을 위해 `HF_AUTH_TOKEN=`
(빈 값) 및 `HF_MODEL_REPO_ID=`(빈 값)으로 설정하거나, 컨테이너를
`--network none`으로 실행하여 connectivity smoke test를 진행하세요.

## Why fully self-contained?

- **호스트 간 결합 제로**: `/opt/mono` bind-mount 없음; validator 호스트에
  `pia_prod` 패키지 설치 불필요. 컨테이너는 FP8 체크포인트가 있는 모든
  GPU 호스트로 portable합니다.
- **Direct publishing**: validator가 `pika.basic_publish` /
  `confluent_kafka.Producer.produce`로 RabbitMQ(또는 Kafka)에 직접
  publish합니다. `pia_prod.AI.utils.init`의 공유 producer 스레드를 거치는
  `match_outputs` indirection 없음.
- **독립적인 스케일링**: load balancer 뒤에 multiple validator replica를
  실행 가능; 각 replica가 자체 MQ/S3/Redis 연결을 보유합니다.

## Wire contract (pe_vqa_2stage와 동일)

### `POST /api/v1/validate`

요청:

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

- `is_start=True`인 경우 `thumbnail_b64` 필수 → 없으면 400.

응답:

```json
{"status": "sent",       "uuid": "abc-123"}
{"status": "discarded",  "uuid": "abc-123", "reason": "vle_rejected"}
{"status": "discarded",  "uuid": "abc-123", "reason": "vle_error"}            // PE_VLE_FAIL_OPEN=false인 경우만
{"status": "discarded",  "uuid": "abc-123", "reason": "start_not_validated"} // is_start=False인데 매칭되는 start 없음
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

| Stage | 입력 | 결과 | `status` | `reason` |
|---|---|---|---|---|
| Boundary | `is_start=True`, `thumbnail_b64` 없음 | 400 (boundary에서 거부) | — | — |
| End-state | `is_start=False`, UUID가 Redis에 **등록됨** | publish + UUID 삭제 | `sent` | — |
| End-state | `is_start=False`, UUID **미등록** | drop (orphan end) | `discarded` | `start_not_validated` |
| vle_id 누락 | `vle_id` 매핑 없음 (예: falldown) | vLLM 건너뜀; UUID 등록; S3 + publish | `sent` | — |
| Anchor classify | `target_max > normal_max` | UUID 등록; S3 + publish | `sent` | — |
| Anchor classify | `target_max ≤ normal_max` | drop | `discarded` | `vle_rejected` |
| vLLM 에러 | `PE_VLE_FAIL_OPEN=true` (기본) | confirmed로 처리; UUID 등록; S3 + publish | `sent` | — |
| vLLM 에러 | `PE_VLE_FAIL_OPEN=false` | drop | `discarded` | `vle_error` |
| Publish 에러 | `pika`/`confluent_kafka` 예외 발생 | server-side 에러 | `error` | exception text |

## 환경변수

### vLLM

| 변수 | 기본값 | 용도 |
|---|---|---|
| `MODEL_PATH` | `/models/Qwen3-VL-Embedding-2B-FP8` | 컨테이너의 FP8 체크포인트 디렉토리 경로. |
| `MODEL_NAME` | `Qwen3-VL-Embedding-2B-FP8` | 호스트 `MODEL_DIR` 아래 서브디렉토리 이름. |
| `MODEL_DIR` | `…/assets/model` | `/models`에 마운트되는 호스트 디렉토리 (read-write — entrypoint가 첫 실행 시 체크포인트를 다운로드). |
| `DTYPE` | `bfloat16` | vLLM 연산 dtype. |
| `GPU_MEMORY_UTILIZATION` | `0.3` | vLLM이 예약할 수 있는 GPU 메모리 비율 (KV cache auto-sizing). |
| `KV_CACHE_MEMORY_BYTES` | `1G` | **KV cache 크기 hard cap** (위의 auto-sizing보다 우선). 형식: `1G`, `512M`, raw int bytes. 빈 문자열로 설정하면 `GPU_MEMORY_UTILIZATION` 기반 vLLM auto-sizing으로 폴백. |
| `MAX_NUM_SEQS` | _unset_ (vLLM 기본) | 엔진 배치의 최대 동시 시퀀스 수. 배치 크기 제한이 필요하면 `benchmark.py`로 튜닝. |
| `MAX_NUM_BATCHED_TOKENS` | _unset_ (vLLM 기본) | scheduler step당 최대 토큰 수. |
| `VLLM_MAX_CONCURRENCY` | `10` | FastAPI 레이어의 in-flight `/api/v1/validate` 호출 cap (asyncio Semaphore). backpressure 다이얼이며 parallelism 다이얼이 아님. |
| `MAX_MODEL_LEN` | `8192` | 최대 시퀀스 길이. |
| `LIMIT_MM_PER_PROMPT_VIDEO/IMAGE` | `1` | 프롬프트당 multi-modal cap. |
| `QWEN3VLE_VLLM_TEXT_FEATURES_PATH` | `/models/.../VLE_FP8_text_features.json` | Anchor JSON 경로. |

### Policy

| 변수 | 기본값 | 용도 |
|---|---|---|
| `PE_VLE_FAIL_OPEN` | `true` | true면 vLLM 에러 발생 시 그래도 publish. |
| `PE_VLE_TO_VLE_CATEGORY_EVENT_MAP_JSON` | `{"fire_pe_vle_ret":"fire_vle_ret",...}` | PE→VLE 카테고리 매핑. |
| `VLE_CATEGORY_EVENT_MAP_JSON` | `{"fire":["fire_vle_ret",...],...}` | vle_id→bucket 매핑. |

### 메시지 백엔드

| 변수 | 기본값 | 용도 |
|---|---|---|
| `MESSAGING_BACKEND` | `kafka` | `kafka` (ES MACS, 기본) 또는 `rabbitmq` (KTT). |

### RabbitMQ (KTT — pia_ai_package와 동일 키)

| 변수 | 기본값 | 용도 |
|---|---|---|
| `BACKEND_RABBITMQ_IP/PORT` | `localhost` / `5672` | 브로커. |
| `BACKEND_RABBITMQ_USER_NAME/PASSWORD` | `guest` / `guest` | 인증 정보. |
| `BACKEND_RABBITMQ_EXCHANGE` | `""` | Exchange (기본 direct). |
| `PYTHON_RABBITMQ_HEARBEAT_INTERVAL` | `60` | Heartbeat 인터벌. |
| `BACKEND_RET_RESULT_RABBITMQ_QUEUE_NAME` | `ret_queue_dev` | Routing key / 큐 이름. |

### Kafka (ES MACS)

| 변수 | 기본값 | 용도 |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | 브로커 bootstrap. |
| `KAFKA_TOPIC_EVENT_PROCESS` | `event.process` | 토픽. |

### S3 (썸네일 업로드 — `BACKEND_S3_SECRET_KEY`)

| 변수 | 기본값 | 용도 |
|---|---|---|
| `BACKEND_S3_ACCESS_KEY/SECRET_KEY` | `""` | 인증 정보. 둘 다 빈 값이면 업로드를 skip합니다 (아래 참고). |
| `BACKEND_S3_THUMBNAIL_BUCKET_REGION` | `""` | 리전. |
| `BACKEND_S3_ENDPOINT` | `""` (아래 경고 참고) | 커스텀 엔드포인트 (예: MinIO). 빈 값으로 두면 AWS 기본. |
| `BACKEND_S3_THUMBNAIL_BUCKET_NAME` | `thumbnail` | 버킷. |

> **Latency 주의 — credential은 채워졌는데 endpoint만 비어있으면 request당 ~2초 페널티.**
> S3 업로드는 `/api/v1/validate` 핸들러 안에서 동기적으로 await됩니다
> (`server.py`의 `s3_uploader.upload_jpeg`에 대한
> `await ... run_in_executor(...)` 호출 참고).
> `BACKEND_S3_ENDPOINT`가 빈 값인 상태에서 `BACKEND_S3_ACCESS_KEY`가 채워져
> 있으면 boto3가 실제 AWS 엔드포인트(`s3.amazonaws.com`)로 fallback 시도하면서
> 매 업로드마다 DNS + TCP/TLS + SigV4 서명 + 기본 retry를 수행한 뒤 실패합니다.
> 측정 결과 ~2000 ms / request이며, 동시 처리량이 약 18배 감소합니다.
>
> 페널티 회피 방법 두 가지:
> 1. **Credential 비우기** (`BACKEND_S3_ACCESS_KEY=""`, `BACKEND_S3_SECRET_KEY=""`)
>    — `S3Uploader`가 생성 시점에 short-circuit합니다. boto3 클라이언트 자체를
>    만들지 않고, `upload_jpeg`는 key만 만들어 반환합니다. 첫 skip 시
>    `[S3] disabled (empty credentials)` 경고를 한 번 남겨서 운영 측에서 disabled
>    상태가 보이도록 합니다.
> 2. **Fail-fast dummy 엔드포인트** (`BACKEND_S3_ENDPOINT=http://127.0.0.1:1`)
>    — boto3 클라이언트는 만들어지지만 모든 PUT이 ConnectionRefused로 수 ms
>    이내에 실패합니다. validator의 request당 latency가 vLLM 단독 baseline
>    근처에 유지됩니다. 업로드 path를 살려두고 나중에 실제 endpoint로 swap-in
>    하려는 경우에 사용하세요.
>
> `.env.example`은 (1)을 기본으로 (credential을 빈 값으로 둠) + (2)도 함께 적용
> (fail-fast endpoint default 지정) 하는 belt-and-braces 구성입니다.
> MinIO/AWS를 실제로 연결할 때 둘 다 덮어쓰면 됩니다.
>
> 측정 환경: RTX PRO 6000 Blackwell + `vllm/vllm-openai:v0.14.0-cu130` +
> Qwen3-VL-Embedding-2B-FP8, fire 영상 thumbnail (1080×1920 JPEG ~570 KB):
>
> | 시나리오 | S3 단계 / req | N=10 throughput |
> |---|---:|---:|
> | 빈 endpoint, credential 채워짐 (AWS fallback) | ~2000 ms | 3.3 req/s |
> | `http://127.0.0.1:1` (fail-fast endpoint) | 5–40 ms | 61.5 req/s |
> | 빈 credential (short-circuit, 본 PR) | ~0 ms | 250+ req/s |
> | 같은 호스트 MinIO | 8–67 ms | 65.4 req/s |

### Redis

| 변수 | 기본값 | 용도 |
|---|---|---|
| `REDIS_IP/PORT/DB` | `localhost` / `6379` / `0` | 연결. |
| `UUID_TTL_SECONDS` | `3600` | in-flight start UUID의 TTL. |
| `UUID_KEY_PREFIX` | `pe_vle_2stage:uuid:` | 키 네임스페이스 (pe_vqa와 구분). |

### 서버 바인드

| 변수 | 기본값 | 용도 |
|---|---|---|
| `VALIDATION_SERVER_HOST/PORT` | `0.0.0.0` / `8200` | FastAPI 바인드. |

## pe_vqa_2stage validation_server와의 차이

| | pe_vqa_2stage | pe_vle (this) |
|---|---|---|
| Verification 엔진 | Chat-completions VLM via vllm-openai (HTTP loopback) | `vllm.AsyncLLMEngine` in-process (HTTP 없음) |
| 컨테이너 구성 | One container, two processes (entrypoint.sh) | One container, one process |
| 카테고리별 프롬프트 | `prompts.py` 텍스트 프롬프트 | 시작 시 anchor JSON 로드 |
| Wire shape | `/api/v1/validate`, `{"status":...}` envelope | 동일 |
| Publishing | Self-contained RabbitMQ/Kafka + S3 + Redis | 동일 (이 commit에서 포팅) |
| Fail-open env | `PE_VQA_2STAGE_FAIL_OPEN` | `PE_VLE_FAIL_OPEN` |
| UUID key prefix | `pe_vqa_2stage:uuid:` | `pe_vle_2stage:uuid:` |
| 기본 백엔드 | `kafka` (ES MACS first) | identical (`kafka` 기본) |
