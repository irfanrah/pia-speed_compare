# PE VQA 2-Stage Validation Server

PE(Perception Encoder)가 1차 탐지한 알람을 VLM(Qwen3.5)으로 2차 검증 후, 확정된 알람만 RabbitMQ로 전송하는 서버.

## 구조

단일 Docker 컨테이너 안에 두 프로세스가 함께 실행된다.

```
컨테이너 (pe-vqa-2stage)
├── vLLM Server      (내부 localhost:8000, 외부 미노출)
└── Validation Server (FastAPI, 외부 노출 기본 8100)
```

전체 알람 흐름:

```
[메인 inference]                       [validation_server 컨테이너]
PE 추론 → EventManager 알람 발생         ├─ /api/v1/validate 수신
  ↓                                    ├─ vLLM 호출 (yes/no 판정)
PE_VQA_2STAGE_VALIDATION_ENABLED=True?    ├─ Redis UUID 등록 (시작 알람)
  ↓ True                               ├─ S3 thumbnail 업로드
fire-and-forget HTTP POST   ────→     ├─ make_alarm_message → RabbitMQ publish
  → /api/v1/validate                   └─ ts/cameraId(int)/키 셋 KTT 표준
  ↓ False (또는 비대상 카테고리)
match_outputs → KTT alarm_producer로 직접 발사
```

## 메시징 백엔드 선택 (KTT vs ES MACS)

`MESSAGING_BACKEND` 환경변수로 두 환경 모두 지원:

| 백엔드 | 사용처 | 메시지 형식 | 인프라 |
|--------|--------|-----------|--------|
| `kafka` (**default**) | ES MACS (com.macs.events.eventStart) | Avro fastavro schemaless, ts long(epoch ms), cameraId string, thumbnail string("" 허용) | Kafka |
| `rabbitmq` | KTT (Product-AI-mono / Package-Common-AI-pia_ai_package) | JSON, ts ISO8601 microsecond, cameraId int | RabbitMQ |

같은 컨테이너 이미지로 두 환경 모두 지원. 빌드 시 둘 다 install되며 운영자가 ENV로 선택.

> **주의**: default가 `kafka`(ES MACS)이므로 KTT 환경에서는 반드시 `.env` 또는 docker-compose env에 `MESSAGING_BACKEND=rabbitmq`를 명시해야 한다. 미설정 시 컨테이너가 KAFKA broker를 찾으려고 시도하다 실패한다.

## 사전 인프라 (이미 운영 중이라 가정)

| 컴포넌트 | 용도 | 컨테이너 접근 | 사용 모드 |
|---------|------|------------|-----------|
| RabbitMQ | 백엔드 알람 큐 | `BACKEND_RABBITMQ_IP:BACKEND_RABBITMQ_PORT` | rabbitmq |
| Kafka | 백엔드 알람 토픽 (Avro fastavro schemaless) | `KAFKA_BOOTSTRAP_SERVERS` | kafka |
| Redis | UUIDTracker 영속화 (재시작 후에도 종료 알람 매칭) | `REDIS_IP:REDIS_PORT` | 공통 |
| S3 / MinIO | thumbnail 저장 | `BACKEND_S3_ENDPOINT` | 공통 |

선택한 백엔드의 인프라가 모두 떠 있고 컨테이너에서 접근 가능해야 한다.

## GPU 아키텍처별 베이스 이미지

`.env`에서 `VLLM_IMAGE`를 GPU에 맞게 설정한다.

| GPU | VLLM_IMAGE |
|-----|-----------|
| Blackwell (B200 등) | `vllm/vllm-openai:cu130-nightly` |
| Hopper / Ampere (H100, A100) | `vllm/vllm-openai:latest` |

`docker build`로 직접 빌드 시 build-arg로도 지정 가능:
```bash
docker build -t pe-vqa-2stage --build-arg BASE_IMAGE=vllm/vllm-openai:latest .
```

## 빠른 시작

### 1. 환경 설정

```bash
cd packages/pia_prod/AI/modules/pe_vqa_2stage/validation_server
cp .env.example .env
vi .env  # GPU 아키, MODEL_HF_CACHE_DIR, Kafka/RabbitMQ/Redis/S3 자격증명 확인
```

### 2. (권장) HF 모델 사전 다운로드

컨테이너 첫 기동 시 모델 다운로드(~3-5분)를 피하려면 호스트 측에서 미리 받는다. Product-AI-mono 다른 모듈과 통일된 경로(`<repo_root>/assets/model/huggingface`)에 캐시.

#### 방법 A — 동봉 스크립트 (권장)

같은 디렉토리의 `.env`를 자동 로드해 `VLLM_MODEL` / `MODEL_HF_CACHE_DIR` / `HF_TOKEN`을 읽는다. docker compose와 같은 모델/캐시 경로를 보장.

```bash
pip install huggingface_hub
python download_model.py                                # .env의 VLLM_MODEL 사용 (또는 default Qwen/Qwen3.5-0.8B)
# .env 무시하고 다른 Qwen3.5 변종으로 한 번만 받기:
VLLM_MODEL=Qwen/Qwen3.5-<variant> python download_model.py
# 캐시 경로 override:
MODEL_HF_CACHE_DIR=/custom/path python download_model.py
# 비공개 모델 (HF_TOKEN 필요):
HF_TOKEN=hf_xxx python download_model.py
```

#### 방법 B — `huggingface-cli` (또는 `hf` CLI)

vLLM이 찾는 위치(`HF_HOME/hub/`)에 직접 받으려면 `--local-dir`가 아니라 `--cache-dir`로 받아야 한다. (`--local-dir`은 평면 구조라 vLLM 캐시 적중 안 됨.)

```bash
pip install -U huggingface_hub

# 모델 (Qwen3.5-0.8B 기준)
huggingface-cli download Qwen/Qwen3.5-0.8B \
  --cache-dir <repo_root>/assets/model/huggingface/hub
```

다른 모델로 받고 싶으면 `Qwen/Qwen3.5-0.8B`만 바꾸면 된다. 비공개 모델은 `--token hf_xxx`.

> `--local-dir`을 쓰면 캐시 형식이 아닌 평면 구조로 받기 때문에 vLLM이 모델을 못 찾고 다시 다운로드한다. 반드시 `--cache-dir <hub>` 형태를 사용.

#### 다운로드 후

세 가지 방식 모두 결과는 같음:
- 호스트: `<repo_root>/assets/model/huggingface/hub/models--<owner>--<repo>/...`
- 컨테이너: `/app/assets/model/huggingface/hub/...` 로 마운트되어 vLLM이 즉시 인식

`.env`의 `MODEL_HF_CACHE_DIR`에 호스트 경로를 절대경로로 설정하면 docker compose가 해당 캐시를 자동 마운트.

k8s Pod 운영 시: PVC를 `/app/assets/model/huggingface`에 마운트하면 첫 1회만 다운로드, 이후 Pod 재시작/스케일아웃에도 재사용됨.

### 3. 빌드 + 실행

```bash
docker compose up --build -d
```

### 4. 상태 확인

```bash
# 컨테이너 로그
docker logs -f pe-vqa-2stage

# health check
curl http://localhost:8100/health
# → {"status":"ok"}
```

### 5. 메인 inference 측 ENV 활성화

컨테이너만 띄워도 자동 활성화 안 된다. 메인 inference 측에서 다음 환경변수 설정 필요:

```bash
# 메인 inference 컨테이너의 .env
PE_VQA_2STAGE_VALIDATION_ENABLED=True
PE_VQA_2STAGE_VALIDATION_HOST=<validation_server 호스트>  # 같은 docker network면 'pe-vqa-2stage'
PE_VQA_2STAGE_VALIDATION_PORT=8100
TWO_STEP_CATEGORIES=fire_pe_vqa,화재_pe_vqa,smoke_pe_vqa,연기_pe_vqa,falldown_pe_vqa,쓰러짐_pe_vqa,smoking_pe_vqa,흡연_pe_vqa
PE_VQA_2STAGE_QUEUE_SIZE=10
PE_VQA_2STAGE_ALARM_DURATION_THRESHOLD=5
```

`PE_VQA_2STAGE_VALIDATION_ENABLED=False`(기본값) 또는 미설정 시 메인 PE가 다른 PE 모듈처럼 KTT alarm_producer로 직접 발사한다 (validation_server 안 거침).

## 환경변수 (validation_server 컨테이너)

### vLLM

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VLLM_IMAGE` | `vllm/vllm-openai:cu130-nightly` | docker-compose 베이스 이미지 (GPU 아키) |
| `VLLM_MODEL` | `Qwen/Qwen3.5-0.8B` | 모델명 (soil_qwen35와 통일) |
| `VLLM_API_BASE` | `http://localhost:8000/v1` | wrapper가 vLLM 호출할 URL (컨테이너 내부) |
| `TENSOR_PARALLEL_SIZE` | `1` | GPU 병렬 수 |
| `MAX_MODEL_LEN` | `8192` | 최대 컨텍스트 길이 |
| `GPU_MEMORY_UTILIZATION` | `0.3` | GPU 메모리 사용률 |
| `KV_CACHE_MEMORY_BYTES` | `1G` | KV Cache 상한 |
| `HF_TOKEN` | (미설정) | 비공개 모델용 HuggingFace 토큰 |
| `MODEL_HF_CACHE_DIR` | `${HOME}/.cache/huggingface` | 호스트 측 HF 모델 캐시 경로. 컨테이너 내부 `/app/assets/model/huggingface`에 마운트됨. Product-AI-mono 다른 모듈과 통일하려면 절대경로(`<repo_root>/assets/model/huggingface`) 권장 |

### Validation Server (FastAPI wrapper)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VALIDATION_SERVER_HOST` | `0.0.0.0` | bind 호스트 |
| `VALIDATION_SERVER_PORT` | `8100` | bind 포트 |
| `VLLM_MAX_CONCURRENCY` | `10` | VLM 동시 요청 수 (semaphore) |
| `VLLM_TIMEOUT` | `120` | VLM 요청 타임아웃(초) |
| `VLLM_MAX_TOKENS` | `64` | VLM 응답 최대 토큰 |
| `PE_VQA_2STAGE_FAIL_OPEN` | `true` | VLM 호출 실패 시 통과 정책 (critical 이벤트 보호) |

### RabbitMQ (Product-AI-mono 표준 키 — Package-Common-AI-pia_ai_package와 동일)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BACKEND_RABBITMQ_IP` | `rabbitmq` | 호스트 |
| `BACKEND_RABBITMQ_PORT` | `5672` | 포트 |
| `BACKEND_RABBITMQ_USER_NAME` | `guest` | 사용자 |
| `BACKEND_RABBITMQ_PASSWORD` | `guest` | 비밀번호 |
| `BACKEND_RABBITMQ_EXCHANGE` | `` | exchange (default direct) |
| `PYTHON_RABBITMQ_HEARBEAT_INTERVAL` | `60` | heartbeat 초 |
| `BACKEND_RET_RESULT_RABBITMQ_QUEUE_NAME` | `ret_queue_dev` | 알람 발사 큐 이름 |

### S3 (MinIO / AWS S3) — KTT 표준 키

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BACKEND_S3_ACCESS_KEY` | (빈값) | 접근 키 |
| `BACKEND_S3_SECRET_KEY` | (빈값) | 비밀 키 |
| `BACKEND_S3_THUMBNAIL_BUCKET_REGION` | (빈값) | 리전 |
| `BACKEND_S3_ENDPOINT` | (빈값) | S3 엔드포인트 URL (MinIO 등) |
| `BACKEND_S3_THUMBNAIL_BUCKET_NAME` | `thumbnail` | thumbnail 버킷 이름 |

### Redis (UUIDTracker 영속화) — KTT 표준 키

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `REDIS_IP` | `localhost` | 호스트 |
| `REDIS_PORT` | `6379` | 포트 |
| `REDIS_DB` | `0` | DB 번호 |
| `UUID_TTL_SECONDS` | `3600` | 시작 알람 UUID 보존 기간 (1h) |

### Kafka (MESSAGING_BACKEND=kafka일 때만 사용 — ES MACS 환경)

ES MACS와 동일한 `com.macs.events.eventStart` Avro 스키마로 `fastavro.schemaless_writer` 직렬화 (ES backend의 `schemaless_reader`와 와이어 포맷 매칭).

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `KAFKA_BOOTSTRAP_SERVERS` | `localhost:9092` | Kafka broker 주소 (host:port,host:port,...) |
| `KAFKA_TOPIC_EVENT_PROCESS` | `event.process` | 알람 발사 토픽명 (ES MACS의 `OutboundTopic.EVENT_PROCESS`와 일치) |

> Schema Registry는 사용하지 않는다. ES MACS backend가 `fastavro.schemaless_reader`로 직접 디코드하므로, validation_server도 동일하게 `fastavro.schemaless_writer`로 인코드해 와이어 포맷이 매칭된다. Confluent Schema Registry 매직 바이트 prefix(0x00 + 4-byte schema_id)를 붙이면 ES consumer가 5바이트만큼 어긋나 디코드 실패하므로 사용 금지.

> **다른 사용처에서 자체 키를 쓰는 경우**: docker-compose의 environment 또는 docker run -e로 KTT 표준 키에 자체 자격증명을 매핑해 주입하면 된다. 예:
> ```yaml
> environment:
>   - BACKEND_RABBITMQ_IP=${YOUR_OWN_MQ_HOST}
> ```

## 알람 메시지 형식 (백엔드별)

### RabbitMQ 모드 (KTT)

다른 PE 기반 모듈(`perception_encoder`, `ft_pe`, `pe_violence` 등)과 동일한 JSON 키 셋:

```json
{
  "cameraId": 6,
  "type": "retEvent",
  "organization": "pia",
  "name": "fire_pe_vqa",
  "isStart": true,
  "thumbnail": "6_pia_2026-04-28T04:51:25.478576.jpg",
  "incidentThresholdSecond": 5,
  "incidentTimeoutSecond": 30,
  "uuid": "06e8f198-1ac4-43df-b680-5be63eeea48f",
  "ts": "2026-04-28T04:51:25.494703"
}
```

- `cameraId` int (다른 모듈과 동일)
- `ts` microsecond 6자리, Z suffix 없음 (KTT `str_UTC_ISO8601_ms_now_time` 표준)
- `thumbnail` 빈 문자열 또는 S3 filename

### Kafka 모드 (ES MACS)

`com.macs.events.eventStart` Avro 스키마로 직렬화. ES MACS 레포의 `outbound_event_start_process.py`와 1:1 일치:

```json
{
  "cameraId": "6",
  "type": "retEvent",
  "organization": "pia",
  "name": "fire_pe_vqa",
  "isStart": true,
  "thumbnail": "6_pia_2026-04-28T04:51:25.478576.jpg",
  "incidentThresholdSecond": 5,
  "incidentTimeoutSecond": 30,
  "uuid": "06e8f198-1ac4-43df-b680-5be63eeea48f",
  "ts": 1745809885495
}
```

- `cameraId` string (Avro 스키마 정의)
- `ts` long, epoch milliseconds (Avro logical type `timestamp-millis`)
- `thumbnail` nullable — 빈 값일 때 null로 변환됨

## API

### POST /api/v1/validate

메인 inference에서 전달받은 알람을 vLLM으로 검증 후 통과한 알람만 RabbitMQ에 publish.

요청:
```json
{
  "thumbnail_b64": "base64_jpeg_string",
  "is_start": true,
  "category_name": "fire_pe_vqa",
  "stream_id": "6_pia-inference",
  "event_uuid": "uuid-string",
  "event_type": "retEvent",
  "user_param": {
    "user_param": {
      "cameraId": 6,
      "organization": "pia",
      "retEvent": {
        "fire_pe_vqa": {
          "name": "fire_pe_vqa",
          "incidentThresholdSecond": 5,
          "incidentTimeoutSecond": 30
        }
      }
    }
  }
}
```

응답:
```json
{"status": "sent", "uuid": "uuid-string"}
{"status": "discarded", "uuid": "uuid-string", "reason": "vlm_rejected"}
{"status": "discarded", "uuid": "uuid-string", "reason": "start_not_validated"}
{"status": "error", "uuid": "uuid-string", "reason": "..."}
```

처리 흐름:
- `is_start=True`: VLM 검증 → "yes"면 Redis UUID 등록 + S3 업로드 + RabbitMQ publish, "no"면 폐기
- `is_start=False`: Redis UUID 매칭 시 RabbitMQ publish + 키 삭제, 미매칭 시 폐기

### GET /health

```json
{"status": "ok"}
```

## 운영 시나리오

### 사고 지속 중 알람 폭주 방지

PE EventManager가 정상 사이클(0→1→2→3→0) 유지. 사고 지속 동안 시작 알람 1회만 발사. 메인 측 `service.py`의 PE state reset 로직을 제거한 결과.

### 컨테이너 재시작 시 종료 알람 매칭

UUIDTracker가 Redis에 보존되므로 컨테이너 재시작 후에도 진행 중 사고의 종료 알람이 정상 매칭되어 발사됨.

### vLLM 호출 실패 (fail-open)

`PE_VQA_2STAGE_FAIL_OPEN=true` (기본): vLLM 호출에 예외 발생 시 통과 처리. 화재/쓰러짐 같은 critical 이벤트가 누락되지 않도록 보수적 정책.

`false`로 설정 시 VLM 실패 알람 폐기.

### RabbitMQ 연결 idle EOF

heartbeat timeout으로 연결이 끊긴 상태에서 publish 시 자동 reconnect + 1회 재시도.

## 중지

```bash
docker compose down
```

## 트러블슈팅

**Q: 첫 기동 시 메인 inference에서 connection refused가 발생합니다.**

validation_server는 첫 기동 시 vLLM 모델 로드 + CUDA 그래프 컴파일 + 워밍업으로 ~3-5분 소요됩니다 (`/health`가 200을 반환할 때까지). 실측 분해:

| 단계 | 캐시 미스 | 캐시 적중 |
|------|---------|----------|
| HF 모델 다운로드 | ~47초 | 0초 (사전 다운로드) |
| 모델 weight + tokenizer 로딩 | ~50초 | ~1.6초 |
| vLLM Engine init + KV cache 프로파일 | ~13초 | ~13초 |
| CUDA 그래프 컴파일 + 워밍업 | ~90초 | ~90초 |
| FastAPI wrapper lifespan (RabbitMQ/Kafka/S3/Redis 연결) | ~10초 | ~60초 |
| **총 ready 시간** | **~211초** | **~166초** |

대응:
- `docker logs pe-vqa-2stage`로 진행 상태 확인 (`vLLM is ready` → `Validation server started` 순서로 출력)
- `MODEL_HF_CACHE_DIR`에 사전 다운로드된 캐시를 두면 ~50초 단축 (위 표 참고)
- 메인 inference 띄우기 전 `curl http://<host>:8100/health`로 200 응답 확인
- k8s에서는 `readinessProbe`로 자동 처리 (권장 설정: `initialDelaySeconds=30`, `periodSeconds=10`, `failureThreshold=30` → 5분까지 대기). Service에 등록되기 전에는 메인 Pod가 호출하지 않으므로 connection refused 회피
- docker-compose.yml의 `healthcheck.start_period: 300s`는 컨테이너 자체가 unhealthy로 죽지 않게 막아주지만, 메인 inference 측 timing은 별도 처리 필요

**Q: 메인 inference에서 알람이 발생하는데 백엔드에 안 옵니다.**
- `PE_VQA_2STAGE_VALIDATION_ENABLED=True`인지 확인
- `TWO_STEP_CATEGORIES`에 해당 카테고리 포함됐는지 확인
- validation_server 로그(`docker logs pe-vqa-2stage`)에 vLLM 응답이 "no"인지 확인 → "no"면 false positive 필터로 폐기된 정상 동작
- `/health` 가용성 확인 (안 되면 컨테이너 다운)

**Q: 종료 알람이 안 옵니다.**
- 시작 알람이 검증 통과해 Redis에 UUID 등록됐는지 확인 (`redis-cli keys 'pe_vqa_2stage:uuid:*'`)
- Redis가 같은 인스턴스인지 확인
- TTL 1시간 내인지 (오래된 사고는 만료될 수 있음)

**Q: 다른 PE 모듈 알람과 형식이 다릅니다.**
- 본 PR의 결함 #2(ts), #3(cameraId)이 해소되어 KTT 표준과 일치. 다르다면 컨테이너 이미지가 옛 빌드일 가능성 → `docker compose build --no-cache` 후 재시작
