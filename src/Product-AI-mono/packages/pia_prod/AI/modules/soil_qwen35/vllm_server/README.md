# Soil Qwen3.5 vLLM Server

SoilQwen35Service가 사용하는 vLLM 서버 배포 구성.

## 빠른 시작

```bash
cd packages/pia_prod/AI/modules/soil_qwen35/vllm_server
cp .env.example .env    # 환경에 맞게 수정
docker compose up -d
```

서버 상태 확인:

```bash
curl http://localhost:9001/v1/models
```

## 파일 구성

| 파일 | 역할 |
|------|------|
| `Dockerfile` | vLLM 베이스 이미지를 확장하여 모델/파라미터 기본값을 박은 컨테이너 빌드 정의 (선택적 사용) |
| `docker-compose.yml` | vLLM 서버 실행 설정 (이미지, GPU, 포트, vLLM 인자) |
| `.env.example` | 환경변수 템플릿 (복사해서 `.env`로 사용) |
| `.env` | 실제 환경변수 (git 미추적) |

기본 흐름은 `docker-compose.yml`이 vLLM 공식 이미지(`vllm/vllm-openai`)를 그대로 사용하므로 별도 빌드 없이 실행 가능합니다. 사내 레지스트리에 모델/파라미터 기본값을 굳혀 배포하고 싶을 때만 `Dockerfile`을 빌드해서 사용하세요.

## 환경변수

`.env.example`에 모든 옵션과 설명이 있습니다. 주요 항목:

### GPU 메모리 설정 (가장 중요)

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `GPU_MEMORY_UTILIZATION` | `0.3` | GPU 메모리 사용 비율 (0.0~1.0). 단독 실행 시 0.9까지 올림, GPU 공유 시 낮춤 |
| `KV_CACHE_MEMORY_BYTES` | `1G` | KV 캐시 메모리 상한. 설정 시 `GPU_MEMORY_UTILIZATION`보다 우선. **미설정 시 GPU 메모리를 거의 전부 사용하므로 반드시 설정** |

모델 크기별 VRAM 참고:
- 0.8B: ~1.6GB (가중치) + KV 캐시
- 2B: ~4GB + KV 캐시
- 4B: ~8GB + KV 캐시
- 9B: ~18GB + KV 캐시

### 모델 / 추론

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VLLM_MODEL` | `Qwen/Qwen3.5-0.8B` | HuggingFace 모델 ID |
| `VLLM_IMAGE` | `vllm/vllm-openai:cu130-nightly` | 도커 이미지. Hopper / Ampere(H100, A100)는 `vllm/vllm-openai:latest` |
| `TENSOR_PARALLEL_SIZE` | `1` | GPU 병렬 수 (멀티 GPU 시 증가) |
| `MAX_MODEL_LEN` | `8192` | 최대 컨텍스트 길이 (토큰) |
| `MAX_NUM_SEQS` | `128` | 동시 처리 가능한 최대 시퀀스 수 |
| `MAX_NUM_BATCHED_TOKENS` | `131072` | 배치당 최대 토큰 수 |
| `EXTRA_VLLM_ARGS` | `--default-chat-template-kwargs '{"enable_thinking": false}'` | 추가 vLLM 인자. thinking 비활성화는 4B 이상 모델에서 필수, 0.8B/2B는 해당 없음 |

### 기타

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VLLM_PORT` | `9001` | 호스트에서 접근할 포트 (`config.py`의 `SOIL_QWEN35_VLLM_PORT`와 일치시킬 것) |
| `HF_TOKEN` | (없음) | gated 모델 접근용 HuggingFace 토큰 |
| `HF_CACHE_DIR` | `~/.cache/huggingface` | 모델 캐시 경로 |
| `NVIDIA_VISIBLE_DEVICES` | `0` | 사용할 GPU 인덱스 (모든 GPU는 `all`) |

## Kubernetes 배포

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: soil-qwen35-vllm
spec:
  replicas: 1
  template:
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:cu130-nightly
          ports:
            - containerPort: 9001
          env:
            - name: NVIDIA_VISIBLE_DEVICES
              value: "0"
          command:
            - python3
            - -m
            - vllm.entrypoints.openai.api_server
            - --port
            - "9001"
            - --model
            - Qwen/Qwen3.5-0.8B
            - --tensor-parallel-size
            - "1"
            - --max-model-len
            - "8192"
            - --kv-cache-memory-bytes
            - "1G"
            - --gpu-memory-utilization
            - "0.3"
            - --limit-mm-per-prompt
            - '{"image": 1}'
            - --seed
            - "0"
          resources:
            limits:
              nvidia.com/gpu: 1
          readinessProbe:
            httpGet:
              path: /v1/models
              port: 9001
            initialDelaySeconds: 300
            periodSeconds: 30
---
apiVersion: v1
kind: Service
metadata:
  name: soil-qwen35-vllm
spec:
  selector:
    app: soil-qwen35-vllm
  ports:
    - port: 9001
      targetPort: 9001
```

## SoilQwen35Service 연결

서비스 모듈의 `config.py`에서 vLLM 서버 URL을 환경변수로 지정:

```bash
# 로컬
export SOIL_QWEN35_VLLM_API_URL="http://localhost:9001/v1"

# K8s
export SOIL_QWEN35_VLLM_API_URL="http://soil-qwen35-vllm:9001/v1"
```
