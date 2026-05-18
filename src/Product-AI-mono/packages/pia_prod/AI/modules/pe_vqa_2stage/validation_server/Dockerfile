# =============================================================
# PE VQA 2-Stage Unified Container
#
# vLLM + Validation Server를 단일 컨테이너에서 실행.
# vLLM이 내부(localhost:8000)에서 모델을 서빙하고,
# Validation Server(외부 노출 포트)가 PE 알람을 받아 VLM 검증 후 RabbitMQ로 전송.
#
# GPU 아키텍처별 베이스 이미지:
#   Blackwell (B200 등):          vllm/vllm-openai:cu130-nightly
#   Hopper / Ampere (H100, A100): vllm/vllm-openai:latest
#
# 빌드:
#   docker build -t pe-vqa-2stage .
#   docker build -t pe-vqa-2stage --build-arg BASE_IMAGE=vllm/vllm-openai:latest .
# =============================================================

ARG BASE_IMAGE=vllm/vllm-openai:cu130-nightly
FROM ${BASE_IMAGE}

WORKDIR /app

# 빌드 컨텍스트는 validation_server 디렉토리. 다른 모듈/패키지를 가져오지 않는다.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .
COPY prompts.py .
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# vLLM 설정
ENV VLLM_MODEL="Qwen/Qwen3.5-0.8B"
ENV TENSOR_PARALLEL_SIZE=1
ENV MAX_MODEL_LEN=8192
ENV MAX_NUM_SEQS=128
ENV MAX_NUM_BATCHED_TOKENS=131072
ENV GPU_MEMORY_UTILIZATION=0.3
ENV KV_CACHE_MEMORY_BYTES="1G"
ENV EXTRA_VLLM_ARGS=""
ENV HF_HOME=/app/assets/model/huggingface

# Validation Server 설정
ENV VLLM_API_BASE="http://localhost:8000/v1"
ENV VALIDATION_SERVER_HOST="0.0.0.0"
ENV VALIDATION_SERVER_PORT=8100

EXPOSE ${VALIDATION_SERVER_PORT}

ENTRYPOINT ["./entrypoint.sh"]
