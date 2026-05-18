# GPU 아키텍처에 따라 빌드 시 BASE_IMAGE를 변경:
#   Hopper / Ampere (H100, A100): vllm/vllm-openai:latest (기본값)
#   Blackwell (B200 등):          vllm/vllm-openai:cu130-nightly
#
# 빌드:
#   docker build -t soil-qwen35-vllm .
#   docker build --build-arg BASE_IMAGE=vllm/vllm-openai:cu130-nightly -t soil-qwen35-vllm .
#
# 실행:
#   docker run --gpus all -p 9000:9000 -e VLLM_PORT=9000 soil-qwen35-vllm

ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

# === 모델 설정 ===
ENV VLLM_MODEL=Qwen/Qwen3.5-0.8B

# === 추론 파라미터 ===
ENV TENSOR_PARALLEL_SIZE=1
ENV MAX_MODEL_LEN=8192
ENV MAX_NUM_SEQS=128
ENV MAX_NUM_BATCHED_TOKENS=131072
ENV GPU_MEMORY_UTILIZATION=0.3

# === 메모리 설정 ===
ENV KV_CACHE_MEMORY_BYTES=1G

# === 포트 설정 ===
ENV VLLM_PORT=9001

EXPOSE ${VLLM_PORT}

CMD /bin/sh -c '\
    KV_CACHE_ARG=""; \
    if [ -n "${KV_CACHE_MEMORY_BYTES}" ]; then \
        KV_CACHE_ARG="--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}"; \
    fi; \
    exec python3 -m vllm.entrypoints.openai.api_server \
        --port ${VLLM_PORT} \
        --model ${VLLM_MODEL} \
        --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
        --max-model-len ${MAX_MODEL_LEN} \
        --max-num-seqs ${MAX_NUM_SEQS} \
        --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
        --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
        --limit-mm-per-prompt '"'"'{"image": 1}'"'"' \
        --seed 0 \
        ${KV_CACHE_ARG}'
