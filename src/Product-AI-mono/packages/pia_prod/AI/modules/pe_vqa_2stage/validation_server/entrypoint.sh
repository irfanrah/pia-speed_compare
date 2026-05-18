#!/bin/bash
set -e

# 컨테이너 내부 vLLM 포트 — host network 사용 시 외부 vLLM과 포트 충돌 방지를 위해
# env로 override 가능. server.py는 VLLM_API_BASE env(예: http://localhost:${VLLM_INTERNAL_PORT}/v1)로
# vLLM에 접근.
VLLM_INTERNAL_PORT="${VLLM_INTERNAL_PORT:-8000}"

# KV Cache 인자 조건부 추가
KV_CACHE_ARG=""
if [ -n "${KV_CACHE_MEMORY_BYTES}" ]; then
    KV_CACHE_ARG="--kv-cache-memory-bytes ${KV_CACHE_MEMORY_BYTES}"
fi

# 1. vLLM 서버 백그라운드 실행
echo "[entrypoint] Starting vLLM server (model: ${VLLM_MODEL}, port: ${VLLM_INTERNAL_PORT})..."
python3 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-seqs ${MAX_NUM_SEQS} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --limit-mm-per-prompt '{"image": 1}' \
    --seed 0 \
    --port ${VLLM_INTERNAL_PORT} \
    ${KV_CACHE_ARG} \
    ${EXTRA_VLLM_ARGS} &

VLLM_PID=$!

# 2. vLLM 준비 대기
echo "[entrypoint] Waiting for vLLM to be ready (port ${VLLM_INTERNAL_PORT})..."
until curl -sf http://localhost:${VLLM_INTERNAL_PORT}/v1/models > /dev/null 2>&1; do
    sleep 3
done
echo "[entrypoint] vLLM is ready."

# 3. Validation server 백그라운드 실행
echo "[entrypoint] Starting validation server on port ${VALIDATION_SERVER_PORT}..."
python3 -c "
import uvicorn
uvicorn.run('server:app', host='${VALIDATION_SERVER_HOST}', port=${VALIDATION_SERVER_PORT}, workers=1)
" &

VALIDATION_PID=$!

# 4. 정리 함수 — 어느 한 프로세스가 죽으면 다른 하나도 깨끗하게 종료
cleanup() {
    echo "[entrypoint] Cleaning up child processes..."
    kill -TERM "$VLLM_PID" "$VALIDATION_PID" 2>/dev/null || true
    wait
}

# SIGTERM/SIGINT 시 정리
trap cleanup TERM INT

# 어느 하나 죽으면 정리 후 종료
wait -n "$VLLM_PID" "$VALIDATION_PID"
echo "[entrypoint] A process exited. Shutting down..."
cleanup
