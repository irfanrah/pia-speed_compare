# =============================================================
# PE-VLE Validation Server
#
# Base image per GPU architecture:
#   Blackwell (e.g. B200):                vllm/vllm-openai:cu130-nightly
#   Hopper / Ampere (H100, A100):         vllm/vllm-openai:latest
#   Stable default for Qwen3VLE:          vllm/vllm-openai:v0.14.0
#
# Why v0.14.0 is the pinned default: this is the version Qwen3VLE was
# tuned and benchmarked against. A side-by-side test on `latest`
# (v0.19.1 at the time of writing) showed ~30% higher per-request
# latency on this pooling/embedding workload — async scheduling,
# chunked prefill, and piecewise cudagraph mode for pooling models
# each add overhead that doesn't exist in v0.14.0. Stay on v0.14.0
# unless you specifically need Blackwell support or a feature only
# present in a newer release; in those cases override BASE_IMAGE.
#
# Build:
#   docker build -t pe-vle-validator .
#   docker build -t pe-vle-validator --build-arg BASE_IMAGE=vllm/vllm-openai:cu130-nightly .
# =============================================================

ARG BASE_IMAGE=vllm/vllm-openai:v0.14.0
FROM ${BASE_IMAGE}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py publishers.py server.py download_model.py entrypoint.sh ./
RUN chmod +x /app/entrypoint.sh /app/download_model.py

ENV MODEL_PATH=/models/Qwen3-VL-Embedding-2B-FP8 \
    DTYPE=bfloat16 \
    GPU_MEMORY_UTILIZATION=0.3 \
    MAX_MODEL_LEN=8192 \
    LIMIT_MM_PER_PROMPT_VIDEO=1 \
    LIMIT_MM_PER_PROMPT_IMAGE=1

ENV HF_TEXT_FEATURES_FILE=VLE_FP8_text_features.json

ENV VALIDATION_SERVER_HOST=0.0.0.0 \
    VALIDATION_SERVER_PORT=8200

EXPOSE 8200
ENTRYPOINT ["/app/entrypoint.sh"]
