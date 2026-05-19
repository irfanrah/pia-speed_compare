#!/usr/bin/env bash
# =============================================================================
# Qwen3VLE Embedding Server entrypoint
#
# 1. Detect whether the FP8 checkpoint exists under $MODEL_PATH. If missing,
#    fetch from HF Hub via download_model.py.
# 2. Re-verify the checkpoint exists after the optional download — fail fast
#    with a clear error message if not.
# 3. Set HF_HUB_OFFLINE=1 so vLLM never reaches out to HF for metadata during
#    normal serving (the download phase, if any, runs *before* this).
# 4. Exec server.py.
# =============================================================================
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/models/Qwen3-VL-Embedding-2B-FP8}"
WEIGHTS_PATH="${MODEL_PATH}/model.safetensors"

if [[ -f "${WEIGHTS_PATH}" ]]; then
    echo "[entrypoint] checkpoint present at ${MODEL_PATH} — skipping HF download"
else
    echo "[entrypoint] missing checkpoint at ${WEIGHTS_PATH}"

    if [[ -z "${HF_MODEL_REPO_ID:-}" ]]; then
        cat <<EOF >&2
ERROR: HF_MODEL_REPO_ID is unset and ${MODEL_PATH} is incomplete.
       Either:
         a) pre-populate the host's mounted dir (run download_model.py on
            an internet-connected host, then rsync to this node), OR
         b) set HF_MODEL_REPO_ID (and HF_AUTH_TOKEN if the repo is gated)
            in the compose env so this entrypoint can fetch it.
EOF
        exit 1
    fi

    echo "[entrypoint] fetching from HF Hub: ${HF_MODEL_REPO_ID}"
    python3 /app/download_model.py \
        --model-dir "${MODEL_PATH}" \
        --repo-id "${HF_MODEL_REPO_ID}" \
        --skip-existing
fi

if [[ ! -f "${WEIGHTS_PATH}" ]]; then
    echo "ERROR: weights still missing after download: ${WEIGHTS_PATH}" >&2
    exit 1
fi

# Lock down HF Hub access for the actual serving phase. Setting this *before*
# the download step would have prevented the snapshot fetch above.
export HF_HUB_OFFLINE=1

echo "[entrypoint] starting embedding server (MODEL_PATH=${MODEL_PATH}, HF_HUB_OFFLINE=1)"
exec python3 /app/server.py "$@"
