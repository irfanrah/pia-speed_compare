"""
Qwen3VLE Embedding Server Config.

vLLM-engine knobs and FastAPI host/port. Mirrors the structure of
`pe_vle_2stage_async/validation_server/config.py` but trimmed down to
the embedding path — no publishers, Redis, or S3, because the qwen3vle
service performs classification + event publishing in-process.
"""

import os
from typing import Optional


# ============================================================================
# Environment Fetchers & Helpers
# ============================================================================

def _parse_bytes(value: Optional[str]) -> Optional[int]:
    """Parse a human-readable byte size (e.g., '1G', '512M', '1024K')."""
    if not value:
        return None

    v = value.strip().upper()
    suffixes = {"G": 1024 ** 3, "M": 1024 ** 2, "K": 1024}

    if v[-1] in suffixes:
        return int(float(v[:-1]) * suffixes[v[-1]])
    return int(v)


def _get_int(key: str, default: Optional[int] = None) -> Optional[int]:
    """Fetch an environment variable and safely cast it to an integer."""
    val = os.getenv(key, "").strip()
    return int(val) if val else default


def _get_float(key: str, default: float) -> float:
    """Fetch an environment variable and safely cast it to a float."""
    val = os.getenv(key, "").strip()
    return float(val) if val else default


# ============================================================================
# vLLM Model Configuration
# ============================================================================

MODEL_PATH                = os.getenv("MODEL_PATH", "/models/Qwen3-VL-Embedding-2B-FP8")
DTYPE                     = os.getenv("DTYPE", "bfloat16")
DEFAULT_INSTRUCTION       = os.getenv("DEFAULT_INSTRUCTION", "Represent the user's input.")

GPU_MEMORY_UTILIZATION    = _get_float("GPU_MEMORY_UTILIZATION", 0.3)
VLLM_MAX_CONCURRENCY      = _get_int("VLLM_MAX_CONCURRENCY", 10)
MAX_MODEL_LEN             = _get_int("MAX_MODEL_LEN", 8192)
LIMIT_MM_PER_PROMPT_VIDEO = _get_int("LIMIT_MM_PER_PROMPT_VIDEO", 1)
LIMIT_MM_PER_PROMPT_IMAGE = _get_int("LIMIT_MM_PER_PROMPT_IMAGE", 1)

KV_CACHE_MEMORY_BYTES     = _parse_bytes(os.getenv("KV_CACHE_MEMORY_BYTES", "1G"))
MAX_NUM_SEQS              = _get_int("MAX_NUM_SEQS", None)
MAX_NUM_BATCHED_TOKENS    = _get_int("MAX_NUM_BATCHED_TOKENS", None)

# Per-request budget for AsyncLLMEngine.encode(). A wedged scheduler holds its
# semaphore slot until this fires, after which the handler returns 504.
EMBED_TIMEOUT_S           = _get_float("EMBED_TIMEOUT_S", 120.0)


# ============================================================================
# Server Configuration
# ============================================================================

SERVER_HOST               = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT               = _get_int("SERVER_PORT", 8210)