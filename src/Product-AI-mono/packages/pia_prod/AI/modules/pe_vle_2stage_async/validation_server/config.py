"""
PE-VLE Validation Server Config.
"""

import json
import os
from typing import Dict, List


# ============================================================================
# Helper Functions
# ============================================================================

def _parse_bytes(value: str) -> int | None:
    """Parse a human-readable byte size (e.g., '1G', '512M', '1024K')."""
    if not value:
        return None
    
    v = value.strip().upper()
    suffixes = {"G": 1024 ** 3, "M": 1024 ** 2, "K": 1024}
    
    if v and v[-1] in suffixes:
        return int(float(v[:-1]) * suffixes[v[-1]])
    return int(v)


def _opt_int(value: str) -> int | None:
    """Parse integer if non-empty, otherwise return None."""
    return int(value) if value else None


# ============================================================================
# vLLM Model Configuration
# ============================================================================

MODEL_PATH = os.getenv("MODEL_PATH", "/models/Qwen3-VL-Embedding-2B-FP8")
DTYPE = os.getenv("DTYPE", "bfloat16")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.3"))
VLLM_MAX_CONCURRENCY = int(os.getenv("VLLM_MAX_CONCURRENCY", "10"))
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))
LIMIT_MM_PER_PROMPT_VIDEO = int(os.getenv("LIMIT_MM_PER_PROMPT_VIDEO", "1"))
LIMIT_MM_PER_PROMPT_IMAGE = int(os.getenv("LIMIT_MM_PER_PROMPT_IMAGE", "1"))
DEFAULT_INSTRUCTION = os.getenv("DEFAULT_INSTRUCTION", "Represent the user's input.")

KV_CACHE_MEMORY_BYTES = _parse_bytes(os.getenv("KV_CACHE_MEMORY_BYTES", "1G"))
MAX_NUM_SEQS = _opt_int(os.getenv("MAX_NUM_SEQS", ""))
MAX_NUM_BATCHED_TOKENS = _opt_int(os.getenv("MAX_NUM_BATCHED_TOKENS", ""))


# ============================================================================
# Classifier & Server Configuration
# ============================================================================

SERVER_HOST = os.getenv("VALIDATION_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("VALIDATION_SERVER_PORT", "8200"))
PE_VLE_FAIL_OPEN = os.getenv("PE_VLE_FAIL_OPEN", "true").lower() == "true"

TEXT_FEATURES_PATH = os.getenv(
    "QWEN3VLE_VLLM_TEXT_FEATURES_PATH",
    "/models/Qwen3-VL-Embedding-2B-FP8/VLE_FP8_text_features.json",
)

PE_VLE_TO_VLE_CATEGORY_EVENT_MAP: Dict[str, str] = json.loads(
    os.getenv(
        "PE_VLE_TO_VLE_CATEGORY_EVENT_MAP_JSON",
        json.dumps({
            "fire_pe_vle_ret":     "fire_vle_ret",
            "화재_pe_vle_ret":     "화재_vle_ret",
            "smoke_pe_vle_ret":    "smoke_vle_ret",
            "연기_pe_vle_ret":     "연기_vle_ret",
            "falldown_pe_vle_ret": "falldown_vle_ret",
            "쓰러짐_pe_vle_ret":   "쓰러짐_vle_ret",
        }),
    )
)

VLE_CATEGORY_EVENT_MAP: Dict[str, List[str]] = json.loads(
    os.getenv(
        "VLE_CATEGORY_EVENT_MAP_JSON",
        json.dumps({
            "fire":     ["fire_vle_ret", "화재_vle_ret"],
            "smoke":    ["smoke_vle_ret", "연기_vle_ret"],
            "falldown": ["falldown_vle_ret", "쓰러짐_vle_ret"],
        }),
    )
)


# ============================================================================
# Infrastructure Configuration
# ============================================================================

MESSAGING_BACKEND = os.getenv("MESSAGING_BACKEND", "kafka").lower()

# --- RabbitMQ ---
RABBITMQ_HOST = os.getenv("BACKEND_RABBITMQ_IP", "localhost")
RABBITMQ_PORT = int(os.getenv("BACKEND_RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("BACKEND_RABBITMQ_USER_NAME", "guest")
RABBITMQ_PASS = os.getenv("BACKEND_RABBITMQ_PASSWORD", "guest")
RABBITMQ_EXCHANGE = os.getenv("BACKEND_RABBITMQ_EXCHANGE", "")
RABBITMQ_HEARTBEAT = int(os.getenv("PYTHON_RABBITMQ_HEARBEAT_INTERVAL", "60"))
RABBITMQ_QUEUE_RET = os.getenv("BACKEND_RET_RESULT_RABBITMQ_QUEUE_NAME", "ret_queue_dev")

# --- Kafka ---
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_EVENT_PROCESS = os.getenv("KAFKA_TOPIC_EVENT_PROCESS", "event.process")

# --- S3 ---
S3_ACCESS_KEY = os.getenv("BACKEND_S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("BACKEND_S3_SECRET_KEY", "")  # Upstream typo 'KET' preserved
S3_REGION = os.getenv("BACKEND_S3_THUMBNAIL_BUCKET_REGION", "")
S3_ENDPOINT = os.getenv("BACKEND_S3_ENDPOINT", "")
S3_BUCKET = os.getenv("BACKEND_S3_THUMBNAIL_BUCKET_NAME", "thumbnail")

# --- Redis ---
REDIS_HOST = os.getenv("REDIS_IP", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
UUID_TTL_SECONDS = int(os.getenv("UUID_TTL_SECONDS", "3600"))
UUID_KEY_PREFIX = os.getenv("UUID_KEY_PREFIX", "pe_vle_2stage:uuid:")