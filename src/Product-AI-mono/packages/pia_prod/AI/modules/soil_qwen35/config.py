import os

# === vLLM API 설정 ===
VLLM_HOST = os.getenv("SOIL_QWEN35_VLLM_HOST", "0.0.0.0")
VLLM_PORT = int(os.getenv("SOIL_QWEN35_VLLM_PORT", "9001"))
VLLM_API_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
VLLM_MODEL = os.getenv("SOIL_QWEN35_VLLM_MODEL", "Qwen/Qwen3.5-0.8B")
VLLM_MAX_TOKENS = int(os.getenv("SOIL_QWEN35_VLLM_MAX_TOKENS", "1"))
VLLM_TEMPERATURE = float(os.getenv("SOIL_QWEN35_VLLM_TEMPERATURE", "0.0"))
VLLM_TIMEOUT = float(os.getenv("SOIL_QWEN35_VLLM_TIMEOUT", "30.0"))

# === 카테고리 정의 ===
FIRE_CATEGORY = ["fire_qwen_vqa", "화재_qwen_vqa"]
SMOKE_CATEGORY = ["smoke_qwen_vqa", "연기_qwen_vqa"]
FALLDOWN_CATEGORY = ["falldown_qwen_vqa", "쓰러짐_qwen_vqa"]
SMOKING_CATEGORY = ["smoking_qwen_vqa", "흡연_qwen_vqa"]
ALL_CATEGORIES = FIRE_CATEGORY + SMOKE_CATEGORY + FALLDOWN_CATEGORY + SMOKING_CATEGORY
SUPPORT_CATEGORIES = ALL_CATEGORIES

# === 이벤트 매니저 설정 ===
QUEUE_SIZE = int(os.environ.get("SOIL_QWEN35_QUEUE_SIZE", 1))
ALARM_DURATION_THRESHOLD = int(os.environ.get("SOIL_QWEN35_ALARM_DURATION_THRESHOLD", 1))
