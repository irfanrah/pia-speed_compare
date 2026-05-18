import os

IMG_SIZE = (336, 336)  # (height, width)
INPUT_SIZE = (3, *IMG_SIZE)
DEVICE = os.getenv("PE_VQA_2STAGE_DEVICE", "cuda")
PERCEPTION_ENCODER_PYTORCH_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_PYTORCH_PATH", "assets/model/PE-Core-L14-336.pt"
)
PERCEPTION_ENCODER_ONNX_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_ONNX_PATH", "assets/model/PE-Core-L14-336.onnx"
)
PERCEPTION_ENCODER_TRT_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_TRT_PATH", "assets/model/PE-Core-L14-336.engine"
)
PERCEPTION_ENCODER_TXT_FEATURE_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_TXT_FEATURE_PATH", "assets/model/text_features.json"
)

INDEX_MAPPING = {0: "normal", 1: "falldown", 2: "fire", 3: "smoke", 4: "smoking"}

FIRE_CATEGORY = ["fire_pe_vqa", "화재_pe_vqa"]
FALLDOWN_CATEGORY = ["falldown_pe_vqa", "쓰러짐_pe_vqa"]
SMOKE_CATEGORY = ["smoke_pe_vqa", "연기_pe_vqa"]
SMOKING_CATEGORY = ["smoking_pe_vqa", "흡연_pe_vqa"]
ALL_CATEGORIES = FIRE_CATEGORY + FALLDOWN_CATEGORY + SMOKE_CATEGORY + SMOKING_CATEGORY

CATEGORY_EVENT_MAP = {
    "fire": FIRE_CATEGORY,
    "falldown": FALLDOWN_CATEGORY,
    "smoke": SMOKE_CATEGORY,
    "smoking": SMOKING_CATEGORY,
}

TEMPORAL_SIZE = 1
INFERENCE_SEQUENCE_SIZE = 1
QUEUE_SIZE = int(os.environ.get("PE_VQA_2STAGE_QUEUE_SIZE", 10))
ALARM_DURATION_THRESHOLD = int(os.environ.get("PE_VQA_2STAGE_ALARM_DURATION_THRESHOLD", 5))

TOP_CANDIDATE = int(os.environ.get("PE_VQA_2STAGE_TOP_CANDIDATE", 13))
TOP_K = 4

IMAGE_DTYPE = "float16"

# === PE_VQA_2stage Validation Server ===
# 2단계 검증 마스터 토글. False(기본값)면 PE 알람을 다른 PE 모듈과 동일하게 직접 발사한다.
# True면 TWO_STEP_CATEGORIES에 포함된 카테고리에 대해 validation server로 비동기 위임한다.
PE_VQA_2STAGE_VALIDATION_ENABLED = os.getenv("PE_VQA_2STAGE_VALIDATION_ENABLED", "False").lower() == "true"

VALIDATION_HOST = os.getenv("PE_VQA_2STAGE_VALIDATION_HOST", "localhost")
VALIDATION_PORT = int(os.getenv("PE_VQA_2STAGE_VALIDATION_PORT", "8100"))
VALIDATION_SERVER_URL = f"http://{VALIDATION_HOST}:{VALIDATION_PORT}"
VALIDATION_SERVER_ENDPOINT = os.getenv("VALIDATION_SERVER_ENDPOINT", "/api/v1/validate")
VALIDATION_SERVER_TIMEOUT = float(os.getenv("VALIDATION_SERVER_TIMEOUT", "30.0"))

# 2단계 검증 대상 카테고리 (쉼표 구분)
# PE_VQA_2STAGE_VALIDATION_ENABLED=True일 때만 의미 있음.
# 이 목록에 포함된 카테고리만 validation server로 전송, 그 외는 직접 발사.
_TWO_STEP_CATEGORIES_RAW = os.getenv("TWO_STEP_CATEGORIES", ",".join(ALL_CATEGORIES))
TWO_STEP_CATEGORIES = set(c.strip() for c in _TWO_STEP_CATEGORIES_RAW.split(",") if c.strip())
