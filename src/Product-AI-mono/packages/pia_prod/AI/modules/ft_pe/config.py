import os
from enum import Enum

IMG_SIZE = (336, 336)
INPUT_SIZE = (8, 3, *IMG_SIZE)
DEVICE = "cuda"
FT_PE_ID = os.getenv("FT_PE_ID", "FT_PE-Core-L14-336_260318")

# pe_violence와 동일한 ONNX/TRT 엔진을 공유한다. 환경변수 키도 pe_violence와 동일하게 맞춰,
# 두 서비스가 동일한 모델 파일을 가리키도록 한다.
FT_PE_MODEL_PYTORCH_PATH = os.getenv(
    "MODEL_PE_VIOLENCE_DETECTION_PYTORCH_PATH",
    f"assets/model/{FT_PE_ID}.pt",
)
FT_PE_MODEL_ONNX_PATH = os.getenv(
    "MODEL_PE_VIOLENCE_DETECTION_ONNX_PATH",
    f"assets/model/{FT_PE_ID}_vision_no_mean_pooling.onnx",
)
FT_PE_MODEL_TRT_PATH = os.getenv(
    "MODEL_PE_VIOLENCE_DETECTION_TRT_PATH",
    f"assets/model/{FT_PE_ID}_vision_no_mean_pooling.engine",
)

# 카테고리별 normal/abnormal text_features(각 1024-d)가 담긴 통합 JSON.
FT_TEXT_FEATURES_PATH = os.getenv(
    "FT_PE_TEXT_FEATURES_JSON",
    "assets/model/FT_text_features.json",
)

NORMAL_CLASS_NAME = "normal"

VIOLENCE_CATEGORY = ["violence_ft_ret", "폭력_ft_ret"]
FALLDOWN_CATEGORY = ["falldown_ft_ret", "쓰러짐_ft_ret"]
FIRE_CATEGORY = ["fire_ft_ret", "화재_ft_ret"]
SMOKE_CATEGORY = ["smoke_ft_ret", "연기_ft_ret"]

CATEGORY_EVENT_MAP = {
    "violence": VIOLENCE_CATEGORY,
    "falldown": FALLDOWN_CATEGORY,
    "fire": FIRE_CATEGORY,
    "smoke": SMOKE_CATEGORY,
}

ABNORMAL_CLASS_NAMES = list(CATEGORY_EVENT_MAP.keys())
ALL_CATEGORIES = [cat for cats in CATEGORY_EVENT_MAP.values() for cat in cats]

INDEX_MAPPING = {0: NORMAL_CLASS_NAME, **{i + 1: name for i, name in enumerate(ABNORMAL_CLASS_NAMES)}}

TEMPORAL_SIZE = 8


class FTPEMode(str, Enum):
    """
    FT_PE 인코더 sliding-window 동작 모드.
    WINDOW_SIZE, SLIDING_WINDOW_SIZE, PREDICTION_SIZE, TIME_INTERVAL이
    모드에 따라 함께 결정되므로 단일 env var로만 전환한다.
    """

    FPS_3 = "3fps"
    FPS_8 = "8fps"


# (WINDOW_SIZE, SLIDING_WINDOW_SIZE, PREDICTION_SIZE, TIME_INTERVAL)
# - WINDOW_SIZE: 한 번의 인코드 패스에 입력되는 프레임 수 (TRT temporal dim)
# - SLIDING_WINDOW_SIZE: 다음 패스와 겹치는 프레임 수 (stride = WINDOW_SIZE - SLIDING_WINDOW_SIZE)
# - PREDICTION_SIZE: 인코드 결과 [W, 1024] 중 frame_buffer에 투입할 뒤쪽 임베딩 개수
# - TIME_INTERVAL: 프레임 피딩 주기 (초). 1 / 타깃 FPS.
MODE_PRESETS: dict[FTPEMode, tuple[int, int, int, float]] = {
    FTPEMode.FPS_3: (3, 2, 1, 0.333),
    FTPEMode.FPS_8: (1, 0, 1, 0.125),
}

FT_PE_MODE = FTPEMode(os.getenv("FT_PE_MODE", FTPEMode.FPS_3.value))
WINDOW_SIZE, SLIDING_WINDOW_SIZE, PREDICTION_SIZE, FT_PE_TIME_INTERVAL = MODE_PRESETS[FT_PE_MODE]

ALARM_QUEUE_SIZE = int(os.getenv("FT_PE_ALARM_QUEUE_SIZE", 3))
ALARM_THRESHOLD = int(os.getenv("FT_PE_ALARM_THRESHOLD", 2))

IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")
