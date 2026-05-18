import os
import torch

IMG_SIZE = (336, 336)  # (height, width)
INPUT_SIZE = (3, *IMG_SIZE)
DEVICE = "cuda"

# -- Model paths ---------------------------------------------------------------
# perception_encoder 모듈과 동일한 PE-Core-L14-336 모델 weight 를 공유한다.
# 환경변수 키는 pe_distribution 모듈 단위로 분리하여 운영에서 독립 설정 가능.
PE_DISTRIBUTION_PYTORCH_PATH = os.getenv(
    "MODEL_PE_DISTRIBUTION_PYTORCH_PATH", "assets/model/PE-Core-L14-336.pt"
)
PE_DISTRIBUTION_ONNX_PATH = os.getenv(
    "MODEL_PE_DISTRIBUTION_ONNX_PATH", "assets/model/PE-Core-L14-336.onnx"
)
PE_DISTRIBUTION_TRT_PATH = os.getenv(
    "MODEL_PE_DISTRIBUTION_TRT_PATH", "assets/model/PE-Core-L14-336.engine"
)
PE_DISTRIBUTION_TXT_FEATURE_PATH = os.getenv(
    "MODEL_PE_DISTRIBUTION_TXT_FEATURE_PATH", "assets/model/text_features_distribution.json"
)

INDEX_MAPPING = {0: "normal", 1: "falldown", 2: "fire", 3: "smoke", 4: "smoking"}

FIRE_CATEGORY = ["fire_ret", "화재_ret"]
FALLDOWN_CATEGORY = ["falldown_ret", "쓰러짐_ret"]
SMOKE_CATEGORY = ["smoke_ret", "연기_ret"]
SMOKING_CATEGORY = ["smoking_ret", "흡연_ret"]
ALL_CATEGORIES = FIRE_CATEGORY + FALLDOWN_CATEGORY + SMOKE_CATEGORY + SMOKING_CATEGORY

CATEGORY_EVENT_MAP = {
    "fire": FIRE_CATEGORY,
    "falldown": FALLDOWN_CATEGORY,
    "smoke": SMOKE_CATEGORY,
    "smoking": SMOKING_CATEGORY,
}

TEMPORAL_SIZE = 1
INFERENCE_SEQUENCE_SIZE = 1
QUEUE_SIZE = int(os.environ.get("PE_QUEUE_SIZE", 10))
ALARM_DURATION_THRESHOLD = int(os.environ.get("PE_ALARM_DURATION_THRESHOLD", 5))

# -- Distribution(IoU) 기반 이벤트 판별 파라미터 -------------------------------
# normal 카테고리 분포와 각 이벤트 카테고리 분포의 hist-IoU 가 IOU_THRESHOLD 미만이면
# 이벤트 후보로 간주한다. IoU 가 작을수록 normal 과 분리도가 높다는 의미.
IOU_THRESHOLD = float(os.environ.get("PE_DISTRIBUTION_IOU_THRESHOLD", 0.3))
# IoU 계산용 히스토그램 bin 수. bin 이 너무 작으면 분포 차이가 묻히고 너무 크면 노이즈 증가.
IOU_HIST_BINS = int(os.environ.get("PE_DISTRIBUTION_IOU_HIST_BINS", 80))
# normal 카테고리의 인덱스 (INDEX_MAPPING 의 key 와 일치).
NORMAL_CATEGORY_ID = 0

INTERNVL3_VQA_CATEGORY = ["화재_vqa", "fire_vqa", "쓰러짐_vqa", "falldown_vqa"]
IMAGE_DTYPE = torch.float16
