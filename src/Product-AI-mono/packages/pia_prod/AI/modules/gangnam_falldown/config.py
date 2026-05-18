import os

INTERNVL3_MODEL_HF_PATH = os.getenv("MODEL_INTERNVL3_PATH", "assets/model/InternVL3-2B")
HOST_IP = os.getenv("HOST_IP", "0.0.0.0")


FIRE_CATEGORY = ["fire_vqa", "화재_vqa"]
FALLDOWN_CATEGORY = ["falldown_vqa", "쓰러짐_vqa"]
MODEL_OUTPUTS = ["falldown", "fire"]
# TODO : 추후 확장시 SUPPORT 카테고리에 추가하는 방향으로 진행 현재는 삭제 ex) FIRE_CATEGORY + FALLDOWN_CATEGORY
## 현재는 단일 카테고리 탐지로만 진행. -> VQA 다중 탐지가 가능할때 진행하는게 효율적
SUPPORT_CATEGORIES = FALLDOWN_CATEGORY

QUEUE_SIZE = int(os.environ.get("VQA_FALLDOWN_QUEUE_SIZE", 1))
ALARM_DURATION_THRESHOLD = int(os.environ.get("VQA_FALLDOWN_ALARM_DURATION_THRESHOLD", 1))

N_GPUS = 1
FRAME_PER_TILE_MAX_NUM = 16
DEVICE = "cuda"
MODEL_INF_DATA_TYPE = "bfloat16"
MAX_NEW_TOKEN = 13
