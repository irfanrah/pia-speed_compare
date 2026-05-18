import os

INTERNVL3_MODEL_HF_PATH = os.getenv(
    "MODEL_INTERNVL3_VIOLENCE_PATH", "assets/model/InternVL3-2B_gangnam"
)
HOST_IP = os.getenv("HOST_IP", "0.0.0.0")

VIOLENCE_CATEGORY = ["violence_vqa", "폭력_vqa"]
MODEL_OUTPUTS = ["violence"]

QUEUE_SIZE = int(os.environ.get("VQA_VIOLENCE_QUEUE_SIZE", 1))
ALARM_DURATION_THRESHOLD = int(os.environ.get("VQA_VIOLENCE_ALARM_DURATION_THRESHOLD", 1))


N_GPUS = 1
FRAME_PER_TILE_MAX_NUM = 16
DEVICE = "cuda"
MODEL_INF_DATA_TYPE = "bfloat16"
MAX_NEW_TOKEN = 15
NUM_SEGMENTS = 12
