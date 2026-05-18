import os

IMG_SIZE = (336, 336)
INPUT_SIZE = (8, 3, *IMG_SIZE)
DEVICE = "cuda"
VIOLENCE_PE_ID = "FT_PE-Core-L14-336_260318"
TEXT_PROMPT_TYPE = "text_prompt_AMPOv2.1.2"
PRED_CLASS = "violence"
# ORGS_NAME = os.getenv("HF_NAMESPACE", "PIA-SPACE-LAB")



VIOLENCE_PE_MODEL_PYTORCH_PATH = os.getenv(
    "MODEL_PE_VIOLENCE_DETECTION_PYTORCH_PATH", "assets/model/FT_PE-Core-L14-336_260318.pt"
)
VIOLENCE_PE_MODEL_ONNX_PATH = os.getenv(
    "MODEL_PE_VIOLENCE_DETECTION_ONNX_PATH", "assets/model/FT_PE-Core-L14-336_260318_vision_no_mean_pooling.onnx"
)
VIOLENCE_PE_MODEL_TRT_PATH = os.getenv(
    "MODEL_PE_VIOLENCE_DETECTION_TRT_PATH", "assets/model/FT_PE-Core-L14-336_260318_vision_no_mean_pooling.engine"
)

VIOLENCE_PE_TXT_FEATURE_PATH = os.getenv(
    "MODEL_PE_VIOLENCE_DETECTION_TXT_FEATURE_PATH", f"assets/model/{TEXT_PROMPT_TYPE}/{PRED_CLASS}_pred_prompts"
)

LIST_OF_NORMAL_TXT_PROMPTS = [
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/falldown_0.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/falldown_36.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/falldown_40.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/fire_2.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/fire_30.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/fire_49.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/normal_17.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/normal_18.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/normal/normal_4.pt",
]

LIST_OF_VIOLENCE_TXT_PROMPTS = [
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/violence/violence_40.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/violence/violence_46.pt",
    "text_prompt_AMPOv2.1.2/violence_pred_prompts/violence/violence_8.pt",
]




INDEX_MAPPING = {0: "normal", 1: "violence"}
VIOLENCE_CATEGORY = ["violence_ret", "폭력_ret"]
ALL_CATEGORIES = VIOLENCE_CATEGORY

CATEGORY_EVENT_MAP = {
    "violence": VIOLENCE_CATEGORY,
}



TEMPORAL_SIZE = 8
PE_RET_VIOLENCE_TIME_INTERVAL = float(os.getenv("PE_RET_VIOLENCE_TIME_INTERVAL", 0.333))

ALARM_QUEUE_SIZE = int(os.getenv("VIOLENCE_ALARM_QUEUE_SIZE", 3)) # Number of recent predictions kept in the alarm queue.
ALARM_THRESHOLD = int(os.getenv("VIOLENCE_ALARM_THRESHOLD", 2)) # Minimum number of anomalous predictions in the queue to trigger an alarm. 
 
IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")
