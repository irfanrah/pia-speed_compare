import os

IMG_SIZE = (336, 336)
INPUT_SIZE = (3, *IMG_SIZE)
DEVICE = "cuda"

HF_REPO_ID = "PIA-SPACE-LAB/PE-Core-L14-336"
HF_ONNX_FILENAME = "onnx/PE-Core-L14-336_vision_dynamic.onnx"

PERCEPTION_ENCODER_ONNX_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_ONNX_PATH",
    "assets/model/PE-Core-L14-336_vision_dynamic.onnx",
)
PERCEPTION_ENCODER_ONNX_FP16_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_ONNX_FP16_PATH",
    "assets/model/PE-Core-L14-336_vision_dynamic_fp16.onnx",
)
PERCEPTION_ENCODER_TRT_PATH = os.getenv(
    "MODEL_PERCEPTION_ENCODER_TRT_PATH",
    "assets/model/PE-Core-L14-336_vision_dynamic.engine",
)

TRT_MIN_BATCH = 1
TRT_OPT_BATCH = 16
TRT_MAX_BATCH = 32

REALTIME_BUDGET_SECONDS = 0.5
TARGET_CHANNELS = 12
BENCHMARK_BATCHES = [1, 4, 8, 12, 16, 24, 32]
BENCHMARK_WARMUP_ITERS = 20
BENCHMARK_MEASURE_ITERS = 100

SAMPLE_IMAGES = [
    "assets/images/dog.jpg",
    "assets/images/cat.jpg",
]

RESULTS_PATH = "results/pe_trt_fp16_benchmark.json"
