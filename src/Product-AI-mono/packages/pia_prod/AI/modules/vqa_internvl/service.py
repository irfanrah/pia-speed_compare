from pia_prod.AI.bases.service_base import ServiceBase
from queue import Queue
from pia.ai.tasks.VQA.base import VQAConfig
from pia.ai.model import PiaTorchModel
from pia_prod.AI.modules.vqa_internvl.event import VQAFalldownEventManager
from pia_prod.AI.modules.vqa_internvl.roi_manager import VQAFalldownRoIManager
from pia_prod.AI.modules.vqa_internvl.config import (
    INTERNVL3_MODEL_HF_PATH,
    HF_REPO_ID,
    ASSETS_MODEL_DIR,
    MODEL_DIR,
    N_GPUS,
    FRAME_PER_TILE_MAX_NUM,
    DEVICE,
    MODEL_INF_DATA_TYPE,
    MAX_NEW_TOKEN,
    FIRE_CATEGORY,
    FALLDOWN_CATEGORY,
    VIT_ONNX_FILE,
    VIT_TRT_FILE,
    VIT_IMAGE_PATH,
    VIT_DTYPE,
    VIT_MIN_BS,
    VIT_OPT_BS,
    VIT_MAX_BS,
    TLLM_CHECKPOINT_DIR,
    TRT_ENGINE_DIR,
    LLM_DTYPE,
    LOAD_MODEL_ON_CPU,
    DISABLE_ROPE_SCALING,
    LLM_MAX_BATCH_SIZE,
    LLM_MAX_INPUT_LEN,
    LLM_MAX_SEQ_LEN,
    LLM_MAX_NUM_TOKENS,
    LLM_GEMM_PLUGIN,
    LLM_MAX_MULTIMODAL_LEN,
    LLM_FORCE_SINGLE_BATCH,
    LLM_BATCH_USE_EXECUTOR,
    LLM_BATCH_USE_ASYNC,
    LLM_USE_INPUT_TOKEN_EXTRA_IDS,
)
from pia_prod.AI.modules.vqa_internvl.prompts import FIRE, FALLDOWN
from pia_prod.AI.global_config import USER_PARAM_KEY, VQA_EVENT_KEY


class InternVL3TrtLlmService(ServiceBase):
    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)

    def _load_model(self):
        trt_build = {
            # ── 경로 ──────────────────────────────────────────────────────────
            "hf_repo_id": HF_REPO_ID,
            "assets_dir": "assets",
            "assets_model_dir": ASSETS_MODEL_DIR,
            "assets_image_dir": "assets/images",
            "model_dir": MODEL_DIR,
            "vit_onnx_file": VIT_ONNX_FILE,
            "vit_trt_file": VIT_TRT_FILE,
            "vit_image_path": VIT_IMAGE_PATH,
            "tllm_checkpoint_dir": TLLM_CHECKPOINT_DIR,
            "trt_engine_dir": TRT_ENGINE_DIR,
            # ── VIT 엔진 빌드 ─────────────────────────────────────────────────
            "vit_dtype": VIT_DTYPE,
            "vit_min_bs": VIT_MIN_BS,
            "vit_opt_bs": VIT_OPT_BS,
            "vit_max_bs": VIT_MAX_BS,
            # ── LLM 체크포인트 / 엔진 빌드 ────────────────────────────────────
            "llm_dtype": LLM_DTYPE,
            "load_model_on_cpu": LOAD_MODEL_ON_CPU,
            "disable_rope_scaling": DISABLE_ROPE_SCALING,
            "llm_max_batch_size": LLM_MAX_BATCH_SIZE,
            "llm_max_input_len": LLM_MAX_INPUT_LEN,
            "llm_max_seq_len": LLM_MAX_SEQ_LEN,
            "llm_max_num_tokens": LLM_MAX_NUM_TOKENS,
            "llm_gemm_plugin": LLM_GEMM_PLUGIN,
            "llm_max_multimodal_len": LLM_MAX_MULTIMODAL_LEN,
            # ── 런타임 동작 ────────────────────────────────────────────────────
            "llm_force_single_batch": LLM_FORCE_SINGLE_BATCH,
            "llm_batch_use_executor": LLM_BATCH_USE_EXECUTOR,
            "llm_batch_use_async": LLM_BATCH_USE_ASYNC,
            "llm_use_input_token_extra_ids": LLM_USE_INPUT_TOKEN_EXTRA_IDS,
        }

        self.config = VQAConfig(
            model_path=INTERNVL3_MODEL_HF_PATH,
            device=DEVICE,
            frame_per_tile_max_num=FRAME_PER_TILE_MAX_NUM,
            data_type=MODEL_INF_DATA_TYPE,
            n_gpus=N_GPUS,
            save_cache_dir=INTERNVL3_MODEL_HF_PATH,
            max_new_tokens=MAX_NEW_TOKEN,
            trt_build=trt_build,
        )
        self.model = PiaTorchModel(
            target_task="VQA", target_model="internvl3trt_llm", config=self.config
        )

    def _init_values(self):
        pass

    def _load_event_manager(self):
        return VQAFalldownEventManager()

    def _load_roi_manager(self):
        return VQAFalldownRoIManager()

    def _match_category2prompt(self, batches, user_params):
        re_batches = []
        prompts = []
        categories = []
        for frame, user_param in zip(batches, user_params):
            events = user_param[USER_PARAM_KEY][VQA_EVENT_KEY]
            temp_categories = []
            for category in events:
                if category in FALLDOWN_CATEGORY:
                    prompts.append(FALLDOWN)
                elif category in FIRE_CATEGORY:
                    prompts.append(FIRE)
                else:
                    continue
                temp_categories.append(category)
                re_batches.append(frame)
            categories.append(temp_categories)
        return re_batches, prompts, categories

    def _detect(self, **datas):
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        batches, prompts, categories = self._match_category2prompt(batches, user_params)
        batches = self.roi_manager.process_batches_with_roi(batches, user_params)
        responses = self.model(batches, prompts)

        alarms = self.alarm_event_manager.update(responses, categories, stream_ids)
        if len(alarms) > 0:
            return alarms, batches, stream_ids, user_params, self.is_needed_cvt_color
        else:
            return None
