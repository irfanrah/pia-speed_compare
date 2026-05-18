"""
internVL3 TRT-LLM 실행에 필요한 설정 상수 모음.

YAML(inference_internvl3.yml)에서 사용하던 값을 상수로 옮겨
런타임에서 YAML 파싱 없이 사용하도록 구성한다.
"""

DEFAULT_SYSTEM_PROMPT = (
    "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。"
)
IMAGE_TAG = "<image>"
IMG_TOKEN = "<IMG_CONTEXT>"

# tokenizer / decoding
IMG_BEGIN_TOKEN_ID = 151674
END_TOKEN_ID = 151643  # YAML 기준 유지
PAD_TOKEN_ID = 151643
SKIP_SPECIAL_TOKENS = [
    151643,  # <|endoftext|>
    151644,  # <|im_start|>
    151645,  # <|im_end|>
    151665,  # <img>
    151666,  # </img>
    151667,  # <IMG_CONTEXT>
]

# sampling config (YAML 기본값)
SAMPLING_CONFIG = {
    "top_k": 1,
    "top_p": 0.1,
    "temperature": 0.1,
}

# TRT-LLM runtime 옵션 (YAML에서 실제 사용하던 것만 유지)
# - kv_cache_free_gpu_mem_fraction: 남는 VRAM 중 KV 캐시에 예약할 비율(0~1).
#   값이 클수록 메모리를 더 잡아 성능은 안정적일 수 있으나, VRAM 사용량이 커진다.
# - max_tokens_in_paged_kv_cache: paged KV 캐시의 "최대 토큰 수" 하드캡.
#   설정하면 해당 토큰 수만큼만 KV 캐시 블록을 할당한다.
#   (배치*시퀀스 길이 상한으로 계산하는 게 일반적)
#  예시 14652 = 7326 * 2
#   None이면 kv_cache_free_gpu_mem_fraction 기준으로 자동 산정된다.
# - enable_chunked_context: 긴 입력을 청크로 나눠 처리하는 옵션.
#   모델/엔진에 따라 유효하며, 켜면 긴 컨텍스트에서 메모리 피크를 줄일 수 있다.
RUNTIME_INFER_ARGS = {
    "kv_cache_free_gpu_mem_fraction": 0.1,
    "max_tokens_in_paged_kv_cache": 14652,
    "enable_chunked_context": None,
}
