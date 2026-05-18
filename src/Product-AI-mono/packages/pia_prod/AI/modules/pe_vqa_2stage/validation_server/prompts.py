"""
PE_VQA_2stage Validation Server - VLM 2차 검증 프롬프트.
카테고리별로 다른 프롬프트를 반환한다.
"""

FIRE_PROMPT = (
    'Is there a real flame or active fire burning in this image?\n\n'
    'Answer "yes" if you see an actual flame or active combustion.\n\n'
    'Answer "no" if the image only contains:\n'
    '- lights, headlights, sunlight, or reflections\n'
    '- fire trucks, fire extinguishers, or fire-related equipment\n'
    '- smoke without a visible flame\n\n'
    'Answer only: "yes" or "no"'
)

SMOKE_PROMPT = (
    'Analyze the image carefully.\n\n'
    'Task:\n'
    'Determine whether the image shows real visible smoke in the scene.\n\n'
    'Return "yes" only if there is actual smoke physically present in the environment, such as:\n'
    '- a localized smoke plume\n'
    '- rising, drifting, or spreading smoke\n'
    '- smoke with a visible origin, direction, or shape\n'
    '- white, gray, or black smoke emitted from a specific area or object\n\n'
    'Return "no" for:\n'
    '- lens fog\n'
    '- condensation\n'
    '- water droplets\n'
    '- humidity haze\n'
    '- camera blur\n'
    '- dirty or smeared lens\n'
    '- low contrast or washed-out frames\n'
    '- glare or overexposure\n'
    '- general misty appearance without a clear smoke source\n'
    '- any ambiguous or uncertain case\n\n'
    'Important rules:\n'
    '- The smoke must be part of the real scene, not caused by the camera or lens condition.\n'
    '- Do not classify smoke based on a globally foggy or low-visibility image alone.\n'
    '- A localized smoke shape, direction, or source must be visible.\n'
    '- If uncertain, return "no".\n\n'
    'Answer only with:\n'
    '"yes"\n'
    'or\n'
    '"no"'
)

FALLDOWN_PROMPT = (
    'Analyze the image and determine whether the person\'s situation should be classified as a fall.\n\n'
    'Focus primarily on the person\'s body posture, balance, support, and whether the pose appears '
    'controlled and intentional or collapsed and unintentional.\n\n'
    'Use the following rules:\n'
    '- Output `no` if the person appears to be standing, walking, sitting, crouching, squatting, '
    'kneeling, bending, reclining, or lying down in a controlled, stable, supported, or clearly '
    'intentional manner.\n'
    '- Output `yes` only if the person appears collapsed, sprawled, limp, tumbled, or unintentionally '
    'down due to loss of body control.\n'
    '- Do not mistake voluntary low postures or resting poses for a fall.\n\n'
    'Output requirements:\n'
    '- Return only one token.\n'
    '- Return `yes` for falldown.\n'
    '- Return `no` for normal.\n'
    '- Do not provide any explanation, description, JSON, punctuation, or additional text.'
)

SMOKING_PROMPT = (
    "Look at this image carefully. Is there a person smoking a cigarette "
    "or similar tobacco product? "
    "Respond with only 'yes' or 'no'."
)

_PROMPT_MAP = {
    "fire_pe_vqa": FIRE_PROMPT,
    "화재_pe_vqa": FIRE_PROMPT,
    "smoke_pe_vqa": SMOKE_PROMPT,
    "연기_pe_vqa": SMOKE_PROMPT,
    "falldown_pe_vqa": FALLDOWN_PROMPT,
    "쓰러짐_pe_vqa": FALLDOWN_PROMPT,
    "smoking_pe_vqa": SMOKING_PROMPT,
    "흡연_pe_vqa": SMOKING_PROMPT,
}

DEFAULT_PROMPT = (
    "Look at this image carefully. Is there any abnormal event or safety hazard visible? "
    "Respond with only 'yes' or 'no'."
)


def get_validation_prompt(category_name: str) -> str:
    return _PROMPT_MAP.get(category_name, DEFAULT_PROMPT)
