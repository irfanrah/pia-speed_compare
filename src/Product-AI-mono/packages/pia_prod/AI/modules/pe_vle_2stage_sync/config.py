import os

# PE Config
TWOSTAGE_PE_QUEUE_SIZE = int(os.getenv("TWOSTAGE_PE_QUEUE_SIZE", 5))
TWOSTAGE_PE_ALARM_DURATION_THRESHOLD = int(os.getenv("TWOSTAGE_PE_ALARM_DURATION_THRESHOLD", 3))

# Qwen3VLE Config
QWEN3VLE_TEMPORAL_SIZE = int(os.getenv("QWEN3VLE_TEMPORAL_SIZE", 1))

# Two-stage event categories
FIRE_CATEGORY = ["fire_pe_vle_ret", "화재_pe_vle_ret"]
SMOKE_CATEGORY = ["smoke_pe_vle_ret", "연기_pe_vle_ret"]
ALL_CATEGORIES = FIRE_CATEGORY + SMOKE_CATEGORY

CATEGORY_EVENT_MAP = {
    "fire": FIRE_CATEGORY,
    "smoke": SMOKE_CATEGORY,
}

# Category mapping: two-stage ↔ PE / VLE
TWOSTAGE_TO_PE_CATEGORY_EVENT_MAP = {
    "fire_pe_vle_ret": "fire_ret",
    "화재_pe_vle_ret": "화재_ret",
    "smoke_pe_vle_ret": "smoke_ret",
    "연기_pe_vle_ret": "연기_ret",
}

TWOSTAGE_TO_VLE_CATEGORY_EVENT_MAP = {
    "fire_pe_vle_ret": "fire_vle_ret",
    "화재_pe_vle_ret": "화재_vle_ret",
    "smoke_pe_vle_ret": "smoke_vle_ret",
    "연기_pe_vle_ret": "연기_vle_ret",
}

PE_TO_TWOSTAGE_CATEGORY_EVENT_MAP = {v: k for k, v in TWOSTAGE_TO_PE_CATEGORY_EVENT_MAP.items()}