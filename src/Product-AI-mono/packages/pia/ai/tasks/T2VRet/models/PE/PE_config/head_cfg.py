from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional


# -------------------------------------------------------------------
# Base template ("load from base_head")
# -------------------------------------------------------------------
BASE_HEAD: Dict[str, Any] = {
    "additional_head": {
        "use": False,  # default; overridden by builder
        "vision": {
            "type": None,         # required
            "embed_dim": None,    # required
            "hidden_dim": None,   # optional
            "dropout": None,      # optional
            "attn_head": None,    # optional
            "attn_dropout": None, # optional
            "use_pos": False,     # optional
            "num_queries": None,  # optional
        },
    }
}


def additional_head(
    *,
    vision_type: str,
    embed_dim: int,
    head_use: bool = True,
    hidden_dim: Optional[int] = None,
    dropout: Optional[float] = None,
    attn_head: Optional[int] = None,
    attn_dropout: Optional[float] = None,
    use_pos: bool = False,
    num_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build a head config by deep-copying BASE_HEAD and overriding fields.
    Keeps the structure exactly:
      {"additional_head": {"use": ..., "vision": {...}}}
    """
    cfg = deepcopy(BASE_HEAD)

    cfg["additional_head"]["use"] = bool(head_use)

    vision = cfg["additional_head"]["vision"]
    vision["type"] = vision_type
    vision["embed_dim"] = int(embed_dim)

    # optional overrides (only set if provided, except use_pos)
    if hidden_dim is not None:
        vision["hidden_dim"] = int(hidden_dim)
    if dropout is not None:
        vision["dropout"] = float(dropout)
    if attn_head is not None:
        vision["attn_head"] = int(attn_head)
    if attn_dropout is not None:
        vision["attn_dropout"] = float(attn_dropout)

    vision["use_pos"] = bool(use_pos)

    if num_queries is not None:
        vision["num_queries"] = int(num_queries)

    return cfg


# -------------------------------------------------------------------
# Your heads (now truly "loaded from base_head")
# -------------------------------------------------------------------

# L14
MHCA_HEAD_L14: Dict[str, Any] = additional_head(
    vision_type="MHCA_1",
    embed_dim=1024,
    dropout=0.1,
    attn_head=8,
    head_use=True,
)

# PE-S16
LINEAR_HEAD_S16: Dict[str, Any] = additional_head(
    vision_type="Linear_1",
    embed_dim=512,
    hidden_dim=256,
    dropout=0.0,
    head_use=True,
)

MHCA_HEAD_S16: Dict[str, Any] = additional_head(
    vision_type="MHCA_1",
    embed_dim=512,
    dropout=0.1,
    attn_head=8,
    head_use=True,
)


EFFICIENTPROBEQ32_HEAD_S16: Dict[str, Any] = additional_head(
    vision_type="vid_efficient_1",
    embed_dim=512,
    num_queries=32,
    head_use=True,
)

PERCIEVER1_HEAD_S16: Dict[str, Any] = additional_head(
    vision_type="perciever_1",
    embed_dim=512,
    head_use=True,
)

PERCIEVER2_HEAD_S16: Dict[str, Any] = additional_head(
    vision_type="perciever_2",
    embed_dim=512,
    head_use=True,
)
