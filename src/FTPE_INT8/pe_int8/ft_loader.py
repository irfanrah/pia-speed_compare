"""Load `FT_PE-Core-L14-336_260318` and return a plain (LoRA-merged)
``pe.CLIP-splitqkv`` model.

The canonical flow in `src.video_embedding_extraction` and `src.PE_FPINT8`
is `PEModelInitializer(load_type="lora_weight_load")`, but that path expects
a separate `<base>-split-qkv.pt` pretrained file under
`PE_FineTuning/other_model/PE/`. That file isn't on every host, and the FT
.pt is already self-contained — base + LoRA weights are both stored under
`base_model.model.*` in the FT state dict — so we replicate the LoRA-load
steps directly, then `merge_and_unload()` to fuse the LoRA back into the
underlying `pe.CLIP`.

Public entry: ``load_ft_clip(ft_pt_path, *, base_model="PE-Core-L14-336",
device="cpu")`` returns a ``pe.CLIP-splitqkv`` model in eval mode.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

import torch

# --- standalone-bundle path setup -------------------------------------------
# Locate the PE vendor (pia-prompt_optimization). The bundle is self-contained
# except for the PE vendor and (for FT) the LoRA-PEFT wheel.
def _resolve_pe_vendor():
    cand = os.environ.get("PE_VENDOR")
    if cand and os.path.isdir(os.path.join(cand, "src", "PE", "perception_models")):
        return cand
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (
        # Common drop-in spots:
        os.path.abspath(os.path.join(here, "..", "vendor", "pia-prompt_optimization")),
        os.path.abspath(os.path.join(here, "..", "..", "..", "pia-prompt_optimization")),
        os.path.abspath(os.path.join(here, "..", "..", "..", "..", "pia-prompt_optimization")),
    ):
        if os.path.isdir(os.path.join(c, "src", "PE", "perception_models")):
            return c
    raise RuntimeError(
        "Cannot locate pia-prompt_optimization. Set PE_VENDOR=/path/to/pia-prompt_optimization "
        "or place the vendor at <exp8>/vendor/pia-prompt_optimization."
    )

_PE_VENDOR = _resolve_pe_vendor()
_PE_PMODELS = os.path.join(_PE_VENDOR, "src", "PE", "perception_models")
_HERE = os.path.dirname(os.path.abspath(__file__))
for p in (_HERE, _PE_VENDOR, _PE_PMODELS):
    if p not in sys.path:
        sys.path.insert(0, p)
# ----------------------------------------------------------------------------

import core.vision_encoder.pe as pe  # noqa: E402
# `find_adapter_config_path` and `merge_lora_if_peft` are only needed for the
# LoRA-PEFT branch (loading the raw FT_PE-Core-L14-336_260318.pt). The deploy
# path loads a post-merge plain state_dict and skips them. We defer the
# imports into _load_peft() so this module works without those repos on hosts
# that only need to load a dequantized deploy checkpoint.

def _load_peft_helpers():
    """Lazy import. Raises a clear error if PEFT helpers aren't importable."""
    try:
        from src.PE.PE_init.utils_paths import find_adapter_config_path  # noqa: E402
        from src.PE_FPINT8.quantizer import merge_lora_if_peft  # noqa: E402
        return find_adapter_config_path, merge_lora_if_peft
    except ImportError as e:
        raise ImportError(
            "ft_loader: loading a LoRA-PEFT checkpoint requires the PE_FPINT8 "
            "and PE_init.utils_paths modules. Either point PYTHONPATH at the "
            "TYPE8_PE_research repo root (which has src/PE_FPINT8/), or use a "
            "post-merge plain state_dict (e.g. qat_deploy_fp32.pt).") from e


def _detect_qkv_layout(sd: dict) -> str:
    """Inspect a state_dict and return "split" if visual transformer blocks
    have separate q_proj/k_proj/v_proj weights, "combined" if they have
    in_proj_weight (the upstream PE-Core-L14-336 layout)."""
    for k in sd:
        # PEFT wrapping prefix won't change the structural suffix.
        kk = k.replace("base_model.model.", "")
        if "visual.transformer.resblocks." in kk:
            if kk.endswith(".attn.q_proj.weight"):
                return "split"
            if kk.endswith(".attn.in_proj_weight"):
                return "combined"
    # Empty or unknown — default to split to preserve current behavior.
    return "split"


def load_ft_clip(ft_pt_path: str, *,
                 base_model: str = "PE-Core-L14-336",
                 device: str = "cpu") -> Any:
    """Build a PE.CLIP arch matching ``ft_pt_path`` and load it. Auto-detects:

    - state_dict has ``visual.transformer.resblocks.N.attn.in_proj_weight``
      → upstream PE-Core-L14-336 layout (combined QKV). Build the default
      arch and load directly. Used for the zero-shot deploy path.
    - state_dict has ``...attn.q_proj.weight``
      → split-QKV layout (what LoRA was trained on and what phase2 dequant
      emits). Build the ``-splitqkv`` arch.

    Within each layout, also detect:
    - PEFT-wrapped (keys start with ``base_model.model.``) → wrap with the
      saved LoraConfig, load, merge_and_unload. This is the original FT path.
    - Plain pe.CLIP state_dict (post-merge or post-QAT-dequant) → load directly.
    """
    if not os.path.isfile(ft_pt_path):
        raise FileNotFoundError(
            f"FT checkpoint not found: {ft_pt_path}. "
            f"Download from huggingface.co/PIA-SPACE-LAB/"
            f"FT_PE-Core-L14-336_260318 or copy it from the NAS."
        )

    sd = torch.load(ft_pt_path, map_location=device)
    layout = _detect_qkv_layout(sd)
    arch_name = f"{base_model}-splitqkv" if layout == "split" else base_model
    print(f"   [ft_loader] detected QKV layout: {layout}  arch: {arch_name}")
    model = pe.CLIP.from_config(arch_name, pretrained=False).to(device)

    sample_keys = list(sd.keys())[:5]
    is_peft = any(k.startswith("base_model.model.") for k in sample_keys)

    if not is_peft:
        # Post-merge plain state_dict (e.g. from QAT dequantize). Load directly.
        m, u = model.load_state_dict(sd, strict=False)
        if m:
            print(f"   [ft_loader] missing keys: {len(m)} (first: {m[:3]})")
        if u:
            print(f"   [ft_loader] unexpected keys: {len(u)} (first: {u[:3]})")
        return model.eval()

    # Original FT path: PEFT wrap + load + merge. Imports are deferred to
    # here so the deploy path (plain post-merge state_dict) doesn't require
    # PE_FPINT8 / utils_paths to be importable.
    from peft import get_peft_model, LoraConfig
    find_adapter_config_path, merge_lora_if_peft = _load_peft_helpers()

    adapter_cfg = find_adapter_config_path(ft_pt_path)
    with open(adapter_cfg) as f:
        j = json.load(f)

    lora_config = LoraConfig(
        r=j["r"],
        lora_alpha=j["lora_alpha"],
        target_modules=j["target_modules"],
        lora_dropout=j.get("lora_dropout", 0.0),
        bias=j.get("bias", "none"),
        modules_to_save=j.get("modules_to_save", None),
        task_type=j.get("task_type", "FEATURE_EXTRACTION"),
    )
    model = get_peft_model(model, lora_config)
    model.load_state_dict(sd)
    merged = merge_lora_if_peft(model)
    return merged.eval()
