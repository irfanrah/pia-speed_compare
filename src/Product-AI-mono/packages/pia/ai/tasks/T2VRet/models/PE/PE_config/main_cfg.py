import os
from dataclasses import dataclass,field
from typing import Optional, Dict, Any

from pia.tests.test_config import ASSETS_MODEL_SAVE_DIR
from pia.ai.tasks.T2VRet.models.PE.PE_config.head_cfg import *

@dataclass
class PECustomConfig:
    original_model_name: str = "PE-Core-L14-336"
    load_type: str = "default"
    weight_path: Optional[str] = None
    split_qkv: bool = False
    pretrained_split_qkv_path: Optional[str] = None
    lora_adapter_path: Optional[str] = None
    additional_head_args: Optional[Dict[str, Any]] = field(default_factory=dict)



HF_NAMESPACE = os.getenv("HF_NAMESPACE") or "PIA-SPACE-LAB"
PRETRAINED_MODEL_PATH = os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "PE-Core-L14-336-splitqkv", "PE-Core-L14-336-splitqkv.pt")

PE_CUSTOM_CONFIG = {
    "PE-Core-L14-336": PECustomConfig(),
    "PE-Core-L14-336-splitqkv": PECustomConfig(split_qkv=True,
                                               pretrained_split_qkv_path=PRETRAINED_MODEL_PATH),
    "FT_PE-Core-L14-336_250804": PECustomConfig(load_type="lora_weight_load", 
                                                weight_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-L14-336_250804", "FT_PE-Core-L14-336_250804.pt"),
                                                lora_adapter_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-L14-336_250804", "FT_PE-Core-L14-336_250804_adapter"),
                                                split_qkv=True,
                                                pretrained_split_qkv_path=PRETRAINED_MODEL_PATH),
    "FT_PE-Core-L14-336_MHCA_250915": PECustomConfig(
        load_type="lora_weight_head_load",
        weight_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-L14-336_MHCA_250915", "FT_PE-Core-L14-336_MHCA_250915.pt"),
        lora_adapter_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-L14-336_MHCA_250915", "FT_PE-Core-L14-336_MHCA_250915_adapter"),
        split_qkv=True,
        pretrained_split_qkv_path=PRETRAINED_MODEL_PATH,
        additional_head_args=MHCA_HEAD_L14
    ),
    "PE-Core-S16-384": PECustomConfig(original_model_name="PE-Core-S16-384"),
    "FT_PE-Core-S16-384": PECustomConfig(original_model_name="PE-Core-S16-384",
                                                load_type="lora_weight_load", 
                                                weight_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_260301", "FT_PE-Core-S16-384_260301.pt"),
                                                lora_adapter_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_260301", "FT_PE-Core-S16-384_260301_adapter"),
                                                split_qkv=True,
                                                pretrained_split_qkv_path=PRETRAINED_MODEL_PATH),

    "FT_PE-Core-S16-384_Linear": PECustomConfig(
        original_model_name="PE-Core-S16-384",
        load_type="lora_weight_head_load",
        weight_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251204_exp1", "FT_PE-Core-S16-384_251204_exp1.pt"),
        lora_adapter_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251204_exp1", "adapter"),
        split_qkv=True,
        pretrained_split_qkv_path=PRETRAINED_MODEL_PATH,
        additional_head_args=LINEAR_HEAD_S16
    ),


    "FT_PE-Core-S16-384_MHCA": PECustomConfig(
        original_model_name="PE-Core-S16-384",
        load_type="lora_weight_head_load",
        weight_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251204_exp2", "FT_PE-Core-S16-384_251204_exp2.pt"),
        lora_adapter_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251204_exp2", "adapter"),
        split_qkv=True,
        pretrained_split_qkv_path=PRETRAINED_MODEL_PATH,
        additional_head_args=MHCA_HEAD_S16
    ),
    "FT_PE-Core-S16-384_EfficientProbe_Q32": PECustomConfig(
        original_model_name="PE-Core-S16-384",
        load_type="lora_weight_head_load",
        weight_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251220_exp2", "FT_PE-Core-S16-384_251220_exp2.pt"),
        lora_adapter_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251220_exp2", "adapter"),
        split_qkv=True,
        pretrained_split_qkv_path=PRETRAINED_MODEL_PATH,
        additional_head_args=EFFICIENTPROBEQ32_HEAD_S16
    ),

    "FT_PE-Core-S16-384_Perciever1": PECustomConfig(
        original_model_name="PE-Core-S16-384",
        load_type="lora_weight_head_load",
        weight_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251208_exp1", "FT_PE-Core-S16-384_251208_exp1.pt"),
        lora_adapter_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251208_exp1", "adapter"),
        split_qkv=True,
        pretrained_split_qkv_path=PRETRAINED_MODEL_PATH,
        additional_head_args=PERCIEVER1_HEAD_S16
    ),
    "FT_PE-Core-S16-384_Perciever2": PECustomConfig(
        original_model_name="PE-Core-S16-384",
        load_type="lora_weight_head_load",
        weight_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251208_exp2", "FT_PE-Core-S16-384_251208_exp2.pt"),
        lora_adapter_path=os.path.join(ASSETS_MODEL_SAVE_DIR, HF_NAMESPACE, "FT_PE-Core-S16-384_251208_exp2", "adapter"),
        split_qkv=True,
        pretrained_split_qkv_path=PRETRAINED_MODEL_PATH,
        additional_head_args=PERCIEVER2_HEAD_S16
    ),
}
