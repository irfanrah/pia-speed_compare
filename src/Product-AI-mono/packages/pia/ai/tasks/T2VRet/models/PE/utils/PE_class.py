import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from peft import get_peft_model, LoraConfig, PeftModel
from collections import OrderedDict

# Safe fallback (only needed if running from deep subfolder)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
# from bluevlm_trainer.modules.vision_text_head import VisionTextHead

from pia.ai.tasks.T2VRet.models.PE.head.vision_text_head import VisionTextHead
import pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.pe as pe
import pia.ai.tasks.T2VRet.models.PE.perception_models.pe_core.vision_encoder.transforms as transforms


class PEModelInitializer:
    def __init__(self,
                model_name: str = 'PE-Core-L14-336',
                load_type: str = 'default',
                device: str = 'cuda:0',
                pretrained: bool = True,
                weight_path = None,
                lora_adapter_path = None,
                split_qkv: bool = False,
                pretrained_split_qkv_path = None,
                head_args = None
                ):
        
        valid_load_types = ['default', 'weight_load', 'lora_weight_load', 'lora_adapter_load', 'lora_weight_head_load']
        assert load_type in valid_load_types, f"Invalid load_type: '{load_type}'. Must be one of: {', '.join(valid_load_types)}"
        self.model_name = model_name
        self.device = device
        self.load_type = load_type
        self.pretrained = pretrained
        self.weight_path = weight_path
        self.lora_adapter_path = lora_adapter_path
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.split_qkv = split_qkv
        self.pretrained_split_qkv_path = pretrained_split_qkv_path
        self.head_args = head_args


        # Define self.constraints for each mode
        self.constraints = {
            "default": {
                "lora_adapter_path": False,
            },
            "weight_load": {
                "weight_path": "required",
                "lora_adapter_path": False,
            },
            "lora_weight_load": {
                "weight_path": "required",
                "lora_adapter_path": "required",
            },
            "lora_adapter_load": {
                "lora_adapter_path": "required",
                "weight_path": False,
            },
            "lora_weight_head_load": {
                "head_args": "required",
                "pretrained_split_qkv_path": "required"            
                },
        }


    def load_base_model(self, ):
        if self.split_qkv:
            self.model_name = self.model_name + "-splitqkv"
            if self.pretrained:
                self.model = pe.CLIP.from_config(self.model_name, 
                                                 checkpoint_path=self.weight_path,
                                                 device=self.device, 
                                                 load_default_weights=False)
                print(f"Loaded weights with split QKV: {self.pretrained_split_qkv_path}")
            else:
                self.model = pe.CLIP.from_config(self.model_name, load_default_weights=False, device=self.device)

        else:
            if self.pretrained:
                self.model = pe.CLIP.from_config(self.model_name, device=self.device)
            else:
                self.model = pe.CLIP.from_config(self.model_name, 
                                                 checkpoint_path=self.weight_path, 
                                                 device=self.device)


    def load_fine_tuned_weights(self):
        self.load_base_model()
        if not os.path.exists(self.weight_path):
            raise FileNotFoundError(f"Fine-tuned weights not found: {self.weight_path}")
        state_dict = torch.load(self.weight_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        suffix = "_".join(self.weight_path.split("/")[-2:])
        self.model_name = f"{self.model_name}_{suffix}"

    def load_lora_weight(self):
        """Load LoRA adapter and optionally pretrained weights."""
        self.load_base_model()

        # Skip if no LoRA weight or not using LoRA mode
        if self.load_type not in ["lora_weight_head_load", "lora_weight_load"] or not self.weight_path:
            return

        # === Load adapter configuration ===
        adapter_config_path = os.path.join(self.lora_adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")

        with open(adapter_config_path, "r") as f:
            config_json = json.load(f)

        lora_config = LoraConfig(
            r=config_json["r"],
            lora_alpha=config_json["lora_alpha"],
            target_modules=config_json["target_modules"],
            lora_dropout=config_json.get("lora_dropout", 0.0),
            bias=config_json.get("bias", "none"),
            modules_to_save=config_json.get("modules_to_save", None),
            task_type=config_json.get("task_type", "FEATURE_EXTRACTION"),
        )

        # === Initialize LoRA model ===
        self.model = get_peft_model(self.model, lora_config)

        # === Load LoRA weights (for eval/inference mode only) ===
        if self.load_type == "lora_weight_load":
            state_dict = torch.load(self.weight_path, map_location=self.device)
            self.model.load_state_dict(state_dict)



    def load_lora_adapter(self):
        self.load_base_model()
        if not os.path.exists(self.pretrained_split_qkv_path):
            raise FileNotFoundError(f"{self.pretrained_split_qkv_path} does not exist.")
        
        adapter_config_path = os.path.join(self.lora_adapter_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError(f"Adapter config not found: {adapter_config_path}")
        self.model = PeftModel.from_pretrained(self.model, self.lora_adapter_path)


    def _load_weight_with_head(self, model_with_head, strict=False):
        """
        Load weights into `model_with_head` from `weight_path`, handling common checkpoint formats.
        Returns (missing_keys, unexpected_keys).
        """
        if not os.path.isfile(self.weight_path):
            raise FileNotFoundError(f"Weight file not found: {self.weight_path}")

        ckpt = torch.load(self.weight_path, map_location=self.device)

        # Support common wrappers: plain state_dict, {'state_dict': ...}, {'model': ...}
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
                state = ckpt["state_dict"]
            elif "model" in ckpt and isinstance(ckpt["model"], dict):
                state = ckpt["model"]
            else:
                # assume it's already a state_dict
                state = ckpt
        else:
            # very rare, but keep fallback
            state = ckpt

        # Strip DistributedDataParallel prefixes if present
        cleaned = OrderedDict()
        for k, v in state.items():
            if k.startswith("module."):
                cleaned[k[len("module."):]] = v
            else:
                cleaned[k] = v

        missing, unexpected = model_with_head.load_state_dict(cleaned, strict=strict)
        return missing, unexpected
 

    def setup_transforms(self):
        self.preprocess = transforms.get_image_transform(self.model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(self.model.context_length)
        self.max_words = self.model.context_length
        self.image_resolution = self.model.image_size

    def initialize(self):

        # Validate self.constraints if load_type exists
        if self.load_type not in self.constraints:
            raise ValueError(f"Unknown load_type: {self.load_type}")

        for attr, expected in self.constraints.get(self.load_type, {}).items():
            value = getattr(self, attr)
            if expected == "required":
                assert value, f"{attr} is required for load_type '{self.load_type}' but was not provided"
            elif expected is None:
                assert value is None, f"{attr} must be None for load_type '{self.load_type}' but got: {value}"
            elif expected is False:
                assert not value, f"{attr} should be False/empty for {self.load_type} load_type"

        # Mode-specific loading logic
        if self.load_type == "default":
            self.load_base_model()
            print("Loaded base model")
            self.setup_transforms()

        elif self.load_type == "weight_load":
            self.load_fine_tuned_weights()
            print("Loaded fine-tuned model")
            self.setup_transforms()

        elif self.load_type == "lora_weight_load":            
            self.load_lora_weight()
            print("Loaded LoRA with weights from lora_weight_load")
            self.setup_transforms()

        elif self.load_type == "lora_adapter_load":
            self.load_lora_adapter()
            print("Loaded LoRA with Adapter")
            self.setup_transforms()
        
        elif self.load_type == "lora_weight_head_load":            
            self.load_lora_weight()
            self.setup_transforms()
            print("Loaded LoRA with weights from lora_weight_head_load")
            base_model = self.model
            model = VisionTextHead(self.head_args, base_model, self.device)
            print("Loaded VisionTextHead layer")
            if self.weight_path:
                missing, unexpected = self._load_weight_with_head(model_with_head = model,
                                                                  strict=True)
                print("Loaded VisionTextHead weight")
            del self.model
            self.model = model
        
        return self.model, self.preprocess, self.tokenizer, self.max_words, self.image_resolution

