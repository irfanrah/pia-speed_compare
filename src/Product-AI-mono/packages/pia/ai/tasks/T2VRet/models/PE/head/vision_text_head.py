import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from pia.ai.tasks.T2VRet.models.PE.head.probe.efficient.head import Efficient_1
from pia.ai.tasks.T2VRet.models.PE.head.probe.linear.head import Linear_1
from pia.ai.tasks.T2VRet.models.PE.head.probe.mhca.head import MHCA_1
from pia.ai.tasks.T2VRet.models.PE.head.probe.perceiver.perceiver_pytorch import Perceiver


def _float_or_none(x):
    return float(x) if x is not None else None

def _int_or_none(x):
    return int(x) if x is not None else None


class VisionTextHead(nn.Module):
    """
    Wraps a base CLIP-like model to customize video encoding.
    Modes:
      - use=False:    delegate to model.encode_video(frames)
      - vision='linear': residual D->D projection on the default video feature
      - vision='attention': per-frame encode_image + attention pooling (keeps D)
    """
    def __init__(self, head_args, base_model, device):
        super().__init__()
        self.head_args = head_args
        self.base_model = base_model
        device = device


        cfg = self.head_args.get("additional_head", {})
        vision = cfg.get("vision", {})

        self.use_extra        = bool(cfg.get("use", False))
        self.vision_head_type = vision.get("type")
        self.embed_dim        = vision.get("embed_dim")
        self.hidden_dim       = vision.get("hidden_dim")
        self.dropout          = _float_or_none(vision.get("dropout"))
        self.attn_dropout     = _float_or_none(vision.get("attn_dropout"))
        self.use_pos          = bool(vision.get("use_pos", False))
        self.num_queries      = _int_or_none(vision.get("num_queries"))
        self.attn_head        = _int_or_none(vision.get("attn_head"))
        
        if self.vision_head_type == "Linear_1":
            if self.attn_dropout:
                raise ValueError("Attention Dropout should be False when using Linear_1")
            if self.use_pos:
                raise ValueError("use_pos should be False when using Linear_1")
            if self.num_queries:
                raise ValueError("num_queries should be False when using Linear_1")
            if self.attn_head:
                raise ValueError("attn_head should be False when using Linear_1")
            
            self.vision_head = Linear_1(
                    dim=self.embed_dim,
                    hidden=self.hidden_dim,
                    dropout=self.dropout)
            
        elif self.vision_head_type == "MHCA_1":
            # Lucas head attention
            if self.attn_dropout:
                raise ValueError("Attention Dropout should be False when using MHCA_1")
            if self.use_pos:
                raise ValueError("use_pos should be False when using MHCA_1")
            if self.hidden_dim:
                raise ValueError("hidden_dim must be None or 0 when using MHCA_1")
            if self.num_queries:
                raise ValueError("num_queries should be False when using MHCA_1")
            
            self.vision_head = MHCA_1(
                    embed_dim=self.embed_dim,
                    num_heads=self.attn_head,
                    dropout=self.dropout)

        elif self.vision_head_type == "vid_efficient_1":
            if self.dropout:
                raise ValueError("dropout should be False when using vid_efficient_1")
            if self.attn_dropout:
                raise ValueError("Attention Dropout should be False when using vid_efficient_1")
            if self.use_pos:
                raise ValueError("use_pos should be False when using vid_efficient_1")
            if self.hidden_dim not in (None, 0):
                raise ValueError("hidden_dim must be None or 0 when using vid_efficient_1")
            if self.attn_head:
                raise ValueError("attn_head should be False when using vid_efficient_1")

            self.vision_head = Efficient_1(
                    embed_dim=self.embed_dim,
                    num_queries=self.num_queries)

        elif "perciever" in self.vision_head_type:
            if self.attn_dropout:
                raise ValueError("Attention Dropout should be False when using perciever")
            if self.use_pos:
                raise ValueError("use_pos should be False when using perciever")
            if self.num_queries:
                raise ValueError("num_queries should be False when using perciever")
            if self.attn_head:
                raise ValueError("attn_head should be False when using perciever")
            if self.hidden_dim:
                raise ValueError("hidden_dim must be None or 0 when using perciever")
            if self.dropout:
                raise ValueError("dropout should be False when using perciever")

            if self.vision_head_type == "perciever_1": #GPT config Recommendation
                self.vision_head = Perceiver(
                    num_freq_bands      = 0,
                    depth              = 1,
                    max_freq            = 1.0,
                    input_channels     = self.embed_dim,
                    input_axis         = 1,
                    num_latents        = 4,
                    latent_dim         = self.embed_dim,
                    cross_heads        = 1,
                    latent_heads       = 4,
                    cross_dim_head     = 64,
                    latent_dim_head    = 64,
                    num_classes        = self.embed_dim,
                    attn_dropout       = 0.0,
                    ff_dropout         = 0.0,
                    weight_tie_layers  = False,
                    fourier_encode_data = False)
                
            elif self.vision_head_type == "perciever_2":
                self.vision_head = Perceiver(
                    num_freq_bands      = 0,
                    depth              = 2,
                    max_freq            = 1.0,
                    input_channels     = self.embed_dim,
                    input_axis         = 1,
                    num_latents        = 4,
                    latent_dim         = self.embed_dim // 2,
                    cross_heads        = 1,
                    latent_heads       = 4,
                    cross_dim_head     = 64,
                    latent_dim_head    = 64,
                    num_classes        = self.embed_dim,
                    attn_dropout       = 0.0,
                    ff_dropout         = 0.0,
                    weight_tie_layers  = False,
                    fourier_encode_data = False)
            elif self.vision_head_type == "perciever_3":
                self.vision_head = Perceiver(
                    num_freq_bands      = 0,
                    depth              = 1,
                    max_freq            = 1.0,
                    input_channels     = self.embed_dim,
                    input_axis         = 1,
                    num_latents        = 8,
                    latent_dim         = self.embed_dim // 4,
                    cross_heads        = 1,
                    latent_heads       = 4,
                    cross_dim_head     = 64,
                    latent_dim_head    = 64,
                    num_classes        = self.embed_dim,
                    attn_dropout       = 0.0,
                    ff_dropout         = 0.0,
                    weight_tie_layers  = False,
                    fourier_encode_data = False)
                
            else:
                raise ValueError(f"Wrong self.vision_head_type {self.vision_head_type}")
        else:
            raise ValueError("Wrong self.vision_head_type")

        self.vision_head = self.vision_head.to(device)
    

    def encode_video(self, video: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        if not self.use_extra:
            return self.base_model.encode_video(video, normalize=normalize)

        b, n, c, h, w = video.shape
        frms = video.reshape(b * n, c, h, w)

        frm_feats = self.base_model.encode_image(frms, normalize=False)  # (B*T, D)
        frm_feats = frm_feats.reshape(b, n, -1)                          # (B, T, D)

        video_feats = self.vision_head(frm_feats)                        # (B, D)

        # keep normalize behavior identical to base path
        if normalize:
            video_feats = F.normalize(video_feats, dim=-1)

        return video_feats

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        text_feats = self.base_model.encode_text(tokens)
        return text_feats

    def logit_scale_exp(self,):        
        return self.base_model.logit_scale.exp()
    