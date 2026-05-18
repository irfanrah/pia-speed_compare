import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Optional, Tuple



########################
## Efficient Probe: Attention, Please! Revisiting Attentive Probing for Masked Image Modeling

class EfficientProbing(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 1,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        num_queries: int = 32,
        d_out: int = 1
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        
        self.d_out = d_out
        self.num_queries = num_queries
        
        self.v = nn.Linear(dim, dim // d_out, bias=qkv_bias)
        self.cls_token = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        
    def forward(self, x: torch.Tensor, cls=None, **_: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        C_prime = C // self.d_out

        if cls is not None:
            cls_token = cls
        else:
            cls_token = self.cls_token.expand(B, -1, -1)  # newly created class token

        q = cls_token.reshape(B, self.num_queries, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = (x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3))
        q = q * self.scale
        v = (self.v(x).reshape(B, N, self.num_queries, C // (self.d_out * self.num_queries)).permute(0, 2, 1, 3))

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x_cls = torch.matmul(attn.squeeze(1).unsqueeze(2), v)
        x_cls = x_cls.view(B, C_prime)
        
        return x_cls


class Efficient_1(nn.Module):
    """
    Pools per-frame features [B, T, C] into a single video vector using EfficientProbing.
    """
    def __init__(self, embed_dim=768, num_queries=32, num_heads=1, d_out=1, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.pool = EfficientProbing(
            dim=embed_dim,
            num_heads=num_heads,      # keep 1 for the given EP code
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            num_queries=num_queries,
            d_out=d_out
        )

    def forward(self, frame_feats, cls=None, attn_mask=None):
        """
        frame_feats: [B, T, C] per-video frame features
        cls: optional external query tokens [B, Q, C] if you want to condition queries
        attn_mask: optional bool mask [B, T] (True = keep / False = pad)  # see note below
        """
        x = frame_feats  # [B, T, C], treat frames as the "tokens"
        # (Optional) If you manage variable T across the batch and have attn_mask,
        # you can adapt EfficientProbing to apply -inf to masked positions before softmax.
        # The provided EP code doesn't include mask handling; see "Masking" section below.
        out = self.pool(x, cls=cls)  # [B, C/d_out]
        return out