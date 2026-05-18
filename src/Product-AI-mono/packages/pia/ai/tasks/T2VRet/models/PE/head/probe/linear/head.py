
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Simple attention head: pools (B, T, D) -> (B, D) with same D ---
class Linear_1(nn.Module):
    def __init__(self, 
                 dim: int, 
                 hidden: int = None, 
                 dropout: float = 0.0):
        super().__init__()
        hidden = hidden or max(64, dim // 2)
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)  # scalar score per frame
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, T, D)
        mask: optional Bool (B, T), True=valid, False=padded
        returns: (B, D)
        """
        B, T, D = x.shape
        scores = self.scorer(x).squeeze(-1)          # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)          # (B, T)
        v = torch.einsum("bt, btd -> bd", attn, x)   # (B, D)
        return v
