from typing import Optional, Tuple

import numpy as np
import torch


def apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: np.ndarray) -> torch.Tensor:
    """
    Interleave T/H/W channels — matches HF's apply_interleaved_mrope.

    freqs:         [3, 1, seq_len, freq_dim]  (torch, on device)
    mrope_section: [3]                        (numpy; read as Python ints)
    Returns:       [1, seq_len, freq_dim]     (torch, on same device)
    """
    result = freqs[0].clone()
    for dim, offset in enumerate((1, 2), start=1):
        length = int(mrope_section[dim]) * 3
        idx = slice(offset, length, 3)
        result[..., idx] = freqs[dim, ..., idx]
    return result


def compute_position_ids(
    seq_len: int,
    image_start_pos: Optional[int],
    vision_embed_size: int,
    height_factor: int,
    width_factor: int,
    temporal_patches: int = 1,
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Build 3-channel mRoPE position IDs.

    For text-only: all 3 channels are simple linear [0, 1, 2, ...].
    For image/video+text: the vision region gets 3D (T, H, W) positions:
      - T channel: temporal index per frame (flat within each frame)
      - H channel: spatial row within each frame
      - W channel: spatial column within each frame

    Returns: [3, 1, seq_len] float32 on device.
    """
    pos = torch.zeros((3, 1, seq_len), dtype=torch.float32, device=device)

    if image_start_pos is None:
        linear = torch.arange(seq_len, dtype=torch.float32, device=device)
        pos[:, 0, :] = linear
        return pos

    img_end = image_start_pos + vision_embed_size
    spatial_size = height_factor * width_factor  # tokens per temporal patch

    # Before image — linear
    pre = torch.arange(image_start_pos, dtype=torch.float32, device=device)
    pos[:, 0, :image_start_pos] = pre

    # Vision region — 3D (T, H, W) positions per temporal patch
    for t in range(temporal_patches):
        base = image_start_pos + t * spatial_size

        # T channel: temporal index (constant within each frame)
        pos[0, 0, base:base + spatial_size] = image_start_pos + t

        # H channel: spatial row, repeated per frame
        for r in range(height_factor):
            s = base + r * width_factor
            e = s + width_factor
            pos[1, 0, s:e] = image_start_pos + r

        # W channel: spatial column, repeated per frame
        width_range = torch.arange(
            image_start_pos, image_start_pos + width_factor,
            dtype=torch.float32, device=device,
        )
        for r in range(height_factor):
            s = base + r * width_factor
            e = s + width_factor
            pos[2, 0, s:e] = width_range

    # After image — resume linear from max position used + 1
    start_id = image_start_pos + max(temporal_patches, height_factor, width_factor)
    after_len = seq_len - img_end
    if after_len > 0:
        tail = torch.arange(
            start_id, start_id + after_len, dtype=torch.float32, device=device,
        )
        pos[:, 0, img_end:] = tail

    return pos


def compute_rotary(
    position_ids: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: np.ndarray,
    head_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute rotary cos/sin from position IDs.

    position_ids: [3, 1, seq_len]  (torch, on device)
    inv_freq:     [freq_dim]       (torch, on same device)
    mrope_section:[3]              (numpy)

    Returns: (cos, sin) each [1, seq_len, 1, 1, head_dim] float32 on device.
    """
    inv_freq_3d = inv_freq[None, :, None].expand(3, -1, 1).contiguous()

    freqs = inv_freq_3d * position_ids                          # [3, freq_dim, seq_len]
    freqs = freqs.permute(0, 2, 1).unsqueeze(1)                 # [3, 1, seq_len, freq_dim]
    freqs = apply_interleaved_mrope(freqs, mrope_section)       # [1, seq_len, freq_dim]

    cos_f = torch.cos(freqs)
    sin_f = torch.sin(freqs)

    # Double for flip-based rotate_half in transformer blocks
    cos_full = torch.cat([cos_f, cos_f], dim=-1).unsqueeze(2).unsqueeze(3)
    sin_full = torch.cat([-sin_f, sin_f], dim=-1).unsqueeze(2).unsqueeze(3)
    return cos_full.to(torch.float32), sin_full.to(torch.float32)


def build_causal_mask(seq_len: int, *, device: torch.device) -> torch.Tensor:
    """Upper-triangular causal mask: [1, 1, 1, seq_len, seq_len] float32 on device."""
    mask = torch.triu(
        torch.full((seq_len, seq_len), -1e9, dtype=torch.float32, device=device),
        diagonal=1,
    )
    return mask[None, None, None, :, :]
