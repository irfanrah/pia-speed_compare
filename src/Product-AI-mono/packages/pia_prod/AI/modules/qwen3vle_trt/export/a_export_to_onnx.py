"""
Export Qwen3-VL-Embedding model to ONNX files for inference.

Produces 2 ONNX files:
  1. Vision.onnx      — Vision encoder (fixed image resolution)
  2. Transformer.onnx — Transformer decoder layers (embedding mode, no KV cache)

Also saves rotary_params.npz, which contains:
  • inv_freq + mrope_section — for Python-side mRoPE computation
  • embed_weight             — token-embedding lookup table (replaces Embed.onnx)
  • image dimensions / config values

Usage:
    python 1_export_to_onnx.py
    python 1_export_to_onnx.py --model_path /path/to/model --output_dir /path/to/output
"""

import math
import os
import gc
import glob
import argparse
import torch
import onnx
import numpy as np
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel,
    Qwen3VLTextRotaryEmbedding,
)
from pia.ai.tasks.T2VRet.models.qwen3_vl_embedding.models.qwen3_vl_embedding import Qwen3VLForEmbedding, MAX_TOTAL_PIXELS


def consolidate_external_data(onnx_path: str):
    """
    Re-save an ONNX model so all external weights live in one .onnx.data file
    instead of hundreds of per-tensor files.
    """
    out_dir  = os.path.dirname(onnx_path)
    base     = os.path.basename(onnx_path)
    data_rel = base + ".data"                      # sibling file: Transformer.onnx.data
    model    = onnx.load(onnx_path, load_external_data=True)

    # Delete ONNX per-tensor external files (e.g.
    # `model.model.language_model.layers.0.self_attn.qk_norm_weight`). They can
    # be named with dots, so we use an allow-list of known suffixes we want to
    # keep. Subdirectories (e.g. `tokenizer/`) are skipped via the isfile check.
    PRESERVE_SUFFIXES = (".onnx", ".onnx.data", ".npz", ".bin")
    for fname in list(os.listdir(out_dir)):
        fpath = os.path.join(out_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if any(fname.endswith(sfx) for sfx in PRESERVE_SUFFIXES):
            continue
        try:
            os.remove(fpath)
        except OSError:
            pass

    # Remove the target .data file if it already exists (stale)
    target_data = os.path.join(out_dir, data_rel)
    if os.path.exists(target_data):
        os.remove(target_data)

    onnx.save_model(
        model, onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_rel,
        size_threshold=1024,
        convert_attribute=False,
    )
    del model
    gc.collect()


# ══════════════════════════════════════════════════════════════════════════════
# Configuration — reads from the module's config.py
# ══════════════════════════════════════════════════════════════════════════════
from pia_prod.AI.modules.qwen3vle_trt.config import (
    IMG_SIZE,
    QWEN3VLE_TRT_PT_MODEL_PATH,
    QWEN3VLE_TRT_ONNX_DIR_PATH,
    TEMPORAL_SIZE,
)

DEFAULT_MODEL_PATH = QWEN3VLE_TRT_PT_MODEL_PATH
DEFAULT_OUTPUT_DIR = QWEN3VLE_TRT_ONNX_DIR_PATH

# Vision — fixed image resolution at export time.
# Must be multiples of (patch_size * merge_size), validated after model loading.
IMAGE_HEIGHT     = IMG_SIZE[0]
IMAGE_WIDTH      = IMG_SIZE[1]

MAX_SEQ_LEN      = MAX_TOTAL_PIXELS                           # 8192
OPSET            = 17


# ══════════════════════════════════════════════════════════════════════════════
# Token Embedding Module
# ══════════════════════════════════════════════════════════════════════════════
class LLM_EMBED(torch.nn.Module):
    """Extract the token-embedding layer and run it in float32."""

    def __init__(self, model):
        super().__init__()
        self.embed_tokens = model.model.language_model.embed_tokens.float()

    def forward(self, input_ids):
        return self.embed_tokens(input_ids)


# ══════════════════════════════════════════════════════════════════════════════
# Vision Encoder Module
# ══════════════════════════════════════════════════════════════════════════════
class LLM_VISION_EMBED(torch.nn.Module):
    """
    ONNX-exportable vision encoder with baked-in positional / rotary
    embeddings for a fixed image resolution.

    Takes the HF processor's pixel_values directly (already CLIP-normalised
    and patched) so the patch ordering, normalisation, and Conv3d behaviour
    are identical to the PyTorch model.

    Input : float32 pixel_values [total_patches, flatten_dim]
    Output: (deepstack_feature_0, …, vision_hidden_states)
    """

    def __init__(self, model, height_factor, width_factor, temporal_patches=1):
        super().__init__()
        visual        = model.model.visual
        vision_config = model.config.vision_config

        self.num_heads     = vision_config.num_heads
        self.head_dim      = vision_config.hidden_size // self.num_heads
        self.head_dim_half = self.head_dim // 2
        self.embed_dim     = visual.patch_embed.embed_dim
        self.merge_size    = visual.spatial_merge_size
        self.t_patches     = temporal_patches

        ms = self.merge_size
        self.grid_h = height_factor * ms
        self.grid_w = width_factor  * ms

        # ── pre-compute positional / rotary embeddings for the fixed grid ─
        grid_thw = torch.tensor(
            [[self.t_patches, self.grid_h, self.grid_w]], dtype=torch.int32
        )

        pos_embeds = Qwen3VLVisionModel.fast_pos_embed_interpolate(
            visual, grid_thw
        ).unsqueeze(0)                                  # [1, N, embed_dim]
        self.register_buffer("pos_embeds", pos_embeds)

        rot_pos_emb = (
            Qwen3VLVisionModel.rot_pos_emb(visual, grid_thw)
            .float()
            .unsqueeze(0).unsqueeze(0).unsqueeze(0)     # [1, 1, 1, N, dim/2]
        )
        cos = rot_pos_emb.cos()
        sin = rot_pos_emb.sin()
        self.register_buffer("rotary_cos", torch.cat([cos, cos], dim=-1))
        self.register_buffer("rotary_sin", torch.cat([-sin, sin], dim=-1))

        # ── ONNX-friendly GELU ──────────────────────────────────────────
        self._replace_gelu(visual)

        # ── fuse layer-norms into the subsequent linear layers ──────────
        scaling = self.head_dim ** -0.25
        for blk in visual.blocks:
            blk.attn.qkv.weight.data[: -self.embed_dim] *= scaling
            blk.attn.qkv.bias.data[: -self.embed_dim]   *= scaling
            self._fuse_norm(blk.norm1, blk.attn.qkv)
            self._fuse_norm(blk.norm2, blk.mlp.linear_fc1)

        for ds_layer in visual.deepstack_merger_list:
            self._fuse_norm(ds_layer.norm, ds_layer.linear_fc1)

        self._fuse_norm(visual.merger.norm, visual.merger.linear_fc1)

        self.visual = visual

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _fuse_norm(norm, linear):
        norm_bias   = norm.bias.data
        norm_weight = norm.weight.data
        if linear.weight.shape[1] != norm_bias.shape[0]:
            repeat_factor = linear.weight.shape[1] // norm_bias.shape[0]
            norm_bias     = norm_bias.repeat(repeat_factor)
            norm_weight   = norm_weight.repeat(repeat_factor)
        linear.bias.data.add_(torch.matmul(linear.weight.data, norm_bias))
        linear.weight.data.mul_(norm_weight.unsqueeze(0))
        norm.elementwise_affine = False
        norm.weight = None
        norm.bias   = None

    @staticmethod
    def _replace_gelu(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate="tanh"))
            else:
                LLM_VISION_EMBED._replace_gelu(child)

    def _rotate_half(self, x, batch_size):
        x = x.view(2, batch_size, self.num_heads, -1, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(2, batch_size, self.num_heads, -1, self.head_dim)

    def _spatial_merge(self, x, target_hidden_size):
        """Explicit spatial reshape: group merge_size×merge_size patches per frame."""
        ms = self.merge_size
        x = x.view(1, self.t_patches, self.grid_h, self.grid_w, -1)
        x = x.view(1, self.t_patches, self.grid_h // ms, ms, self.grid_w // ms, ms, -1)
        x = x.permute(0, 1, 2, 4, 3, 5, 6)
        return x.reshape(1, -1, target_hidden_size)

    # ── forward ──────────────────────────────────────────────────────────
    def forward(self, pixel_values):
        """
        pixel_values : float32 [total_patches, flatten_dim]
            Already CLIP-normalised, from the HF processor.
        """
        batch_size = 1

        # patch embedding (original HF module)
        hidden = self.visual.patch_embed(pixel_values)           # [N, embed_dim]
        hidden = hidden.unsqueeze(0)                             # [1, N, embed_dim]
        hidden = hidden + self.pos_embeds

        # transformer blocks
        deepstack_features = []
        ds_indices = self.visual.deepstack_visual_indexes
        ds_modules = self.visual.deepstack_merger_list

        for layer_num, blk in enumerate(self.visual.blocks):
            # --- self-attention (manual, ONNX-safe) ---
            h_norm = blk.norm1(hidden)
            qkv    = blk.attn.qkv(h_norm)
            qkv    = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
            qkv    = qkv.permute(2, 0, 3, 1, 4)          # [3, B, heads, seq, dim]
            qk, v  = qkv.split([2, 1], dim=0)

            qk_rot = (
                qk * self.rotary_cos
                + self._rotate_half(qk, batch_size) * self.rotary_sin
            )
            q_rot, k_rot = qk_rot.split([1, 1], dim=0)
            attn = torch.matmul(q_rot, k_rot.transpose(-1, -2))
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v)
            attn = attn.transpose(2, 3).reshape(
                batch_size, -1, blk.attn.proj.in_features
            )
            hidden = hidden + blk.attn.proj(attn)

            # --- feed-forward ---
            mlp_out = blk.mlp.linear_fc1(blk.norm2(hidden))
            mlp_out = blk.mlp.act_fn(mlp_out)
            mlp_out = blk.mlp.linear_fc2(mlp_out)
            hidden  = hidden + mlp_out

            # --- deepstack (explicit spatial merge) ---
            if layer_num in ds_indices:
                idx      = ds_indices.index(layer_num)
                ds_layer = ds_modules[idx]
                x_ds = self._spatial_merge(hidden, ds_layer.hidden_size)
                x_ds = ds_layer.norm(x_ds)
                x_ds = ds_layer.linear_fc1(x_ds)
                x_ds = ds_layer.act_fn(x_ds)
                x_ds = ds_layer.linear_fc2(x_ds)
                deepstack_features.append(x_ds)

        # merger (explicit spatial merge)
        hidden = self.visual.merger.norm(hidden)
        hidden = self._spatial_merge(hidden, self.visual.merger.hidden_size)
        hidden = self.visual.merger.linear_fc1(hidden)
        hidden = self.visual.merger.act_fn(hidden)
        hidden = self.visual.merger.linear_fc2(hidden)

        return *deepstack_features, hidden


# ══════════════════════════════════════════════════════════════════════════════
# Transformer Module (embedding mode — no KV cache, no lm_head)
# ══════════════════════════════════════════════════════════════════════════════
class LLM_MAIN_EMBED(torch.nn.Module):
    """
    All decoder layers in a single ONNX graph.

    Differences from the generation-oriented LLM_MAIN:
      • No KV-cache inputs / outputs  (single forward pass)
      • No lm_head projection         (we need hidden states, not logits)
      • Final RMSNorm is kept         (not fused into lm_head)
      • Outputs the full sequence      (not just last token)

    Input layout  (via *all_inputs):
        hidden_states, ds_feat_0 … ds_feat_N, rotary_cos, rotary_sin, attn_mask

    Output:
        last_hidden_state  [batch, seq_len, hidden_size]
    """

    def __init__(
        self,
        model,
        num_heads,
        num_key_value_heads,
        head_dim,
        num_layers,
        hidden_size,
        deepstack_features_len,
    ):
        super().__init__()
        self.model = model

        self.head_dim            = head_dim
        self.head_dim_half       = head_dim // 2
        self.num_heads           = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.qk_heads            = num_heads + num_key_value_heads
        self.num_layers          = num_layers
        self.deepstack_features_len = deepstack_features_len

        # 3 trailing args after deepstack features: cos, sin, mask
        self._ds_offset = 3 + deepstack_features_len

        # ── save the final RMSNorm (before weight fusion touches layers) ─
        lang = model.model.language_model
        self.register_buffer(
            "final_norm_weight", lang.norm.weight.data.clone()
        )
        self.final_norm_eps = float(lang.norm.variance_epsilon)

        # ── ONNX-friendly GELU ──────────────────────────────────────────
        self._replace_gelu(model)

        # ── fuse weights (same optimisations as reference) ──────────────
        self._fuse_weights(hidden_size)

    # ══════════════════════════════════════════════════════════════════════
    # Weight fusion (runs once at init)
    # ══════════════════════════════════════════════════════════════════════
    def _fuse_weights(self, hidden_size):
        scale_factor   = self.head_dim ** -0.25
        norm_factor    = hidden_size ** 0.5
        norm_factor_qk = self.head_dim ** 0.5

        with torch.no_grad():
            for layer in self.model.model.language_model.layers:
                self._fuse_qkv(layer, scale_factor, norm_factor, norm_factor_qk)
                self._fuse_gate_up(layer, norm_factor)
            # NOTE: we do NOT fuse the final norm into lm_head (no lm_head).

    def _fuse_qkv(self, layer, scale_factor, norm_factor, norm_factor_qk):
        attn = layer.self_attn
        q, k, v = attn.q_proj, attn.k_proj, attn.v_proj

        in_f  = int(q.in_features)
        out_f = int(q.out_features + k.out_features + v.out_features)
        has_bias = any(p.bias is not None for p in (q, k, v))

        qkv = torch.nn.Linear(in_f, out_f, bias=has_bias)
        qkv.weight.copy_(torch.cat([q.weight, k.weight, v.weight], dim=0))
        if has_bias:
            def _b(p):
                return p.bias if p.bias is not None else torch.zeros(
                    p.out_features, dtype=qkv.weight.dtype
                )
            qkv.bias.copy_(torch.cat([_b(q), _b(k), _b(v)], dim=0))

        attn.q_out_features = int(q.out_features)
        attn.k_out_features = int(k.out_features)
        attn.v_out_features = int(v.out_features)
        del attn.q_proj, attn.k_proj, attn.v_proj

        # fuse QK norms + attention scaling
        combined_scale = scale_factor * norm_factor_qk
        attn.q_norm.weight.mul_(combined_scale)
        attn.k_norm.weight.mul_(combined_scale)
        q_norm_rep = attn.q_norm.weight.repeat(self.num_heads)
        k_norm_rep = attn.k_norm.weight.repeat(self.num_key_value_heads)
        attn.qk_norm_weight = torch.nn.Parameter(
            torch.cat([q_norm_rep, k_norm_rep], dim=0).view(
                1, 1, 1, -1, self.head_dim
            )
        )
        del attn.q_norm, attn.k_norm

        # absorb input LayerNorm into QKV
        input_norm_w = layer.input_layernorm.weight.unsqueeze(0) * norm_factor
        qkv.weight.mul_(input_norm_w)
        attn.qkv = qkv
        del layer.input_layernorm

    def _fuse_gate_up(self, layer, norm_factor):
        post_norm_w = layer.post_attention_layernorm.weight.unsqueeze(0) * norm_factor
        gate, up = layer.mlp.gate_proj, layer.mlp.up_proj
        gate_up = torch.nn.Linear(
            gate.in_features, gate.out_features + up.out_features, bias=False
        )
        gate_up.weight.copy_(
            torch.cat([gate.weight * post_norm_w, up.weight * post_norm_w], dim=0)
        )
        layer.mlp.gate_up_proj = gate_up
        del layer.mlp.gate_proj, layer.mlp.up_proj, layer.post_attention_layernorm

    # ══════════════════════════════════════════════════════════════════════
    # Utility
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _replace_gelu(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.GELU):
                setattr(module, name, torch.nn.GELU(approximate="tanh"))
            else:
                LLM_MAIN_EMBED._replace_gelu(child)

    def _rms_norm(self, x):
        """Sum-based RMS norm (weight is absorbed into the subsequent linear)."""
        return x * torch.rsqrt(x.square().sum(-1, keepdim=True))

    def _rotate_half(self, x, batch_size):
        x = x.view(batch_size, -1, 1, self.qk_heads, 2, self.head_dim_half)
        x = x.flip(-2)
        return x.view(batch_size, -1, 1, self.qk_heads, self.head_dim)

    # ══════════════════════════════════════════════════════════════════════
    # Forward
    # ══════════════════════════════════════════════════════════════════════
    def forward(self, *all_inputs):
        """
        all_inputs layout:
            hidden_states          [batch, seq, hidden]
            deepstack_feat_0 …     [1, seq, hidden]   (×deepstack_features_len)
            rotary_cos             [1, seq, 1, 1, head_dim]
            rotary_sin             [1, seq, 1, 1, head_dim]
            attention_mask         [1, 1, 1, seq, seq]
        """
        hidden_states  = all_inputs[0]
        rotary_cos     = all_inputs[-3]
        rotary_sin     = all_inputs[-2]
        attention_mask  = all_inputs[-1]
        batch_size     = hidden_states.shape[0]

        for i, layer in enumerate(self.model.model.language_model.layers):
            # ── self-attention ───────────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            qkv = layer.self_attn.qkv(hidden_states)
            qkv = qkv.reshape(
                batch_size, -1, 1,
                self.qk_heads + self.num_key_value_heads,
                self.head_dim,
            )
            qk, v = torch.split(
                qkv, [self.qk_heads, self.num_key_value_heads], dim=-2
            )

            qk     = self._rms_norm(qk) * layer.self_attn.qk_norm_weight
            qk_rot = (
                qk * rotary_cos
                + self._rotate_half(qk, batch_size) * rotary_sin
            )

            q, k = torch.split(
                qk_rot, [self.num_heads, self.num_key_value_heads], dim=-2
            )
            q = q.reshape(
                batch_size, -1,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            )
            q = q.permute(0, 2, 3, 1, 4)      # [B, kv_h, g, seq, dim]
            k = k.permute(0, 3, 2, 4, 1)       # [B, kv_h, 1, dim, seq]
            v = v.transpose(1, 3)               # [B, kv_h, 1, seq, dim]

            # no KV cache — direct attention over the full sequence
            attn = torch.matmul(q, k) + attention_mask
            attn = torch.softmax(attn, dim=-1)
            attn = torch.matmul(attn, v)

            attn = attn.permute(0, 3, 1, 2, 4).reshape(
                batch_size, -1, layer.self_attn.o_proj.in_features
            )
            hidden_states = residual + layer.self_attn.o_proj(attn)

            # ── feed-forward ─────────────────────────────────────────
            residual      = hidden_states
            hidden_states = self._rms_norm(hidden_states)

            gate_up = layer.mlp.gate_up_proj(hidden_states)
            gate, up = torch.split(
                gate_up,
                [layer.mlp.down_proj.in_features, layer.mlp.down_proj.in_features],
                dim=-1,
            )
            hidden_states = residual + layer.mlp.down_proj(
                layer.mlp.act_fn(gate) * up
            )

            # ── deepstack feature injection ──────────────────────────
            if i < self.deepstack_features_len:
                hidden_states = all_inputs[1 + i] + hidden_states

        # ── final RMSNorm (standard, not fused) ─────────────────────────
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.final_norm_eps
        )
        hidden_states = hidden_states * self.final_norm_weight

        return hidden_states


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════
def main(model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    onnx_vision      = os.path.join(output_dir, "Vision.onnx")
    onnx_transformer = os.path.join(output_dir, "Transformer.onnx")
    rotary_params    = os.path.join(output_dir, "rotary_params.npz")

    # ── Save processor files (tokenizer + image processor config) ───────
    # Stored in a `tokenizer/` subdirectory so the output layout stays tidy
    # and the directory remains self-contained — no HF repo pull at inference.
    print("Saving processor files …")
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    tokenizer_dir = os.path.join(output_dir, "tokenizer")
    os.makedirs(tokenizer_dir, exist_ok=True)
    Qwen3VLProcessor.from_pretrained(model_path).save_pretrained(tokenizer_dir)
    print(f"  Processor saved to {tokenizer_dir}")

    print("Loading model …")
    model = Qwen3VLForEmbedding.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu", low_cpu_mem_usage=True
    ).eval()

    # ── extract config ──────────────────────────────────────────────────
    text_cfg = model.config.text_config
    num_heads           = text_cfg.num_attention_heads
    num_key_value_heads = text_cfg.num_key_value_heads
    head_dim            = text_cfg.head_dim
    num_layers          = text_cfg.num_hidden_layers
    hidden_size         = text_cfg.hidden_size
    deepstack_features_len = len(model.model.visual.deepstack_visual_indexes)

    patch_size  = model.model.visual.patch_size
    merge_size  = model.model.visual.spatial_merge_size
    temporal_patch_size = model.model.visual.patch_embed.temporal_patch_size
    unit        = patch_size * merge_size

    # Compute or validate IMAGE_HEIGHT / IMAGE_WIDTH
    if IMAGE_HEIGHT is None:
        IMAGE_HEIGHT_val = 16 * unit   # sensible default
    else:
        IMAGE_HEIGHT_val = IMAGE_HEIGHT
    if IMAGE_WIDTH is None:
        IMAGE_WIDTH_val = 16 * unit
    else:
        IMAGE_WIDTH_val = IMAGE_WIDTH

    assert IMAGE_HEIGHT_val % unit == 0, (
        f"IMAGE_HEIGHT ({IMAGE_HEIGHT_val}) must be a multiple of "
        f"patch_size * merge_size = {unit}"
    )
    assert IMAGE_WIDTH_val % unit == 0, (
        f"IMAGE_WIDTH ({IMAGE_WIDTH_val}) must be a multiple of "
        f"patch_size * merge_size = {unit}"
    )

    HEIGHT_FACTOR     = IMAGE_HEIGHT_val // unit
    WIDTH_FACTOR      = IMAGE_WIDTH_val  // unit

    # Number of temporal patches after Conv3d with stride=temporal_patch_size.
    # The HF processor pads frames to multiples of temporal_patch_size.
    temporal_patches  = math.ceil(TEMPORAL_SIZE / temporal_patch_size)
    vision_embed_size = temporal_patches * HEIGHT_FACTOR * WIDTH_FACTOR

    print(f"  text  : layers={num_layers}  heads={num_heads}  kv_heads={num_key_value_heads}  "
          f"head_dim={head_dim}  hidden={hidden_size}")
    print(f"  vision: deepstack={deepstack_features_len}  patch={patch_size}  merge={merge_size}")
    print(f"  TEMPORAL_SIZE={TEMPORAL_SIZE}  temporal_patch_size={temporal_patch_size}"
          f"  →  temporal_patches={temporal_patches}")
    print(f"  image size: {IMAGE_HEIGHT_val}x{IMAGE_WIDTH_val}  →  {vision_embed_size} vision tokens")

    with torch.inference_mode():

        # ══════════════════════════════════════════════════════════════════
        # 1. Save rotary parameters + token embedding weight (to skip Embed.onnx)
        # ══════════════════════════════════════════════════════════════════
        rotary_emb = model.model.language_model.rotary_emb
        inv_freq      = rotary_emb.inv_freq.cpu().numpy()
        mrope_section = np.array(rotary_emb.mrope_section, dtype=np.int64)
        embed_weight = (
            model.model.language_model.embed_tokens.weight
            .detach().cpu().float().numpy()
        )
        np.savez(
            rotary_params,
            inv_freq=inv_freq,
            mrope_section=mrope_section,
            embed_weight=embed_weight,
            image_height=IMAGE_HEIGHT_val,
            image_width=IMAGE_WIDTH_val,
            height_factor=HEIGHT_FACTOR,
            width_factor=WIDTH_FACTOR,
            patch_size=patch_size,
            merge_size=merge_size,
            hidden_size=hidden_size,
            head_dim=head_dim,
            max_seq_len=MAX_SEQ_LEN,
            temporal_patch_size=temporal_patch_size,
            temporal_patches=temporal_patches,
            temporal_size=TEMPORAL_SIZE,
        )
        print(f"Saved rotary params → {rotary_params}")

        # ══════════════════════════════════════════════════════════════════
        # 2. Export Vision
        # ══════════════════════════════════════════════════════════════════
        print("Exporting Vision …")
        total_patches = temporal_patches * HEIGHT_FACTOR * merge_size * WIDTH_FACTOR * merge_size
        flatten_dim   = 3 * temporal_patch_size * patch_size * patch_size
        dummy_pixels  = torch.randn(total_patches, flatten_dim, dtype=torch.float32)

        vision_output_names = []
        vision_dynamic_axes = {}
        for i in range(deepstack_features_len):
            name = f"deepstack_feature_{i}"
            vision_output_names.append(name)
        vision_output_names.append("vision_hidden_states")

        torch.onnx.export(
            LLM_VISION_EMBED(model, HEIGHT_FACTOR, WIDTH_FACTOR, temporal_patches),
            (dummy_pixels,),
            onnx_vision,
            input_names=["pixel_values"],
            output_names=vision_output_names,
            dynamic_axes=vision_dynamic_axes if vision_dynamic_axes else None,
            opset_version=OPSET,
            dynamo=False,
        )
        del dummy_pixels
        gc.collect()
        print(f"  → {onnx_vision}")
        print("  Consolidating external weights …")
        consolidate_external_data(onnx_vision)

        # ══════════════════════════════════════════════════════════════════
        # 3. Export Transformer
        # ══════════════════════════════════════════════════════════════════
        print("Exporting Transformer …")

        # dummy sequence length for tracing
        dummy_seq   = 10 + vision_embed_size
        dummy_hidden = torch.ones((1, dummy_seq, hidden_size), dtype=torch.float32)
        dummy_ds     = torch.ones((1, dummy_seq, hidden_size), dtype=torch.float32)
        dummy_cos    = torch.zeros((1, dummy_seq, 1, 1, head_dim), dtype=torch.float32)
        dummy_sin    = torch.zeros((1, dummy_seq, 1, 1, head_dim), dtype=torch.float32)
        dummy_mask   = torch.zeros((1, 1, 1, dummy_seq, dummy_seq), dtype=torch.float32)

        all_inputs  = [dummy_hidden]
        input_names = ["hidden_states"]
        dynamic_axes = {
            "hidden_states":     {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
            "rotary_cos":        {1: "seq_len"},
            "rotary_sin":        {1: "seq_len"},
            "attention_mask":    {3: "seq_len", 4: "seq_len"},
        }

        for i in range(deepstack_features_len):
            name = f"deepstack_features_{i}"
            input_names.append(name)
            all_inputs.append(dummy_ds)
            dynamic_axes[name] = {1: "seq_len"}

        all_inputs.extend([dummy_cos, dummy_sin, dummy_mask])
        input_names.extend(["rotary_cos", "rotary_sin", "attention_mask"])

        model_main = LLM_MAIN_EMBED(
            model, num_heads, num_key_value_heads, head_dim,
            num_layers, hidden_size, deepstack_features_len,
        )
        del model
        gc.collect()

        torch.onnx.export(
            model_main,
            tuple(all_inputs),
            onnx_transformer,
            input_names=input_names,
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            dynamo=False,
        )
        del model_main, all_inputs
        gc.collect()
        print(f"  → {onnx_transformer}")
        print("  Consolidating external weights …")
        consolidate_external_data(onnx_transformer)

    print("Export complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Qwen3-VL-Embedding model to ONNX files"
    )
    parser.add_argument(
        "--model_path", type=str, default=DEFAULT_MODEL_PATH,
        help=f"Path to the HF model (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save ONNX files (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    main(args.model_path, args.output_dir)
