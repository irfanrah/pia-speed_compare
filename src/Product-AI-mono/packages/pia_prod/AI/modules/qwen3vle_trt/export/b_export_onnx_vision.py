"""
Direct HF export of the Qwen3-VL vision encoder to ONNX.

Instead of manually reimplementing the vision encoder (with norm fusion,
GELU replacement, etc.), this script wraps the HF model's own
Qwen3VLVisionModel and exports it via torch.onnx.export.

This produces a Vision.onnx that is numerically identical to PyTorch
because it traces through the exact same code path.

Usage:
    python 2_export_onnx_vision.py
    python 2_export_onnx_vision.py --model_path /path/to/model --output_dir /path/to/output

This ONLY re-exports Vision.onnx.  The Embed.onnx, Transformer.onnx,
and rotary_params.npz from export_embedding_onnx.py are reused as-is.
"""

import math
import os
import gc
import argparse
import torch
import onnx
import numpy as np
from pia.ai.tasks.T2VRet.models.qwen3_vl_embedding.models.qwen3_vl_embedding import Qwen3VLForEmbedding
from pia_prod.AI.modules.qwen3vle_trt.export.a_export_to_onnx import consolidate_external_data


# ══════════════════════════════════════════════════════════════════════════════
# Configuration  (must match export_embedding_onnx.py)
# ══════════════════════════════════════════════════════════════════════════════
from pia_prod.AI.modules.qwen3vle_trt.config import (
    QWEN3VLE_TRT_PT_MODEL_PATH,
    QWEN3VLE_TRT_ONNX_DIR_PATH,
    TEMPORAL_SIZE,
)

DEFAULT_MODEL_PATH = QWEN3VLE_TRT_PT_MODEL_PATH
DEFAULT_OUTPUT_DIR = QWEN3VLE_TRT_ONNX_DIR_PATH
OPSET              = 17


# ══════════════════════════════════════════════════════════════════════════════
# Vision Wrapper — thin shell around the HF model
# ══════════════════════════════════════════════════════════════════════════════
class VisionEncoderWrapper(torch.nn.Module):
    """
    Wraps Qwen3VLVisionModel for ONNX export.

    The HF vision model's forward() takes (hidden_states, grid_thw) and
    returns (merged_features, deepstack_features_list).

    For ONNX we bake in a fixed grid_thw and return flat outputs.
    """

    def __init__(self, visual, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual
        # Bake in the fixed grid as a buffer so it's part of the ONNX graph
        self.register_buffer("grid_thw", grid_thw)

    def forward(self, pixel_values):
        """
        pixel_values : float32 [total_patches, flatten_dim]
            From the HF processor (already CLIP-normalised).

        Returns: (deepstack_feature_0, …, vision_hidden_states)
        """
        hidden_states, deepstack_features = self.visual(
            pixel_values, grid_thw=self.grid_thw
        )
        # Return deepstack features first, then merged vision features
        return *deepstack_features, hidden_states


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════
def main(model_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    onnx_vision = os.path.join(output_dir, "Vision.onnx")

    # ── Load rotary_params to get the grid config ────────────────────────
    rp = np.load(os.path.join(output_dir, "rotary_params.npz"))
    height_factor = int(rp["height_factor"])
    width_factor  = int(rp["width_factor"])
    patch_size    = int(rp["patch_size"])
    merge_size    = int(rp["merge_size"])

    grid_h = height_factor * merge_size
    grid_w = width_factor  * merge_size
    image_height  = int(rp["image_height"])
    image_width   = int(rp["image_width"])

    temporal_patch_size_npz = int(rp["temporal_patch_size"]) if "temporal_patch_size" in rp else 2
    temporal_patches = math.ceil(TEMPORAL_SIZE / temporal_patch_size_npz)
    total_patches = temporal_patches * grid_h * grid_w

    print(f"Grid: {temporal_patches}x{grid_h}x{grid_w} = {total_patches} patches")
    print(f"Image: {image_height}x{image_width}  TEMPORAL_SIZE={TEMPORAL_SIZE}")

    # ── Load model ──────────────────────────────────────────────────────
    print("Loading model …")
    model = Qwen3VLForEmbedding.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True,
        _attn_implementation="eager",
    ).eval()

    visual = model.model.visual
    temporal_patch_size = visual.patch_embed.temporal_patch_size
    flatten_dim = 3 * temporal_patch_size * patch_size * patch_size

    deepstack_features_len = len(visual.deepstack_visual_indexes)
    print(f"  deepstack features: {deepstack_features_len}")
    print(f"  temporal_patch_size: {temporal_patch_size}")
    print(f"  flatten_dim: {flatten_dim}")

    # ── Build wrapper ───────────────────────────────────────────────────
    grid_thw = torch.tensor([[temporal_patches, grid_h, grid_w]], dtype=torch.int64)
    wrapper = VisionEncoderWrapper(visual, grid_thw)

    # ── Build dummy input ───────────────────────────────────────────────
    dummy_pixels = torch.randn(total_patches, flatten_dim, dtype=torch.float32)

    # ── Output names ────────────────────────────────────────────────────
    output_names = []
    for i in range(deepstack_features_len):
        output_names.append(f"deepstack_feature_{i}")
    output_names.append("vision_hidden_states")

    # ── Export ──────────────────────────────────────────────────────────
    print("Exporting Vision (direct HF) …")
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy_pixels,),
            onnx_vision,
            input_names=["pixel_values"],
            output_names=output_names,
            opset_version=OPSET,
            dynamo=False,
        )

    del wrapper, model
    gc.collect()
    print(f"  → {onnx_vision}")
    print("Consolidating external weights …")
    consolidate_external_data(onnx_vision)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Qwen3-VL vision encoder to ONNX (direct HF)"
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
