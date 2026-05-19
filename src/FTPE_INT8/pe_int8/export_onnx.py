"""Export the FT-PE-Core-L14-336_260318 vision tower to ONNX with a fixed
trace batch of B*T frames.

Mirrors claude_exp2/export_onnx.py with two changes:

1. The PE checkpoint is the public pretrained one *plus* a `load_ckpt(...)`
   call that overlays the FT_PE-Core-L14-336_260318 weights downloaded
   from `huggingface.co/PIA-SPACE-LAB/FT_PE-Core-L14-336_260318`. Architecture
   is unchanged, so the ONNX graph and Reshape volumes are identical to
   claude_exp2's at the same trace batch — only the weight tensors differ.

2. Trace batch is `--batch_videos B * --frames_per_video T`. PE's
   `encode_video` flattens (B, T, C, H, W) -> (B*T, C, H, W) before calling
   `encode_image`, so the ONNX we export is the per-frame image tower at
   B*T. Mean-pooling over T is done host-side after inference.

Trace happens on GPU (claude_exp2 fix); host RAM doesn't need to hold
ViT-L activations for B*T = up to 96.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

# --- standalone-bundle path setup -------------------------------------------
# Locate the PE vendor (pia-prompt_optimization). The bundle is self-contained
# except for the PE vendor and (for FT) the LoRA-PEFT wheel.
def _resolve_pe_vendor():
    cand = os.environ.get("PE_VENDOR")
    if cand and os.path.isdir(os.path.join(cand, "src", "PE", "perception_models")):
        return cand
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (
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
import core.vision_encoder.pe as pe  # noqa: E402,F401  (loaded for image_size config)
from ft_loader import load_ft_clip  # noqa: E402


_DEFAULT_FT_PT = (
    "/home/piawsa6000/nas192/Research_materials/Kur/Blue-VLMTF-PVLM/code/"
    "Research-AI-mono/PE_FineTuning/assets/models/"
    "FT_PE-Core-L14-336_260318/FT_PE-Core-L14-336_260318.pt"
)


class VisionEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames: torch.Tensor):
        # Input is already flat (BT, C, H, W); encode each frame with
        # L2-normalize matching PE.encode_image(..., normalize=True).
        return self.model.encode_image(frames, normalize=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", default="PE-Core-L14-336")
    p.add_argument("--ft_ckpt", default=_DEFAULT_FT_PT,
                   help="Path to the FT_PE-Core-L14-336_260318.pt checkpoint")
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "onnx"))
    p.add_argument("--out_name", default="fp32_b1t3.onnx")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--batch_videos", type=int, default=1,
                   help="B in the (B, T, C, H, W) video clip layout")
    p.add_argument("--frames_per_video", type=int, default=3,
                   help="T in the (B, T, C, H, W) video clip layout")
    return p.parse_args()


def consolidate_external(onnx_path: str) -> None:
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    base_name = os.path.splitext(os.path.basename(onnx_path))[0]
    data_fname = f"{base_name}.data"

    model = onnx.load(onnx_path, load_external_data=True)
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_fname,
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.save_model(
        model, onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_fname,
        size_threshold=0,
    )


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.isfile(args.ft_ckpt):
        raise FileNotFoundError(
            f"FT checkpoint not found: {args.ft_ckpt}. "
            f"Download from huggingface.co/PIA-SPACE-LAB/"
            f"FT_PE-Core-L14-336_260318 and pass --ft_ckpt /path/to/.pt"
        )

    bt = args.batch_videos * args.frames_per_video
    print(f"[export] config={args.config_name}  (split-qkv + LoRA merged)")
    print(f"[export] B={args.batch_videos}  T={args.frames_per_video}  "
          f"trace BT={bt}")
    print(f"[export] FT checkpoint: {args.ft_ckpt}")

    model = load_ft_clip(args.ft_ckpt, base_model=args.config_name, device="cpu")
    image_size = model.image_size
    print(f"[export] image_size={image_size}")
    vmodel = VisionEncoderWrapper(model).eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        vmodel = vmodel.to(device)

    out_path = os.path.join(args.out_dir, args.out_name)
    dummy = torch.randn(bt, 3, image_size, image_size,
                        dtype=torch.float32, device=device)

    print(f"[export] writing ONNX -> {out_path}  (trace device={device})")
    with torch.no_grad():
        # Force the legacy TorchScript exporter. torch 2.9+ defaults the
        # `dynamo=` kwarg to True; on this model the dynamo path fails with
        # "Unhandled FakeTensor Device Propagation for aten.mul.Tensor",
        # whereas the legacy path traces cleanly.
        torch.onnx.export(
            vmodel, dummy, out_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    del vmodel, model, dummy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"[export] consolidating external weights ...")
    consolidate_external(out_path)

    onnx_size = os.path.getsize(out_path)
    base = os.path.splitext(args.out_name)[0]
    data_path = os.path.join(args.out_dir, f"{base}.data")
    data_size = os.path.getsize(data_path) if os.path.isfile(data_path) else 0
    print(f"[export] DONE")
    print(f"  graph file: {out_path}  ({onnx_size / 1024:.1f} KiB)")
    if data_size:
        print(f"  data  file: {data_path}  ({data_size / 1024**2:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
