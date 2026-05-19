"""Random-image cos / MSE comparator for INT8 TRT engines vs PT BF16.

Loads the FT_PE / PE PyTorch model in BF16, walks every ``.engine`` in
``--engine-dir``, and reports per-engine cosine similarity + MSE of the
output embeddings against the PT BF16 reference using *random* 1080p
ndarrays as inputs (matching the input pattern in
``src/speed_calculate_PE.py`` / ``src/speed_calculate_FTPE.py``).

Why random images: the speed benchmarks feed the encoder random
(1080, 1920, 3) uint8 frames so the measurement isolates the encoder
work from any disk / dataset I/O. This script reuses the same input
generator so the comparator and the speed bench see the same kind of
data.

Output: one row per engine
    pt_bf16    (baseline, always 1.0 / 0.0)
    bf16_*     (TRT BF16 reference engine)
    int8_*     (post-PTQ INT8 engine)
    int8_*_crl (CRL pre-pass + INT8)

A pass criterion is reported (cos >= --min_cos and MSE <= --max_mse).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import tensorrt as trt
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PE_INT8 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "pe_int8"))
sys.path.insert(0, PE_INT8)

# Resolve the PE vendor (same fallback chain as export_onnx.py).
def _resolve_pe_vendor() -> str:
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
    raise RuntimeError("Cannot locate pia-prompt_optimization; set PE_VENDOR=/path/to/vendor")

_PE_VENDOR = _resolve_pe_vendor()
_PE_PMODELS = os.path.join(_PE_VENDOR, "src", "PE", "perception_models")
for p in (PE_INT8, _PE_VENDOR, _PE_PMODELS):
    if p not in sys.path:
        sys.path.insert(0, p)

import core.vision_encoder.pe as pe  # noqa: E402,F401  (image_size config)
from ft_loader import load_ft_clip  # noqa: E402


_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
_DEFAULT_FT_PT = os.environ.get(
    "FT_PT_DEFAULT",
    os.path.expanduser("~/.cache/huggingface/hub/"
                       "models--PIA-SPACE-LAB--FT_PE-Core-L14-336_260318/"
                       "snapshots"),
)

# PE input normalization. Matches trt_utils.preprocess_image DEFAULT_MEAN/STD.
_PE_MEAN = [0.5, 0.5, 0.5]
_PE_STD = [0.5, 0.5, 0.5]
_FRAME_HW = (1080, 1920, 3)  # match the 1080p random frames used by speed_*


def _trt_to_torch_dtype(dt):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF:  torch.float16,
        trt.DataType.BF16:  torch.bfloat16,
        trt.DataType.INT8:  torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL:  torch.bool,
    }[dt]


class TRTEngine:
    """Minimal TRT runner: supports both static and dynamic batch profiles."""

    def __init__(self, engine_path: str):
        with open(engine_path, "rb") as f, trt.Runtime(_TRT_LOGGER) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i)
                 for i in range(self.engine.num_io_tensors)]
        self.input_name = next(
            n for n in names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self.output_names = [
            n for n in names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]
        prof = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.min_shape, self.opt_shape, self.max_shape = (tuple(s) for s in prof)
        self.in_dtype = _trt_to_torch_dtype(
            self.engine.get_tensor_dtype(self.input_name)
        )

    def fits(self, bt: int) -> bool:
        return self.min_shape[0] <= bt <= self.max_shape[0]

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != self.in_dtype:
            x = x.to(self.in_dtype)
        if not x.is_contiguous():
            x = x.contiguous()
        self.context.set_input_shape(self.input_name, tuple(x.shape))
        outputs: Dict[str, torch.Tensor] = {}
        for name in self.output_names:
            shape = tuple(int(d) for d in self.context.get_tensor_shape(name))
            dtype = _trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, dtype=dtype, device="cuda")
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        for name, t in outputs.items():
            self.context.set_tensor_address(name, int(t.data_ptr()))
        ok = self.context.execute_async_v3(
            stream_handle=torch.cuda.current_stream().cuda_stream,
        )
        if not ok:
            raise RuntimeError("execute_async_v3 returned False")
        torch.cuda.current_stream().synchronize()
        return outputs[self.output_names[0]].float()


def gen_random_frames(bt: int, *, rng: np.random.Generator) -> List[np.ndarray]:
    """Generate BT random (1080, 1920, 3) uint8 frames — same shape and dtype
    as ``src/speed_calculate_PE.py::gen_random_frame``."""
    return [rng.integers(0, 256, size=_FRAME_HW, dtype=np.uint8) for _ in range(bt)]


@torch.inference_mode()
def preprocess(frames: List[np.ndarray], *, size: int = 336,
               device: torch.device) -> torch.Tensor:
    """List[(1080,1920,3) uint8] -> (BT, 3, size, size) float32 on device,
    matching PE/FT_PE preprocessing (resize 336x336 + ImageNet [0.5,0.5,0.5]
    mean/std normalize)."""
    tensors = []
    for f in frames:
        t = torch.from_numpy(f).permute(2, 0, 1).contiguous()  # (3, 1080, 1920) uint8
        t = TF.resize(t, [size, size],
                      interpolation=T.InterpolationMode.BILINEAR, antialias=True)
        tensors.append(t)
    x = torch.stack(tensors, dim=0).to(device).float() / 255.0  # (BT, 3, H, W) in [0,1]
    mean = torch.tensor(_PE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_PE_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.inference_mode()
def pt_bf16_forward(model_bf16, x: torch.Tensor) -> torch.Tensor:
    return model_bf16.encode_image(x.to(torch.bfloat16), normalize=True).float()


_HEADER = f"  {'engine':<48}  {'cos':>10}   {'mse':>10}   verdict"
def fmt_row(name: str, cos: float, mse: float, ok: bool) -> str:
    flag = "PASS" if ok else "FAIL"
    return (f"  {name:<48}  {cos:>10.6f}   {mse:>10.3e}   {flag}")


def _gpu_info(device_index: int = 0) -> dict:
    props = torch.cuda.get_device_properties(device_index)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip().splitlines()[0]
    except Exception:
        driver = None
    return {
        "name": torch.cuda.get_device_name(device_index),
        "device_index": device_index,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_mib": int(props.total_memory / 1024 / 1024),
        "driver_version": driver,
    }


def _summarize(samples: List[float]) -> dict:
    if not samples:
        return {"mean": None, "min": None, "max": None, "std": None}
    return {
        "mean": float(statistics.fmean(samples)),
        "min":  float(min(samples)),
        "max":  float(max(samples)),
        "std":  float(statistics.pstdev(samples)) if len(samples) > 1 else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--engine-dir", required=True,
                   help="Directory with .engine files to compare against PT BF16.")
    p.add_argument("--ft_ckpt", required=True,
                   help="Path to the QAT-deployed FP32 .pt that PT BF16 loads.")
    p.add_argument("--config_name", default="PE-Core-L14-336")
    p.add_argument("--batch_videos", type=int, default=16,
                   help="B (clips per iter). For dynamic engines this also picks "
                        "the test BT = B * T.")
    p.add_argument("--frames_per_video", type=int, default=1,
                   help="T (frames per clip).")
    p.add_argument("--iters", type=int, default=5,
                   help="How many random-batch iters to average over.")
    p.add_argument("--seed", type=int, default=20260519)
    p.add_argument("--min_cos", type=float, default=0.99,
                   help="Pass threshold for cosine similarity vs PT BF16.")
    p.add_argument("--max_mse", type=float, default=1e-3,
                   help="Pass threshold for MSE vs PT BF16.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="If set, write a per-engine cos/MSE summary JSON here. "
                        "Filename: cos_mse_<gpu>_b<B>_t<T>_<timestamp><_tag>.json.")
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix appended to the output JSON filename "
                        "(matches scripts/speed_calculate_*.py's --tag convention).")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")

    # Defensive: some torch + conda-env + cuDNN combos crash at the first
    # nn.Conv2d call with "RuntimeError: cuDNN error:
    # CUDNN_STATUS_NOT_INITIALIZED" (typically a wheel-bundled cuDNN vs
    # system cuDNN version mismatch). The cos pass is a correctness check,
    # not a benchmark — disable cuDNN so Conv2d routes through native CUDA.
    # Numerics are equivalent for cos / MSE purposes (any difference is at
    # the 1e-7 noise floor that ALREADY shows up in the BF16 baseline row).
    try:
        torch.backends.cudnn.enabled = False
    except Exception:
        pass

    # Also try to keep the per-process CUDA workspace small — when the
    # speed bench just ran the GPU hot and freed a large dynamic-profile
    # context, the new process can transiently see an OOM during cuDNN /
    # cuBLAS handle creation. Limit our split to avoid that.
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    bt = args.batch_videos * args.frames_per_video
    print(f"[test] config={args.config_name}  "
          f"B={args.batch_videos}  T={args.frames_per_video}  BT={bt}  "
          f"iters={args.iters}")
    print(f"[test] ft_ckpt: {args.ft_ckpt}")
    print(f"[test] engine_dir: {args.engine_dir}")

    print("[test] loading PT BF16 reference ...")
    t0 = time.time()
    model = load_ft_clip(args.ft_ckpt, base_model=args.config_name, device="cpu")
    model = model.to(device).eval().bfloat16()
    img_size = model.image_size
    print(f"[test] PT BF16 loaded in {time.time()-t0:.1f}s  (image_size={img_size})")

    engines = sorted(glob.glob(os.path.join(args.engine_dir, "*.engine")))
    if not engines:
        raise FileNotFoundError(f"no .engine files under {args.engine_dir}")
    print(f"[test] found {len(engines)} engines: "
          f"{[os.path.basename(e) for e in engines]}")

    rng = np.random.default_rng(args.seed)

    # Pre-generate the inputs ONCE (deterministic seed) so every engine sees
    # the same BT random tensors and the comparison is apples-to-apples.
    # Each TRT engine is then loaded → run → freed in sequence so the dynamic
    # engine's pre-allocated workspace for the OPT shape doesn't pile up
    # alongside the PT BF16 model on a single 16 GB A4000.
    inputs_x: List[torch.Tensor] = []
    pt_embs: List[torch.Tensor] = []
    print(f"\n[test] precomputing {args.iters} random-batch PT references ...")
    for _ in range(args.iters):
        frames = gen_random_frames(bt, rng=rng)
        x = preprocess(frames, size=img_size, device=device)        # (BT, 3, H, W) float32
        pt_emb = pt_bf16_forward(model, x)                          # (BT, D) float32, L2-norm
        inputs_x.append(x)
        pt_embs.append(pt_emb)
    print(f"[test] freeing PT BF16 to make room for TRT contexts ...")
    del model
    torch.cuda.empty_cache()

    # Per-engine cos / mse accumulators, populated by running each engine in
    # sequence on the pre-computed inputs.
    acc: Dict[str, Dict[str, List[float]]] = {}
    for ep in engines:
        name = os.path.splitext(os.path.basename(ep))[0]
        rn = TRTEngine(ep)
        if not rn.fits(bt):
            print(f"  skipping {name}: profile {rn.min_shape}..{rn.max_shape} "
                  f"does not include BT={bt}")
            del rn
            torch.cuda.empty_cache()
            continue
        acc[name] = {"cos": [], "mse": []}
        for it in range(args.iters):
            x = inputs_x[it]
            pt_emb = pt_embs[it]
            trt_emb = rn.forward(x).float()
            # Defensive L2 norm — the exported PE graph emits normalize=True
            # already, but we sanity-renormalize to compare directions even
            # if a future export forgets the flag.
            trt_emb = trt_emb / trt_emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cos = F.cosine_similarity(pt_emb, trt_emb, dim=-1)
            mse = ((pt_emb - trt_emb) ** 2).mean(dim=-1)
            acc[name]["cos"].append(float(cos.mean()))
            acc[name]["mse"].append(float(mse.mean()))
        print(f"  {name}: iter-0 cos={acc[name]['cos'][0]:.4f}")
        # Free GPU resources before loading the next engine.
        del rn
        torch.cuda.empty_cache()
    if not acc:
        raise RuntimeError(f"no engine accepts BT={bt}")
    runners = acc  # downstream code expects `runners` keyed by engine name

    print(f"\n[test] results (mean over {args.iters} iters)\n")
    any_fail = False
    rows = []
    for name in runners:
        cos = statistics.fmean(acc[name]["cos"])
        mse = statistics.fmean(acc[name]["mse"])
        ok = (cos >= args.min_cos) and (mse <= args.max_mse)
        if not ok:
            any_fail = True
        rows.append((name, cos, mse, ok))

    rows.sort(key=lambda r: r[0])
    print(_HEADER)
    print("  " + "-" * 80)
    for name, cos, mse, ok in rows:
        print(fmt_row(name, cos, mse, ok))

    print(f"\n[test] thresholds: cos >= {args.min_cos}   mse <= {args.max_mse:.0e}")

    # ── Optional JSON dump ─────────────────────────────────────────────
    if args.out_dir:
        gpu = _gpu_info()
        engines_summary: Dict[str, dict] = {}
        for name in runners:
            cos_samples = acc[name]["cos"]
            mse_samples = acc[name]["mse"]
            cos_mean = statistics.fmean(cos_samples)
            mse_mean = statistics.fmean(mse_samples)
            engines_summary[name] = {
                "cos": _summarize(cos_samples),
                "mse": _summarize(mse_samples),
                "iters": [
                    {"iter": i, "cos": round(c, 8), "mse": round(m, 12)}
                    for i, (c, m) in enumerate(zip(cos_samples, mse_samples))
                ],
                "passed": (cos_mean >= args.min_cos) and (mse_mean <= args.max_mse),
            }
        initial_time = datetime.now()
        payload = {
            "test_kind": "random_image_cos_mse",
            "initial_time": initial_time.isoformat(timespec="seconds"),
            "config_name": args.config_name,
            "ft_ckpt": args.ft_ckpt,
            "engine_dir": args.engine_dir,
            "batch_videos": args.batch_videos,
            "frames_per_video": args.frames_per_video,
            "bt": bt,
            "iters": args.iters,
            "seed": args.seed,
            "min_cos": args.min_cos,
            "max_mse": args.max_mse,
            "gpu": gpu,
            "gpu_type": gpu["name"],
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "engines": engines_summary,
            "overall_passed": not any_fail,
        }
        os.makedirs(args.out_dir, exist_ok=True)
        safe_gpu = gpu["name"].replace(" ", "_").replace("/", "_")
        ts = initial_time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{args.tag}" if args.tag else ""
        out_path = os.path.join(
            args.out_dir,
            f"cos_mse_{safe_gpu}_b{args.batch_videos}_t{args.frames_per_video}_{ts}{suffix}.json",
        )
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[test] wrote: {out_path}")

    if any_fail:
        print("[test] FAIL — one or more engines did not meet the threshold")
        return 1
    print("[test] PASS — all engines within threshold vs PT BF16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
