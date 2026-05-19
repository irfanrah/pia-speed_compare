"""Benchmark BF16 .pt vs TRT BF16 vs TRT INT8 on the FT_PE-Core-L14-336_260318
vision tower with real video clips.

For every engine in `engines/` we group its trace batch BT into video clips
of length T=3, so the engine call processes B = BT/T videos per invocation.
For each batch we report:
  - cos / MSE vs the FT BF16 PyTorch reference (per-frame and per-video)
  - median latency (CUDA events, `--iters` after `--warmup`)
  - throughput (videos/s and frames/s)
  - speedup vs the TRT BF16 baseline at the same batch
  - speedup vs the PT BF16 baseline at the same batch

The reference is the FT .pt loaded via `pe.CLIP.from_config + load_ckpt`,
moved to CUDA + BF16. PyTorch fp32 is intentionally not used — the FT
checkpoint shipped in fp32 but PT BF16 is the production baseline we care
about beating.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import tensorrt as trt

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
from ft_loader import load_ft_clip  # noqa: E402
from video_utils import build_clip_tensor, resolve_samples  # noqa: E402


_TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
_TARGET_SPEEDUP = 1.5

_DEFAULT_FT_PT = (
    "/home/piawsa6000/nas192/Research_materials/Kur/Blue-VLMTF-PVLM/code/"
    "Research-AI-mono/PE_FineTuning/assets/models/"
    "FT_PE-Core-L14-336_260318/FT_PE-Core-L14-336_260318.pt"
)


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
        self.output_names = [n for n in names
                             if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        prof = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.min_shape, self.opt_shape, self.max_shape = (tuple(s) for s in prof)
        self.in_dtype = _trt_to_torch_dtype(
            self.engine.get_tensor_dtype(self.input_name)
        )

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


def parse_args():
    here = os.path.dirname(__file__)
    p = argparse.ArgumentParser()
    p.add_argument("--config_name", default="PE-Core-L14-336")
    p.add_argument("--ft_ckpt", default=_DEFAULT_FT_PT)
    p.add_argument("--frames_per_video", type=int, default=3)
    p.add_argument("--engine_dir", default=os.path.join(here, "engines"))
    p.add_argument("--manifest",
                   default=os.path.join(here, "calib", "manifest.json"))
    p.add_argument("--dataset_root",
                   default="/home/piawsa6000/nas192/Research_materials/Kur/"
                           "PIA_clip_dataset/train_val_master_v2")
    p.add_argument("--n_per_split", type=int, default=5)
    p.add_argument("--seed", type=int, default=20260508)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--min_cos", type=float, default=0.999)
    p.add_argument("--max_mse", type=float, default=1e-5)
    p.add_argument("--out_dir", default=os.path.join(here, "results"))
    p.add_argument("--img_size", type=int, default=336)
    p.add_argument("--stratified", action="store_true")
    p.add_argument("--split", default=None,
                   help="Restrict the eval video pool to a specific dataset "
                        "split (e.g. 'test'). Default: legacy train+val pool. "
                        "Use 'test' for the data-hygiene pipeline so the "
                        "final cos benchmark is on videos the model never "
                        "saw during QAT or PTQ calibration.")
    return p.parse_args()


@torch.inference_mode()
def time_callable(fn, *, n_warmup: int, n_iters: int) -> float:
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
    for i in range(n_iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    ms = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return ms[len(ms) // 2]


def infer_precision(name: str) -> str:
    n = name.lower()
    if "int8" in n:
        return "int8"
    if "bf16" in n:
        return "bf16"
    if "fp16" in n:
        return "fp16"
    return "unknown"


def diagnose(int8_speedup: float) -> str:
    if int8_speedup >= _TARGET_SPEEDUP:
        return f"INT8 >= {_TARGET_SPEEDUP}x: validate accuracy on real data, then deploy"
    if int8_speedup > 1.05:
        return ("INT8 only marginally faster: re-run surgery.py to clear residual "
                "Q/DQ around layout-only ops")
    if int8_speedup >= 0.95:
        return ("INT8 ~= BF16: confirm --high_precision_dtype=fp16 was passed; "
                "without it, non-quantized regions fall back to FP32")
    return ("INT8 slower than BF16: GPU lacks fast INT8 tensor cores or many "
            "tensors fell back to FP32 -- check build log")


def _build_eval_clips(samples, *, B: int, T: int, img_size: int,
                      device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Return (clips_BTHW_float32, clips_BTCHW_for_PT, video_keys).

    clips_BTHW: (B*T, 3, H, W) float32 on `device` — TRT engine input.
    clips_BTCHW_for_PT: (B, T, 3, H, W) float32 on `device` — PT reference.
    video_keys: per-position video labels (split/cls/basename) aligned to B.
    """
    needed = B
    n_unique = len(samples)
    keys = []
    picks = []
    for i in range(needed):
        v = samples[i % n_unique]
        keys.append(f"{v.split}/{v.cls}/{os.path.basename(v.path)}")
        picks.append(v)
    clip_np = build_clip_tensor(picks, n_frames=T, img_size=img_size)  # (B, T, 3, H, W)
    pt_tensor = torch.from_numpy(clip_np).to(device)
    flat = pt_tensor.reshape(B * T, 3, img_size, img_size).contiguous()
    return flat, pt_tensor, keys


@torch.inference_mode()
def _pt_reference(model_bf16, frames_flat: torch.Tensor) -> torch.Tensor:
    """Run the PT BF16 reference on (BT, C, H, W) and return float32 (BT, D)."""
    out = model_bf16.encode_image(frames_flat.to(torch.bfloat16),
                                  normalize=True)
    return out.float()


def _video_pool(per_frame: torch.Tensor, *, B: int, T: int) -> torch.Tensor:
    """(B*T, D) -> (B, D) by L2-normalize-then-mean (matches encode_video)."""
    return per_frame.reshape(B, T, -1).mean(dim=1)


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda")

    print(f"[bench] loading FT BF16 reference: {args.config_name}  "
          f"<- {args.ft_ckpt}")
    model_bf16 = load_ft_clip(args.ft_ckpt, base_model=args.config_name,
                              device="cpu")
    model_bf16 = model_bf16.to(device).eval().bfloat16()

    splits_kw = {"splits": (args.split,)} if args.split else {}
    samples = resolve_samples(manifest_path=args.manifest,
                              dataset_root=args.dataset_root,
                              n_per_split=args.n_per_split, seed=args.seed,
                              stratified=args.stratified,
                              **splits_kw)
    print(f"[bench] eval set: {len(samples)} videos "
          f"(seed={args.seed}, T={args.frames_per_video})")
    for v in samples:
        print(f"   {v.split:<5}  {v.cls:<18}  {os.path.basename(v.path)}")

    engines: Dict[str, TRTEngine] = {}
    for engine_path in sorted(glob.glob(os.path.join(args.engine_dir, "*.engine"))):
        try:
            eng = TRTEngine(engine_path)
        except Exception as e:
            print(f"  [skip] {engine_path}: {e}")
            continue
        size = os.path.getsize(engine_path) / 1024**2
        print(f"  [load] {os.path.basename(engine_path):30s}  "
              f"opt_BT={eng.opt_shape[0]}  size={size:.1f} MiB")
        engines[os.path.basename(engine_path)] = eng

    if not engines:
        print("[bench] no engines -- nothing to do.")
        return 1

    by_bt: Dict[int, List[str]] = {}
    for name, eng in engines.items():
        by_bt.setdefault(eng.opt_shape[0], []).append(name)

    payload: Dict = {
        "config_name": args.config_name,
        "ft_ckpt": args.ft_ckpt,
        "frames_per_video": args.frames_per_video,
        "iters": args.iters,
        "warmup": args.warmup,
        "thresholds": {
            "min_cos_video": args.min_cos,
            "max_mse_video": args.max_mse,
            "target_speedup_int8_vs_trt_bf16": _TARGET_SPEEDUP,
        },
        "eval_videos": [
            {"split": v.split, "cls": v.cls, "name": os.path.basename(v.path)}
            for v in samples
        ],
        "batches": {},
    }

    T = args.frames_per_video
    for bt in sorted(by_bt):
        if bt % T != 0:
            print(f"[bench] WARN: engine BT={bt} not divisible by T={T}; "
                  f"skipping")
            continue
        B = bt // T
        print(f"\n===== B={B}  T={T}  (engine BT={bt}) =====")
        flat_fp32, _clip_BTCHW, video_keys = _build_eval_clips(
            samples, B=B, T=T, img_size=args.img_size, device=device,
        )

        # PT BF16 reference (baseline against which TRT cos/MSE are measured).
        with torch.inference_mode():
            ref_per_frame = _pt_reference(model_bf16, flat_fp32)   # (BT, D)
        ref_per_video = _video_pool(ref_per_frame, B=B, T=T)        # (B, D)

        engine_results: Dict[str, dict] = {}

        # PT BF16 row uses ref vs ref so cos/MSE are 1.0/0; its purpose is
        # the speed baseline.
        pt_ms = time_callable(
            lambda: model_bf16.encode_image(flat_fp32.to(torch.bfloat16),
                                            normalize=True),
            n_warmup=args.warmup, n_iters=args.iters,
        )
        engine_results["pt_bf16"] = {
            "precision": "bf16",
            "kind": "pytorch_ft",
            "median_ms": pt_ms,
            "videos_per_s": B * 1000.0 / pt_ms,
            "frames_per_s": bt * 1000.0 / pt_ms,
            "size_MiB": None,
            "cos_frame_mean": 1.0,
            "mse_frame_mean": 0.0,
            "cos_video_mean": 1.0,
            "mse_video_mean": 0.0,
        }
        print(f"  pt_bf16                         "
              f"ms={pt_ms:7.2f}  "
              f"({B*1000.0/pt_ms:6.1f} vid/s, {bt*1000.0/pt_ms:6.1f} img/s)")

        for name in sorted(by_bt[bt]):
            eng = engines[name]
            try:
                trt_per_frame = eng.forward(flat_fp32)               # (BT, D)
                trt_per_video = _video_pool(trt_per_frame, B=B, T=T)  # (B, D)
                cos_f = F.cosine_similarity(ref_per_frame, trt_per_frame, dim=-1)
                mse_f = ((ref_per_frame - trt_per_frame) ** 2).mean(dim=-1)
                cos_v = F.cosine_similarity(ref_per_video, trt_per_video, dim=-1)
                mse_v = ((ref_per_video - trt_per_video) ** 2).mean(dim=-1)
            except Exception as e:
                engine_results[name] = {"error": str(e)[:200]}
                print(f"  {name:30s}  RUN-FAILED: {e}")
                continue
            ms = time_callable(lambda: eng.forward(flat_fp32),
                               n_warmup=args.warmup, n_iters=args.iters)
            engine_results[name] = {
                "precision": infer_precision(name),
                "median_ms": ms,
                "videos_per_s": B * 1000.0 / ms,
                "frames_per_s": bt * 1000.0 / ms,
                "size_MiB": os.path.getsize(os.path.join(args.engine_dir, name)) / 1024**2,
                "cos_frame_mean": float(cos_f.mean()),
                "cos_frame_min":  float(cos_f.min()),
                "mse_frame_mean": float(mse_f.mean()),
                "mse_frame_max":  float(mse_f.max()),
                "cos_video_mean": float(cos_v.mean()),
                "cos_video_min":  float(cos_v.min()),
                "mse_video_mean": float(mse_v.mean()),
                "mse_video_max":  float(mse_v.max()),
            }
            r = engine_results[name]
            print(f"  {name:30s}  ms={ms:7.2f}  "
                  f"cos_v={r['cos_video_mean']:.6f}  mse_v={r['mse_video_mean']:.3e}  "
                  f"({B*1000.0/ms:6.1f} vid/s, {bt*1000.0/ms:6.1f} img/s)")

        bf16_entry = next((r for n, r in engine_results.items()
                           if r.get("precision") == "bf16"
                           and r.get("kind") != "pytorch_ft"
                           and "error" not in r),
                          None)
        pt_bf16_entry = engine_results.get("pt_bf16")
        if bf16_entry:
            for r in engine_results.values():
                if "error" in r:
                    continue
                r["speedup_vs_trt_bf16"] = bf16_entry["median_ms"] / r["median_ms"]
        if pt_bf16_entry:
            for r in engine_results.values():
                if "error" in r:
                    continue
                r["speedup_vs_pt_bf16"] = pt_bf16_entry["median_ms"] / r["median_ms"]

        verdict = None
        int8_entry = next((r for r in engine_results.values()
                           if r.get("precision") == "int8" and "error" not in r),
                          None)
        if bf16_entry and int8_entry:
            sp = int8_entry["speedup_vs_trt_bf16"]
            acc_ok = (int8_entry["cos_video_mean"] >= args.min_cos
                      and int8_entry["mse_video_mean"] <= args.max_mse)
            verdict = {
                "int8_vs_trt_bf16_speedup": sp,
                "int8_video_accuracy_passes": acc_ok,
                "diagnosis": diagnose(sp),
                "deploy_int8": acc_ok and sp >= _TARGET_SPEEDUP,
            }
            print(f"  [verdict] int8/bf16 = {sp:.2f}x  "
                  f"video_acc={'PASS' if acc_ok else 'FAIL'}  "
                  f"-> {verdict['diagnosis']}")

        payload["batches"][str(B)] = {
            "engine_BT": bt,
            "T": T,
            "video_keys": video_keys,
            "engines": engine_results,
            "verdict": verdict,
        }

    json_path = os.path.join(args.out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n[bench] wrote {json_path}")

    md_path = os.path.join(args.out_dir, "results.md")
    with open(md_path, "w") as f:
        f.write("# claude_exp3_phase2 -- FT BF16 / TRT BF16 / TRT INT8 video sweep "
                "(patch-embed exclusion + simplify + optional autotune)\n\n")
        f.write(f"- model: `{args.config_name}` + FT weights from `{os.path.basename(args.ft_ckpt)}`\n")
        f.write(f"- video clips: {args.n_per_split} train + {args.n_per_split} val "
                f"(seed={args.seed}); T={T}\n")
        f.write(f"- engine BT = B*T; iters: {args.iters} after {args.warmup} warmup, CUDA events\n")
        f.write(f"- video accuracy gate: cos >= {args.min_cos}, mse <= {args.max_mse:.1e}\n")
        f.write(f"- INT8 deployment target: >= {_TARGET_SPEEDUP}x vs TRT BF16\n\n")
        f.write("| Mode | B | BT | Precision | Size | cos (frame) | MSE (frame) | "
                "cos (video) | MSE (video) | ms | vid/s | img/s | "
                "vs TRT BF16 | vs PT BF16 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for B in sorted(payload["batches"], key=int):
            entry = payload["batches"][B]
            bt = entry["engine_BT"]
            for name, r in entry["engines"].items():
                if "error" in r:
                    f.write(f"| {name} | {B} | {bt} | err | - | - | - | - | - | - | - | - | - | "
                            f"_{r['error']}_ |\n")
                    continue
                sz = f"{r['size_MiB']:.1f} MiB" if r.get("size_MiB") else "-"
                sp_bf = r.get("speedup_vs_trt_bf16")
                sp_pt = r.get("speedup_vs_pt_bf16")
                sp_bf_s = f"{sp_bf:.2f}x" if sp_bf is not None else "-"
                sp_pt_s = f"{sp_pt:.2f}x" if sp_pt is not None else "-"
                f.write(
                    f"| {name} | {B} | {bt} | {r['precision']} | {sz} | "
                    f"{r['cos_frame_mean']:.6f} | {r['mse_frame_mean']:.2e} | "
                    f"{r['cos_video_mean']:.6f} | {r['mse_video_mean']:.2e} | "
                    f"{r['median_ms']:.2f} | {r['videos_per_s']:.1f} | "
                    f"{r['frames_per_s']:.1f} | {sp_bf_s} | {sp_pt_s} |\n"
                )
            v = entry["verdict"]
            if v:
                f.write(f"\n**B={B} verdict:** int8/bf16 = "
                        f"{v['int8_vs_trt_bf16_speedup']:.2f}x; "
                        f"video accuracy {'PASS' if v['int8_video_accuracy_passes'] else 'FAIL'}; "
                        f"deploy={v['deploy_int8']}; _{v['diagnosis']}_\n\n")
    print(f"[bench] wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
