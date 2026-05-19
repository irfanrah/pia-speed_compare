"""Sample videos from the PIA train_val_master_v2 dataset and turn each into a
fixed-T frame tensor that the PE-L14-336 vision tower expects.

A single fixed RNG seed reproduces the same 5-train + 5-val pick across the
export, quantize, and bench stages, so the calibration set and the accuracy
eval set are byte-identical between calls.
"""

from __future__ import annotations

import glob
import json
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


_DEFAULT_DATASET_ROOT = (
    "/home/piawsa6000/nas192/Research_materials/Kur/PIA_clip_dataset/"
    "formated_dataset/PIA_clip_outdoor_v5/populated_processed_videos_384_v1"
)
_VIDEO_GLOBS = ("**/*.mp4", "**/*.avi", "**/*.mkv")

# CLIP normalization (matches the values in PE/perception_models/transforms.py
# and the calibration tensor built in claude_exp2/quantize_onnx.py).
_CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073],
                      dtype=np.float32).reshape(3, 1, 1)
_CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711],
                     dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class VideoSample:
    """One video in the eval/calibration pool, identified by its split + path."""
    split: str               # "train" or "val"
    path: str                # absolute path to the .mp4
    cls: str                 # parent dir name, e.g. "fire", "smoke", ...


def list_videos(dataset_root: str = _DEFAULT_DATASET_ROOT,
                splits: Sequence[str] = ("train", "val")) -> List[VideoSample]:
    """Walk the dataset and return every video file, keeping the split + class
    label. Handles .mp4 / .avi / .mkv (PIA's outdoor_v5 set is .avi)."""
    out: List[VideoSample] = []
    for split in splits:
        split_dir = os.path.join(dataset_root, split)
        seen = set()
        candidates = []
        for pat in _VIDEO_GLOBS:
            candidates.extend(glob.glob(os.path.join(split_dir, pat),
                                         recursive=True))
        for path in candidates:
            if path in seen:
                continue
            seen.add(path)
            cls = os.path.basename(os.path.dirname(path))
            out.append(VideoSample(split=split, path=path, cls=cls))
    return out


def sample_videos(*,
                  dataset_root: str = _DEFAULT_DATASET_ROOT,
                  n_per_split: int = 5,
                  seed: int = 20260508,
                  stratified: bool = False,
                  splits: Sequence[str] = ("train", "val")
                  ) -> List[VideoSample]:
    """Pick `n_per_split` random videos from each of the requested splits.

    `splits` can be any subset of the dataset's split subdirs — e.g.,
    ("train",) for QAT, ("val",) for CRL calibration, ("test",) for the
    final cos benchmark. This keeps the three pools physically disjoint
    when the dataset has train/val/test subdirs (which outdoor_v5 does).

    Deterministic for a given (root, n_per_split, seed, stratified, splits).
    """
    rng = random.Random(seed)
    pool = list_videos(dataset_root, splits=splits)
    by_split = {s: [] for s in splits}
    for v in pool:
        if v.split in by_split:
            by_split[v.split].append(v)

    picked: List[VideoSample] = []
    for split in splits:
        if len(by_split[split]) < n_per_split:
            raise RuntimeError(
                f"only {len(by_split[split])} videos in {split} split — "
                f"can't sample {n_per_split}")
        if not stratified:
            picked.extend(rng.sample(by_split[split], n_per_split))
            continue

        by_cls: dict = {}
        for v in by_split[split]:
            by_cls.setdefault(v.cls, []).append(v)
        classes = sorted(by_cls)
        per_class = max(1, n_per_split // max(1, len(classes)))

        chosen: List[VideoSample] = []
        for cls in classes:
            quota = min(per_class, len(by_cls[cls]))
            chosen.extend(rng.sample(by_cls[cls], quota))

        if len(chosen) < n_per_split:
            chosen_set = {v.path for v in chosen}
            leftover = [v for v in by_split[split]
                        if v.path not in chosen_set]
            need = n_per_split - len(chosen)
            chosen.extend(rng.sample(leftover, min(need, len(leftover))))
        elif len(chosen) > n_per_split:
            rng.shuffle(chosen)
            chosen = chosen[:n_per_split]
        picked.extend(chosen)
    return picked


def _read_frames_decord(path: str, n_frames: int) -> np.ndarray:
    """Return `n_frames` evenly-spaced frames from `path` as uint8 HxWxC numpy.

    Uses decord for speed; falls back to torchvision.io if decord isn't built
    against the system ffmpeg (rare in this env).
    """
    try:
        import decord
        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(path)
        total = len(vr)
        if total < 1:
            raise RuntimeError(f"empty video: {path}")
        if total <= n_frames:
            idx = list(range(total)) + [total - 1] * (n_frames - total)
        else:
            # Evenly spaced indices including endpoints (mirrors the WS=3 sampling
            # used by VideoEmbeddingExtraction for these clips).
            idx = np.linspace(0, total - 1, n_frames, dtype=np.int64).tolist()
        frames = vr.get_batch(idx).asnumpy()  # (T, H, W, 3) uint8
        return frames
    except ImportError:
        from torchvision.io import read_video
        v, _, _ = read_video(path, pts_unit="sec")
        v = v.numpy()  # (T, H, W, 3) uint8
        total = v.shape[0]
        if total < 1:
            raise RuntimeError(f"empty video: {path}")
        if total <= n_frames:
            idx = list(range(total)) + [total - 1] * (n_frames - total)
        else:
            idx = np.linspace(0, total - 1, n_frames, dtype=np.int64).tolist()
        return v[idx]


def _resize_uint8(frames_uhwc: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize to (size, size); avoids a torchvision dependency here."""
    from PIL import Image
    out = np.empty((frames_uhwc.shape[0], size, size, 3), dtype=np.uint8)
    for i in range(frames_uhwc.shape[0]):
        out[i] = np.asarray(
            Image.fromarray(frames_uhwc[i]).resize((size, size),
                                                   Image.BILINEAR),
        )
    return out


def load_video_frames(path: str, *, n_frames: int, img_size: int) -> np.ndarray:
    """Return a CLIP-normalized (T, 3, H, W) float32 tensor for one video."""
    frames = _read_frames_decord(path, n_frames)         # (T, H, W, 3) u8
    frames = _resize_uint8(frames, img_size)             # (T, H, W, 3) u8
    arr = frames.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    arr = (arr - _CLIP_MEAN) / _CLIP_STD
    return np.ascontiguousarray(arr, dtype=np.float32)


def build_clip_tensor(samples: Sequence[VideoSample], *,
                      n_frames: int, img_size: int) -> np.ndarray:
    """Stack per-video tensors into (N_videos, T, 3, H, W) float32."""
    out = np.empty((len(samples), n_frames, 3, img_size, img_size),
                   dtype=np.float32)
    for i, v in enumerate(samples):
        out[i] = load_video_frames(v.path, n_frames=n_frames, img_size=img_size)
    return out


def build_calibration_npy(samples: Sequence[VideoSample], *,
                          n_frames: int, img_size: int,
                          target_n: int, out_path: str) -> str:
    """Write the (target_n, 3, H, W) flat-frame .npy modelopt expects.

    `target_n` should be `>= max engine BT` so n_itr = target_n // BT >= 1
    for every batch in the sweep. We tile the unique frames cyclically to
    fill `target_n`.
    """
    clips = build_clip_tensor(samples, n_frames=n_frames, img_size=img_size)
    flat = clips.reshape(-1, 3, img_size, img_size)        # (N*T, 3, H, W)
    if flat.shape[0] >= target_n:
        flat = flat[:target_n]
    else:
        reps = (target_n + flat.shape[0] - 1) // flat.shape[0]
        flat = np.tile(flat, (reps, 1, 1, 1))[:target_n]
    flat = np.ascontiguousarray(flat, dtype=np.float32)
    np.save(out_path, flat)
    return out_path


def write_manifest(samples: Sequence[VideoSample], out_path: str) -> str:
    """Persist the sampled video list so every stage uses the same picks."""
    payload = {
        "samples": [
            {"split": v.split, "cls": v.cls, "path": v.path}
            for v in samples
        ],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def read_manifest(path: str) -> List[VideoSample]:
    with open(path) as f:
        payload = json.load(f)
    return [VideoSample(**s) for s in payload["samples"]]


def resolve_samples(*, manifest_path: Optional[str], dataset_root: str,
                    n_per_split: int, seed: int,
                    stratified: bool = False,
                    splits: Sequence[str] = ("train", "val")
                    ) -> List[VideoSample]:
    """Read the manifest if it exists, otherwise sample fresh and write it.

    `splits` lets callers restrict the manifest to a subset of the dataset's
    split subdirs — e.g., ("test",) for hold-out evaluation in step3, or
    ("val",) for CRL calibration in step2. Defaults to the legacy ("train",
    "val") pool so older callers keep their behavior.
    """
    if manifest_path and os.path.isfile(manifest_path):
        return read_manifest(manifest_path)
    samples = sample_videos(dataset_root=dataset_root,
                            n_per_split=n_per_split, seed=seed,
                            stratified=stratified, splits=splits)
    if manifest_path:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        write_manifest(samples, manifest_path)
    return samples
