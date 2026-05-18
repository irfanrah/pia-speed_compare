"""
Qwen3-VL-Embedding TensorRT model.

Replaces PiaTorchModel(Qwen3VLEmbedding) with a TensorRT inference pipeline.
Called by service.py the same way as the torch model:

    model(video=batched_videos, text=None)  →  torch.Tensor [B, D]

Internally the pipeline processes one video (1 or more frames) at a time
through 6 steps:

    1. Tokenise        — HF Qwen3VLProcessor turns PIL frame(s) into token IDs +
                         pixel patches. Uses the VIDEO path (same as the torch
                         model) so that <|video_pad|> tokens are produced.
    2. Token embed     — GPU lookup in embed_weight (loaded from rotary_params.npz).
    3. Vision encode   — Vision.engine turns pixel patches → vision features +
                         deepstack features (stays on GPU).
    4. Splice          — Replace <|video_pad|> token embeddings with vision features.
                         Pad deepstack features to the full sequence length.
    5. Rotary + mask   — Compute mRoPE cos/sin + causal attention mask on GPU.
    6. Transformer     — Transformer.engine processes the merged sequence →
                         last_hidden_state. Pool the last non-pad token → embedding.

Steps 2-6 are GPU-resident: the vision and transformer engines exchange torch
CUDA tensors directly, avoiding per-frame GPU→CPU→GPU round-trips.
"""

import logging
import os
import unicodedata
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .engine import VisionEngine, TransformerEngine
from .utils import build_causal_mask, compute_position_ids, compute_rotary

logger = logging.getLogger(__name__)


class Qwen3VLETrtModel:
    """
    Parameters
    ----------
    trt_dir         : directory with Vision.engine, Transformer.engine, rotary_params.npz
    processor_path  : HF model dir with tokenizer / processor config files
    device          : torch device for returned tensors
    max_length      : tokenizer truncation length
    """

    def __init__(
        self,
        trt_dir: str,
        processor_path: str,
        device: str = "cuda",
        max_length: int = 8192,
        default_instruction: str = "Represent the user's input.",
    ):
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        self.device = device
        self.max_length = max_length
        self.default_instruction = default_instruction

        # ── Load rotary_params.npz ────────────────────────────────────────
        # Contains: token embedding weight, RoPE inverse frequencies,
        # and the image geometry constants the ONNX was exported with.
        # Tensors used in the hot path are uploaded to `device` once at init
        # so the per-frame pipeline stays GPU-resident.
        rp = np.load(os.path.join(trt_dir, "rotary_params.npz"))
        self.inv_freq = torch.from_numpy(rp["inv_freq"]).to(device)
        self.mrope_section = rp["mrope_section"]
        self.embed_weight = torch.from_numpy(rp["embed_weight"]).to(device)  # [vocab, hidden]
        self.image_height = int(rp["image_height"])
        self.image_width = int(rp["image_width"])
        self.height_factor = int(rp["height_factor"])
        self.width_factor = int(rp["width_factor"])
        self.hidden_size = int(rp["hidden_size"])
        self.head_dim = int(rp["head_dim"])

        self.temporal_patches = int(rp["temporal_patches"]) if "temporal_patches" in rp else 1
        self.vision_embed_size = self.temporal_patches * self.height_factor * self.width_factor

        # ── Load TRT engines ──────────────────────────────────────────────
        self.vision_engine = VisionEngine(os.path.join(trt_dir, "Vision.engine"))
        self.transformer_engine = TransformerEngine(os.path.join(trt_dir, "Transformer.engine"))
        self.deepstack_count = self.transformer_engine.deepstack_count

        # ── Load processor (tokenizer + image processor) ──────────────
        self.processor = Qwen3VLProcessor.from_pretrained(
            processor_path, padding_side="right"
        )
        self._video_pad_id = self.processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")

    # =====================================================================
    # Step 1: Tokenise — turn a PIL image into token IDs + pixel patches
    # =====================================================================

    def _format_conversation(
        self,
        image: Optional[Union[Image.Image, str]] = None,
        frames: Optional[List[Image.Image]] = None,
        text: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        """
        Build a HF chat-template conversation with system instruction.

        Accepts either a single ``image`` (wrapped as a 1-frame video) or a
        list of ``frames`` (multi-frame video).  Both paths produce
        ``<|video_pad|>`` tokens — matching the torch model's code path.
        """
        inst = (instruction or self.default_instruction).strip()
        if inst and not unicodedata.category(inst[-1]).startswith("P"):
            inst += "."

        content: List[Dict[str, Any]] = []
        if frames is not None:
            content.append({"type": "video", "video": frames})
        elif image is not None:
            if isinstance(image, str):
                pil = Image.open(image).convert("RGB")
            else:
                pil = image
            content.append({"type": "video", "video": [pil]})
        if text is not None:
            content.append({"type": "text", "text": text})
        if not content:
            content.append({"type": "text", "text": "NULL"})

        return [
            {"role": "system", "content": [{"type": "text", "text": inst}]},
            {"role": "user", "content": content},
        ]

    def _tokenise(self, conversation: List[Dict]) -> Dict[str, np.ndarray]:
        """Run HF processor → input_ids, attention_mask, pixel_values (via video path)."""
        from qwen_vl_utils.vision_process import process_vision_info

        template = self.processor.apply_chat_template(
            [conversation], add_generation_prompt=True, tokenize=False
        )

        try:
            images, video_inputs, video_kwargs = process_vision_info(
                [conversation],
                image_patch_size=16,
                return_video_metadata=True,
                return_video_kwargs=True,
            )
        except (ValueError, RuntimeError) as e:
            # Input-shaped failures only (corrupt image, format mismatch).
            # Anything else (ImportError, AttributeError, ...) should crash hard.
            # Fallback path runs the transformer WITHOUT vision features, so the
            # resulting embedding is text-only — loud warning so callers can spot
            # the degraded output instead of silently trusting it.
            logger.warning(
                "process_vision_info failed (%s: %s); falling back to text-only "
                "tokenisation — transformer will run without vision features. "
                "Check input frames for corruption or format mismatches.",
                type(e).__name__, e,
            )
            images, video_inputs, video_kwargs = None, None, {"do_sample_frames": False}

        if video_inputs is not None:
            videos, video_metadata = zip(*video_inputs)
            videos, video_metadata = list(videos), list(video_metadata)
        else:
            videos, video_metadata = None, None

        inputs = self.processor(
            text=template, images=images, videos=videos,
            video_metadata=video_metadata, truncation=True,
            max_length=self.max_length, padding=True,
            do_resize=False, return_tensors="pt", **video_kwargs,
        )

        result: Dict[str, np.ndarray] = {
            "input_ids": inputs["input_ids"].numpy().astype(np.int64),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
        }
        # Video path produces pixel_values_videos (same pixel data as pixel_values)
        if "pixel_values_videos" in inputs:
            result["pixel_values"] = inputs["pixel_values_videos"].numpy().astype(np.float32)
        elif "pixel_values" in inputs:
            result["pixel_values"] = inputs["pixel_values"].numpy().astype(np.float32)
        return result

    # =====================================================================
    # Steps 2-6: Forward pass
    # =====================================================================

    def _forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        pixel_values: Optional[np.ndarray],
    ) -> torch.Tensor:
        """
        Full forward pass through the TRT pipeline.

        All tensor ops run on `self.device`; only the HF-tokenizer outputs
        (input_ids, attention_mask, pixel_values) come in as numpy from CPU.

        Returns: last_hidden_state torch.Tensor [1, seq_len, hidden_size] on self.device
        """
        # ── Step 2: Token embeddings (GPU lookup) ────────────────────────
        input_ids_t = torch.from_numpy(input_ids).to(self.device)
        text_embed = self.embed_weight[input_ids_t].to(torch.float32)  # [1, seq, hidden]

        vision_embed: Optional[torch.Tensor] = None
        deepstack_features: List[torch.Tensor] = []
        image_start_pos: Optional[int] = None

        # ── Step 3: Vision encode ────────────────────────────────────────
        if pixel_values is not None:
            pixel_values_t = torch.from_numpy(pixel_values).to(self.device, non_blocking=True)
            vision_outs = self.vision_engine(pixel_values_t)
            vision_names = list(vision_outs.keys())
            ds_names = sorted(
                [n for n in vision_names if n.startswith("deepstack_feature_")],
                key=lambda s: int(s.split("_")[-1]),
            )
            vh_names = [n for n in vision_names if n not in ds_names]
            assert len(vh_names) == 1, f"Expected 1 vision output, got {vh_names}"

            vision_embed = vision_outs[vh_names[0]]
            deepstack_features = [vision_outs[n] for n in ds_names]

            # Ensure 3D: [1, num_patches, hidden]
            if vision_embed.ndim == 2:
                vision_embed = vision_embed.unsqueeze(0)
            deepstack_features = [
                ds.unsqueeze(0) if ds.ndim == 2 else ds for ds in deepstack_features
            ]
            image_start_pos = self._find_video_token_start(input_ids[0])

        # ── Step 4: Splice vision into text embeddings ───────────────────
        if vision_embed is not None and image_start_pos is not None:
            img_end = image_start_pos + self.vision_embed_size
            text_embed = torch.cat([
                text_embed[:, :image_start_pos, :],
                vision_embed.to(torch.float32),
                text_embed[:, img_end:, :],
            ], dim=1)

            # Pad deepstack features with zeros outside the image region
            pre_zeros = torch.zeros(
                (1, image_start_pos, self.hidden_size),
                dtype=torch.float32, device=self.device,
            )
            post_len = text_embed.shape[1] - image_start_pos - self.vision_embed_size
            post_zeros = torch.zeros(
                (1, post_len, self.hidden_size),
                dtype=torch.float32, device=self.device,
            )
            deepstack_features = [
                torch.cat([pre_zeros, ds.to(torch.float32), post_zeros], dim=1)
                for ds in deepstack_features
            ]

        # ── Step 5: Rotary embeddings + causal mask ──────────────────────
        total_seq = text_embed.shape[1]
        position_ids = compute_position_ids(
            total_seq, image_start_pos, self.vision_embed_size,
            self.height_factor, self.width_factor, self.temporal_patches,
            device=self.device,
        )
        rotary_cos, rotary_sin = compute_rotary(
            position_ids, self.inv_freq, self.mrope_section, self.head_dim,
        )
        attn_mask = build_causal_mask(total_seq, device=self.device)

        # ── Step 6: Transformer engine ───────────────────────────────────
        feeds: Dict[str, torch.Tensor] = {
            "hidden_states": text_embed,
            "rotary_cos": rotary_cos,
            "rotary_sin": rotary_sin,
            "attention_mask": attn_mask,
        }
        if deepstack_features:
            for i, ds in enumerate(deepstack_features):
                feeds[f"deepstack_features_{i}"] = ds
        else:
            for i in range(self.deepstack_count):
                feeds[f"deepstack_features_{i}"] = torch.zeros(
                    (1, total_seq, self.hidden_size),
                    dtype=torch.float32, device=self.device,
                )

        return self.transformer_engine(feeds)  # [1, seq, hidden] on self.device

    # =====================================================================
    # Helpers
    # =====================================================================

    def _find_video_token_start(self, input_ids_1d: np.ndarray) -> Optional[int]:
        """Find the first <|video_pad|> token position."""
        positions = np.where(input_ids_1d == self._video_pad_id)[0]
        return int(positions[0]) if len(positions) else None

    @staticmethod
    def _pool_last(hidden: torch.Tensor, attention_mask: np.ndarray) -> torch.Tensor:
        """Select the last non-padding hidden state per batch sample.

        `hidden` stays on GPU; the indices are tiny so we compute them on CPU
        (numpy) and upload as index tensors.
        """
        flipped = np.flip(attention_mask, axis=1)
        last_pos = np.argmax(flipped, axis=1)
        col = attention_mask.shape[1] - last_pos - 1
        col_idx = torch.from_numpy(col.astype(np.int64)).to(hidden.device)
        row_idx = torch.arange(hidden.shape[0], device=hidden.device)
        return hidden[row_idx, col_idx]

    def _tensor_frame_to_pil(self, frame: torch.Tensor) -> Image.Image:
        """Convert a torch frame [C,H,W] or [H,W,C] → PIL Image at the fixed TRT resolution."""
        arr = frame.detach().cpu().numpy()
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D frame [C,H,W] or [H,W,C], got {arr.shape}")

        if arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = arr.transpose(1, 2, 0)

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0 + 1e-3:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        img = Image.fromarray(arr, mode="RGB")
        return img.resize((self.image_width, self.image_height), Image.BILINEAR)

    # =====================================================================
    # Public API
    # =====================================================================

    def _encode_frames_tensor(self, frames: List[Image.Image]) -> torch.Tensor:
        """Internal: encode a list of PIL frames → [1, D] torch tensor on self.device."""
        conversation = self._format_conversation(frames=frames)
        tok = self._tokenise(conversation)
        last_hidden = self._forward(
            tok["input_ids"], tok["attention_mask"], tok.get("pixel_values"),
        )
        return self._pool_last(last_hidden, tok["attention_mask"])

    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """Encode a single PIL image or file path → [1, D] numpy embedding."""
        conversation = self._format_conversation(image=image, text=None, instruction=None)
        tok = self._tokenise(conversation)
        last_hidden = self._forward(tok["input_ids"], tok["attention_mask"], tok.get("pixel_values"))
        pooled = self._pool_last(last_hidden, tok["attention_mask"])
        return pooled.detach().cpu().numpy()

    def encode_frames(self, frames: List[Image.Image]) -> np.ndarray:
        """Encode a list of PIL frames (video) → [1, D] numpy embedding."""
        return self._encode_frames_tensor(frames).detach().cpu().numpy()

    def encode_video(self, batched_videos: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of videos.

        batched_videos: [B, T, C, H, W] torch tensor.
        Returns: [B, D] torch tensor on self.device (unnormalised).
        """
        if batched_videos.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got {tuple(batched_videos.shape)}")
        B, T = batched_videos.shape[:2]

        embeddings: List[torch.Tensor] = []
        for b in range(B):
            frames = [self._tensor_frame_to_pil(batched_videos[b, t]) for t in range(T)]
            embeddings.append(self._encode_frames_tensor(frames))
        return torch.cat(embeddings, dim=0)

    def __call__(self, video: Optional[torch.Tensor] = None, text: Optional[Any] = None) -> torch.Tensor:
        """PiaTorchModel-compatible interface: model(video=tensor, text=None)."""
        if text is not None:
            raise NotImplementedError("Only video=tensor inference is supported.")
        if video is None:
            raise ValueError("`video` must be provided.")
        return self.encode_video(video)
