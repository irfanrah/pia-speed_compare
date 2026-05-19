"""
Qwen3VLE Embedding Server (vLLM AsyncLLMEngine).

One container, one process. Strips down `pe_vle_2stage_async/validation_server`
to just the embedding path — no publishers/Redis/S3, because the qwen3vle
service does classification + event publishing in-process.

Wire contract:
    POST /v1/embeddings        — single video (list of frames)
    POST /v1/embeddings/batch  — batch of videos (list of list of frames)
    GET  /health               — readiness probe

Frames are base64-encoded JPEGs (or PNG; PIL auto-detects). All requests
take the video modality regardless of T — T=1 inputs are padded to T=2 to
satisfy the Qwen3-VL video processor's temporal_factor=2 constraint. This
keeps the single-frame and multi-frame call paths byte-compatible with
Qwen3VLE_HF_Model.encode_video and avoids a ~0.13 cosine gap that the
image-modality shortcut used to introduce.
"""

import asyncio
import base64
import inspect
import io
import logging
import uuid
from contextlib import asynccontextmanager, suppress
from typing import List, Tuple

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from config import (
    DEFAULT_INSTRUCTION,
    DTYPE,
    EMBED_TIMEOUT_S,
    GPU_MEMORY_UTILIZATION,
    KV_CACHE_MEMORY_BYTES,
    LIMIT_MM_PER_PROMPT_IMAGE,
    LIMIT_MM_PER_PROMPT_VIDEO,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MODEL_PATH,
    SERVER_HOST,
    SERVER_PORT,
    VLLM_MAX_CONCURRENCY,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qwen3vle_embedding_server")


# === Request / Response Schemas ==============================================

class EmbedRequest(BaseModel):
    frames: List[str]  # one video, T base64 frames

class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int

class BatchEmbedRequest(BaseModel):
    videos: List[List[str]]  # batch of videos, each a list of T base64 frames

class BatchEmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dim: int


# === Qwen3VLE Embedder Core ==================================================

class Qwen3VLEEmbedder:
    """`AsyncLLMEngine`-backed pooled embedder.

    Uses vLLM's async engine so multiple `embed_frames()` coroutines can be
    in flight concurrently — the engine batches them on its scheduler and
    runs the GPU work without serializing at the FastAPI layer.
    """
    
    TEMPORAL_FACTOR = 2

    def __init__(self):
        self.engine = None
        self._tokenizer = None
        self._pooling_params = None

    async def load(self) -> None:
        """Build the AsyncLLMEngine and resolve the tokenizer."""
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.pooling_params import PoolingParams

        kwargs = {
            "model": MODEL_PATH,
            "runner": "pooling",
            "dtype": DTYPE,
            "trust_remote_code": True,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "max_model_len": MAX_MODEL_LEN,
            "limit_mm_per_prompt": {
                "video": LIMIT_MM_PER_PROMPT_VIDEO,
                "image": LIMIT_MM_PER_PROMPT_IMAGE,
            },
        }
        
        if KV_CACHE_MEMORY_BYTES is not None:
            kwargs["kv_cache_memory_bytes"] = KV_CACHE_MEMORY_BYTES
        if MAX_NUM_SEQS is not None:
            kwargs["max_num_seqs"] = MAX_NUM_SEQS
        if MAX_NUM_BATCHED_TOKENS is not None:
            kwargs["max_num_batched_tokens"] = MAX_NUM_BATCHED_TOKENS

        engine_args = AsyncEngineArgs(**kwargs)
        self._silence_vllm_loggers()
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        self._pooling_params = PoolingParams(task="embed")

        # Handle version differences where get_tokenizer() might be a coroutine
        tok = self.engine.get_tokenizer()
        self._tokenizer = await tok if inspect.isawaitable(tok) else tok

    async def shutdown(self) -> None:
        """Safely shuts down the vLLM engine to prevent GPU memory leaks."""
        if self.engine is None:
            return

        for name in ("shutdown", "shutdown_background_loop"):
            shutdown_func = getattr(self.engine, name, None)
            if shutdown_func is None:
                continue
            try:
                result = shutdown_func()
                if inspect.isawaitable(result):
                    await result
            except Exception as e:
                logger.warning(f"Engine shutdown raised an exception: {e}")
            break

    @staticmethod
    def _silence_vllm_loggers():
        """Raise logging level for noisy vLLM modules."""
        noisy_loggers = [
            "vllm.engine.async_llm_engine",
            "vllm.v1.engine.async_llm",
            "vllm.engine.metrics",
            "vllm.core.scheduler",
        ]
        for logger_name in noisy_loggers:
            logging.getLogger(logger_name).setLevel(logging.WARNING)

    @staticmethod
    def _decode_frames(frames_b64: List[str]) -> List[Image.Image]:
        return [
            Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            for b64 in frames_b64
        ]

    def _pad_frames(self, imgs: List[Image.Image]) -> List[Image.Image]:
        """Ensure minimum frame count for Qwen3-VL processor constraints."""
        if len(imgs) < self.TEMPORAL_FACTOR:
            return imgs + [imgs[-1]] * (self.TEMPORAL_FACTOR - len(imgs))
        return imgs

    @staticmethod
    def _build_video_metadata(num_frames: int) -> dict:
        """Synthesize metadata to satisfy vLLM's internal parser requirements."""
        return {
            "fps": 2.0,
            "duration": num_frames / 2.0,
            "total_num_frames": num_frames,
            "frames_indices": list(range(num_frames)),
            "video_backend": "opencv",
            "do_sample_frames": False,
        }

    def _prepare_request(self, frames_b64: List[str]) -> Tuple[str, dict]:
        """Sync preprocessing: decodes, pads, and builds the model payload."""
        imgs = self._decode_frames(frames_b64)
        if not imgs:
            raise ValueError("frames must be non-empty")

        imgs = self._pad_frames(imgs)
        video_array = np.stack([np.asarray(img) for img in imgs], axis=0)
        
        video_metadata = self._build_video_metadata(num_frames=video_array.shape[0])
        modal_value = (video_array, video_metadata)
        
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "video", "video": imgs}]},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        return prompt_text, {"video": modal_value}

    async def _abort_engine_request(self, request_id: str) -> None:
        """Notifies vLLM to drop a cancelled request to free up scheduler slots."""
        for name in ("abort", "abort_request"):
            abort = getattr(self.engine, name, None)
            if abort is None:
                continue
            with suppress(Exception):
                result = abort(request_id)
                if inspect.isawaitable(result):
                    await result
            break

    async def _encode(self, prompt_text: str, multi_modal_data: dict) -> List[float]:
        """Drives AsyncLLMEngine.encode() to completion for a single request."""
        request_id = uuid.uuid4().hex

        try:
            async for output in self.engine.encode(
                {"prompt": prompt_text, "multi_modal_data": multi_modal_data},
                pooling_params=self._pooling_params,
                request_id=request_id,
            ):
                if output.finished:
                    pooling_out = output.outputs
                    
                    # Cross-version compatibility for single vs list outputs
                    if isinstance(pooling_out, list):
                        pooling_out = pooling_out[0]

                    vec = pooling_out.data
                    return vec.tolist() if hasattr(vec, "tolist") else list(vec)
                    
        except (asyncio.CancelledError, GeneratorExit):
            await self._abort_engine_request(request_id)
            raise

        raise RuntimeError(
            f"AsyncLLMEngine.encode() finished without yielding a final output "
            f"for request_id={request_id}"
        )

    async def embed_frames(self, frames_b64: List[str]) -> List[float]:
        """Preprocesses frames and yields the pooled vector via vLLM."""
        # Caveat: cancelling this coroutine (timeout, batch sibling failure)
        # raises CancelledError in the awaiting context but does NOT stop the
        # worker thread — Python has no thread cancellation. The preprocessing
        # finishes anyway. Acceptable because the GPU semaphore slot is freed
        # downstream (in _encode's abort path) and CPU is the cheap resource.
        prompt_text, multi_modal_data = await asyncio.to_thread(
            self._prepare_request, frames_b64
        )
        return await asyncio.wait_for(
            self._encode(prompt_text, multi_modal_data),
            timeout=EMBED_TIMEOUT_S,
        )


# === App State & Lifespan ====================================================

embedder = Qwen3VLEEmbedder()
_embed_semaphore: "asyncio.Semaphore | None" = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_semaphore
    _embed_semaphore = asyncio.Semaphore(VLLM_MAX_CONCURRENCY)

    await embedder.load()
    logger.info(
        f"Qwen3VLE embedding server ready: model={MODEL_PATH}, "
        f"vllm_max_concurrency={VLLM_MAX_CONCURRENCY}"
    )
    
    try:
        yield
    finally:
        await embedder.shutdown()


app = FastAPI(title="Qwen3VLE Embedding Server", lifespan=lifespan)


# === API Endpoints ===========================================================

def _require_ready():
    """Dependency check to prevent requests before the model is loaded."""
    if _embed_semaphore is None:
        raise HTTPException(status_code=503, detail="server not ready")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_loaded": embedder._tokenizer is not None,
    }


@app.post("/v1/embeddings", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames must not be empty")
    _require_ready()

    try:
        async with _embed_semaphore:
            vec = await embedder.embed_frames(req.frames)
    except HTTPException:
        raise
    except asyncio.TimeoutError:
        logger.warning(f"embed timed out after {EMBED_TIMEOUT_S}s")
        raise HTTPException(status_code=504, detail="embedding timed out")
    except Exception as e:
        logger.error(f"embed failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="embedding failed")

    return EmbedResponse(embedding=vec, dim=len(vec))


@app.post("/v1/embeddings/batch", response_model=BatchEmbedResponse)
async def embed_batch(req: BatchEmbedRequest):
    if not req.videos:
        raise HTTPException(status_code=400, detail="videos must not be empty")
    for i, frames in enumerate(req.videos):
        if not frames:
            raise HTTPException(status_code=400, detail=f"videos[{i}].frames must not be empty")

    _require_ready()

    async def _one(frames_b64: List[str]) -> List[float]:
        async with _embed_semaphore:
            return await embedder.embed_frames(frames_b64)

    tasks = [asyncio.create_task(_one(v)) for v in req.videos]

    def _cancel_tasks():
        for t in tasks:
            if not t.done():
                t.cancel()

    try:
        results = await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        _cancel_tasks()
        logger.warning(f"batch embed timed out after {EMBED_TIMEOUT_S}s")
        raise HTTPException(status_code=504, detail="embedding timed out")
    except Exception as e:
        _cancel_tasks()
        logger.error(f"batch embed failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="batch embedding failed")

    return BatchEmbedResponse(
        embeddings=results,
        dim=len(results[0]) if results else 0,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)