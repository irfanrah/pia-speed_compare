"""
Custom FastAPI server that serves Qwen3-VL-Embedding via the TensorRT pipeline.

Frames may be sent at any resolution; the server resizes each one to the
fixed `IMG_SIZE` baked into the TRT engines (a static-shape requirement).

Request shape (POST /v1/embeddings):
  {
    "frames": ["<base64_jpeg>", ...]   # T frames, BGR JPEG bytes in base64
  }

Response shape:
  {
    "embedding": [float, ...],
    "dim": 2048
  }
"""
import asyncio
import base64
import io
import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from PIL import Image
from pydantic import BaseModel

from pia_prod.AI.modules.qwen3vle_trt.config import IMG_SIZE
from pia_prod.AI.modules.qwen3vle_trt.trt.model import Qwen3VLETrtModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pia_qwen3vle_trt_server")

# === Config (env vars) ===
TRT_DIR = os.getenv("TRT_DIR", "/models/Qwen3VLE-TRT")
PROCESSOR_PATH = os.getenv("PROCESSOR_PATH", "/models/Qwen3VLE-TRT/tokenizer")
DEVICE = os.getenv("DEVICE", "cuda")
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "9002"))
# Caps queue depth for backpressure / OOM safety. This is NOT a GPU-concurrency
# knob: TRT IExecutionContext is single-threaded, and the engine wrapper holds
# a lock around every set_input_shape → execute → synchronize sequence, so GPU
# work is serialised on a single stream regardless of how many requests are
# admitted. Tune this to bound how many decode + memory-resident requests sit
# waiting for the engine, not to raise inference parallelism.
MAX_INFLIGHT_REQUESTS = int(os.getenv("MAX_INFLIGHT_REQUESTS", "2"))

# IMG_SIZE in qwen3vle_trt/config.py is (h, w). PIL.Image.resize
# takes (w, h) — keep both forms ready so we don't fight the conventions.
TARGET_SIZE_HW = IMG_SIZE
TARGET_SIZE_WH = (IMG_SIZE[1], IMG_SIZE[0])


class EmbedRequest(BaseModel):
    frames: List[str]


class EmbedResponse(BaseModel):
    embedding: List[float]
    dim: int


class ModelHolder:
    def __init__(self):
        self.model = None

    def load(self):
        logger.info(f"Loading Qwen3VLETrtModel from {TRT_DIR}")
        self.model = Qwen3VLETrtModel(
            trt_dir=TRT_DIR,
            processor_path=PROCESSOR_PATH,
            device=DEVICE,
        )
        logger.info("Qwen3VLETrtModel ready")


model_holder = ModelHolder()
gpu_semaphore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global gpu_semaphore
    gpu_semaphore = asyncio.Semaphore(MAX_INFLIGHT_REQUESTS)
    model_holder.load()
    yield
    model_holder.model = None


app = FastAPI(title="PIA Qwen3-VL-Embedding TRT server", lifespan=lifespan)


def _decode_frame_to_pil(b64: str) -> Image.Image:
    """base64 JPEG/PNG → RGB PIL Image, resized to the engine's fixed input
    size. The TRT engines are compiled with a static `pixel_values` shape;
    skipping this resize lets the HF processor pick its own resolution and
    setInputShape fails with a dimension mismatch. Direct resize (no
    letterbox) mirrors `TF.resize` used in the in-process service path."""
    raw = base64.b64decode(b64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.size != TARGET_SIZE_WH:
        img = img.resize(TARGET_SIZE_WH, Image.Resampling.BILINEAR)
    return img


def _process_and_infer(req: EmbedRequest) -> List[float]:
    """Decodes + resizes frames, then runs TRT inference. Runs in a threadpool
    so the asyncio event loop stays responsive during GPU work."""
    pil_frames = [_decode_frame_to_pil(f) for f in req.frames]
    emb_numpy = model_holder.model.encode_frames(pil_frames)
    return emb_numpy.squeeze(0).tolist()


@app.post("/v1/embeddings", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    if not req.frames:
        raise HTTPException(status_code=400, detail="frames must not be empty")

    try:
        async with gpu_semaphore:
            vec = await run_in_threadpool(_process_and_infer, req)

        return EmbedResponse(embedding=vec, dim=len(vec))
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model_holder.model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)