"""
PE-VLE Validation Server (Qwen3VLE in-process, fully self-contained).

One container, one process. No `pia_prod` dependency, no monorepo
bind-mount. Bundles:

  - Qwen3VLE FP8 vLLM model (loaded in-process at startup).
  - Anchor-based classifier (numpy cosine vs FP8 text features).
  - Self-contained publish stack: RabbitMQ / Kafka via fastavro,
    S3 thumbnail upload, Redis UUID pairing dedup.

Wire contract is identical to pe_vqa_2stage's `validation_server`:
- POST /api/v1/validate ← `ValidateRequest`
  - `is_start=True` requires `thumbnail_b64` (else 400).
- 200 { "status": "sent",       "uuid": ... }
- 200 { "status": "discarded",  "uuid": ..., "reason": "vle_rejected"
                                                       | "vle_error"
                                                       | "start_not_validated" }
- 200 { "status": "error",      "uuid": ..., "reason": ... }
"""

import asyncio
import base64
import io
import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

from config import (
    DEFAULT_INSTRUCTION,
    DTYPE,
    GPU_MEMORY_UTILIZATION,
    KV_CACHE_MEMORY_BYTES,
    LIMIT_MM_PER_PROMPT_IMAGE,
    LIMIT_MM_PER_PROMPT_VIDEO,
    MAX_MODEL_LEN,
    MAX_NUM_BATCHED_TOKENS,
    MAX_NUM_SEQS,
    MESSAGING_BACKEND,
    MODEL_PATH,
    PE_VLE_FAIL_OPEN,
    PE_VLE_TO_VLE_CATEGORY_EVENT_MAP,
    SERVER_HOST,
    SERVER_PORT,
    TEXT_FEATURES_PATH,
    UUID_TTL_SECONDS,
    VLLM_MAX_CONCURRENCY,
    VLE_CATEGORY_EVENT_MAP,
)
from publishers import (
    MessagePublisher,
    RedisUUIDTracker,
    S3Uploader,
    build_publisher,
    make_alarm_message_compatible,
    resolve_topic_or_queue,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pe_vle_validation_server")


# === Request schema (matches pe_vqa_2stage's _build_validation_payload 1:1) ===
class ValidateRequest(BaseModel):
    thumbnail_b64: Optional[str] = None  # base64 encoded JPEG (BGR pixels)
    is_start: bool
    category_name: str
    stream_id: str
    event_uuid: str
    event_type: str
    user_param: Dict[str, Any]


# === In-process Qwen3VLE wrapper (AsyncLLMEngine — concurrent-safe) ==========
class Qwen3VLEEmbedder:
    """`AsyncLLMEngine`-backed pooled embedder.

    Uses vLLM's async engine instead of the synchronous `LLM` wrapper so
    multiple `embed_image()` coroutines can be in flight concurrently — the
    engine batches them on its scheduler and runs the GPU work without
    serializing at the FastAPI layer. Mirrors how `vllm-openai`'s embedding
    endpoint handles concurrency, just with the engine in-process so we
    skip the loopback HTTP hop pe_vqa pays.

    Loaded once in `lifespan` (synchronous construction is fine — engine
    init happens before the FastAPI app accepts traffic).
    """

    def __init__(self):
        self.engine = None        # populated in load()
        self._tokenizer = None    # tokenizer for chat-template prompt build
        self._pooling_params = None

    def load(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.pooling_params import PoolingParams

        # Required args first; optional vLLM-tuning kwargs only when explicitly
        # set so empty defaults don't override vLLM's auto-sizing.
        kwargs: dict = {
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
        # disable_log_requests was removed from AsyncLLM.from_engine_args() in
        # v0.14 — quiet the per-request INFO spam by raising the log level on
        # the engine_core / scheduler loggers instead. Harmless if those
        # loggers don't exist (older versions just keep logging).
        for noisy in ("vllm.engine.async_llm_engine", "vllm.v1.engine.async_llm",
                       "vllm.engine.metrics", "vllm.core.scheduler"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        # vLLM v1 requires `task` to be explicit on PoolingParams — submitting
        # with task=None raises "Unsupported task: None" inside EngineCore and
        # kills the engine subprocess. 'embed' = whole-sequence pooled vector
        # (what we want); 'token_embed' = per-token vectors (not what we want).
        self._pooling_params = PoolingParams(task="embed")

        # Tokenizer is needed for the chat-template render. AsyncLLMEngine
        # exposes it via get_tokenizer(); we cache it once at startup.
        # The await happens inside lifespan via _async_post_load().

    async def _async_post_load(self) -> None:
        """Async portion of load — must run on the event loop. Called from
        lifespan after `load()` finishes the synchronous engine init."""
        import inspect
        tok = self.engine.get_tokenizer()
        if inspect.isawaitable(tok):
            tok = await tok
        self._tokenizer = tok

    async def embed_image(self, image_b64: str) -> List[float]:
        """Decode the b64 JPEG, build the chat-template conversation, submit
        to vLLM's async engine, return the pooled vector. Multiple concurrent
        callers are batched by the engine — no manual locking required."""
        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_INSTRUCTION}]},
            {"role": "user", "content": [{"type": "image", "image": img}]},
        ]
        prompt_text = self._tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

        request_id = uuid.uuid4().hex
        # AsyncLLMEngine.encode() is the pooling/embedding equivalent of
        # generate(). It yields one or more RequestOutput objects; only the
        # final one (output.finished) carries the pooled embedding.
        async for output in self.engine.encode(
            {"prompt": prompt_text, "multi_modal_data": {"image": img}},
            pooling_params=self._pooling_params,
            request_id=request_id,
        ):
            if output.finished:
                # AsyncLLMEngine yields PoolingRequestOutput where `.outputs`
                # is a single PoolingOutput (not a list). The synchronous
                # LLM.embed() returns a list[PoolingOutput], hence the
                # difference in subscripting from the older code path.
                pooling_out = output.outputs
                if isinstance(pooling_out, list):
                    pooling_out = pooling_out[0]  # belt-and-suspenders for older vllm
                vec = pooling_out.data
                if hasattr(vec, "tolist"):
                    return [float(x) for x in vec.tolist()]
                return [float(x) for x in vec]
        raise RuntimeError(
            f"AsyncLLMEngine.encode() finished without yielding a final output "
            f"for request_id={request_id}"
        )


# === Anchor classifier =======================================================
class AnomalyClassifier:
    """Cosine sim against per-bucket normal/target unit-vector matrices
    loaded from the FP8 text-features JSON."""

    def __init__(self):
        self.anchors: Dict[str, Dict[str, np.ndarray]] = {}

    def load(self) -> None:
        with open(TEXT_FEATURES_PATH) as f:
            data = json.load(f)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for class_name, block in data.items():
            feats = block.get("text_features", {})
            normal, target = feats.get("normal", []), feats.get(class_name, [])
            if not normal or not target:
                continue
            n = self._l2(np.asarray(normal, dtype=np.float32), axis=-1).T
            t = self._l2(np.asarray(target, dtype=np.float32), axis=-1).T
            out[class_name] = {
                "normal": np.ascontiguousarray(n),
                "target": np.ascontiguousarray(t),
            }
        if not out:
            raise RuntimeError(f"No text_features loaded from {TEXT_FEATURES_PATH}")
        self.anchors = out

    @staticmethod
    def _l2(x: np.ndarray, axis: int = -1) -> np.ndarray:
        return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)

    def classify(self, embedding: List[float], vle_id: str) -> bool:
        bucket = next(
            (b for b, ids in VLE_CATEGORY_EVENT_MAP.items() if vle_id in ids),
            None,
        )
        anchors = self.anchors.get(bucket) if bucket else None
        if not anchors:
            return True
        emb = self._l2(np.asarray(embedding, dtype=np.float32)[None, :], axis=1)
        normal_max = float((emb @ anchors["normal"]).max())
        target_max = float((emb @ anchors["target"]).max())
        return target_max > normal_max


# === App state ===============================================================
embedder = Qwen3VLEEmbedder()
classifier = AnomalyClassifier()
mq_publisher: MessagePublisher = build_publisher()
s3_uploader = S3Uploader()
uuid_tracker = RedisUUIDTracker(ttl_seconds=UUID_TTL_SECONDS)

# Bounded by VLLM_MAX_CONCURRENCY in lifespan() — initialized there because
# asyncio primitives must be created on the running loop. Until then it's a
# placeholder; the validate() handler holds a reference to the live one.
import asyncio  # local import to keep top imports tidy
_embed_semaphore: "asyncio.Semaphore | None" = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_semaphore
    _embed_semaphore = asyncio.Semaphore(VLLM_MAX_CONCURRENCY)

    classifier.load()
    embedder.load()
    # AsyncLLMEngine.get_tokenizer() is async, so finish embedder init here
    # rather than inside the synchronous load() method.
    await embedder._async_post_load()
    backend_label = "Kafka" if MESSAGING_BACKEND == "kafka" else "RabbitMQ"
    for component_name, connect_fn in (
        (backend_label, mq_publisher.connect),
        ("S3", s3_uploader.connect),
        ("Redis", uuid_tracker.connect),
    ):
        try:
            connect_fn()
        except Exception as e:
            logger.warning(
                f"{component_name} connect failed (server still starting; will retry on use): {e}"
            )
    # Kafka backend gets a background drain task (poll/flush every 100ms).
    # No-op for RabbitMQ (start_drain doesn't exist on RabbitMQPublisher).
    if hasattr(mq_publisher, "start_drain"):
        await mq_publisher.start_drain()

    logger.info(
        f"Anchor buckets: {sorted(classifier.anchors.keys())}; "
        f"backend={MESSAGING_BACKEND}; PE_VLE_FAIL_OPEN={PE_VLE_FAIL_OPEN}; "
        f"vllm_max_concurrency={VLLM_MAX_CONCURRENCY}"
    )
    yield
    if hasattr(mq_publisher, "stop_drain"):
        await mq_publisher.stop_drain()
    mq_publisher.close()


app = FastAPI(title="PE-VLE Validation Server", lifespan=lifespan)


# === Helpers =================================================================
def _resolve_vle_id(category_name: str) -> Optional[str]:
    return PE_VLE_TO_VLE_CATEGORY_EVENT_MAP.get(category_name)


# === Endpoints ===============================================================
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "anchor_buckets": sorted(classifier.anchors.keys()),
        "model_path": MODEL_PATH,
        "messaging_backend": MESSAGING_BACKEND,
        "fail_open": PE_VLE_FAIL_OPEN,
    }


@app.post("/api/v1/validate")
async def validate(req: ValidateRequest):
    # Reuse one timestamp for both message["ts"] and the S3 key — matches
    # ES MQProducer's behavior of using message.ts as the S3 key timestamp.
    ts_epoch_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    if req.is_start:
        if req.thumbnail_b64 is None:
            raise HTTPException(
                status_code=400, detail="is_start=True 일 때 thumbnail_b64 필수"
            )

        # Decide whether to verify with vLLM. PE categories without a VLE
        # counterpart (e.g. falldown) skip the embed step entirely and trust
        # the PE verdict — same fall-through pe_vqa applies for unmapped
        # prompts.
        vle_id = _resolve_vle_id(req.category_name)
        if vle_id is not None:
            try:
                # Semaphore caps in-flight embed calls — pure backpressure
                # for OOM safety / queue-depth bounding. AsyncLLMEngine
                # batches concurrent requests internally, so the semaphore
                # doesn't reduce throughput; it just prevents the validator
                # from accepting more work than it can hold in memory.
                async with _embed_semaphore:
                    embedding = await embedder.embed_image(req.thumbnail_b64)
            except Exception as e:
                if PE_VLE_FAIL_OPEN:
                    logger.warning(
                        f"vLLM embed failed — fail-open: stream={req.stream_id}, "
                        f"uuid={req.event_uuid}, error={e}"
                    )
                else:
                    logger.error(
                        f"vLLM embed failed (discarded): stream={req.stream_id}, error={e}"
                    )
                    return {
                        "status": "discarded",
                        "uuid": req.event_uuid,
                        "reason": "vle_error",
                    }
            else:
                if not classifier.classify(embedding, vle_id):
                    logger.info(
                        f"False positive discarded: stream={req.stream_id}, "
                        f"category={req.category_name}, uuid={req.event_uuid}"
                    )
                    return {
                        "status": "discarded",
                        "uuid": req.event_uuid,
                        "reason": "vle_rejected",
                    }

        # Confirmed (or fall-through): register the UUID for end-pairing.
        try:
            uuid_tracker.add(req.event_uuid)
        except Exception as e:
            logger.error(f"Redis UUID register failed: {e}")

        # Best-effort S3 upload. The key is always reserved on the message
        # even if the upload fails; the consumer falls back to a placeholder
        # thumbnail in that case (same policy ES MQProducer follows).
        up = req.user_param["user_param"]
        thumbnail_filename = S3Uploader.build_key(
            str(up["cameraId"]), up["organization"], ts_epoch_ms
        )
        try:
            jpeg_bytes = base64.b64decode(req.thumbnail_b64)
            await asyncio.get_running_loop().run_in_executor(
                None,
                s3_uploader.upload_jpeg,
                jpeg_bytes,
                str(up["cameraId"]),
                up["organization"],
                ts_epoch_ms,
            )
        except Exception as e:
            logger.error(f"S3 upload failed (key reserved on message anyway): {e}")

    else:
        # End-state: only publish if the start was registered. Drops orphan
        # ends from rejected starts (consumer never sees a half-paired event).
        try:
            registered = uuid_tracker.contains(req.event_uuid)
        except Exception as e:
            logger.error(f"Redis UUID lookup failed: {e}")
            registered = False
        if not registered:
            logger.info(
                f"End discarded (start not validated): stream={req.stream_id}, "
                f"uuid={req.event_uuid}"
            )
            return {
                "status": "discarded",
                "uuid": req.event_uuid,
                "reason": "start_not_validated",
            }
        try:
            uuid_tracker.remove(req.event_uuid)
        except Exception as e:
            logger.error(f"Redis UUID delete failed: {e}")
        thumbnail_filename = ""

    # Build the alarm message + publish.
    try:
        message = make_alarm_message_compatible(
            user_param=req.user_param,
            thumbnail_filename=thumbnail_filename,
            is_start=req.is_start,
            category_name=req.category_name,
            event_type=req.event_type,
            event_uuid=req.event_uuid,
            ts_epoch_ms=ts_epoch_ms,
        )
    except KeyError as e:
        logger.error(f"user_param missing key: {e}, payload={req.user_param}")
        return {"status": "error", "uuid": req.event_uuid, "reason": f"missing_key:{e}"}

    target = resolve_topic_or_queue()
    try:
        await asyncio.get_running_loop().run_in_executor(
            None, mq_publisher.publish, message, target
        )
    except Exception as e:
        logger.error(f"{MESSAGING_BACKEND} publish failed: {e}")
        return {"status": "error", "uuid": req.event_uuid, "reason": str(e)}

    logger.info(
        f"Alarm sent: stream={req.stream_id}, category={req.category_name}, "
        f"uuid={req.event_uuid}, is_start={req.is_start}"
    )
    return {"status": "sent", "uuid": req.event_uuid}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT)
