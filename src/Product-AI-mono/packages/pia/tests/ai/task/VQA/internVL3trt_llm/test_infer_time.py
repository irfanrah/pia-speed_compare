"""InternVL3 TRT-LLM VQA 추론 시간 측정 테스트."""

import os
import time

import pytest
import torch

from pia.ai.tasks.VQA.base import VQAConfig
from pia.ai.tasks.VQA.models.internVL3trt_llm.main import InternVL3trt_llm

def _maybe_sync(enabled: bool):
    """선택적으로 CUDA 동기화를 수행한다."""
    if enabled:
        torch.cuda.synchronize()


def _timed_infer(model, images, prompts, repeats=5, sync=False):
    """지정 횟수만큼 추론 시간을 측정해 ms 리스트로 반환한다."""
    times_ms = []
    for _ in range(repeats):
        _maybe_sync(sync)
        start = time.perf_counter()
        _ = model.forward(images, prompts)
        _maybe_sync(sync)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed_ms)
    return times_ms


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_infer_time(image_300, red_image_300, falldown_prompt):
    """배치 1/2 추론 시간을 측정해 콘솔에 출력한다."""
    cfg = VQAConfig()
    model = InternVL3trt_llm(cfg)

    img = image_300
    red_img = red_image_300
    prompt_single = falldown_prompt
    prompt_batch = "explain this image, only english"

    sync = os.environ.get("VQA_TIME_SYNC", "0") == "1"
    warmup = int(os.environ.get("VQA_TIME_WARMUP", "1"))
    repeats = int(os.environ.get("VQA_TIME_REPEATS", "5"))

    # Warmup (exclude engine warmup from timing)
    for _ in range(warmup):
        _ = model.forward(img, prompt_single)
        _ = model.forward([img, red_img], [prompt_batch, prompt_batch])
    _maybe_sync(sync)

    # Batch size 1 timing (test_infer.py 기준)
    times_bs1 = _timed_infer(model, img, prompt_single, repeats=repeats, sync=sync)
    print("---------------------------------------------------")
    print(
        f"batch=1 times_ms={times_bs1} avg_ms={sum(times_bs1)/len(times_bs1):.2f} "
        f"(sync={sync}, warmup={warmup})"
    )

    # Batch size 2 timing (test_infer.py 기준)
    times_bs2 = _timed_infer(
        model, [img, red_img], [prompt_batch, prompt_batch], repeats=repeats, sync=sync
    )
    print("---------------------------------------------------")
    print(
        f"batch=2 times_ms={times_bs2} avg_ms={sum(times_bs2)/len(times_bs2):.2f} "
        f"(sync={sync}, warmup={warmup})"
    )
