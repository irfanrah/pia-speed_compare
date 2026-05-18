import os
import time

import pytest
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.VQA.base import VQAConfig


@pytest.fixture(scope="module")
def vqa_model():
    config = VQAConfig(
        device="cuda",
        data_type="bfloat16",
        api_host="127.0.0.1",
        api_port=9997,
        max_new_tokens= 14
    )
    return PiaTorchModel(target_task="VQA", target_model="internvl3trt", config=config)


def _get_batch_sizes():
    raw = os.getenv("INTERNVL3TRT_BATCH_SIZES")
    if raw:
        sizes = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            sizes.append(int(part))
        sizes = [size for size in sizes if size > 0]
        if not sizes:
            raise ValueError("INTERNVL3TRT_BATCH_SIZES must contain positive integers.")
        return list(dict.fromkeys(sizes))

    max_batch = int(os.getenv("INTERNVL3TRT_MAX_BATCH_SIZE", "5"))
    if max_batch < 1:
        raise ValueError("INTERNVL3TRT_MAX_BATCH_SIZE must be >= 1.")
    return list(range(1, max_batch + 1))


def test_internvl3trt_batch_latency_asset_300(vqa_model, image_300):
    prompt = "describe this image"
    _ = vqa_model.forward([image_300], [prompt])

    for batch_size in _get_batch_sizes():
        images = [image_300] * batch_size
        prompts = [prompt] * batch_size

        start = time.perf_counter()
        response = vqa_model.forward(images, prompts)
        elapsed = time.perf_counter() - start

        assert isinstance(response, list)
        assert len(response) == batch_size
        assert all(isinstance(item, str) and item for item in response)

        per_image = elapsed / batch_size
        print(f"batch_size={batch_size} elapsed={elapsed:.3f}s per_image={per_image:.3f}s")
        # if batch_size == 1:
        #     print(f"batch_size=1 response: {response[0]}")
        # else:
        #     for idx, item in enumerate(response, start=1):
        #         print(f"batch_size={batch_size} response_{idx}: {item}")
