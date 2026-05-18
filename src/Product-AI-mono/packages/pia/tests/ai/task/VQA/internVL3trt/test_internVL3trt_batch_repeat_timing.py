import time


REPEAT_COUNT = 10
REQUEST_INTERVAL_SEC = 0.1
BATCH_SIZE = 1


def test_internvl3trt_batch_repeat_avg_latency(vqa_model, image_300):
    if REPEAT_COUNT < 1:
        raise ValueError("REPEAT_COUNT must be >= 1.")
    if BATCH_SIZE < 1:
        raise ValueError("BATCH_SIZE must be >= 1.")

    prompt = "describe this image"
    images = [image_300] * BATCH_SIZE
    prompts = [prompt] * BATCH_SIZE

    _ = vqa_model.forward(images, prompts)

    durations = []
    for idx in range(REPEAT_COUNT):
        start = time.perf_counter()
        response = vqa_model.forward(images, prompts)
        elapsed = time.perf_counter() - start
        durations.append(elapsed)

        assert isinstance(response, list)
        assert len(response) == BATCH_SIZE
        assert all(isinstance(item, str) and item for item in response)

        print(f"repeat={idx + 1}/{REPEAT_COUNT} batch_size={BATCH_SIZE} elapsed={elapsed:.3f}s")
        if BATCH_SIZE == 1:
            print(f"repeat={idx + 1} response: {response[0]}")
        else:
            for resp_idx, item in enumerate(response, start=1):
                print(f"repeat={idx + 1} response_{resp_idx}: {item}")

        if idx < REPEAT_COUNT - 1 and REQUEST_INTERVAL_SEC > 0:
            time.sleep(REQUEST_INTERVAL_SEC)

    avg = sum(durations) / len(durations)
    per_image_avg = avg / BATCH_SIZE
    print(
        f"average_elapsed={avg:.3f}s per_image_avg={per_image_avg:.3f}s "
        f"(repeats={REPEAT_COUNT}, batch_size={BATCH_SIZE}, interval={REQUEST_INTERVAL_SEC}s)"
    )
