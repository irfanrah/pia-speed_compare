def test_internVL3trt_single_batch(blue_image, big_blue_image, vqa_model):
    """단일 배치(이미지 1개)에 대한 응답을 테스트합니다."""
    print("\n--- Running Single Batch Test ---")

    prompt = "what color is this image?"
    response = vqa_model.forward(blue_image, prompt)

    print(f"224 * 224: ")
    print(f"REQUEST: '{prompt}'")
    print(f"RESPONSE: {response}")

    prompt = "what color is this image?"
    response = vqa_model.forward(big_blue_image, prompt)

    print(f"1080 * 1920: ")
    print(f"REQUEST: '{prompt}'")
    print(f"RESPONSE: {response}")

    assert isinstance(response, list), "응답은 리스트여야 합니다."
    assert len(response) == 1, "응답 리스트 길이는 1이어야 합니다."
    assert isinstance(response[0], str), "응답은 문자열이어야 합니다."
    assert len(response[0]) > 0, "응답 문자열이 비어있으면 안 됩니다."

def test_internVL3trt_multi_batch(blue_image, red_image , vqa_model):
    """다중 배치(이미지 2개)에 대한 응답을 테스트합니다."""
    print("\n--- Running Multi Batch Test ---")

    images = [blue_image, red_image]
    prompts = ["what is the color of the first image?", "what is the color of the second image?"]

    response = vqa_model.forward(images, prompts)

    assert isinstance(response, list), "응답은 리스트여야 합니다."
    assert len(response) == 2, "응답 리스트의 길이는 이미지 개수와 같아야 합니다."

    for i, res in enumerate(response):
        print(f"REQUEST {i}: '{prompts[i]}'")
        print(f"RESPONSE {i}: {res}")
        assert isinstance(res, str), f"{i}번째 응답이 문자열이 아닙니다."


def test_internVL3trt_single_peoplo_img(img, vqa_model):
    """단일 배치(이미지 1개)에 대한 응답을 테스트합니다."""
    print("\n--- Running Single Batch Test ---")

    prompt = "what color do you see a lot of people wearing?"
    response = vqa_model.forward(img, prompt)

    print(f"REQUEST: '{prompt}'")
    print(f"RESPONSE: {response}")

    assert isinstance(response, list), "응답은 리스트여야 합니다."
    assert len(response) == 1, "응답 리스트 길이는 1이어야 합니다."
    assert isinstance(response[0], str), "응답은 문자열이어야 합니다."
    assert len(response[0]) > 0, "응답 문자열이 비어있으면 안 됩니다."
