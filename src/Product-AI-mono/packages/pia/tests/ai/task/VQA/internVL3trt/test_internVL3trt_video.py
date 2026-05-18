import numpy as np
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.VQA.base import VQAConfig


def test_internVL3trt_single_video(vqa_model, rainbow_video):
    """단일 비디오 (T,H,W,C)에 대한 응답을 테스트합니다."""
    print("\n--- Running Single Video Test ---")

    prompt = "Describe the color changes in this video sequence."

    response = vqa_model.forward(rainbow_video, prompt)

    print(f"REQUEST: '{prompt}'")
    print(f"RESPONSE: {response}")
    assert len(response) > 0, "응답 문자열이 비어있으면 안 됩니다."
    assert isinstance(response[0], str), "응답은 문자열이어야 합니다."
    print("✅ Single video test passed!")

def test_internVL3trt_video_list(vqa_model, rainbow_video, gradient_video):
    """비디오 리스트 [(T,H,W,C), (T,H,W,C)]에 대한 응답을 테스트합니다."""
    print("\n--- Running Video List Test ---")

    videos = [rainbow_video, gradient_video]
    prompts = [
        "What colors do you see in this video?",
        "Describe the visual pattern in this video."
    ]

    response = vqa_model.forward(videos, prompts)

    assert isinstance(response, list) and len(response) == 2
    for i, res in enumerate(response):
        print(f"REQUEST {i}: '{prompts[i]}'")
        print(f"RESPONSE {i}: {res}")
        assert isinstance(res, str) and len(res) > 0
    print("✅ Video list test passed!")

def test_internVL3trt_mixed_prompts(vqa_model, text_overlay_video):
    """단일 비디오에 여러 프롬프트를 적용하는 테스트입니다."""
    print("\n--- Running Mixed Prompts Test ---")

    videos = [text_overlay_video, text_overlay_video]
    prompts = [
        "How many frames are in this video?",
        "What text appears in the video?"
    ]

    response = vqa_model.forward(videos, prompts)

    assert isinstance(response, list) and len(response) == 2
    for i, res in enumerate(response):
        print(f"REQUEST {i}: '{prompts[i]}'")
        print(f"RESPONSE {i}: {res}")
    print("✅ Mixed prompts test passed!")

def test_internVL3trt_video_dimensions(vqa_model):
    """다양한 차원의 비디오 입력을 테스트합니다."""
    print("\n--- Testing Video Dimension Handling ---")

    videos = [
        np.random.randint(0, 255, (3, 448, 448, 3), dtype=np.uint8),   # 3 프레임
        np.random.randint(0, 255, (8, 448, 448, 3), dtype=np.uint8),  # 8 프레임
        np.random.randint(0, 255, (24, 448, 448, 3), dtype=np.uint8)   # 24 프레임
    ]
    prompts = ["Describe this video", "What do you see?", "Analyze the content"]

    response = vqa_model.forward(videos, prompts)

    assert isinstance(response, list) and len(response) == 3
    for i, res in enumerate(response):
        print(f"Video {i+1} ({videos[i].shape[0]} frames) - Response length: {len(res)}")
    print("✅ Video dimensions test passed!")


def test_size_comparison():
    """
    224x224 vs 448x448 성능 비교
    (참고: 이 테스트는 다른 config를 사용하므로 별도의 fixture를 만들거나 내부에서 생성합니다)
    """
    video_224 = np.random.randint(0, 255, (12, 224, 224, 3), dtype=np.uint8)
    video_448 = np.random.randint(0, 255, (12, 448, 448, 3), dtype=np.uint8)
    video_1080 = np.random.randint(0, 255, (12, 1080, 1920, 3), dtype=np.uint8)

    # 이 테스트는 고유한 Config를 사용하므로 Fixture를 사용하지 않습니다.
    config = VQAConfig(api_host="127.0.0.1", api_port=9997)
    model = PiaTorchModel(target_task="VQA", target_model="internvl3trt", config=config)
    prompt = "Describe this video"

    print("\n🔴 1920*1080 비디오 테스트:")
    result_1080 = model.forward(video_1080, prompt)
    assert result_1080

    print("\n🔴 448x448 비디오 테스트:")
    result_448 = model.forward(video_448, prompt)
    assert result_448

    print("\n🔵 224x224 비디오 테스트:")
    result_224 = model.forward(video_224, prompt)
    assert result_224

def test_internVL3trt_single_cat_video(vqa_model, video):
    """단일 비디오 (T,H,W,C)에 대한 응답을 테스트합니다."""
    print("\n--- Running Single Video Test ---")
    prompt = "Describe the video"
    response = vqa_model.forward(video, prompt)

    print(f"REQUEST: '{prompt}'")
    print(f"RESPONSE: {response}")
    assert len(response) > 0, "응답 문자열이 비어있으면 안 됩니다."
    print("✅ Single video test passed!")

def test_internVL3trt_single_cat_violence_video(vqa_model, videos):
    """단일 비디오 (T,H,W,C)에 대한 응답을 테스트합니다."""
    print("\n--- Running Single Video Test ---")

    prompts = ["Describe the video" , "What is the predominant color tone in the video? Is it mostly gray, reddish, or whitish?"]

    response = vqa_model.forward(videos, prompts)

    assert isinstance(response, list) and len(response) == 2
    for i, res in enumerate(response):
        print(f"REQUEST {i}: '{prompts[i]}'")
        print(f"RESPONSE {i}: {res}")
        assert isinstance(res, str) and len(res) > 0
    print("✅ Video list test passed!")
