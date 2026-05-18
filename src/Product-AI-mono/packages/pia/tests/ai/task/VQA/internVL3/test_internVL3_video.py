from pia.ai.model import PiaTorchModel
from pia.ai.tasks.VQA.base import VQAConfig


def test_internVL3_single_video(model_path, video):
    """단일 비디오 테스트"""
    config = VQAConfig(
        model_path=model_path,
        device="cuda",
        frame_per_tile_max_num=1,
        data_type="bfloat16",
        n_gpus=1,
        save_cache_dir=model_path,
        num_segments=3
    )
    model = PiaTorchModel(target_task="VQA", target_model="internVL3", config=config)

    prompt = "explain this video."
    response = model(video, prompt)

    assert isinstance(response, str), "response should be str"
    print(f"Single video response: {response}")



def test_internVL3_multi_video(model_path, videos):
    """복수 비디오 배치 테스트"""
    config = VQAConfig(
        model_path=model_path,
        device="cuda",
        frame_per_tile_max_num=1,
        data_type="bfloat16",
        n_gpus=1,
        save_cache_dir=model_path,
        num_segments=3

    )
    model = PiaTorchModel(target_task="VQA", target_model="internVL3", config=config)

    prompts = [
        "explain this video.",
        "what safety issues can you identify in this video?"
    ]

    response = model(videos, prompts)

    assert len(response) == 2, "response length should be 2"
    for i, res in enumerate(response):
        print(f"RESPONSE {i}: {res}")
