from pia.ai.model import PiaTorchModel
from pia.ai.tasks.VQA.base import VQAConfig


def test_internVL3_single_batch(model_path, img):
    config = VQAConfig(
        model_path=model_path,
        device="cuda",
        frame_per_tile_max_num=16,
        data_type="bfloat16",
        n_gpus=1,
        save_cache_dir=model_path,
    )
    model = PiaTorchModel(target_task="VQA", target_model="internVL3", config=config)

    prompt = "describe this image"
    response = model(img, prompt)

    assert isinstance(response, str) , "response should be str"
    print(response)


def test_internVL3_multi_batch(model_path, img):
    config = VQAConfig(
        model_path=model_path,
        device="cuda",
        frame_per_tile_max_num=16,
        data_type="bfloat16",
        n_gpus=1,
        save_cache_dir=model_path,
    )

    model = PiaTorchModel(target_task="VQA", target_model="internVL3", config=config)
    prompt = "describe this image"
    imgs = [img, img]
    prompts = [prompt, prompt]
    response = model(imgs, prompts)

    assert len(response) == 2, "response length should be 2"
    for i, res in enumerate(response):
        print(f"RESPONSE {i} : {res}")
