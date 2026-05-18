import numpy as np
import torch
from pia.ai.device import load_model_backend
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.T2VRet.base import T2VRetConfig, T2VRetONNXConfig
from pia.tests.ai.task.T2VRet.models.clip_vip.conftest import ASSETS_MODEL_SAVE_DIR


def test_clip_vip(model_path):
    device = load_model_backend("cuda", type="str")

    config = T2VRetConfig(
        model_path=model_path,
        device=device,
        tile_config=None,
    )
    model = PiaTorchModel(
        target_task=1,  # T2VRet
        target_model=1,  # clip-vip
        config=config,  # T2VRet  # clipvip
    )
    ######################    VIDEO    ###########################
    video = np.arange(0, config.temporal_size * 224 * 224 * 3, dtype=np.uint8)
    video = video.reshape(config.temporal_size, 224, 224, 3)

    # numpy 확인
    vis_vector = model(video=video)

    video = np.arange(0, config.temporal_size * 1080 * 1920 * 3, dtype=np.uint8)
    video = video.reshape(config.temporal_size, 1080, 1920, 3)

    # numpy 전처리 확인
    vis_vector = model(video=video)

    # torch 확인 - torch는 전처리가 되어있다고 가정
    video = torch.rand(
        size=(1, config.temporal_size, 3, 224, 224),
        dtype=torch.float32,
        device=config.device,
    )

    # numpy 전처리 확인
    vis_vector = model(video=video)
    assert vis_vector.shape == (1, 1, 512)

    ######################    TEXT    ###########################
    # 텍스트 하나만 입력
    txt = "self attention is good choice"
    txt_vector = model(text=txt)
    assert txt_vector.shape == (1, 512)

    # 텍스트 여러개 입력
    txts = [txt, txt]
    txt_vector = model(text=txts)
    assert txt_vector.shape == (len(txts), 512)

    # 전처리 후 입력
    txt_ids, txt_mask = model.text_preprocess(txts)
    txt_vector = model(text=txt_ids, txt_mask=txt_mask)
    assert txt_vector.shape == (len(txts), 512)

    ######################    SIMILARITY    ###########################
    similarity = model(video=video, text=txts)

    ######################    TILED    ###########################
    config = T2VRetConfig(
        model_path=model_path,
        device=device,
        tile_config="S",
    )
    model = PiaTorchModel(
        target_task=1,  # T2VRet
        target_model=1,  # clip-vip
        config=config,  # T2VRet  # clipvip
    )
    similarity = model(video=video, text=txts)
    assert similarity.shape == (len(txts), video.shape[0])

    ########## half model load ############
    config = T2VRetConfig(
        model_path=model_path,
        device=device,
        use_half_precision=True,
    )
    model = PiaTorchModel(
        target_task=1,  # T2VRet
        target_model=1,  # clip-vip
        config=config,  # T2VRet  # clipvip
    )


def test_clip_vip_onnx_export(model_path):
    device = load_model_backend("cuda", type="str")
    config = T2VRetConfig(model_path=model_path, device=device)
    model = PiaTorchModel(
        target_task=1,  # T2VRet
        target_model=1,  # clip-vip
        config=config,  # T2VRet  # clipvip
    )
    opset = 14
    # visual
    onnx_config = T2VRetONNXConfig(output_dir=ASSETS_MODEL_SAVE_DIR, split_part="visual", opset=opset)
    model.export(onnx_config)
    # textual
    onnx_config = T2VRetONNXConfig(output_dir=ASSETS_MODEL_SAVE_DIR, split_part="textual", opset=opset)
    model.export(onnx_config)
    # embedding
    onnx_config = T2VRetONNXConfig(output_dir=ASSETS_MODEL_SAVE_DIR, split_part="embedding", opset=opset)
    model.export(onnx_config)


if __name__ == "__main__":
    test_clip_vip("cuda")
    test_clip_vip_onnx_export("cuda")
