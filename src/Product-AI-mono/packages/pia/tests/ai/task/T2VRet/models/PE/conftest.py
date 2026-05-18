from typing import List, Tuple

import pytest
from pia.ai.device import load_model_backend
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.tests.test_config import ASSETS_MODEL_SAVE_DIR
from pia.utils.api.hugging_face import HFModelDownloader


PE_SPLITQKV_REPO_NAME = "PE-Core-L14-336-splitqkv"
PE_SPLITQKV_FILE_NAME = f"{PE_SPLITQKV_REPO_NAME}.pt"

PE_FT_LORA_REPO_NAME = "FT_PE-Core-L14-336_250804"
PE_FT_LORA_FILE_NAME = f"{PE_FT_LORA_REPO_NAME}.pt"
PE_FT_LORA_ADAPTER = PE_FT_LORA_REPO_NAME + "_adapter"

PE_FT_LORA_MHCA_REPO_NAME = "FT_PE-Core-L14-336_MHCA_250915"
PE_FT_LORA_MHCA_FILE_NAME = f"{PE_FT_LORA_MHCA_REPO_NAME}.pt"




PESMALL_FT_LORA_REPO_NAME = "FT_PE-Core-S16-384_260301"
PESMALL_FT_LORA_FILE_NAME = f"{PESMALL_FT_LORA_REPO_NAME}.pt"
PESMALL_FT_LORA_ADAPTER = PESMALL_FT_LORA_REPO_NAME + "_adapter"

@pytest.fixture
def setup_dummy_texts() -> List[str]:    
    texts = [
        "A person in white forcibly pulls a person in yellow, who is on the ground.",
        "Someone in black roughly grabs a seated person in white.",
        "A man in red, appearing drunk, is being held up by someone in beige pants.",
    ]
    return texts


@pytest.fixture(scope="module", autouse=True)
def download_PE_splitqkv_model():
    hf_downloader = HFModelDownloader()
    hf_downloader.download(
        repo_id=PE_SPLITQKV_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        snapshot=True,
    )

@pytest.fixture(scope="module", autouse=True)
def download_PE_FT_lora_model():
    hf_downloader = HFModelDownloader()
    hf_downloader.download(
        repo_id=PE_FT_LORA_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        snapshot=True,
    )


@pytest.fixture(scope="module", autouse=True)
def download_PESmall_FT_lora_model():
    hf_downloader = HFModelDownloader()
    hf_downloader.download(
        repo_id=PESMALL_FT_LORA_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        snapshot=True,
    )



@pytest.fixture(scope="module", autouse=True)
def download_PE_FT_lora_MHCA_model():
    hf_downloader = HFModelDownloader()
    hf_downloader.download(
        repo_id=PE_FT_LORA_MHCA_REPO_NAME,
        save_dir=ASSETS_MODEL_SAVE_DIR,
        snapshot=True,
    )

@pytest.fixture
def setup_pe_model() -> Tuple[PiaTorchModel, T2VRetConfig]:
    device = load_model_backend("cuda", type="str")
    config = T2VRetConfig(
        model_path=None,
        device=device,
        tile_config=None,
        model_name="PE-Core-L14-336",
        temporal_size=8,
        img_size=[None, None],
    )
    model = PiaTorchModel(
        target_task="RET",
        target_model="PerceptionEncoder",
        config=config,
    )
    return model, config
@pytest.fixture
def setup_pe_splitqkv_model() -> Tuple[PiaTorchModel, T2VRetConfig]:
    device = load_model_backend("cuda", type="str")
    config = T2VRetConfig(
        model_path=None,
        device=device,
        tile_config=None,
        model_name="PE-Core-L14-336-splitqkv",
        temporal_size=8,
        img_size=[None, None],
    )
    model = PiaTorchModel(
        target_task="RET",
        target_model="PerceptionEncoder",
        config=config,
    )
    return model, config


@pytest.fixture
def setup_pe_FT_lora_model() -> Tuple[PiaTorchModel, T2VRetConfig]:
    device = load_model_backend("cuda", type="str")
    config = T2VRetConfig(
        model_path=None,
        device=device,
        tile_config=None,
        model_name="FT_PE-Core-L14-336_250804",
        temporal_size=8,
        img_size=[None, None],
    )
    model = PiaTorchModel(
        target_task="RET",
        target_model="PerceptionEncoder",
        config=config,
    )
    return model, config



@pytest.fixture
def setup_pe_lora_MHCA_model() -> Tuple[PiaTorchModel, T2VRetConfig]:
    device = load_model_backend("cuda", type="str")
    config = T2VRetConfig(
        model_path=None,
        device=device,
        tile_config=None,
        model_name="FT_PE-Core-L14-336_MHCA_250915",
        temporal_size=8,
        img_size=[None, None],
    )
    model = PiaTorchModel(
        target_task="RET",
        target_model="PerceptionEncoder",
        config=config,
    )
    return model, config
