from typing import Tuple

import numpy as np
import torch
from pia.ai.model import PiaTorchModel
from pia.ai.tasks.T2VRet.base import T2VRetConfig
from pia.tests.ai.task.T2VRet.models.qwen3_vl_embedding.conftest import setup_dummy_video


# ---------- helpers (NOT collected as tests) ----------
def check_video_vector(model_setup: Tuple[PiaTorchModel, T2VRetConfig], setup_name: str):
    model, config = model_setup
    dummy_video = setup_dummy_video(sequence_length=config.temporal_size)
    video_feat = model(video=dummy_video, text=None)

    assert isinstance(video_feat, torch.Tensor)
    assert video_feat.ndim == 2
    assert video_feat.shape[0] == 1
    print(f"✅ Complete video vector test of {setup_name}")


def check_text_vector(model_setup: Tuple[PiaTorchModel, T2VRetConfig], dummy_texts, setup_name: str):
    model, _ = model_setup
    text_feat = model(video=None, text=dummy_texts)

    assert isinstance(text_feat, torch.Tensor)
    assert text_feat.ndim == 2
    assert text_feat.shape[0] == len(dummy_texts)
    print(f"✅ Complete text vector test of {setup_name}")


def check_similarity(model_setup: Tuple[PiaTorchModel, T2VRetConfig], dummy_texts, setup_name: str):
    model, config = model_setup
    dummy_video = setup_dummy_video(sequence_length=config.temporal_size)
    sim_score = model(video=dummy_video, text=dummy_texts)
    print(f"sim_score: {sim_score}")

    if isinstance(sim_score, torch.Tensor):
        sim_score = sim_score.detach().cpu().numpy()
    else:
        sim_score = np.asarray(sim_score)

    assert sim_score.shape == (len(dummy_texts), 1)
    assert np.all((sim_score >= -1) & (sim_score <= 1))
    print(f"✅ Complete similarity test of {setup_name}")


# ---------- actual tests (collected by pytest) ----------
def test_qwen3vl_video_vector(setup_qwen3vl_model):
    check_video_vector(setup_qwen3vl_model, "Qwen3-VL-Embedding-2B")


def test_qwen3vl_fp8_video_vector(setup_qwen3vl_fp8_model):
    check_video_vector(setup_qwen3vl_fp8_model, "Qwen3-VL-Embedding-2B-FP8")


def test_qwen3vl_text_vector(setup_qwen3vl_model, setup_dummy_texts):
    check_text_vector(setup_qwen3vl_model, setup_dummy_texts, "Qwen3-VL-Embedding-2B")


def test_qwen3vl_fp8_text_vector(setup_qwen3vl_fp8_model, setup_dummy_texts):
    check_text_vector(setup_qwen3vl_fp8_model, setup_dummy_texts, "Qwen3-VL-Embedding-2B-FP8")


def test_qwen3vl_similarity(setup_qwen3vl_model, setup_dummy_texts):
    check_similarity(setup_qwen3vl_model, setup_dummy_texts, "Qwen3-VL-Embedding-2B")


def test_qwen3vl_fp8_similarity(setup_qwen3vl_fp8_model, setup_dummy_texts):
    check_similarity(setup_qwen3vl_fp8_model, setup_dummy_texts, "Qwen3-VL-Embedding-2B-FP8")