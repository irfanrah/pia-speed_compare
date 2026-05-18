import numpy as np
import torch
from pia.tests.ai.task.T2VRet.models.clip4clip.conftest import (
    setup_dummy_video,
)

# ---------- helpers (NOT collected as tests) ----------
def check_video_vector(model_setup, setup_name: str):
    model, config = model_setup
    dummy_video = setup_dummy_video(sequence_length=config.temporal_size)
    video_feat = model(video=dummy_video)

    assert isinstance(video_feat, torch.Tensor)
    assert video_feat.shape[0] == dummy_video.shape[0]
    print(f"✅ Complete video vector test of {setup_name}")


def check_text_vector(model_setup, dummy_texts, setup_name: str):
    model, _ = model_setup
    text_feat = model(text=dummy_texts)

    assert isinstance(text_feat, torch.Tensor)
    assert text_feat.shape[0] == len(dummy_texts)
    print(f"✅ Complete text vector test of {setup_name}")


def check_similarity(model_setup, dummy_texts, setup_name: str):
    model, config = model_setup
    dummy_video = setup_dummy_video(sequence_length=config.temporal_size)
    sim_score = model(video=dummy_video, text=dummy_texts)
    print(f"sim_score: {sim_score}")

    # accept torch.Tensor or np.ndarray
    if isinstance(sim_score, torch.Tensor):
        sim_score = sim_score.detach().cpu().numpy()
    else:
        sim_score = np.asarray(sim_score)

    assert sim_score.shape == (len(dummy_texts), dummy_video.shape[0])
    assert np.all((sim_score >= 0) & (sim_score <= 1))
    print(f"✅ Complete similarity test of {setup_name}")


# ---------- actual tests (collected by pytest) ----------
def test_pe_video_vector(setup_pe_model, setup_pe_splitqkv_model, setup_pe_FT_lora_model, setup_pe_lora_MHCA_model):
    check_video_vector(setup_pe_model, "PE")
    # check_video_vector(setup_pe_splitqkv_model, "PE split QKV")
    check_video_vector(setup_pe_FT_lora_model, "PE FT LoRA (250804)")
    check_video_vector(setup_pe_lora_MHCA_model, "PE LoRA MHCA (250915)")


def test_pe_text_vector(setup_pe_model, setup_pe_splitqkv_model, setup_pe_FT_lora_model, setup_pe_lora_MHCA_model, setup_dummy_texts):
    check_text_vector(setup_pe_model, setup_dummy_texts, "PE")
    # check_text_vector(setup_pe_splitqkv_model, setup_dummy_texts, "PE split QKV")
    check_text_vector(setup_pe_FT_lora_model, setup_dummy_texts, "PE FT LoRA (250804)")
    check_text_vector(setup_pe_lora_MHCA_model, setup_dummy_texts, "PE LoRA MHCA (250915)")


def test_pe_similarity(setup_pe_model, setup_pe_splitqkv_model, setup_pe_FT_lora_model, setup_pe_lora_MHCA_model, setup_dummy_texts):
    check_similarity(setup_pe_model, setup_dummy_texts, "PE")
    check_similarity(setup_pe_FT_lora_model, setup_dummy_texts, "PE FT LoRA (250804)")
    check_similarity(setup_pe_lora_MHCA_model, setup_dummy_texts, "PE LoRA MHCA (250915)")