"""HuggingFace 의 fire prompt 데이터와 config 의 파일명이 일치하는지 검증.

PR #410 Docker 검증에서 qwen3_vl_embedding / two_stage_pe_qwen3vle 의 fire
관련 5개 테스트가 ``ValueError: torch.cat(): expected a non-empty list of
Tensors`` 로 실패. 추적 결과 원인은 다음 mismatch 였음.

config 가 기대 (단일 파일):
    text_prompt_APOv2.2.1/fire_pred_prompts/normal/normal.pt
    text_prompt_APOv2.2.1/fire_pred_prompts/fire/fire.pt

HF (PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8) 에 실제 존재 (인덱스 붙은 다수):
    fire_pred_prompts/normal/normal_0.pt, normal_1.pt, ...
    fire_pred_prompts/fire/fire_16.pt, fire_17.pt, ...

비교 (정상 동작하는 smoke):
    text_prompt_APOv2.2.1/smoke_pred_prompts/normal/normal.pt  -- 단일, 일치

이 테스트는 service 인스턴스화 / 모델 로딩 없이 HF API 와 config 변수만으로
mismatch 를 reproducer 로 증명한다. (요구 의존성: requests 만 — 베이스 이미지
공통 deps).
"""

from __future__ import annotations

import os
from typing import List, Set

import pytest
import requests

HF_REPO = "PIA-SPACE-LAB/Qwen3-VL-Embedding-2B-FP8"
HF_API = "https://huggingface.co/api/models"


def _hf_listdir(rel_path: str) -> List[str]:
    """HF 레포의 한 디렉토리 내 파일 이름 목록 (basename) 을 가져온다."""
    url = f"{HF_API}/{HF_REPO}/tree/main/{rel_path}"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    return [
        os.path.basename(entry["path"])
        for entry in resp.json()
        if entry.get("type") == "file" and not entry["path"].endswith(".DS_Store")
    ]


@pytest.fixture(scope="module")
def hf_fire_files() -> dict:
    """HF 의 fire_pred_prompts 하위 normal/fire 파일 목록."""
    return {
        "normal": _hf_listdir("text_prompt_APOv2.2.1/fire_pred_prompts/normal"),
        "fire": _hf_listdir("text_prompt_APOv2.2.1/fire_pred_prompts/fire"),
    }


@pytest.fixture(scope="module")
def hf_smoke_files() -> dict:
    """HF 의 smoke_pred_prompts (정상 동작하는 비교군)."""
    return {
        "normal": _hf_listdir("text_prompt_APOv2.2.1/smoke_pred_prompts/normal"),
        "smoke": _hf_listdir("text_prompt_APOv2.2.1/smoke_pred_prompts/smoke"),
    }


def _config_basenames(list_attr: str) -> Set[str]:
    """config 의 PROMPT 리스트에서 basename 만 추출."""
    from pia_prod.AI.modules.qwen3_vl_embedding import config as cfg

    rel_paths: List[str] = getattr(cfg, list_attr)
    return {os.path.basename(p) for p in rel_paths}


def test_hf_smoke_files_match_config(hf_smoke_files):
    """비교군: smoke 는 정상 동작 — config 와 HF 의 파일명이 일치해야 함."""
    config_normal = _config_basenames("LIST_OF_NORMAL_SMOKE_TXT_PROMPTS")
    config_smoke = _config_basenames("LIST_OF_TARGET_SMOKE_TXT_PROMPTS")

    hf_normal = set(hf_smoke_files["normal"])
    hf_smoke = set(hf_smoke_files["smoke"])

    assert config_normal.issubset(hf_normal), (
        f"smoke/normal: config 가 가리키는 {config_normal} 가 HF 의 {hf_normal} "
        f"에 없음 — 정상 동작하는 비교군이 깨짐"
    )
    assert config_smoke.issubset(hf_smoke), (
        f"smoke/smoke: config 가 가리키는 {config_smoke} 가 HF 의 {hf_smoke} "
        f"에 없음 — 정상 동작하는 비교군이 깨짐"
    )


def test_hf_fire_files_match_config(hf_fire_files):
    """버그 reproducer: fire 의 config 가 HF 에 실제로 없는 파일명을 가리키는 것을 증명.

    이 테스트는 PR #410 의 5+5=10 fail 의 정확한 근본 원인을 다음 단언으로 고정한다:
    - config 가 기대: ``normal.pt``, ``fire.pt`` (단일)
    - HF 에 존재: ``normal_<idx>.pt``, ``fire_<idx>.pt`` (인덱스 붙은 다수)

    이 mismatch 가 해결되면 (publisher 가 normal.pt/fire.pt 단일 파일을
    업로드하거나, config 가 인덱스 붙은 파일명을 가리키도록 수정되면) 테스트는
    통과한다.
    """
    config_normal = _config_basenames("LIST_OF_NORMAL_FIRE_TXT_PROMPTS")
    config_fire = _config_basenames("LIST_OF_TARGET_FIRE_TXT_PROMPTS")

    hf_normal = set(hf_fire_files["normal"])
    hf_fire = set(hf_fire_files["fire"])

    missing_normal = config_normal - hf_normal
    missing_fire = config_fire - hf_fire

    assert not missing_normal, (
        f"fire/normal: config 가 가리키는 {sorted(missing_normal)} 가 HF 에 없음.\n"
        f"  HF 에 실제로 있는 파일: {sorted(hf_normal)}\n"
        f"  → publisher 가 단일 파일 업로드 또는 config 수정 필요"
    )
    assert not missing_fire, (
        f"fire/fire: config 가 가리키는 {sorted(missing_fire)} 가 HF 에 없음.\n"
        f"  HF 에 실제로 있는 파일: {sorted(hf_fire)}\n"
        f"  → publisher 가 단일 파일 업로드 또는 config 수정 필요"
    )


def test_hf_fire_directory_not_empty(hf_fire_files):
    """fire_pred_prompts 디렉토리 자체에 파일이 없는 케이스 (가설 확인)."""
    assert hf_fire_files["normal"], "fire_pred_prompts/normal/ 가 비어있음"
    assert hf_fire_files["fire"], "fire_pred_prompts/fire/ 가 비어있음"
