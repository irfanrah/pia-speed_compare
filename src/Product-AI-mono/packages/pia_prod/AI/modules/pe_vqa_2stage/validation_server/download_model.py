#!/usr/bin/env python3
"""validation_server vLLM 모델 사전 다운로드 스크립트.

컨테이너 첫 기동 시 모델 다운로드(~3-5분)를 피하기 위해 호스트에서 미리 받아둘 때 사용.
다운로드 위치는 Product-AI-mono의 다른 모듈과 통일된 `<repo_root>/assets/model/huggingface/`.
이 경로를 컨테이너의 `/app/assets/model/huggingface`에 마운트하면 vLLM이 캐시를 재사용한다.

같은 디렉토리의 `.env` 파일을 자동 로드해 `VLLM_MODEL`/`MODEL_HF_CACHE_DIR`/`HF_TOKEN` 등의
설정을 읽는다 (docker compose가 .env를 읽는 것과 동일하게 동작). 따라서 사용자는 .env만
편집하면 download_model.py와 docker compose가 같은 모델/캐시 경로를 사용한다.

Usage:
    pip install huggingface_hub
    python download_model.py                                 # .env 자동 로드, 기본 default 모델 (Qwen/Qwen3.5-0.8B)
    VLLM_MODEL=Qwen/Qwen3.5-<variant> python download_model.py       # 다른 Qwen3.5 변종 override
    MODEL_HF_CACHE_DIR=/custom/path python download_model.py         # 캐시 경로 override
    HF_TOKEN=hf_xxx python download_model.py                          # 비공개 모델
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 컨테이너 ↔ 호스트 양쪽에서 동일하게 사용하는 default 값.
# .env.example / Dockerfile / docker-compose.yml의 default와 일치해야 한다.
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"
CONTAINER_MOUNT_PATH = "/app/assets/model/huggingface"


def _load_dotenv_if_present() -> None:
    """같은 디렉토리의 .env를 환경변수로 로드 (단순 KEY=VALUE 파싱).

    이미 환경변수에 값이 있으면 .env가 덮어쓰지 않는다 (CLI override 우선).
    docker-compose의 `.env` 파일 처리와 동일한 시맨틱 — 외부 의존성 없이 동작.
    """
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # 이미 environ에 있으면 (CLI에서 명시 export됨) 안 덮어씀
        os.environ.setdefault(key, value)


def _resolve_default_cache_dir() -> Path:
    """Product-AI-mono 레포 root의 assets/model/huggingface."""
    here = Path(__file__).resolve().parent
    # validation_server -> pe_vqa_2stage -> modules -> AI -> pia_prod -> packages -> repo_root
    repo_root = here.parents[5]
    return repo_root / "assets" / "model" / "huggingface"


def main() -> int:
    _load_dotenv_if_present()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "[ERROR] huggingface_hub 미설치. 다음으로 설치 후 재시도:\n"
            "  pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    model_id = os.getenv("VLLM_MODEL", DEFAULT_MODEL)
    cache_dir_env = os.getenv("MODEL_HF_CACHE_DIR")
    target = (
        Path(cache_dir_env).expanduser().resolve()
        if cache_dir_env
        else _resolve_default_cache_dir()
    )
    target.mkdir(parents=True, exist_ok=True)

    # vLLM/transformers는 HF_HOME 환경변수를 읽고 그 하위 hub/에서 모델을 찾는다
    # (HF_HUB_CACHE = HF_HOME/hub). 컨테이너의 HF_HOME과 동일한 트리에 저장하기 위해
    # cache_dir을 target/hub로 설정한다. 결과 구조:
    #   <target>/hub/models--<owner>--<repo>/...
    hub_dir = target / "hub"
    hub_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("HF_TOKEN") or None

    print(f"[download] model     : {model_id}")
    print(f"[download] HF_HOME   : {target}")
    print(f"[download] hub cache : {hub_dir}")
    print(f"[download] hf_token  : {'set' if token else '(none)'}")
    print("[download] start ... (~3-5분 소요, 모델 크기에 따라 다름)")

    snapshot_path = snapshot_download(
        repo_id=model_id,
        cache_dir=str(hub_dir),
        token=token,
    )

    print()
    print("[download] OK")
    print(f"[download] snapshot  : {snapshot_path}")
    print()
    print("validation_server 컨테이너에 다음과 같이 마운트하면 vLLM이 캐시를 재사용한다:")
    print(f"  docker run -v {target}:{CONTAINER_MOUNT_PATH} ... pe-vqa-2stage:latest")
    print(f"  또는 .env에 MODEL_HF_CACHE_DIR={target} 설정 후 docker compose up")
    return 0


if __name__ == "__main__":
    sys.exit(main())
