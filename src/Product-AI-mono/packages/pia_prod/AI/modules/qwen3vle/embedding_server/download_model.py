#!/usr/bin/env python3
"""
Download the Qwen3-VL-Embedding FP8 checkpoint from HuggingFace Hub.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


# ============================================================================
# Helper Functions
# ============================================================================

def _load_dotenv_if_present() -> None:
    """Load `.env` next to this script into os.environ if it exists."""
    env_file = Path(__file__).resolve().parent / ".env"
    if not env_file.exists():
        return

    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _resolve_token() -> str | None:
    """Retrieve the HuggingFace auth token from the environment."""
    return os.getenv("HF_AUTH_TOKEN") or os.getenv("HF_TOKEN") or None


# ============================================================================
# Download Operations
# ============================================================================

def download_checkpoint(repo_id: str, model_dir: Path) -> None:
    """Snapshot-download the FP8 checkpoint into the target directory."""
    print(f"[download] checkpoint: {repo_id} → {model_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(model_dir),
        token=_resolve_token(),
    )


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    _load_dotenv_if_present()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir", required=True,
        help="Local directory for the checkpoint snapshot.",
    )
    parser.add_argument(
        "--repo-id", default=os.getenv("HF_MODEL_REPO_ID"),
        help="HF Hub repo ID for the FP8 checkpoint.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip download if model.safetensors already exists.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        sys.exit("ERROR: --repo-id (or HF_MODEL_REPO_ID env var) is required.")

    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "model.safetensors"

    if args.skip_existing and weights_path.exists():
        print(f"[skip] checkpoint already present at {model_dir}")
        return

    if weights_path.exists():
        print(f"[skip] {weights_path.name} already present")
    else:
        download_checkpoint(args.repo_id, model_dir)

    if not weights_path.exists():
        sys.exit(f"ERROR: weights missing after download: {weights_path}")

    print(f"[done] model ready at {model_dir}")


if __name__ == "__main__":
    main()
