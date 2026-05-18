#!/usr/bin/env python3
"""
Download the Qwen3-VL-Embedding FP8 checkpoint and text-features JSON from HuggingFace Hub.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


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


def download_text_features(repo_id: str, file_name: str, dest_dir: Path) -> Path:
    """Download a single file (text-features JSON) into the target directory."""
    print(f"[download] text features: {repo_id}/{file_name} → {dest_dir}")
    local = hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        local_dir=str(dest_dir),
        token=_resolve_token(),
    )
    return Path(local)


# ============================================================================
# Main Execution
# ============================================================================

def main() -> None:
    _load_dotenv_if_present()

    # --- CLI Setup ---
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
        "--text-features-repo-id", default=os.getenv("HF_TEXT_FEATURES_REPO_ID"),
        help="HF Hub repo ID for the text-features JSON. Defaults to --repo-id.",
    )
    parser.add_argument(
        "--text-features-file", default=os.getenv("HF_TEXT_FEATURES_FILE", "VLE_FP8_text_features.json"),
        help="File name of the text-features JSON.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip download if model.safetensors and text-features JSON exist.",
    )
    args = parser.parse_args()

    if not args.repo_id:
        sys.exit("ERROR: --repo-id (or HF_MODEL_REPO_ID env var) is required.")

    # --- Directory & Path Resolution ---
    model_dir = Path(args.model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)

    weights_path = model_dir / "model.safetensors"
    anchors_path = model_dir / args.text_features_file

    weights_present = weights_path.exists()
    anchors_present = anchors_path.exists()

    # --- Download Logic ---
    if args.skip_existing and weights_present and anchors_present:
        print(f"[skip] checkpoint + text features already present at {model_dir}")
        return

    if weights_present:
        print(f"[skip] {weights_path.name} already present")
    else:
        download_checkpoint(args.repo_id, model_dir)

    if anchors_present:
        print(f"[skip] {anchors_path.name} already present")
    else:
        tf_repo = args.text_features_repo_id or args.repo_id
        download_text_features(tf_repo, args.text_features_file, model_dir)

    # --- Final Validation ---
    if not weights_path.exists():
        sys.exit(f"ERROR: weights missing after download: {weights_path}")
    if not anchors_path.exists():
        sys.exit(f"ERROR: text features missing after download: {anchors_path}")

    print(f"[done] model + text features ready at {model_dir}")


if __name__ == "__main__":
    main()