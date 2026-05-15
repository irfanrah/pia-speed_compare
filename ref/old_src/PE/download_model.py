import os
import shutil

from huggingface_hub import hf_hub_download

from .config import HF_ONNX_FILENAME, HF_REPO_ID, PERCEPTION_ENCODER_ONNX_PATH


def ensure_onnx(local_path: str = PERCEPTION_ENCODER_ONNX_PATH) -> str:
    if os.path.exists(local_path):
        print(f"ONNX already present at {local_path}, skipping download.")
        return local_path

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    print(f"Downloading {HF_ONNX_FILENAME} from {HF_REPO_ID} ...")
    cached = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_ONNX_FILENAME)

    if os.path.abspath(cached) != os.path.abspath(local_path):
        shutil.copy(cached, local_path)

    size_mb = os.path.getsize(local_path) / 1024 / 1024
    print(f"ONNX ready at {local_path} ({size_mb:.2f} MB)")
    return local_path


if __name__ == "__main__":
    ensure_onnx()
