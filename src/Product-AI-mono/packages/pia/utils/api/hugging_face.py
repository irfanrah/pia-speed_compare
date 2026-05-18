import os

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.utils import HfHubHTTPError


class HFModelDownloader:
    def __init__(self, namespace: str = None, auth_token: str = None, cache_dir: str = "assets/"):
        self.namespace = namespace or os.getenv("HF_NAMESPACE")
        self.auth_token = auth_token or os.getenv("HF_AUTH_TOKEN")
        self.cache_dir = cache_dir

    @staticmethod
    def extract_model_name(path: str) -> str:
        """경로에서 모델 파일명 또는 디렉토리 이름 추출."""
        return os.path.basename(path).replace(".engine", "").replace(".pt", "").replace(".onnx", "")

    def download_snapshot(self, repo_id: str, save_dir: str = None) -> str:
        if save_dir is None:
            save_dir = os.path.join(self.cache_dir, os.path.basename(repo_id))
        else :
            save_dir = os.path.join(save_dir, repo_id)
        if os.path.isdir(save_dir):
            print(f"✅파일이 이미 존재합니다: {save_dir}")
            return save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"⬇️ 스냅샷 다운로드 시도: {repo_id} → {save_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=save_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                token=self.auth_token,
            )
            print(f"✅ 다운로드 완료: {save_dir}")
        except Exception as e:
            print(f"❌ 스냅샷 다운로드 실패: {repo_id} → {e}")
        return save_dir

    def download_file(self, repo_id: str, file_name: str, save_dir: str = None) -> str:
        if save_dir is None:
            save_dir = self.cache_dir

        save_path = os.path.join(save_dir, file_name)
        if os.path.isfile(save_path):
            print(f"✅파일이 이미 존재합니다: {save_path}")
            return save_path
        os.makedirs(save_dir, exist_ok=True)
        print(f"⬇️ 다운로드 시도: {repo_id}/{file_name} → {save_dir}")
        try:
            path = hf_hub_download(
                repo_id=repo_id,
                filename=file_name,
                local_dir=save_dir,
                token=self.auth_token,
            )
            print(f"✅ 다운로드 완료: {path}")
            return save_path
        except HfHubHTTPError as e:
            print(f"❌ HTTPError 파일 다운로드 실패: {repo_id}/{file_name} → {e}")
        except Exception as e:
            print(f"❌ Exception 파일 다운로드 실패: {repo_id}/{file_name} → {e}")

    def download(
        self, repo_id: str, save_dir: str = None, file_name=None, snapshot: bool = False
    ) -> str:
        """모델 파일 또는 스냅샷을 다운로드"""
        if (
            "/" not in repo_id
        ):  # repo_id에서 네임스페이스가 명시되어 있으면 그대로 repo_id로 반영, 아니면 namespace와 결합
            repo_id = f"{self.namespace}/{repo_id}"
        else:
            repo_id = repo_id

        if snapshot:
            save_path = self.download_snapshot(repo_id, save_dir)
        else:
            save_path = self.download_file(repo_id, file_name, save_dir)
        return save_path
