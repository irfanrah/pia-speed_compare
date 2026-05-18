import os
import sys
import time
from dataclasses import dataclass

import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError


@dataclass
class NasInfoBaseConfig:
    NAS_PATH_FILE: str = None
    NAS_ID: str = None
    NAS_PW: str = None


@dataclass
class NasConfig(NasInfoBaseConfig):
    NAS_PATH_FILE: str = "http://172.168.47.36/Package-Common-AI-pia_ai_package/"
    NAS_ID: str = "piaspace"
    NAS_PW: str = "piaspace@418"
    DUMMY_FILE: str = "dummy/dummy.txt"


class NasManager:
    LOGGING_TIME = 1  # seconds

    def __init__(self, config: NasConfig):
        self.config = config

    def get_nas_path(self, file_path: str) -> str:
        return os.path.join(self.config.NAS_PATH_FILE, file_path)

    def get_nas_id(self) -> str:
        return self.config.NAS_ID

    def get_nas_pw(self) -> str:
        return self.config.NAS_PW

    def _get_response(self, url=None) -> requests.Response:
        url = self.config.NAS_PATH_FILE + "/" + self.config.DUMMY_FILE if url is None else url  # Default NAS path
        auth = HTTPBasicAuth(self.get_nas_id(), self.get_nas_pw())
        try:
            response = requests.get(url, auth=auth, stream=True)
            response.raise_for_status()
        except HTTPError as e:
            print(f"Can't find file : {url} -> {e}")
            sys.exit(1)

    def download_file(self, url: str, save_path: str, overwrite=False) -> str:
        if os.path.isfile(save_path) and overwrite is False:
            print(f"✅파일이 이미 존재합니다: {save_path}")
            return save_path
        auth = HTTPBasicAuth(self.get_nas_id(), self.get_nas_pw())
        response = requests.get(url, auth=auth, stream=True)
        try:
            response.raise_for_status()
        except HTTPError as e:
            print(f"Can't find file : {url} -> {e}")
            sys.exit(1)

        total_downloaded = 0
        chunk_size = 8192  # 8KB
        last_log_time = time.time()
        if not os.path.isdir(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    total_downloaded += len(chunk)

                    now = time.time()
                    if now - last_log_time >= NasManager.LOGGING_TIME:
                        if total_downloaded < 1024 * 1024:
                            msg = f"Downloaded: {total_downloaded / 1024:.2f} KB"
                        else:
                            msg = f"Downloaded: {total_downloaded / (1024 * 1024):.2f} MB"
                        print(f"\r{msg:<40}", end="")
                        last_log_time = now

        print(f"✅ 다운로드 완료: {save_path}")
        return save_path

    # def download_nas_video(self, local_video_save_dir: str, video_name: str) -> str:
    #     local_video_path = os.path.join(local_video_save_dir, video_name)
    #     if not os.path.isfile(local_video_path):
    #         nas_url = self.get_nas_path(video_name)
    #         self.download_file(nas_url, local_video_path)
    #     return local_video_path
