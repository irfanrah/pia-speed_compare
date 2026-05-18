import os
from pathlib import Path

import gdown


def google_drive_get_model(save_dir, save_file_name: str, file_id: str):
    if not os.path.exists(os.path.join(save_dir, save_file_name)):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        gdown.download(
            url=f"https://drive.google.com/u/0/uc?id={file_id}&export=download",
            output=os.path.join(save_dir, save_file_name),
            quiet=False,
        )
