import shutil
from pathlib import Path

from pia.utils.api.check_internet import require_internet
from pia.utils.api.gdrive import google_drive_get_model

DUMMY_GDRIVE_ID = "1odbg-qGJ0EKnmp6SZcQH2v4GfCC4bXSsEOq6QwPJdmg"
FILE_NAME = "dummy_gdrive_file.docx"

@require_internet
def test_google_drive_get_model(tmp_path: Path):
    model_path = tmp_path / FILE_NAME

    # 다운로드 시도
    google_drive_get_model(
        save_dir=tmp_path,
        save_file_name=FILE_NAME,
        file_id=DUMMY_GDRIVE_ID,
    )

    # 파일이 존재하는지 확인
    assert model_path.exists(), "Model file was not downloaded"

    # 파일 크기가 0보다 큰지 확인 (빈 파일이 아닌지)
    assert model_path.stat().st_size > 0, "Downloaded file is empty"

    # 테스트 후 임시 디렉토리 삭제
    shutil.rmtree(tmp_path)
    assert not tmp_path.exists()
