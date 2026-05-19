# Coding Convention

## 네이밍 규칙

### 클래스
- **CamelCase** 사용
- Service: `{Name}Service` (예: `FallService`, `HelmetService`, `InternVL3TrtLlmService`)
- EventManager: `{Name}EventManager` (예: `VQAFalldownEventManager`)
- RoIManager: `{Name}RoIManager` (예: `VQAFalldownRoIManager`)
- DTO: `{Name}Model` / `{Name}Base` (예: `AddStreamModel`, `CategoryBase`, `ROIModel`)

### 함수 / 변수
- **snake_case** 사용
- private 메서드: `_method_name()`
- 예: `_load_model()`, `_detect()`, `_init_values()`, `_match_category2prompt()`

### 상수
- **UPPER_SNAKE_CASE** 사용
- 예: `SUPPORT_CATEGORIES`, `FALLDOWN_CATEGORY`, `MODEL_OUTPUTS`, `QUEUE_SIZE`

### 환경변수
- config.py에서 `os.getenv()` 또는 `os.environ.get()`으로 관리
- 접두사로 모듈 구분 (예: `VQA_LLM_MAX_BATCH_SIZE`, `VQA_VIT_DTYPE`)

---

## 임포트 순서

1. 표준 라이브러리
2. 서드파티 (numpy, torch, cv2, pydantic 등)
3. pia 패키지 (공통 AI 패키지)
4. pia_prod 패키지

```python
# 1. 표준 라이브러리
import os
from typing import List
from collections import defaultdict, deque

# 2. 서드파티
import numpy as np
import torch

# 3. pia 패키지 (공통)
from pia.ai.tasks.VQA.base import VQAConfig
from pia.ai.model import PiaTorchModel
from pia.utils.devtools.debug_tools import print_only_debug_mode

# 4. pia_prod 패키지
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.global_config import USER_PARAM_KEY, VQA_EVENT_KEY
```

### 규칙
- pia / pia_prod 패키지는 **절대 경로** 사용
- 같은 모듈 내에서도 절대 경로 사용 (예: `from pia_prod.AI.modules.vqa_internvl.config import ...`)
- 다른 모듈의 config를 import하지 않는다 — 자체 모듈 config에서 가져올 것

---

## 타입 힌트

함수 시그니처에 타입 힌트를 사용한다.

```python
def update(self, responses: List[str], categories_per_stream: List[str], stream_ids: List[str]):
    ...

def process_batches_with_roi(self, batches: List[np.array], user_params: List) -> List[np.array]:
    ...
```

---

## Docstring

Google 스타일을 사용한다.

```python
class VQAFalldownRoIManager(ROIManagerBase):
    """
    RoI(Region of Interest)을 관리하는 클래스.

    역할:
    - vqaEvent 기반 ROI 정보를 카메라별로 캐시하고 변경 시 갱신
    - ROI polygonCoordinates 기준으로 ROI 영역만 crop
    """
```

---

## 에러 처리

### 데코레이터 패턴
모델 로딩 등에서 `@raise_exception_decorator`를 사용한다.

```python
from pia.utils.exception.model_handler import raise_exception_decorator

@raise_exception_decorator(FileNotFoundError)
def _load_model(self):
    ...
```

### 선택적 의존성
런타임에만 필요한 패키지는 try-except로 처리한다.

```python
try:
    from tensorrt_llm import ...
except Exception:
    ...
```

---

## 테스트

### 위치
- `packages/pia_prod/AI/tests/modules/` 하위에 모듈별 테스트

### 규칙
- 파일명: `test_{module_name}.py`
- 프레임워크: pytest
- 영상 소스: `nas_downloader` fixture로 NAS에서 다운로드
- 모델 다운로드: `hf_downloader` fixture 사용
- 서비스 종료: `stop_thread()` 헬퍼로 정리
- 환경변수: `conftest.py`의 `set_env` fixture에서 `TEAM=ai` 설정

```python
@pytest.fixture(scope="module")
def local_video_path(nas_downloader, video_save_dir):
    video_name = "test_video.mp4"
    local_path = os.path.join(video_save_dir, video_name)
    nas_video_path = nas_downloader.get_nas_path(video_name)
    nas_downloader.download_file(nas_video_path, local_path)
    return local_path
```
