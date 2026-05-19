# New Category Guide

> **대상 독자**: 새로운 AI 감지 카테고리를 추가해야 하는 개발자  
> **참고 자료**: 실제 Merge된 PR (`aramco_loitering` PR#361/362, `vanguard_patient` PR#359) 분석 기반  
> **사전 준비**: [repo_structure.md](./repo_structure.md) 를 먼저 읽어주세요

---

## 1. 작업 흐름 전체 요약

신규 카테고리 추가는 아래 **6단계**를 순서대로 진행합니다.

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ STEP 1  │───→│ STEP 2   │───→│ STEP 3   │───→│ STEP 4   │───→│ STEP 5   │───→│ STEP 6   │
│ 모듈     │    │ 핵심 파일  │    │ 시스템     │    │ 모델 파일  │    │ 테스트     │    │ PR 제출   │
│ 디렉토리  │    │ 구현      │    │ 등록      │    │ 배치      │    │ 작성       │    │   │
│ 생성     │    │ (6개 필수) │    │ (2개 수정)│    │          │    │           │    │     │
└─────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                
   신규 생성        신규 생성       기존 파일 수정      assets/에      tests/에       브랜치 →
   1개 디렉토리     6개 파일        2개 파일            모델 배치      테스트 작성    커밋 → PR
```

**최종 결과**: 신규 6개 파일 생성 + 기존 2개 파일 수정 + 테스트 1개 파일 = **총 9개 파일**

---

## 2. 수정이 필요한 파일 한눈에 보기

아래 다이어그램에서 `[NEW]`는 새로 생성, `[MOD]`는 기존 파일 수정을 의미합니다.

```
packages/pia_prod/AI/
│
├─ __init__.py                              [MOD] 3곳 수정 (STEP 3-1)
│
├─ DTO/
│  └─ stream_params.py                      [MOD] 4곳 수정 (STEP 3-2)
│
├─ modules/
│  └─ <새_모듈명>/                           [NEW] 디렉토리 생성
│     ├─ __init__.py                        [NEW] 빈 파일 (STEP 2-6)
│     ├─ config.py                          [NEW] 설정 상수 (STEP 2-1)
│     ├─ param.py                           [NEW] 파라미터 모델 (STEP 2-2)
│     ├─ event.py                           [NEW] 이벤트 매니저 (STEP 2-3)
│     ├─ service.py                         [NEW] 서비스 클래스 (STEP 2-4)
│     └─ roi_manager.py                     [NEW] ROI 매니저 (STEP 2-5)
│
└─ tests/modules/
   └─ test_<새_모듈명>.py                   [NEW] 테스트 코드 (STEP 5)
```

---

## 3. STEP 1 - 모듈 디렉토리 생성

### 네이밍 규칙

```
┌─────────────────────────────────────────────────────────────────────┐
│ 형식:  <프로젝트/고객사명>_<카테고리명>                             │
│                                                                     │
│ 예시:                                                               │
│ aramco_loitering       → 아람코 프로젝트의 배회 감지                │
│ kumho_pinch            → 금호타이어 프로젝트의 협착 감지            │
│ vanguard_patient       → 뱅가드 헬스케어의 환자 배회                │
│ khonkaen_helmet        → 콘깬 프로젝트의 헬멧 미착용 감지           │
│                                                                     │
│ 특수 케이스:                                                      │
│ 한 모델이 여러 카테고리 커버 시 → 모델명 기준                     │
│ 예: samsung_internvl3  → 삼성 프로젝트의 InternVL3 범용 모델      │
└─────────────────────────────────────────────────────────────────────┘
```

### 실행

```bash
mkdir -p packages/pia_prod/AI/modules/<새_모듈명>
touch packages/pia_prod/AI/modules/<새_모듈명>/__init__.py
```

---

## 4. STEP 2 - 핵심 파일 구현 (6개 필수)

### 4.1 `config.py` - 설정 상수 정의

모든 모듈의 설정값을 한 곳에 모아 관리합니다. 실제 `aramco_loitering/config.py` 를 참고한 템플릿입니다.

```python
import os

# ─────────────────────────────────────────────────────────
# 디바이스
# ─────────────────────────────────────────────────────────
DEVICE = "cuda"

# ─────────────────────────────────────────────────────────
# 모델 경로
#   반드시 os.environ.get() 또는 os.getenv()로 감싸서
#   환경변수로 오버라이드 가능하도록 합니다.
# ─────────────────────────────────────────────────────────
MODEL_ONNX_PATH = os.getenv(
    "MODEL_YOUR_DETECTION_ONNX_PATH",
    "assets/model/YourModel_v0.1.0.onnx"
)
MODEL_TRT_PATH = os.getenv(
    "MODEL_YOUR_DETECTION_TRT_PATH",
    "assets/model/YourModel_v0.1.0.engine"
)

# ─────────────────────────────────────────────────────────
# Object Detection 파라미터
# ─────────────────────────────────────────────────────────
OD_INPUT_SIZE = [640, 640]              # 모델 입력 크기 [H, W]
OD_CONFIDENCE_THRESHOLD = 0.5           # OD confidence 임계값
OD_NMS_THRESHOLD = 0.35                 # NMS IoU 임계값
OD_TARGET_CLASSES = [0]                 # 타겟 클래스 인덱스
OD_TIME_INTERVAL_SECOND = 0.3           # 추론 시간 간격(초)

# ─────────────────────────────────────────────────────────
# 리소스 제한
# ─────────────────────────────────────────────────────────
DEFAULT_FPS = 15
LIMITED_NUM_OF_CAMERA = int(os.getenv("LIMITED_NUM_OF_CAMERA", "16"))
LIMITED_NUM_OF_PERSON_PER_CAMERA = int(os.getenv("LIMITED_NUM_OF_PERSON_PER_CAMERA", 16))

# ─────────────────────────────────────────────────────────
# 카테고리 이름 (한글 + 영문)
#   백엔드 API에서 전달되는 name 필드와 매칭됩니다.
#   반드시 한글/영문 모두 포함해야 합니다.
# ─────────────────────────────────────────────────────────
YOUR_CV_CATEGORY = ["카테고리_cv", "category_cv"]

# ─────────────────────────────────────────────────────────
# 디버그
# ─────────────────────────────────────────────────────────
IMAGE_SAVE_PATH = os.getenv("IMAGE_SAVE_PATH", "logs")
```

> **주의사항**
> - `torch.device("cuda:0")` 같은 하드코딩 금지 → `DEVICE` 상수 또는 환경변수 사용
> - 모델 경로는 반드시 `os.getenv()` 래핑
> - 카테고리 이름 리스트에는 한글/영문 모두 포함

---

### 4.2 `param.py` - 파라미터 모델 정의

`CategoryBase`를 상속하여 카테고리 고유 파라미터를 정의합니다. 실제 `aramco_loitering/param.py`를 참고합니다.

```python
from pia_prod.AI.DTO.param_base import CategoryBase
from pia_prod.AI.modules.<모듈명>.config import (
    OD_CONFIDENCE_THRESHOLD,
    OD_NMS_THRESHOLD,
)


class YourModel(CategoryBase):
    confidence_threshold: float = OD_CONFIDENCE_THRESHOLD
    nms_threshold: float = OD_NMS_THRESHOLD
    # 카테고리 고유 파라미터가 있으면 여기에 추가
```

**CategoryBase가 이미 제공하는 기본 필드:**

```
┌───────────────────────────────────────────────────────────────────┐
│ CategoryBase (param_base.py)                                      │
├───────────────────────┬───────────────────────────────────────────┤
│ 필드                   │ 설명                                     │
├───────────────────────┼───────────────────────────────────────────┤
│ name: str             │ 카테고리 식별자 (예: "helmet_cv")         │
│ incidentThresholdSecond: int │ 이벤트 판정 시간 (초)              │
│ incidentTimeoutSecond: int   │ 이벤트 타임아웃 (초)               │
│ roi: Optional[ROIModel]      │ ROI 폴리곤 좌표                    │
│   ├─ roiId: int              │   ROI 고유 ID                      │
│   ├─ polygonCoordinates: []  │   폴리곤 꼭짓점 좌표 리스트        │
│   └─ divideCoordinates: []   │   분할선 좌표 리스트               │
└───────────────────────┴───────────────────────────────────────────┘
```

> ROI는 다양한 형태로 입력되어도 `validate_roi()` 에서 자동 변환됩니다 (리스트, dict, 중첩 등).

---

### 4.3 `event.py` - 이벤트 매니저 구현

`EventBase`를 상속하여 감지 결과를 이벤트 상태 전이로 변환합니다.

> **중요**: 실제 구현에서는 `update()`와 `get_alarm()` 을 **분리**하여 사용하는 패턴이 주로 쓰입니다.
> - `update()`: 내부 상태만 갱신 (반환값 없음)
> - `get_alarm()`: 상태 전이 행렬을 적용하여 알람 dict 반환
>
> EventBase의 추상 메서드 선언에는 `update()`가 `[["stream_id", is_start]]`를 반환하도록 되어 있지만,
> 실제 모듈들은 위의 분리 패턴을 따릅니다.

**실제 구현 예시** (`aramco_loitering/event.py` 핵심 패턴):

```python
from collections import defaultdict
from pia_prod.AI.bases.event_base import EventBase


class YourEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "category_cv"

        # 스트림별 상태 추적
        self.event_status = defaultdict(int)    # {stream_id: 상태코드}
        # 카테고리 특화 데이터 (예: 큐, 카운터 등)
        # ...

    def update(self, results, stream_id, **kwargs):
        """
        감지 결과를 받아 내부 상태만 갱신합니다. (반환값 없음)
        실제 알람 판정은 get_alarm()에서 수행합니다.
        """
        # 여기서 카테고리 특화 로직 수행
        # 예: 큐에 결과 누적, 트래커 ID별 점수 관리 등
        pass

    def get_alarm(self) -> dict:
        """
        STATUS_TRANSITION 행렬을 적용하여 상태를 전이시키고,
        전이가 발생한 스트림에 대해서만 알람을 반환합니다.

        service.py의 _detect() 에서 update() 호출 후 이 메서드를 호출합니다.
        """
        alarms = {}
        for stream_id, data in self.내부데이터.items():
            is_trigger = (판정 로직)  # True 또는 False

            # ★ 핵심: 상태 전이 행렬 적용
            before_status = self.event_status[stream_id]
            now_status = self.STATUS_TRANSITION[before_status][int(is_trigger)]
            self.event_status[stream_id] = now_status

            # 상태 1(시작) 또는 3(종료)일 때만 알람 발생
            if now_status in [1, 3]:
                alarms[stream_id] = [self.EVENT_STATUS_DICT[now_status], None]

        return alarms
```

**service.py에서의 호출 패턴:**

```python
# _detect() 내부에서
self.alarm_event_manager.update(results=True, stream_id=stream_id, ...)  # 상태 갱신만
alarms = self.alarm_event_manager.get_alarm()  # 알람 판정은 별도 호출
```

**상태 전이 행렬 활용 핵심:**

```
                           is_trigger
                        False(0)  True(1)
                        ┌────────┬────────┐
  현재상태 0 (no_event) │ → 0    │ → 1 ★  │  ★ = 알람 발생
                        ├────────┼────────┤
  현재상태 1 (True)     │ → 3 ★  │ → 2    │
                        ├────────┼────────┤
  현재상태 2 (continue) │ → 3 ★  │ → 2    │
                        ├────────┼────────┤
  현재상태 3 (False)    │ → 0    │ → 1 ★  │
                        └────────┴────────┘

사용법:
  now_status = self.STATUS_TRANSITION[before_status][int(is_trigger)]
```

---

### 4.4 `service.py` - 서비스 클래스 구현 (핵심 파일)

`ServiceBase`의 **4개 추상 메서드**를 구현합니다. 이 파일이 모듈의 핵심입니다.

**실제 구현 패턴** (`aramco_loitering/service.py` 기반):

```python
from pia_prod.AI.bases.service_base import ServiceBase
from pia_prod.AI.modules.<모듈명>.config import (
    DEVICE, MODEL_TRT_PATH, OD_INPUT_SIZE, OD_CONFIDENCE_THRESHOLD, ...
)
from pia_prod.AI.modules.<모듈명>.event import YourEventManager
from pia_prod.AI.modules.<모듈명>.roi_manager import YourRoIManager
from pia.vision.preprocessing.resize import LetterBoxTorch
from pia.ai.model import PiaONNXTensorRTModel
from pia_prod.AI.utils.utils import free_autobackend


class YourService(ServiceBase):

    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)    # ← 부모 생성자가 4개 추상 메서드를 순서대로 호출
        self.category_name = "category"

    def __del__(self):
        """GPU 메모리 해제"""
        if hasattr(self, "model_instance"):
            free_autobackend(self.model_instance)

    # ──────────────────────────────────────────────────────
    # 추상 메서드 1: 초기값 설정
    # ──────────────────────────────────────────────────────
    def _init_values(self):
        self.od_letterbox_instance = LetterBoxTorch(
            max_batch=LIMITED_NUM_OF_CAMERA,
            target_size=OD_INPUT_SIZE,
            device=DEVICE,
        )
        # 필요한 추가 초기화 (트래커, 전처리기 등)

    # ──────────────────────────────────────────────────────
    # 추상 메서드 2: 모델 로딩
    # ──────────────────────────────────────────────────────
    def _load_model(self):
        self.model_instance = PiaONNXTensorRTModel(
            model_path=MODEL_TRT_PATH,
            device=DEVICE,
        )

    # ──────────────────────────────────────────────────────
    # 추상 메서드 3: 이벤트 매니저 반환
    # ──────────────────────────────────────────────────────
    def _load_event_manager(self):
        return YourEventManager()

    # ──────────────────────────────────────────────────────
    # ROI 매니저 오버라이드 (대부분의 모듈에서 오버라이드함)
    #   기본 ServiceBase는 ROIManagerBase()를 반환하지만,
    #   커스텀 ROI 처리가 필요하면 여기서 오버라이드합니다.
    #   ※ @final 데코레이터가 붙어있지만 런타임에서 강제되지
    #     않으므로, 실제로 14개 이상의 모듈이 오버라이드 중입니다.
    # ──────────────────────────────────────────────────────
    def _load_roi_manager(self):
        return YourRoIManager()

    # ──────────────────────────────────────────────────────
    # 추상 메서드 4: 메인 감지 로직 (★ 핵심)
    # ──────────────────────────────────────────────────────
    def _detect(self, **datas):
        batches = datas["batches"]           # List[np.array] - N개 프레임
        stream_ids = datas["stream_ids"]     # List[str] - 카메라 ID
        user_params = datas["user_params"]   # List[dict] - 사용자 파라미터

        # ── 1. ROI 크롭 ────────────────────────────────────
        cropped, categories = self.roi_manager.process_batches_with_roi(
            batches=batches, stream_ids=stream_ids, user_params=user_params
        )

        # ── 2. 전처리 (Letterbox + 정규화) ──────────────────
        resized = self.od_letterbox_instance(imgs=cropped)
        preprocessed = resized / 255.0

        # ── 3. 모델 추론 ───────────────────────────────────
        raw_results = self.model_instance(preprocessed)

        # ── 4. 후처리 (NMS 등) ──────────────────────────────
        # nms_results = ...

        # ── 5. 이벤트 판정 ──────────────────────────────────
        # self.alarm_event_manager.update(...)
        # alarms = self.alarm_event_manager.get_alarm()

        # ── 6. 결과 반환 ───────────────────────────────────
        if len(alarms) > 0:
            return alarms, batches, stream_ids, user_params, self.is_needed_cvt_color
        else:
            return None
```

**`_detect()` 반환값 규격:**

```
성공 시:  (alarms_dict, batches, stream_ids, user_params, is_needed_cvt_color)
          │              │           │            │              │
          │              │           │            │              └─ bool: BGR→RGB 변환 필요 여부
          │              │           │            └─ 원본 user_params (변경 없이 전달)
          │              │           └─ 원본 stream_ids (변경 없이 전달)
          │              └─ 원본 batches (썸네일 추출용)
          └─ dict: {stream_id: [True/False, category_id]}

감지 없음: None
```

---

### 4.5 `roi_manager.py` - ROI 매니저 구현

ROI(Region of Interest) 영역을 관리하고, 입력 이미지를 ROI 기준으로 크롭합니다.

```python
from collections import defaultdict
from typing import List
import numpy as np
import torch
from pia.vision.roi.roi_manager import (
    ROIManagerBase,
    calc_expand_coord,
    batch_crop_region,
)
from pia_prod.AI.modules.<모듈명>.config import DEVICE
from pia_prod.AI.global_config import (
    ROI_POLYGON_COORDINATES_KEY,
    EXPANDED_ROI_KEY,
    ROI_KEY,
)


class YourRoIManager(ROIManagerBase):
    def __init__(self):
        super().__init__()
        self.roi_dict = defaultdict(dict)
        self.category_list = ["category_cv", "카테고리_cv"]

    def process_batches_with_roi(
        self, batches: List[np.array], stream_ids, user_params: List
    ) -> List[np.array]:
        """
        배치 이미지를 ROI 기준으로 크롭합니다.

        Returns:
            cropped_images: ROI 크롭된 이미지 리스트
            cat_results: 각 크롭이 원본 배치의 몇 번째 인덱스인지
        """
        cat_results = []
        regions = []
        gpu_batches = []

        for idx, (batch, stream_id, user_param) in enumerate(
            zip(batches, stream_ids, user_params)
        ):
            cv_event = user_param["user_param"]["cvEvent"]

            # 카테고리 키 찾기
            found_key = None
            for key in self.category_list:
                if key in cv_event:
                    found_key = key
                    break

            # ROI 정보 가져오기 (없으면 전체 이미지)
            roi_dict = self.get_roi_info(
                camera_id=stream_id,
                roi_raw_info=cv_event[found_key],
                image_wh=tuple(reversed(batch.shape[:2])),
            )

            # GPU 텐서 변환
            gpu_batch = (
                torch.from_numpy(batch).to(DEVICE)
                if isinstance(batch, np.ndarray) else batch
            )
            gpu_batches.append(gpu_batch)
            regions.append(roi_dict[EXPANDED_ROI_KEY])
            cat_results.append(idx)

        # GPU 배치 크롭
        cropped_images = batch_crop_region(gpu_batches, regions)

        return cropped_images, cat_results
```

**ROI 처리 흐름:**

```
원본 이미지 (1920x1080)
┌───────────────────────────────────────┐
│                                       │
│ ╱╲                                    │
│ ╱  ╲  ← 폴리곤 ROI                    │
│ ╱    ╲   (polygonCoordinates)         │
│ ╱      ╲                              │
│ ╱________╲                            │
│                                       │
└───────────────────────────────────────┘
                                        │
                    │ calc_expand_coord()
                    │ 폴리곤 → 직사각형으로 확장
                    ▼
┌──────────────────┐
│ ┌──────────┐     │
│   │ 확장된    │  │
│   │ 직사각형  │   │  ← expanded_roi
│   │ ROI 영역  │  │
│ └──────────┘     │
└──────────────────┘
                   │
                    │ batch_crop_region()
                    │ 크롭 수행
                    ▼
              ┌──────────┐
              │ 크롭된   │
              │ 이미지   │  ← 이것이 모델에 입력됨
              └──────────┘
```

---

### 4.6 `__init__.py` - 빈 파일

```python
# (빈 파일 - Python 패키지 마커)
```

---

## 5. STEP 3 - 시스템 등록

새로 만든 모듈을 시스템에 인식시키기 위해 기존 파일을 수정합니다.

> **중요**: 등록 방식은 **이벤트 타입(CV / RET / VQA)에 따라 다릅니다.**
> 어떤 이벤트 타입인지에 따라 수정해야 할 파일 수와 내용이 달라집니다.

```
┌───────────────────────────────────────────────────────────────────────────┐
│ 이벤트 타입별 등록 차이 요약                                              │
├──────────┬──────────────────┬─────────────────────────────────────────────┤
│ 이벤트    │ __init__.py 수정 │ stream_params.py 수정                      │
├──────────┼──────────────────┼─────────────────────────────────────────────┤
│ cvEvent  │ 3곳 (필수)       │ 4곳 (필수) - 개별 Param 모델 + name 라우팅  │
│          │                  │ 카테고리명 접미사: _cv                      │
├──────────┼──────────────────┼─────────────────────────────────────────────┤
│ retEvent │ 3곳 (필수)       │ 수정 불필요 - 모든 RET 이벤트가 자동으로    │
│          │                  │ RetrievalBase로 파싱됨 (name 라우팅 없음)   │
│          │                  │ 카테고리명 접미사: _ret                     │
├──────────┼──────────────────┼─────────────────────────────────────────────┤
│ vqaEvent │ 3곳 (필수)       │ 수정 불필요 - 모든 VQA 이벤트가 자동으로    │
│          │                  │ VQABase로 파싱됨 (name 라우팅 없음)         │
│          │                  │ 카테고리명 접미사: _vqa                     │
└──────────┴──────────────────┴─────────────────────────────────────────────┘

참고 - 이벤트 타입별 대표 모듈:
  CV:  ax_fall, khonkaen_helmet, aramco_loitering 등 (대부분)
  RET: perception_encoder, pe_violence, hyundai_esfalldown
  VQA: gangnam_falldown, gangnam_violence, samsung_internvl3
```

---

### 5.1 `AI/__init__.py` 수정 (3곳) - 모든 이벤트 타입 공통

**파일 경로**: `packages/pia_prod/AI/__init__.py`

이 파일은 CV / RET / VQA **관계없이 항상 동일하게** 수정합니다.

```python
# ──────────────────────────────────────────────────────────────────
# 수정 1: __all__ 리스트에 서비스 클래스명 추가
# ──────────────────────────────────────────────────────────────────
__all__ = [
    "FallService",
    "HarnessService",
    # ... 기존 서비스들 ...
    "LoiteringService",
    "YourService",           # ← ★ 추가
]

# ──────────────────────────────────────────────────────────────────
# 수정 2: _SERVICE_MODULE_OVERRIDE 딕셔너리에 매핑 추가
# ──────────────────────────────────────────────────────────────────
_SERVICE_MODULE_OVERRIDE = {
    "FallService": "ax_fall",
    # ... 기존 매핑 ...
    "LoiteringService": "aramco_loitering",
    "YourService": "<모듈_디렉토리명>",    # ← ★ 추가
}

# ──────────────────────────────────────────────────────────────────
# 수정 3: TYPE_CHECKING 블록에 import 추가 (IDE 자동완성용)
# ──────────────────────────────────────────────────────────────────
if TYPE_CHECKING:
    # ... 기존 import ...
    from .modules.aramco_loitering.service import LoiteringService
    from .modules.<모듈명>.service import YourService    # ← ★ 추가
```

---

### 5.2 `DTO/stream_params.py` 수정 - cvEvent인 경우만 (4곳)

> **RET / VQA 모듈이면 이 섹션은 건너뛰세요.**
> `retEvent`와 `vqaEvent`는 각각 `RetrievalBase`, `VQABase`로 일괄 파싱되므로
> stream_params.py 수정이 필요 없습니다.

**파일 경로**: `packages/pia_prod/AI/DTO/stream_params.py`

```python
# ──────────────────────────────────────────────────────────────────
# 수정 1: param 모델 import 추가 (파일 상단)
# ──────────────────────────────────────────────────────────────────
from pia_prod.AI.modules.<모듈명>.param import YourModel              # ← ★ 추가

# ──────────────────────────────────────────────────────────────────
# 수정 2: config의 카테고리 상수 import 추가 (파일 상단)
# ──────────────────────────────────────────────────────────────────
from pia_prod.AI.modules.<모듈명>.config import YOUR_CV_CATEGORY      # ← ★ 추가


class AddStreamModel(BaseModel):
    # ...

    # ──────────────────────────────────────────────────────────────
    # 수정 3: cvEvent Union 타입에 YourModel 추가
    # ──────────────────────────────────────────────────────────────
    cvEvent: Optional[
        List[
            Union[
                IntrusionModel,
                FalldownModel,
                # ... 기존 모델들 ...
                LoiteringModel,
                YourModel,           # ← ★ 추가
            ]
        ]
    ] = []

    @field_validator("cvEvent", mode="before")
    @classmethod
    def parse_cv_events(cls, v):
        # ...
        for event in v:
            name = event.get("name")

            if name in INTRUSION_CV_CATEGORY:
                parsed_event_list.append(IntrusionModel(**event))
            # ... 기존 분기들 ...
            elif name in LOITERING_CV_CATEGORY:
                parsed_event_list.append(LoiteringModel(**event))

            # ──────────────────────────────────────────────────────
            # 수정 4: parse_cv_events() 분기 추가
            # ──────────────────────────────────────────────────────
            elif name in YOUR_CV_CATEGORY:                           # ← ★ 추가
                parsed_event_list.append(YourModel(**event))         # ← ★ 추가

            else:
                raise ValueError(f"Unknown category name: {event}")
```

---

### 5.3 이벤트 타입별 param.py & 서비스 내부 차이

각 이벤트 타입에 따라 param.py와 service.py 내부에서 접근하는 키가 다릅니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CV 모듈 (cvEvent)                                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ config.py:    YOUR_CV_CATEGORY = ["카테고리_cv", "category_cv"]             │
│ param.py:     class YourModel(CategoryBase): ...  ← 개별 Param 모델 정의    │
│ service.py:   user_param[USER_PARAM_KEY][CV_EVENT_KEY]  ← CV_EVENT_KEY 사용 │
│                                                                             │
│ 등록:         stream_params.py 4곳 수정 필수                                │
│ 참고 모듈:    khonkaen_helmet, aramco_loitering, ax_fall 등                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ RET 모듈 (retEvent)                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ config.py:    YOUR_RET_CATEGORY = ["카테고리_ret", "category_ret"]          │
│ param.py:     불필요 - RetrievalBase 직접 사용                              │
│ (abnormalText, normalText, topCandidates 필드 포함)                         │
│ service.py:   user_param[USER_PARAM_KEY][RET_EVENT_KEY]  ← RET_EVENT_KEY    │
│                                                                             │
│ 등록:         stream_params.py 수정 불필요 (자동으로 RetrievalBase 파싱)    │
│ 참고 모듈:    perception_encoder, pe_violence, hyundai_esfalldown           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ VQA 모듈 (vqaEvent)                                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ config.py:    YOUR_VQA_CATEGORY = ["카테고리_vqa", "category_vqa"]          │
│ param.py:     불필요 - VQABase 직접 사용                                    │
│ (CategoryBase와 동일한 필드)                                                │
│ service.py:   user_param[USER_PARAM_KEY][VQA_EVENT_KEY]  ← VQA_EVENT_KEY    │
│                                                                             │
│ 등록:         stream_params.py 수정 불필요 (자동으로 VQABase 파싱)          │
│ 참고 모듈:    gangnam_falldown, gangnam_violence, samsung_internvl3         │
└─────────────────────────────────────────────────────────────────────────────┘
```

**카테고리 이름 접미사 규칙:**

```
  CV 이벤트:   xxx_cv      (예: helmet_cv, loitering_cv, 헬멧_cv)
  RET 이벤트:  xxx_ret     (예: fire_ret, violence_ret, 화재_ret)
  VQA 이벤트:  xxx_vqa     (예: falldown_vqa, fire_vqa, 쓰러짐_vqa)
```

**API 요청 예시 (이벤트 타입별):**

```json
// CV 이벤트 - cvEvent에 전달
{
  "cvEvent": [{"name": "helmet_cv", "incidentThresholdSecond": 5, ...}]
}

// RET 이벤트 - retEvent에 전달
{
  "retEvent": [{"name": "fire_ret", "abnormalText": ["fire"], "normalText": ["normal"], "topCandidates": 3, ...}]
}

// VQA 이벤트 - vqaEvent에 전달
{
  "vqaEvent": [{"name": "falldown_vqa", "incidentThresholdSecond": 5, ...}]
}
```

---

## 6. STEP 4 - 모델 파일 배치

```
assets/model/
├─ YourModelDet_v0.1.0.onnx        ← OD 모델 (필수)
├─ YourModelDet_v0.1.0.engine      ← TRT 엔진 (테스트 시 자동 변환됨)
├─ YourModelCls_v0.1.0.onnx        ← CLS 모델 (필요 시)
└─ text_features_your.json          ← 텍스트 프롬프트 (CLIP 사용 시)
```

**네이밍 규칙**: `<모델명>_v<Major>.<Minor>.<Patch>.onnx`

> **중요**: 모델 파일(.onnx, .engine, .pt)은 `.gitignore`에 의해 Git에 커밋되지 않습니다.  
> 모델 다운로드 경로를 테스트 fixture에 명시하여 테스트 시 자동 다운로드되도록 합니다.

---

## 7. STEP 5 - 테스트 코드 작성

**파일 경로**: `packages/pia_prod/AI/tests/modules/test_<모듈명>.py`

실제 `test_aramco_loitering.py` 패턴을 참고한 템플릿:

```python
import cv2
import pytest
from queue import Queue
from pia_prod.AI.DTO.stream_params import AddStreamModel
from pia_prod.AI.utils.utils import AddStreamModel2dict


# ────────────────────────────────────────────────────────────
# Fixture: 모델 다운로드 & TensorRT 변환
# ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def model_download():
    from huggingface_hub import hf_hub_download
    # ONNX 다운로드
    hf_hub_download(repo_id="TeamPIA/...", filename="...", local_dir="assets/model")
    # TensorRT 변환
    from pia.ai.model import PiaONNXTensorRTModel
    PiaONNXTensorRTModel(model_path="assets/model/YourModel_v0.1.0.engine", device="cuda")
    yield


# ────────────────────────────────────────────────────────────
# Fixture: 테스트 영상 다운로드
# ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def local_video_path():
    # NAS 또는 HuggingFace에서 테스트 영상 다운로드
    video_path = "assets/videos/test_video.mp4"
    yield video_path


# ────────────────────────────────────────────────────────────
# Fixture: AddStreamModel 생성
# ────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def get_AddStreamModel():
    stream_model = AddStreamModel(
        cameraId=1,
        cameraUrl="rtsp://...",
        organization="pia",
        cvEvent=[{
            "name": "category_cv",
            "incidentThresholdSecond": 5,
            "incidentTimeoutSecond": 10,
            "roi": {"polygonCoordinates": [], "divideCoordinates": []}
        }],
        timestamp="2026-01-01T00:00:00.000000",
    )
    yield AddStreamModel2dict(stream_model)


# ────────────────────────────────────────────────────────────
# 테스트 1: Import 테스트
# ────────────────────────────────────────────────────────────
def test_import_module():
    from pia_prod.AI import YourService
    assert YourService is not None


# ────────────────────────────────────────────────────────────
# 테스트 2: 단일 카메라 추론
# ────────────────────────────────────────────────────────────
def test_single_camera(model_download, local_video_path, get_AddStreamModel):
    from pia_prod.AI import YourService

    q = Queue(1)
    user_param = get_AddStreamModel
    stream_id = "1_pia"

    cap = cv2.VideoCapture(local_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 0.3
    service = YourService(q)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % round(fps * interval) == 0:
            q.put({
                "batches": [frame],
                "stream_ids": [stream_id],
                "user_params": [{"user_param": user_param}],
            })

    # 쓰레드 종료 시그널
    q.put({"batches": None, "stream_ids": None, "user_params": None})
    cap.release()


# ────────────────────────────────────────────────────────────
# 테스트 3: 멀티 카메라 배치 테스트 (batch_size=4)
# ────────────────────────────────────────────────────────────
def test_batches(model_download, local_video_path, get_AddStreamModel):
    from pia_prod.AI import YourService

    batch_size = 4
    q = Queue(1)
    service = YourService(q)
    # 4개 스트림 동시 처리 테스트
    # ...
```

---

## 8. STEP 6 - 커밋 & PR 제출

### 커밋 메시지 규칙

```
┌──────────────────────────────────────────────────────────────────┐
│ 형식: type: Subject line (50자 이내)                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ 초기 파일:  chore: initial commit for aiprod-XXX-프로젝트-설명   │
│ 설정 정의:  chore: 카테고리 관련 컨피그 정의                         │
│ 기능 구현:  feat: 이벤트매니저 정의                                  │
│ 등록 수정:  feat: addstream에 카테고리 추가                          │
│ 테스트:     test: 테스트코드 추가                                    │
│ 버그 수정:  fix: 오타 수정                                           │
│                                                                      │
│ type 목록: feat / fix / refactor / style / docs / test / chore       │
└──────────────────────────────────────────────────────────────────┘
```

### PR 제출

- **타겟 브랜치**: `dev`
- **브랜치 네이밍**: `aiprod-<Linear이슈번호>-<설명>` (Issue 할당 시 자동 생성)
- **PR 생성 후**: 리뷰어 자동 지정됨 (ready-pr.yaml)

---

## 9. 실제 PR 커밋 이력 참고

실제 Merge된 PR의 커밋 순서를 보면, 개발 흐름을 이해하는 데 도움이 됩니다.

### 배회 감지 (`aramco_loitering`) - PR #361, #362

```
 커밋순서                                                    대응 STEP
 ──────────────────────────────────────────────────────────────────────
 1  chore: initial commit for aiprod-219                    STEP 1
 2  chore: 배회 관련 컨피그 정의                              STEP 2 (config.py)
 3  chore: 배회 모델의 param 정의                             STEP 2 (param.py)
 4  feat: 배회 이벤트매니저 정의                               STEP 2 (event.py)
 5  feat: 배회 roi manager 추가                              STEP 2 (roi_manager.py)
 6  feat: 배회 Service manager 개발                          STEP 2 (service.py)
 7  feat: addstream에 배회 추가                               STEP 3 (등록)
 8  test: 배회 테스트코드 추가                                 STEP 5 (테스트)
 9  feat: od+tracker 전용 디버그 유틸 추가                     선택 파일
10  chore: 일부 주석 제거                                     정리
11  HOTFIX: 종속성에 의한 yolox 제거 및 이관                   핫픽스
```

### 환자 배회 (`vanguard_patient`) - PR #359

```
 커밋순서                                                    대응 STEP
 ──────────────────────────────────────────────────────────────────────
 1  chore: 기본 파일 추가                                    STEP 1+2 (스캐폴딩)
 2  feat: 환자배회 서비스 초안 작성                            STEP 2 (service.py)
 3  feat: 환자배회 roi_manager 초안 작성                      STEP 2 (roi_manager.py)
 4  feat: PIA octracker 정의                                 선택 파일 (tracker.py)
 5  chore: stream_params에 patient 추가                      STEP 3 (등록)
 6  feat: __init__.py에 PatientService 추가                   STEP 3 (등록)
 7  test: 테스트 코드 초안 작성                                STEP 5 (테스트)
 8  (이후 20+ 커밋으로 개선/수정)                               반복 개선
```

**공통 패턴:**

```
기본 파일 생성 → config/param 정의 → event 매니저 → service 구현
                                                       │
    ┌──────────────────────────────────────────────────┘
    ▼
  등록 코드 수정 (stream_params.py + __init__.py) → 테스트 작성 → 반복 개선
```

---

## 10. 선택적 추가 파일 가이드

필요에 따라 아래 파일을 추가합니다. 어떤 경우에 필요한지 참고하세요.

```
┌──────────────────┬────────────────────────────┬────────────────────────────┐
│ 파일             │ 필요한 경우                │ 참고 모듈                  │
├──────────────────┼────────────────────────────┼────────────────────────────┤
│ postprocess.py   │ 커스텀 NMS, 좌표 변환 등   │ aramco_loitering           │
│                  │ 표준 후처리로 부족할 때    │                            │
├──────────────────┼────────────────────────────┼────────────────────────────┤
│ preprocess.py    │ 색상 필터, 배경 제거,      │ vanguard_patient           │
│                  │ 이미지 증강 등 특수 전처리 │ yeonsei_smoke              │
├──────────────────┼────────────────────────────┼────────────────────────────┤
│ tracker.py       │ 객체 추적이 필요한 경우    │ vanguard_patient           │
│                  │ (배회, 추적 기반 감지)     │ (MultiStreamPiaOCSort)     │
├──────────────────┼────────────────────────────┼────────────────────────────┤
│ debug_utils.py   │ 바운딩박스 시각화,         │ aramco_loitering           │
│                  │ 프레임 저장 등 디버깅      │ khonkaen_helmet            │
├──────────────────┼────────────────────────────┼────────────────────────────┤
│ func.py          │ 모듈 전용 유틸리티 함수    │ khonkaen_helmet            │
│                  │                            │ Daegu_intrusion            │
├──────────────────┼────────────────────────────┼────────────────────────────┤
│ prompts.py       │ VLM/CLIP 텍스트 프롬프트   │ gangnam_falldown           │
│                  │ 정의가 필요한 경우         │ perception_encoder         │
└──────────────────┴────────────────────────────┴────────────────────────────┘
```

---

## 11. 자주 하는 실수 & 주의사항

```
┌──────────────────────────────────────────────────────────────────────┐
│ ★ 반드시 지켜야 할 규칙                                              │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ 1. 추상 클래스 상속 필수                                             │
│ ServiceBase, EventBase를 상속하지 않으면 커밋 체크리스트에 걸림      │
│                                                                      │
│ 2. GPU/CPU 디바이스 하드코딩 금지                                    │
│ ✗ torch.device("cuda:0")                                             │
│ ✓ DEVICE 상수 사용 (config.py에서 정의)                              │
│                                                                      │
│ 3. 모델 파일 커밋 금지                                               │
│ .onnx, .engine, .pt 파일은 .gitignore에 포함                         │
│                                                                      │
│ 4. pre-commit 훅 통과 필수                                           │
│ 커밋 전에 반드시: pre-commit run --all-files                         │
│                                                                      │
│ 5. Submodule 사용 금지                                               │
│ 외부 코드는 직접 복사하거나 pip 의존성으로 관리                      │
│                                                                      │
│ 6. 모델 경로 환경변수 래핑 필수                                      │
│ ✗ MODEL_PATH = "assets/model/abc.onnx"                               │
│ ✓ MODEL_PATH = os.getenv("MODEL_PATH", "assets/model/abc.onnx")      │
│ ※ os.getenv()와 os.environ.get() 모두 사용됨 (기능 동일)             │
│                                                                      │
│ 7. 카테고리 이름 리스트에 한글/영문 모두 포함                        │
│ ✗ YOUR_CV_CATEGORY = ["category_cv"]                                 │
│ ✓ YOUR_CV_CATEGORY = ["카테고리_cv", "category_cv"]                  │
│                                                                      │
│ 8. stream_params.py 등록 누락 주의                                   │
│ import 2개 + Union 타입 1개 + parse 분기 1개 = 총 4곳 수정           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```
