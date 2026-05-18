# 신규 카테고리 추가 체크리스트

> **사용법**: 이 체크리스트를 복사하여 PR 또는 Issue에 붙여넣고, 항목을 하나씩 완료하며 체크합니다.
> **참고 문서**: [02_신규_카테고리_추가_가이드.md](./02_신규_카테고리_추가_가이드.md)에서 각 항목의 상세 설명을 확인할 수 있습니다.

---

## 전체 진행 흐름

```
Phase 1          Phase 2          Phase 3          Phase 4          Phase 5          Phase 6          Phase 7          Phase 8
사전 준비   ───→  테스트 작성  ───→  파일 생성    ───→  시스템 등록  ───→  모델 배치   ───→  테스트 추론  ───→  품질 확인   ───→  PR 제출
                  (1개 신규)        (6개 신규)        (2개 수정)                       (STEP 6)

[준비사항]       [테스트 코드]    [모듈 핵심]       [기존 파일]       [assets/]       [pytest 실행]    [pre-commit]      [dev 브랜치]
[파이프라인      [fixture 정의]   [config.py]       [__init__.py]     [.onnx 파일]    [단일 카메라]    [상속 확인]       [커밋 메시지]
 유형 결정]      [테스트 케이스]  [param.py]        [stream_params    [.engine 파일]  [배치 테스트]    [하드코딩 확인]   [리뷰어 지정]
                             [event.py]         .py]
                              [service.py]
                              [roi_manager.py]
                              [__init__.py]
```

---

## Phase 1: 사전 준비

> 코드 작성 전에 확인해야 할 사항입니다.

| | 항목 | 비고 |
|:---:|:---|:---|
| [ ] | Linear 이슈 생성 및 담당자 할당 완료 | Issue 할당 시 브랜치 자동 생성됨 |
| [ ] | 자동 생성된 브랜치(`aiprod-XXX-...`) 확인 | `.github/workflows/create-branch.yaml` |
| [ ] | 사용할 ONNX 모델 파일 준비 완료 | 필요한 모델 |
| [ ] | 테스트용 영상 파일 확보 완료 | NAS 또는 HuggingFace 경로 확인 |
| [ ] | 파이프라인 유형 결정 | 아래 표 참고 |

**파이프라인 유형 선택 가이드:**

```
┌─ 질문 1: 감지에 필요한 모델이 몇 개인가?
│
├─ 1개 (OD만)
│  └─ 질문 2: 객체 추적이 필요한가?
│     ├─ 아니오 → [유형 A] OD 단독           (참고: ax_fall)
│     └─ 예    → [유형 C] OD + Tracker       (참고: aramco_loitering)
│
├─ 2개 (OD + CLS)
│  └─ 질문 2: 객체 추적이 필요한가?
│     ├─ 아니오 → [유형 B] OD + CLS           (참고: khonkaen_helmet)
│     └─ 예    → [유형 D] OD + Tracker + CLS (참고: vanguard_patient)
│
├─ VLM 기반 (InternVL3)
│  └─ [유형 E] VLM 기반                       (참고: gangnam_falldown)
│
├─ CLIP 기반
│  └─ [유형 F] CLIP Retrieval 기반             (참고: perception_encoder)
│
└─ 특수 전처리 필요
   └─ [유형 G] 전처리 집약형                   (참고: yeonsei_smoke, samsung_fire)
```

---

## Phase 2: 테스트 코드 작성

> 테스트 코드를 먼저 작성하여 구현해야 할 인터페이스를 정의합니다.
> `conftest.py`의 공유 fixture(`hf_downloader`, `nas_downloader`, `video_save_dir`)를 활용합니다.

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `packages/pia_prod/AI/tests/modules/test_<모듈명>.py` 파일 생성 | |
| [ ] | `model_download` fixture 작성 | `hf_downloader` 공유 fixture 사용 + `onnx2trt()` 변환 포함 |
| [ ] | `local_video_path` fixture 작성 | `nas_downloader` + `video_save_dir` 공유 fixture 사용 |
| [ ] | `test_import_module` 작성 | `from pia_prod.AI import YourService` + `assert` |
| [ ] | `test_single_camera` 작성 | AddStreamModel 구성 + Queue 기반 프레임 입력 + `stop_thread(q)` |
| [ ] | `test_batches` 작성 (권장) | 멀티 카메라 배치 테스트 (batch_size=4) |

---

## Phase 3: 모듈 디렉토리 및 파일 생성

> 새 모듈의 핵심 파일을 생성합니다.
> **이벤트 타입(CV / RET / VQA)에 따라 파일 구성과 내용이 다릅니다.** 해당하는 섹션만 따라가세요.

### 3-0. 공통 - 디렉토리 및 `__init__.py` 생성

| | 항목 | 비고 |
|:---:|:---|:---|
| [ ] | `packages/pia_prod/AI/modules/<프로젝트>_<카테고리>/` 디렉토리 생성 | `mkdir -p ...` |
| [ ] | 네이밍이 `프로젝트명_카테고리` 규칙을 따르는지 확인 | 예: `aramco_loitering` |
| [ ] | `__init__.py` 빈 파일 생성 | |

---

### 3-A. cvEvent 모듈 (OD/CLS/Tracker 기반 — 대부분의 모듈)

> 참고 모듈: `aramco_loitering`, `khonkaen_helmet`, `vanguard_patient`, `ax_fall`

#### `config.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | 모델 ONNX 경로 정의 | `os.getenv()` 로 감싸져 있는가? |
| [ ] | 모델 TensorRT 엔진 경로 정의 | `os.getenv()` 로 감싸져 있는가? |
| [ ] | `OD_INPUT_SIZE` 정의 | 예: `[640, 640]` |
| [ ] | `OD_CONFIDENCE_THRESHOLD` 정의 | 예: `0.5` |
| [ ] | `OD_NMS_THRESHOLD` 정의 | 예: `0.35` |
| [ ] | `OD_TIME_INTERVAL_SECOND` 정의 | 예: `0.3` |
| [ ] | 카테고리 이름 리스트 정의 (접미사 `_cv`) | **한글 + 영문 모두**: `["카테고리_cv", "category_cv"]` |
| [ ] | `DEVICE` 상수 정의, 하드코딩 없음 | `cuda:0` 직접 사용 금지 |
| [ ] | (CLS 모델 필요 시) CLS 관련 상수 정의 | 입력 사이즈, threshold 등 |
| [ ] | (Tracker 필요 시) Tracker 관련 상수 정의 | `TRACKER_DICT` (max_age, min_hits, iou 등) |

#### `param.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `CategoryBase` 를 상속하였는가? | `from pia_prod.AI.DTO.param_base import CategoryBase` |
| [ ] | 카테고리 고유 threshold 필드 정의 | 예: `confidence_threshold`, `nms_threshold` 등 |
| [ ] | 기본값이 `config.py` 상수를 참조하는가? | 하드코딩된 숫자 없이 상수 사용 |

#### `event.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `EventBase` 를 상속하였는가? | `from pia_prod.AI.bases.event_base import EventBase` |
| [ ] | `update()` 메서드 구현 | 감지 결과 → 내부 상태 갱신 |
| [ ] | `get_alarm()` 메서드 구현 | `STATUS_TRANSITION` 행렬 활용, 상태 1/3에서만 알람 반환 |
| [ ] | 스트림별 독립 상태 관리 | `defaultdict(int)` 등으로 스트림 분리 |
| [ ] | (Tracker 사용 시) 트랙 ID별 상태 관리 | duration 기반 판정 |

#### `service.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `ServiceBase` 를 상속하였는가? | |
| [ ] | `_init_values()` 구현 | `LetterBoxTorch` 인스턴스, (Tracker 필요 시) 트래커 초기화 |
| [ ] | `_load_model()` 구현 | `PiaONNXTensorRTModel(model_path, device)` |
| [ ] | `_load_event_manager()` 구현 | 이벤트 매니저 인스턴스를 **return** |
| [ ] | `_load_roi_manager()` 오버라이드 | 커스텀 ROI 매니저를 return |
| [ ] | `_detect()` 파이프라인 | ROI 크롭 → Letterbox → OD 추론 → NMS → 이벤트 판정 |
| [ ] | `_detect()` 반환값 dict 형식 | `{ALARMS_KEY: ..., BATCHES_KEY: ..., ...}` |
| [ ] | `global_config` import 포함 | `ALARMS_KEY`, `BATCHES_KEY` 등 |
| [ ] | `user_param[USER_PARAM_KEY][CV_EVENT_KEY]` 사용 | CV_EVENT_KEY로 이벤트 접근 |
| [ ] | `__del__()` 에서 GPU 메모리 해제 | `free_autobackend()` 호출 |

#### `roi_manager.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `ROIManagerBase` 를 상속하였는가? | |
| [ ] | `process_batches_with_roi()` 구현 | GPU 배치 크롭 (`batch_crop_region`) |
| [ ] | `category_list`에 한글/영문 카테고리명 모두 포함 | `["category_cv", "카테고리_cv"]` |
| [ ] | ROI가 비어있을 때 전체 이미지 사용 폴백 | |

#### 선택 파일

| | 파일 | 필요 조건 |
|:---:|:---|:---|
| [ ] | `postprocess.py` | 커스텀 NMS, 좌표 변환 등 |
| [ ] | `preprocess.py` | 특수 전처리 (색상 필터, 배경 제거 등) |
| [ ] | `tracker.py` | 객체 추적 필요 시 (OC-SORT 등) |
| [ ] | `debug_utils.py` | 바운딩박스 시각화, 프레임 저장 등 |

---

### 3-B. retEvent 모듈 (PE/CLIP Retrieval 기반)

> 참고 모듈: `perception_encoder`, `pe_violence`, `hyundai_esfalldown`

#### `config.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | 모델 PyTorch/ONNX/TRT 경로 정의 | `os.getenv()` 로 감싸져 있는가? (3종 모두) |
| [ ] | `IMG_SIZE` 정의 | 예: `(336, 336)` |
| [ ] | `INPUT_SIZE` 정의 (시간축 포함) | 예: `(8, 3, 336, 336)` 또는 `(3, 336, 336)` |
| [ ] | `TEMPORAL_SIZE` 정의 | 예: `8` (프레임 버퍼 크기) |
| [ ] | 텍스트 프롬프트 경로 정의 | `TXT_FEATURE_PATH`, 프롬프트 리스트 경로 |
| [ ] | `INDEX_MAPPING` 정의 | `{0: "normal", 1: "abnormal"}` 형태 |
| [ ] | `ALARM_QUEUE_SIZE` / `ALARM_THRESHOLD` 정의 | 큐 기반 알람 판정용 |
| [ ] | 카테고리 이름 리스트 정의 (접미사 `_ret`) | **한글 + 영문 모두**: `["카테고리_ret", "category_ret"]` |
| [ ] | `CATEGORY_EVENT_MAP` 정의 | 예측 클래스 → 카테고리 매핑 |
| [ ] | `DEVICE` 상수 정의, 하드코딩 없음 | |
| [ ] | 추론 정밀도 정의 | 예: `torch.float16` |

#### `param.py`

| | 항목 |
|:---:|:---|
| [ ] | **불필요** — `RetrievalBase`를 직접 사용 (별도 파일 생성 불필요) |

#### `event.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `EventBase` 를 상속하였는가? | |
| [ ] | `update()` 메서드 구현 | 유사도 점수 → 큐에 누적 |
| [ ] | 큐 기반 알람 판정 | `ALARM_QUEUE_SIZE` 중 `ALARM_THRESHOLD` 이상이면 트리거 |
| [ ] | `STATUS_TRANSITION` 행렬 활용 | 상태 1/3에서만 알람 반환 |
| [ ] | 스트림별 독립 상태 관리 | |

#### `service.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `ServiceBase` 를 상속하였는가? | |
| [ ] | `_init_values()` 구현 | 전처리 인스턴스, 시간축 버퍼 초기화 |
| [ ] | `_load_model()` 구현 | PE 모델 + 텍스트 피처 로딩 |
| [ ] | `_load_event_manager()` 구현 | 이벤트 매니저 **return** |
| [ ] | `_detect()` 파이프라인 | BGR→RGB → 전처리 → 임베딩 추출 → 시간축 버퍼링 → 유사도 계산 → 이벤트 판정 |
| [ ] | `_detect()` 반환값 dict 형식 | `{ALARMS_KEY: ..., BATCHES_KEY: ..., ...}` |
| [ ] | `global_config` import 포함 | `ALARMS_KEY`, `BATCHES_KEY` 등 |
| [ ] | `user_param[USER_PARAM_KEY][RET_EVENT_KEY]` 사용 | RET_EVENT_KEY로 이벤트 접근 |

#### `roi_manager.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `ROIManagerBase` 를 상속하였는가? | |
| [ ] | `process_batches_with_roi()` 구현 | ROI 크롭 + 카테고리별 텍스트 벡터 매핑 |
| [ ] | `category_list`에 한글/영문 카테고리명 포함 | `["category_ret", "카테고리_ret"]` |
| [ ] | `retEvent`에서 카테고리 정보 추출 | `user_param[USER_PARAM_KEY][RET_EVENT_KEY]` |

---

### 3-C. vqaEvent 모듈 (InternVL3 VLM 기반)

> 참고 모듈: `gangnam_falldown`, `gangnam_violence`, `samsung_internvl3`

#### `config.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | HuggingFace 모델 경로 정의 | `os.getenv()`: 예: `"assets/model/InternVL3-2B"` |
| [ ] | 카테고리 이름 리스트 정의 (접미사 `_vqa`) | **한글 + 영문 모두**: `["카테고리_vqa", "category_vqa"]` |
| [ ] | `SUPPORT_CATEGORIES` 정의 | 지원하는 감지 카테고리 목록 |
| [ ] | `QUEUE_SIZE` / `ALARM_DURATION_THRESHOLD` 정의 | 예측 큐 크기, 알람 판정 임계값 |
| [ ] | `DEVICE` 상수 정의, 하드코딩 없음 | |
| [ ] | `MODEL_INF_DATA_TYPE` 정의 | 예: `"bfloat16"` |
| [ ] | `MAX_NEW_TOKEN` 정의 | VQA 응답 최대 토큰 수 |
| [ ] | (영상 버퍼 사용 시) `FRAME_PER_TILE_MAX_NUM` 정의 | 프레임 타일링 설정 |

#### `param.py`

| | 항목 |
|:---:|:---|
| [ ] | **불필요** — `VQABase`를 직접 사용 (별도 파일 생성 불필요) |

#### `event.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `EventBase` 를 상속하였는가? | |
| [ ] | `update()` 메서드 구현 | VQA 텍스트 응답 → 카테고리 매칭 → 상태 갱신 |
| [ ] | 단일 예측 기반 알람 판정 | `ALARM_DURATION_THRESHOLD` 활용 |
| [ ] | `STATUS_TRANSITION` 행렬 활용 | 상태 1/3에서만 알람 반환 |
| [ ] | 스트림별 독립 상태 관리 | |

#### `service.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `ServiceBase` 를 상속하였는가? | |
| [ ] | `_init_values()` 구현 | VQA 모델 초기화 설정 |
| [ ] | `_load_model()` 구현 | InternVL3 모델 로딩 (HuggingFace 경로) |
| [ ] | `_load_event_manager()` 구현 | 이벤트 매니저 **return** |
| [ ] | `_detect()` 파이프라인 | 프레임 → (버퍼링) → VQA 추론 → 텍스트 응답 해석 → 이벤트 판정 |
| [ ] | `_detect()` 반환값 dict 형식 | `{ALARMS_KEY: ..., BATCHES_KEY: ..., ...}` |
| [ ] | `global_config` import 포함 | `ALARMS_KEY`, `BATCHES_KEY` 등 |
| [ ] | `user_param[USER_PARAM_KEY][VQA_EVENT_KEY]` 사용 | VQA_EVENT_KEY로 이벤트 접근 |
| [ ] | (영상 버퍼 사용 시) 별도 쓰레드 구현 | 프레임 수집 → 버퍼 → 추론 분리 |

#### `roi_manager.py`

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `ROIManagerBase` 를 상속하였는가? | |
| [ ] | `process_batches_with_roi()` 구현 | ROI 크롭 + 카테고리별 프롬프트 매핑 |
| [ ] | `category_list`에 한글/영문 카테고리명 포함 | `["category_vqa", "카테고리_vqa"]` |
| [ ] | `vqaEvent`에서 카테고리 정보 추출 | `user_param[USER_PARAM_KEY][VQA_EVENT_KEY]` |
| [ ] | 카테고리-프롬프트 매핑 로직 | `_match_category2prompt()` 구현 |

#### 선택 파일

| | 파일 | 필요 조건 |
|:---:|:---|:---|
| [ ] | `prompts.py` | VQA 프롬프트 정의 (카테고리별 질의문) |

---

## Phase 4: 시스템 등록

> 이 단계를 빠뜨리면 모듈이 시스템에서 인식되지 않습니다.
> **중요**: 등록 방식은 이벤트 타입(CV / RET / VQA)에 따라 다릅니다.

### 4-0. 먼저 이벤트 타입을 확인하세요 (택 1)

| | 타입 | 해당하는 경우 | 카테고리명 접미사 | stream_params.py 수정 |
|:---:|:---|:---|:---:|:---:|
| [ ] | **cvEvent** | OD/CLS/Tracker 기반 감지 (대부분) | `_cv` | **필요 (4곳)** |
| [ ] | **retEvent** | PE(Perception Encoder) Retrieval 기반 | `_ret` | 불필요 |
| [ ] | **vqaEvent** | InternVL3 VQA 기반 | `_vqa` | 불필요 |

### 4-1. `packages/pia_prod/AI/__init__.py` (3곳 수정) - 모든 타입 공통

| | 수정 위치 | 추가 내용 |
|:---:|:---|:---|
| [ ] | `__all__` 리스트 | `"YourService"` 추가 |
| [ ] | `_SERVICE_MODULE_OVERRIDE` 딕셔너리 | `"YourService": "<모듈_디렉토리명>"` 추가 |
| [ ] | `TYPE_CHECKING` 블록 | `from .modules.<모듈명>.service import YourService` 추가 |

### 4-2. `packages/pia_prod/AI/DTO/stream_params.py` - cvEvent인 경우만 (4곳 수정)

> **retEvent / vqaEvent 모듈이면 이 섹션은 건너뛰세요.**
> RET은 `RetrievalBase`, VQA는 `VQABase`로 자동 파싱되어 수정이 불필요합니다.

| | 수정 위치 | 추가 내용 |
|:---:|:---|:---|
| [ ] | import 영역 (상단) | `from pia_prod.AI.modules.<모듈명>.param import YourModel` |
| [ ] | import 영역 (상단) | `from pia_prod.AI.modules.<모듈명>.config import YOUR_CV_CATEGORY` |
| [ ] | `AddStreamModel.cvEvent` Union 타입 | `YourModel` 추가 |
| [ ] | `parse_cv_events()` 메서드 | `elif name in YOUR_CV_CATEGORY:` 분기 추가 |

### 4-3. 이벤트 타입별 param.py & service.py 내부 차이

| | 항목 | CV | RET | VQA |
|:---:|:---|:---:|:---:|:---:|
| [ ] | param.py 모델 | `CategoryBase` 상속하여 개별 모델 정의 | 불필요 (`RetrievalBase` 직접 사용) | 불필요 (`VQABase` 직접 사용) |
| [ ] | config.py 카테고리 접미사 | `_cv` | `_ret` | `_vqa` |
| [ ] | service.py에서 사용하는 이벤트 키 | `CV_EVENT_KEY` | `RET_EVENT_KEY` | `VQA_EVENT_KEY` |
| [ ] | 백엔드 API 전달 위치 | `cvEvent` 필드 | `retEvent` 필드 | `vqaEvent` 필드 |

---

## Phase 5: 모델 파일 배치

| | 항목 | 비고 |
|:---:|:---|:---|
| [ ] | ONNX 모델 파일을 `assets/model/` 에 배치 | |
| [ ] | 파일 네이밍 규칙 준수 | `<모델명>_v<Major>.<Minor>.<Patch>.onnx` |
| [ ] | (CLS 모델 사용 시) CLS ONNX 도 배치 | |
| [ ] | (CLIP 사용 시) text_features JSON 배치 | `text_features_*.json` |
| [ ] | (VLM 사용 시) 프롬프트 파일 배치 | `text_prompt_*` 디렉토리 |
| [ ] | `.gitignore`에 의해 커밋에서 제외되는지 확인 | `.onnx`, `.engine`, `.pt` 제외됨 |
| [ ] | 테스트 fixture에 모델 다운로드 경로 명시 | HuggingFace 등 |

---

## Phase 6: 테스트 추론

> Phase 2에서 작성한 테스트 코드를 실행하여 구현이 정상 동작하는지 검증합니다.

| | 항목 | 명령어 |
|:---:|:---|:---|
| [ ] | import 테스트 통과 | `pytest test_<모듈명>.py::test_import_module -v` |
| [ ] | 단일 카메라 테스트 통과 | `pytest test_<모듈명>.py::test_single_camera -v` |
| [ ] | 배치 테스트 통과 | `pytest test_<모듈명>.py::test_batches -v` |
| [ ] | 전체 모듈 테스트 통과 | `pytest packages/pia_prod/AI/tests/modules/test_<모듈명>.py -v` |
| [ ] | 전체 테스트에 영향 없는지 확인 | `pytest .` |

> **체크리스트:**
> - `test_import_module` 통과 → `__init__.py` 등록 정상
> - `test_single_camera` 통과 → 모델 로딩 + 추론 파이프라인 정상
> - 알람 발생/종료 로그 확인 → 이벤트 매니저 상태 전이 정상

---

## Phase 7: 코드 품질 확인

> PR 제출 전 최종 확인 사항입니다.

| | 항목 | 확인 방법 |
|:---:|:---|:---|
| [ ] | pre-commit 통과 | `pre-commit run --all-files` |
| [ ] | ServiceBase 상속 확인 | `service.py` 에서 `class ...Service(ServiceBase)` |
| [ ] | EventBase 상속 확인 | `event.py` 에서 `class ...EventManager(EventBase)` |
| [ ] | GPU 디바이스 하드코딩 없음 | `grep -r "cuda:0"` 결과 없음 |
| [ ] | 불필요한 파일 미포함 | 로그, 캐시, 임시 스크립트 제외 |
| [ ] | Submodule 미사용 | `.gitmodules` 파일 없음 |
| [ ] | 모델 파일 미커밋 | `git status`에서 .onnx/.engine/.pt 없음 |

---

## Phase 8: 커밋 및 PR 제출


### 8-1. 커밋

| | 항목 | 예시 |
|:---:|:---|:---|
| [ ] | 커밋 메시지 규칙 준수 | `type: Subject` (50자 이내) |
| [ ] | 초기 커밋 | `chore: initial commit for aiprod-XXX-<설명>` |
| [ ] | 기능 커밋 | `feat: <기능명> 개발` |
| [ ] | 테스트 커밋 | `test: <테스트 설명>` |
| [ ] | 수정 커밋 | `fix: <수정 내용>` |

### 8-2. PR 제출

| | 항목 | 비고 |
|:---:|:---|:---|
| [ ] | `dev` 브랜치 대상으로 PR 생성 | |
| [ ] | PR 설명에 변경 사항 요약 작성 | |
| [ ] | 리뷰어 자동 지정 확인 | ready-pr.yaml 동작 확인 |
| [ ] | CI 체크 통과 확인 | |

---

## 빠른 참조: 수정 파일 총정리

### cvEvent 모듈인 경우 (대부분)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│ [NEW] modules/<모듈명>/__init__.py          빈 파일                             │
│ [NEW] modules/<모듈명>/config.py            설정 상수 (카테고리명 접미사: _cv)  │
│ [NEW] modules/<모듈명>/param.py             파라미터 모델 (CategoryBase 상속)   │
│ [NEW] modules/<모듈명>/event.py             이벤트 매니저 (EventBase 상속)      │
│ [NEW] modules/<모듈명>/service.py           서비스 클래스 (ServiceBase 상속)    │
│ [NEW] modules/<모듈명>/roi_manager.py       ROI 매니저 (ROIManagerBase 상속)    │
│                                                                                 │
│ [MOD] AI/__init__.py                        __all__ + override + TYPE_CHECKING  │
│ [MOD] DTO/stream_params.py                  import + Union + parse 분기         │
│                                                                                 │
│ [NEW] tests/modules/test_<모듈명>.py        import + 단일 + 배치 테스트         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 총 파일 수:  신규 생성 6개  +  기존 수정 2개  +  테스트 1개  =  9개 파일        │
│ 총 수정 지점: __init__.py 3곳  +  stream_params.py 4곳  =  7곳                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### retEvent / vqaEvent 모듈인 경우

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                                                                 │
│ [NEW] modules/<모듈명>/__init__.py          빈 파일                             │
│ [NEW] modules/<모듈명>/config.py            설정 상수 (접미사: _ret 또는 _vqa)  │
│ [---] modules/<모듈명>/param.py             불필요 (RetrievalBase/VQABase 사용) │
│ [NEW] modules/<모듈명>/event.py             이벤트 매니저 (EventBase 상속)      │
│ [NEW] modules/<모듈명>/service.py           서비스 (RET_EVENT_KEY/VQA_EVENT_KEY)│
│ [NEW] modules/<모듈명>/roi_manager.py       ROI 매니저 (ROIManagerBase 상속)    │
│                                                                                 │
│ [MOD] AI/__init__.py                        __all__ + override + TYPE_CHECKING  │
│ [---] DTO/stream_params.py                  수정 불필요                         │
│                                                                                 │
│ [NEW] tests/modules/test_<모듈명>.py        import + 단일 + 배치 테스트         │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│ 총 파일 수:  신규 생성 5개  +  기존 수정 1개  +  테스트 1개  =  7개 파일        │
│ 총 수정 지점: __init__.py 3곳                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```
