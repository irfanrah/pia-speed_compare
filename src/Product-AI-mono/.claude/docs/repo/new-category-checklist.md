# New Category Checklist

> **사용법**: 이 체크리스트를 복사하여 PR 또는 Issue에 붙여넣고, 항목을 하나씩 완료하며 체크합니다.  
> **참고 문서**: [new-category-guide.md](./new-category-guide.md)에서 각 항목의 상세 설명을 확인할 수 있습니다.

---

## 전체 진행 흐름

```
Phase 1          Phase 2          Phase 3          Phase 4          Phase 5          Phase 6          Phase 7
사전 준비   ───→  파일 생성    ───→  시스템 등록  ───→  모델 배치   ───→  테스트      ───→  품질 확인   ───→  PR 제출
                  (6개 신규)        (2개 수정)                       (1개 신규)
                                                                                    
[준비사항]       [모듈 핵심]       [기존 파일]       [assets/]       [pytest]         [pre-commit]      [dev 브랜치]
[파이프라인      [config.py]       [__init__.py]     [.onnx 파일]    [단일 카메라]    [상속 확인]       [커밋 메시지]
 유형 결정]      [param.py]        [stream_params    [.engine 파일]  [배치 테스트]    [하드코딩 확인]   [리뷰어 지정]
                 [event.py]         .py]                             [import 테스트]
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
│     ├─ 아니오 → [유형 A] OD 단독           (참고: ax_fall, samsung_fire)
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
   └─ [유형 G] 전처리 집약형                   (참고: yeonsei_smoke)
```

---

## Phase 2: 모듈 디렉토리 및 파일 생성

> 새 모듈의 핵심 파일 6개를 생성합니다.

### 2-1. 디렉토리 생성

| | 항목 | 비고 |
|:---:|:---|:---|
| [ ] | `packages/pia_prod/AI/modules/<프로젝트>_<카테고리>/` 디렉토리 생성 | `mkdir -p ...` |
| [ ] | 네이밍이 `프로젝트명_카테고리` 규칙을 따르는지 확인 | 예: `aramco_loitering` |

### 2-2. `__init__.py` (빈 파일)

| | 항목 |
|:---:|:---|
| [ ] | `__init__.py` 빈 파일 생성 |

### 2-3. `config.py` (설정 상수)

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | 모델 ONNX 경로 정의 | `os.getenv()` 로 감싸져 있는가? |
| [ ] | 모델 TensorRT 엔진 경로 정의 | `os.getenv()` 로 감싸져 있는가? |
| [ ] | 입력 사이즈 정의 | 예: `OD_INPUT_SIZE = [640, 640]` |
| [ ] | OD confidence threshold 정의 | 예: `OD_CONFIDENCE_THRESHOLD = 0.5` |
| [ ] | NMS IoU threshold 정의 | 예: `OD_NMS_THRESHOLD = 0.35` |
| [ ] | 추론 시간 간격 정의 | 예: `OD_TIME_INTERVAL_SECOND = 0.3` |
| [ ] | 카테고리 이름 리스트 정의 | **한글 + 영문 모두** 포함되어 있는가? |
| | | 예: `YOUR_CV_CATEGORY = ["카테고리_cv", "category_cv"]` |
| [ ] | GPU/CPU 디바이스 하드코딩이 없는지 확인 | `cuda:0` 직접 사용 금지 |
| [ ] | (CLS 모델 필요 시) CLS 관련 상수 정의 | 입력 사이즈, threshold 등 |
| [ ] | (Tracker 필요 시) Tracker 관련 상수 정의 | max_age, min_hits, iou 등 |

### 2-4. `param.py` (파라미터 모델)

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `CategoryBase` 를 상속하였는가? | `from pia_prod.AI.DTO.param_base import CategoryBase` |
| [ ] | 카테고리 고유 threshold 필드 정의 | 예: `od_threshold`, `cls_threshold` 등 |
| [ ] | 기본값이 `config.py` 상수를 참조하는가? | 하드코딩된 숫자 없이 상수 사용 |

### 2-5. `event.py` (이벤트 매니저)

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `EventBase` 를 상속하였는가? | `from pia_prod.AI.bases.event_base import EventBase` |
| [ ] | `update()` 메서드가 구현되었는가? | 추상 메서드 반드시 구현 |
| [ ] | `get_alarm()` 메서드가 구현되었는가? | `update()`는 상태 갱신만, `get_alarm()`에서 알람 판정 |
| [ ] | `STATUS_TRANSITION` 행렬을 활용하여 상태 전이를 판정하는가? | `self.STATUS_TRANSITION[before][int(trigger)]` |
| [ ] | 상태 1(시작) 또는 3(종료) 시에만 알람을 반환하는가? | `if now_status in [1, 3]` |
| [ ] | 스트림별(카메라별) 독립 상태 관리가 되는가? | `defaultdict(int)` 등으로 스트림 분리 |
| [ ] | (Tracker 사용 시) 트랙 ID별 상태 관리 구현 | 배회 감지의 경우 필수 |

### 2-6. `service.py` (서비스 클래스 - 핵심)

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `ServiceBase` 를 상속하였는가? | `from pia_prod.AI.bases.service_base import ServiceBase` |
| [ ] | `_init_values()` 구현 | Letterbox 인스턴스, 트래커 등 초기화 |
| [ ] | `_load_model()` 구현 | `PiaONNXTensorRTModel()` 등으로 모델 로딩 |
| [ ] | `_load_event_manager()` 구현 | 이벤트 매니저 인스턴스를 **return** 하는가? |
| [ ] | `_load_roi_manager()` 오버라이드 (필요 시) | 커스텀 ROI 매니저를 return |
| [ ] | `_detect(**datas)` 구현 | 메인 감지 파이프라인 |
| [ ] | `_detect()` 반환값 형식 준수 | 성공: `(alarms, batches, stream_ids, user_params, is_needed_cvt_color)` |
| | | 감지 없음: `None` |
| [ ] | `__del__()` 에서 GPU 메모리 해제 | `free_autobackend()` 호출 |
| [ ] | `logging_flag` 활용한 디버그 출력 구현 (권장) | `if getattr(self, "logging_flag", False):` |

### 2-7. `roi_manager.py` (ROI 매니저)

| | 항목 | 확인 포인트 |
|:---:|:---|:---|
| [ ] | `ROIManagerBase` 를 상속하였는가? | `from pia.vision.roi.roi_manager import ROIManagerBase` |
| [ ] | `process_batches_with_roi()` 구현 | 배치 크롭 및 카테고리 인덱스 반환 |
| [ ] | ROI가 비어있을 때 전체 이미지를 사용하는 폴백 처리 | `if len(coordinates) == 0: → 전체 이미지` |
| [ ] | `category_list`에 한글/영문 카테고리명 모두 포함 | config.py의 카테고리와 일치 |

### 2-8. 선택 파일 (필요한 것만 체크)

| | 파일 | 필요 조건 |
|:---:|:---|:---|
| [ ] | `postprocess.py` | 커스텀 NMS, 좌표 변환 등 표준 후처리로 부족할 때 |
| [ ] | `preprocess.py` | 특수 전처리 (색상 필터, 배경 제거 등) |
| [ ] | `tracker.py` | 객체 추적 필요 시 (OC-SORT 등) |
| [ ] | `debug_utils.py` | 바운딩박스 시각화, 프레임 저장 등 디버깅 |
| [ ] | `func.py` | 모듈 전용 유틸리티 함수 |
| [ ] | `prompts.py` | VLM/CLIP 텍스트 프롬프트 정의 |

---

## Phase 3: 시스템 등록

> 이 단계를 빠뜨리면 모듈이 시스템에서 인식되지 않습니다.
> **중요**: 등록 방식은 이벤트 타입(CV / RET / VQA)에 따라 다릅니다.

### 3-0. 먼저 이벤트 타입을 확인하세요 (택 1)

| | 타입 | 해당하는 경우 | 카테고리명 접미사 | stream_params.py 수정 |
|:---:|:---|:---|:---:|:---:|
| [ ] | **cvEvent** | OD/CLS/Tracker 기반 감지 (대부분) | `_cv` | **필요 (4곳)** |
| [ ] | **retEvent** | PE(Perception Encoder) Retrieval 기반 | `_ret` | 불필요 |
| [ ] | **vqaEvent** | InternVL3 VQA 기반 | `_vqa` | 불필요 |

### 3-1. `packages/pia_prod/AI/__init__.py` (3곳 수정) - 모든 타입 공통

| | 수정 위치 | 추가 내용 |
|:---:|:---|:---|
| [ ] | `__all__` 리스트 | `"YourService"` 추가 |
| [ ] | `_SERVICE_MODULE_OVERRIDE` 딕셔너리 | `"YourService": "<모듈_디렉토리명>"` 추가 |
| [ ] | `TYPE_CHECKING` 블록 | `from .modules.<모듈명>.service import YourService` 추가 |

### 3-2. `packages/pia_prod/AI/DTO/stream_params.py` - cvEvent인 경우만 (4곳 수정)

> **retEvent / vqaEvent 모듈이면 이 섹션은 건너뛰세요.**
> RET은 `RetrievalBase`, VQA는 `VQABase`로 자동 파싱되어 수정이 불필요합니다.

| | 수정 위치 | 추가 내용 |
|:---:|:---|:---|
| [ ] | import 영역 (상단) | `from pia_prod.AI.modules.<모듈명>.param import YourModel` |
| [ ] | import 영역 (상단) | `from pia_prod.AI.modules.<모듈명>.config import YOUR_CV_CATEGORY` |
| [ ] | `AddStreamModel.cvEvent` Union 타입 | `YourModel` 추가 |
| [ ] | `parse_cv_events()` 메서드 | `elif name in YOUR_CV_CATEGORY:` 분기 추가 |

### 3-3. 이벤트 타입별 param.py & service.py 내부 차이

| | 항목 | CV | RET | VQA |
|:---:|:---|:---:|:---:|:---:|
| [ ] | param.py 모델 | `CategoryBase` 상속하여 개별 모델 정의 | 불필요 (`RetrievalBase` 직접 사용) | 불필요 (`VQABase` 직접 사용) |
| [ ] | config.py 카테고리 접미사 | `_cv` | `_ret` | `_vqa` |
| [ ] | service.py에서 사용하는 이벤트 키 | `CV_EVENT_KEY` | `RET_EVENT_KEY` | `VQA_EVENT_KEY` |
| [ ] | 백엔드 API 전달 위치 | `cvEvent` 필드 | `retEvent` 필드 | `vqaEvent` 필드 |

---

## Phase 4: 모델 파일 배치

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

## Phase 5: 테스트 작성 및 실행

### 5-1. 테스트 파일 생성

| | 항목 |
|:---:|:---|
| [ ] | `packages/pia_prod/AI/tests/modules/test_<모듈명>.py` 파일 생성 |

### 5-2. 테스트 케이스 구현

| | 테스트 | 내용 |
|:---:|:---|:---|
| [ ] | `test_import_module` | `from pia_prod.AI import YourService` + `assert` |
| [ ] | `test_single_camera` | 단일 카메라 추론 (모델 다운로드 fixture 포함) |
| | | - 모델 다운로드 fixture 작성 |
| | | - 영상 다운로드 fixture 작성 |
| | | - AddStreamModel 구성 fixture 작성 |
| | | - Queue 기반 프레임 입력 |
| | | - `stop_thread(q)` 로 쓰레드 종료 (`thread_stop_signal` 모듈 사용) |
| [ ] | `test_batches` | 멀티 카메라 배치 테스트 (batch_size=4) |

> **참고**: 테스트 fixture의 `scope`는 모듈마다 다릅니다.
> - 모델 다운로드: `scope="function"` (aramco_loitering) 또는 `scope="module"` (khonkaen_helmet)
> - AddStreamModel: 대부분 `scope="module"`
> - 기존 모듈의 테스트 파일을 참고하여 적절한 scope를 선택하세요.

### 5-3. 테스트 실행

| | 항목 | 명령어 |
|:---:|:---|:---|
| [ ] | 모듈 테스트 통과 | `pytest packages/pia_prod/AI/tests/modules/test_<모듈명>.py -v` |
| [ ] | 전체 테스트에 영향 없는지 확인 | `pytest .` |

---

## Phase 6: 코드 품질 확인

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

## Phase 7: 커밋 및 PR 제출

### 7-1. 커밋

| | 항목 | 예시 |
|:---:|:---|:---|
| [ ] | 커밋 메시지 규칙 준수 | `type: Subject` (50자 이내) |
| [ ] | 초기 커밋 | `chore: initial commit for aiprod-XXX-<설명>` |
| [ ] | 기능 커밋 | `feat: <기능명> 개발` |
| [ ] | 테스트 커밋 | `test: <테스트 설명>` |
| [ ] | 수정 커밋 | `fix: <수정 내용>` |

### 7-2. PR 제출

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
```
