# Product-AI-mono 프로젝트 구조 가이드

> **대상 독자**: 이 프로젝트에 처음 합류하는 개발자
> **목적**: 코드를 읽기 전에 전체 구조와 설계 원리를 빠르게 파악하기 위한 문서

---

## 1. 프로젝트 한 줄 요약

> **CCTV 영상을 실시간으로 분석하여 이상 상황(낙상, 화재, 무기, 배회 등)을 감지하고, 알람을 발생시키는 AI 추론 파이프라인의 통합 저장소(Monorepo)입니다.**

각 고객사/프로젝트별 AI 감지 카테고리를 **독립된 모듈**로 관리하되, 공통 추상 클래스를 상속하여 **동일한 구조와 인터페이스**를 보장합니다.

---

## 2. 기술 스택

| 분류 | 기술 | 용도 |
|:---:|:-----|:-----|
| **언어** | Python 3.10+ | 전체 코드베이스 |
| **딥러닝** | PyTorch, Ultralytics | 모델 추론 및 전처리 |
| **최적화** | TensorRT, ONNX Runtime | GPU 추론 가속 |
| **VLM** | InternVL3, CLIP | 비전-언어 모델 기반 감지 |
| **메시징** | RabbitMQ | 알람 메시지 전송 (AI -> Backend) |
| **캐시** | Redis | 로깅 상태 관리 |
| **스토리지** | AWS S3 (DataLake) | 썸네일/이미지 저장 |
| **테스트** | pytest | 단위/통합 테스트 |
| **CI/CD** | GitHub Actions | PR 자동 할당, 브랜치 생성 |

---

## 3. 최상위 디렉토리 구조

```
Product-AI-mono/
│
├─ packages/                        # ★ 메인 소스코드 (이 안이 핵심)
│   └─ pia_prod/
│       └─ AI/                      #    모든 AI 로직이 여기에 있음
│
├─ assets/                          # 모델 가중치 & 테스트 영상 (테스트 실행 시 Hugging Face/NAS에서 자동 다운로드)
│   ├─ model/                       #   .onnx / .engine / .pt / VLM 모델
│   └─ videos/                      #   테스트용 .mp4 파일
│
├─ logs/                            # 추론 로그, pytest 로그
├─ build/                           # 빌드 산출물 (자동 생성)
│
├─ .github/workflows/               # CI/CD 워크플로우 3개
├─ .githooks/                       # 커밋 메시지 템플릿 & 훅 스크립트
│
├─ setup.py                         # pip install . 의존성 정의
├─ Makefile                         # make init → pre-commit 훅 설정
├─ pyproject.toml                   # 빌드 시스템 설정
├─ pytest.ini                       # pytest 설정
├─ .pre-commit-config.yaml          # pre-commit 설정
├─ .gitignore                       # 모델/영상/로그 등 제외 규칙
└─ README.md                        # 초기 설정 가이드
```

> **핵심**: 실제 코드는 거의 전부 `packages/pia_prod/AI/` 안에 있습니다.

---

## 4. 핵심 패키지 상세 구조 (`packages/pia_prod/AI/`)

```
AI/
│
│  ┌─────────────────────── 진입점 ───────────────────────┐
├─ __init__.py              │ 서비스 클래스 Lazy Import     │
├─ global_config.py         │ 전역 상수 키 (문자열 상수)    │
│  └────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────── 추상 베이스 ──────────────────┐
├─ bases/                   │                               │
│  ├─ service_base.py       │ ServiceBase (모든 서비스의 부모)      │
│  └─ event_base.py         │ EventBase (이벤트 상태 머신)  │
│  └────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────── 데이터 모델 ──────────────────┐
├─ DTO/                     │                               │
│  ├─ param_base.py         │ 서비스 파라미터 공통 Base 모델 (Pydantic)│
│  ├─ stream_params.py      │ AddStreamModel (API 요청 모델)│
│  └─ output_handler.py     │ 알람 출력 핸들링              │
│  └────────────────────────────────────────────────────────┘
│
│  ┌─────────────── ★ AI 카테고리 모듈 (21개) ────────────┐
├─ modules/                 │                               │
│  ├─ aramco_loitering/     │ 아람코 - 배회 감지            │
│  ├─ ax_fall/              │ AX - 낙상 감지                │
│  ├─ ax_harness/           │ AX - 안전벨트 미착용          │
│  ├─ DaeGu_crowd_people/   │ 대구 - 군중 밀집              │
│  ├─ Daegu_intrusion/      │ 대구 - 침입 감지              │
│  ├─ gangnam_falldown/     │ 강남 - 낙상 (InternVL3)       │
│  ├─ gangnam_violence/     │ 강남 - 폭력 (InternVL3)       │
│  ├─ hyundai_esfalldown/   │ 현대 - 에스컬레이터 낙상 (PE)     │
│  ├─ khonkaen_helmet/      │ 콘깬 - 헬멧 미착용            │
│  ├─ khonkaen_weapon/      │ 콘깬 - 무기 감지              │
│  ├─ kumho_pinch/          │ 금호타이어 - 협착 감지        │
│  ├─ perception_encoder/   │ CLIP 기반 Perception Encoder  │
│  ├─ pe_violence/          │ PE 기반 폭력 감지             │
│  ├─ samsung_fire/         │ 삼성 - 화재 감지              │
│  ├─ samsung_internvl3/    │ 삼성 - InternVL3 범용         │
│  ├─ vanguard_patient/     │ 뱅가드 - 환자 배회            │
│  ├─ vehicle_reverse/      │ 차량 역주행 감지              │
│  ├─ yeonsei_falldown/     │ 연세 - 낙상 감지              │
│  ├─ yeonsei_smoke/        │ 연세 - 연기 감지              │
│  ├─ yonsei_tailgate/      │ 연세 - 테일게이트             │
│  └─ yonsei_walljump/      │ 연세 - 담장 넘기              │
│  └────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────── 유틸리티 ─────────────────────┐
├─ utils/                   │                               │
│  ├─ init.py               │ 인프라 컴포넌트 초기화 및 전역 선언   │
│  │                        │ (Logger/Redis/RabbitMQ/S3)   │
│  ├─ utils.py              │ 헬퍼 함수 (파라미터 변환, ROI 추출, │
│  │                        │ threshold 체크, 모델 메모리 해제) │
│  └─ log_templates.py      │ 표준 로그 메시지 템플릿       │
│  └────────────────────────────────────────────────────────┘
│
│  ┌─────────────────────── 테스트 ───────────────────────┐
├─ tests/                   │                               │
│  ├─ conftest.py           │ 공통 fixture                  │
│  ├─ thread_stop_signal.py │ 쓰레드 종료 시그널 유틸       │
│  └─ modules/              │ 모듈별 테스트 (23개 파일)     │
│  └────────────────────────────────────────────────────────┘
│
└─ validate/
   └─ calc_score.py         # 정확도 검증 (현재 미구현/TODO)
```

---

## 5. 핵심 아키텍처 이해하기

### 5.1 전체 데이터 흐름

아래 다이어그램은 백엔드에서 스트림이 등록되고, 실시간 영상 분석 후 알람이 발생하기까지의 **전체 데이터 흐름**을 보여줍니다.

```
 ┌──────────────────────────────────────────────────────────────────┐
 │ 백엔드 서버                                                     │
 │ POST /api/v1/stream/add                                         │
 │ { cameraId, cameraUrl, organization, cvEvent: [...] }           │
 └───────────────────────┬──────────────────────────────────────────┘
                         │ JSON 요청
                         ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ AddStreamModel (DTO/stream_params.py)                           │
 │                                                                 │
 │ cvEvent의 name 필드를 기반으로 올바른 Param 모델로 자동 라우팅  │
 │ 예: name="helmet_cv" → HelmetModel 로 파싱                      │
 └───────────────────────┬──────────────────────────────────────────┘
                         │ 파싱된 파라미터
                         ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ RTSP 프레임 수신 (외부 모듈)                                    │
 │                                                                 │
 │ 카메라 URL로부터 실시간 프레임을 받아                           │
 │ analysis_data_queue (Queue) 에 적재                             │
 │ { batches: [frame1, frame2, ...],                               │
 │ stream_ids: ["cam1", "cam2", ...],                              │
 │ user_params: [param1, param2, ...] }                            │
 └───────────────────────┬──────────────────────────────────────────┘
                         │ Queue.get()
                         ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ ServiceBase.thread_ai_inference()  [데몬 쓰레드]                │
 │                                                                 │
 │ while True:                                                     │
 │ ┌─────────────────────────────────────────────────────────┐     │
 │    │ 1. 데이터 수신 (Queue에서 배치 가져오기)                  ││
 │    │                     ▼                                    │ │
 │    │ 2. _detect(**datas) 호출 [추상 메서드 - 각 모듈이 구현]   ││
 │    │    ┌─────────────────────────────────────────────┐       │ │
 │    │    │ a. ROI 크롭 (roi_manager)                   │       │ │
 │    │    │ b. 전처리 (Letterbox, 정규화)                │       ││
 │    │    │ c. 모델 추론 (OD / CLS / VLM)               │       │ │
 │    │    │ d. 후처리 (NMS, threshold)                   │       ││
 │    │    │ e. 이벤트 판정 (event_manager.update())      │       ││
 │    │    └─────────────────────────────────────────────┘       │ │
 │    │                     ▼                                    │ │
 │    │ 3. send_alarm() → 상태 전이가 발생한 경우에만 알람 전송   ││
 │ └─────────────────────────────────────────────────────────┘     │
 └───────────────────────┬──────────────────────────────────────────┘
                         │ 알람 발생 시
                         ▼
 ┌──────────────────────────────────────────────────────────────────┐
 │ output_handler.py → match_outputs()                             │
 │                                                                 │
 │ ┌─ PROD 환경 ─────────────────────────────────────────────┐     │
 │  │  RabbitMQ로 알람 메시지 전송                             │   │
 │  │  { cameraId, isStart, thumbnail, uuid, ... }             │   │
 │  │  → cv_queue / ret_queue / vqa_queue                      │   │
 │ └──────────────────────────────────────────────────────────┘    │
 │ ┌─ DEV 환경 ──────────────────────────────────────────────┐     │
 │  │  calc_score() 호출 (개발용 검증)                          │  │
 │ └──────────────────────────────────────────────────────────┘    │
 └──────────────────────────────────────────────────────────────────┘
```

---

### 5.2 추상 클래스 상속 구조

모든 AI 모듈은 아래 두 추상 클래스를 **반드시 상속**해야 합니다.

#### ServiceBase (`bases/service_base.py`)

```
┌──────────────────────────────────────────────────────────────────┐
│ ServiceBase (ABC)                                                │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ [생성자에서 자동 실행되는 초기화 순서]                           │
│ ┌──────────────────────────────────────────────────────────┐     │
│  │ 1. _init_values()          ← 추상 메서드 (개발자 구현)     │  │
│  │ 2. _load_model()           ← 추상 메서드 (개발자 구현)     │  │
│  │ 3. _load_roi_manager()     ← final (ROIManagerBase 반환)  │   │
│  │ 4. _load_event_manager()   ← 추상 메서드 (개발자 구현)     │  │
│  │ 5. thread_ai_inference()   ← final (추론 쓰레드 시작)      │  │
│  │ 6. thread_run_get_logging_state() ← final (로깅 상태 감시) │  │
│ └──────────────────────────────────────────────────────────┘     │
│                                                                  │
│ ┌──── 개발자가 구현해야 할 추상 메서드 (4개) ─────────────────┐  │
│  │                                                             │ │
│  │  _init_values()                                             │ │
│  │    서비스 고유 초기값 설정                                    ││
│  │    예: Letterbox 인스턴스, 트래커, 전처리기 초기화             │  │
│  │                                                             │ │
│  │  _load_model()                                              │ │
│  │    추론 모델 로딩 (ONNX/TensorRT)                            ││
│  │    예: PiaONNXTensorRTModel(model_path=..., device=...)     │ │
│  │                                                             │ │
│  │  _load_event_manager()                                      │ │
│  │    이벤트 매니저 인스턴스 반환                                 │    │
│  │    예: return HelmetEventManager()                           ││
│  │                                                             │ │
│  │  _detect(**datas)                                           │ │
│  │    메인 감지 로직 (핵심!)                                     │     │
│  │    입력: batches, stream_ids, user_params                   │ │
│  │    출력: (alarms, batches, stream_ids, user_params,         │ │
│  │           is_needed_cvt_color) 또는 None                    │ │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ ┌──── 이미 구현된 final 메서드 (건드리지 않음) ───────────────┐  │
│  │                                                             │ │
│  │  thread_ai_inference()   추론 루프 (Queue → _detect → 알람) │ │
│  │  send_alarm()            RabbitMQ 알람 전송 로직             ││
│  │  get_alarm_with_uuid()   UUID로 True/False 매칭 (중복 방지) │ │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ ┌──── 오버라이드 가능한 메서드 ───────────────────────────────┐  │
│  │                                                             │ │
│  │  _load_roi_manager()     기본 ROIManagerBase 반환           │ │
│  │    ※ 코드상 @final이 붙어있지만, 대부분의 모듈이             ││
│  │       커스텀 ROI 매니저를 반환하도록 오버라이드합니다.         │    │
│  │       (Python @final은 런타임에서 강제되지 않음)              │     │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ [주요 인스턴스 변수]                                             │
│ - analysis_data_queue : 프레임 수신 큐                           │
│ - frame_cnt           : 프레임 카운터                            │
│ - logging_flag        : Redis 기반 로깅 ON/OFF                   │
│ - roi_manager         : ROI 처리기                               │
│ - alarm_event_manager : 이벤트 상태 머신                         │
│ - alarm_dict_with_uuid: 알람 UUID 추적 (deque)                   │
└──────────────────────────────────────────────────────────────────┘
```

#### EventBase (`bases/event_base.py`)

```
┌──────────────────────────────────────────────────────────────────┐
│ EventBase (ABC)                                                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│ [개발자가 구현해야 할 추상 메서드 (1개)]                         │
│ ┌─────────────────────────────────────────────────────────────┐  │
│  │  update(*args, **kwds) → list                               │ │
│  │                                                             │ │
│  │  - 프레임 분석 결과를 받아 이벤트 상태를 갱신                  │    │
│  │  - 상태 전이가 발생하면: [["stream_id", is_start]] 반환       │     │
│  │  - 전이가 없으면:       [] 반환 (빈 리스트)                   │     │
│ └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│ [내장 상태 머신]                                                 │
│                                                                  │
│ EVENT_STATUS_DICT:                                               │
│ ┌──────────┬─────────────┬──────────────────────────────────┐    │
│  │ 상태코드  │ 값           │ 의미                             │ │
│ ├──────────┼─────────────┼──────────────────────────────────┤    │
│  │    0     │ "no_event"  │ 이벤트 없음 (평상시)               │ │
│  │    1     │ True        │ 이벤트 시작됨 → ★ 알람 발생       │  │
│  │    2     │ "continue"  │ 이벤트 지속 중 (알람 없음)         │ │
│  │    3     │ False       │ 이벤트 종료됨 → ★ 알람 발생       │  │
│ └──────────┴─────────────┴──────────────────────────────────┘    │
│                                                                  │
│ STATUS_TRANSITION (상태 전이 행렬):                              │
│ ┌─────────────────┬──────────────────┬────────────────────────┐  │
│  │ 현재 상태        │ 이상 미감지 (0)   │ 이상 감지됨 (1)        │     │
│ ├─────────────────┼──────────────────┼────────────────────────┤  │
│  │ 0 (no_event)    │ → 0 (no_event)   │ → 1 (True) ★시작 알람  │ │
│  │ 1 (True)        │ → 3 (False) ★종료 │ → 2 (continue)        │ │
│  │ 2 (continue)    │ → 3 (False) ★종료 │ → 2 (continue)        │ │
│  │ 3 (False)       │ → 0 (no_event)   │ → 1 (True) ★시작 알람  │ │
│ └─────────────────┴──────────────────┴────────────────────────┘  │
│                                                                  │
│ ★ 표시된 곳에서만 알람 메시지가 발송됩니다.                      │
│ 즉, "상태가 전이되는 순간"에만 알람이 발생합니다.                │
└──────────────────────────────────────────────────────────────────┘
```

**상태 전이 시각화 (시간 순서):**

```
시간 →  t1    t2    t3    t4    t5    t6    t7    t8    t9
감지    미감지  감지   감지   감지   미감지  미감지  감지   감지
        │      │      │      │      │      │      │          │
상태    0──→──1──→──2──→──2──→──2──→──3──→──0──→──1──→──2
              │                        │                     │
              ▼                        ▼              ▼
         ★ 시작 알람              ★ 종료 알람     ★ 시작 알람
         (True 전송)             (False 전송)    (True 전송)
```

---

### 5.3 개별 모듈의 표준 파일 구성

각 AI 카테고리 모듈은 아래 구조를 따릅니다.

```
modules/<프로젝트>_<카테고리>/
│
│  ┌─────── 필수 파일 (6개) ─────────────────────────────────────────────┐
│  │                                                                     │
├─ __init__.py          빈 파일 (Python 패키지 마커)                     │
│  │                                                                     │
├─ config.py            설정 상수 정의                                   │
│  │                    - 모델 경로 (ONNX, TensorRT)                     │
│  │                    - 임계값 (OD confidence, NMS IoU, CLS 등)        │
│  │                    - 카테고리 이름 리스트 (한글/영문)               │
│  │                    - 입력 사이즈, 타겟 클래스 등                    │
│  │                                                                     │
├─ param.py             파라미터 모델 (Pydantic)                         │
│  │                    - CategoryBase 상속                              │
│  │                    - 카테고리 고유 임계값 필드 정의                 │
│  │                                                                     │
├─ event.py             이벤트 매니저                                    │
│  │                    - EventBase 상속                                 │
│  │                    - update() 메서드 구현                           │
│  │                    - 스트림별 상태 관리 & 전이 판정                 │
│  │                                                                     │
├─ service.py           서비스 클래스 (★ 핵심 파일)                      │
│  │                    - ServiceBase 상속                               │
│  │                    - _init_values(), _load_model(),                 │
│  │                      _load_event_manager(), _detect() 구현          │
│  │                                                                     │
├─ roi_manager.py       ROI 매니저                                       │
│  │                    - ROIManagerBase 상속                            │
│  │                    - 폴리곤 좌표 → 직사각형 확장 → 배치 크롭        │
│  └─────────────────────────────────────────────────────────────────────┘
│
│  ┌─────── 선택 파일 (필요 시 추가) ────────────────────────────────────┐
│  │                                                                          │
├─ postprocess.py       후처리 함수 (커스텀 NMS, 좌표 변환 등)                │
├─ preprocess.py        전처리 함수 (색상 필터, 배경 제거 등)                 │
├─ tracker.py           객체 추적기 (OC-SORT 커스텀 등)                       │
├─ debug_utils.py       디버그 시각화 (bbox 그리기, 스냅샷 저장)              │
├─ func.py              모듈 전용 유틸리티 함수                               │
├─ prompts.py           VLM 프롬프트 정의 (InternVL3, CLIP용)                 │
│  └─────────────────────────────────────────────────────────────────────┘
```

---

### 5.4 DTO(Data Transfer Object) 계층 구조

백엔드 API와 내부 모듈 사이의 데이터 전달 구조입니다.

```
                          ┌─────────────────────┐
                          │ AddStreamModel      │  ← 백엔드 API 요청 전체
                          │ (stream_params.py)             │
                          ├─────────────────────┤
                          │ cameraId: int                  │
                          │ cameraUrl: str                 │
                          │ organization: str              │
                          │ timestamp: str                 │
                          │ cvEvent: [...]   ───┼──→ CV 이벤트 목록
                          │ retEvent: [...]  ───┼──→ Retrieval 이벤트 목록
                          │ vqaEvent: [...]  ───┼──→ VQA 이벤트 목록
                          └─────────────────────┘
                                                           │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
          ┌─────────────┐  ┌──────────────┐  ┌──────────┐
          │ CategoryBase │  │RetrievalBase │  │ VQABase    │
          │(param_base.py)│ │(param_base.py)│ │(param_base)│
          ├─────────────┤  ├──────────────┤  ├──────────┤
          │name          │  │name          │  │name        │
          │incidentThres │  │incidentThres │  │incidentT   │
          │incidentTimeo │  │abnormalText  │  │roi         │
          │roi       │          │normalText    │  └──────────┘
          └──────┬──────┘  │topCandidates                  │
                 │          └──────────────┘
      ┌──────────┼──────────┬──────────────┐
      ▼          ▼          ▼              ▼
┌──────────┐┌──────────┐┌───────────┐┌──────────┐
│HelmetModel││LoiteringM││ PatientM  ││ FallModel│  ...
│           ││odel      ││odel       ││                    │
├──────────┤├──────────┤├───────────┤├──────────┤
│od_thres   ││confidence││od_thres   ││od_thres            │
│iou_thres  ││nms_thres ││iou_thres  ││iou_thres           │
│cls_thres  ││          ││cls_thres  ││                    │
└──────────┘└──────────┘└───────────┘└──────────┘
```

**카테고리 라우팅 메커니즘** (`stream_params.py` 내부):

```python
# 백엔드에서 전달된 이벤트의 "name" 필드를 기반으로 올바른 Pydantic 모델로 자동 파싱
# 예시:
#   name = "helmet_cv"    → HELMET_CV_CATEGORY에 포함 → HelmetModel(**event)
#   name = "배회_cv"       → LOITERING_CV_CATEGORY에 포함 → LoiteringModel(**event)
#   name = "환자배회_cv"   → PATIENT_CV_CATEGORY에 포함 → PatientModel(**event)
```

---

### 5.5 Lazy Import 메커니즘

`AI/__init__.py`는 **서비스 클래스를 요청 시에만 로딩**하여 불필요한 GPU 메모리 점유를 방지합니다.

```
사용자 코드:  from pia_prod.AI import WeaponService
                                                │
                                          ▼
              __getattr__("WeaponService") 호출
                                                           │
                         ┌────────────────┴────────────────┐
                         │ 1. __all__ 목록에 있는지 확인   │
                         │ 2. _SERVICE_MODULE_OVERRIDE에서 │
                         │    모듈명 조회                  │
                         │    "WeaponService"              │
                         │     → "khonkaen_weapon"         │
                         │ 3. 동적 import 수행             │
                         │    import_module(               │
                         │      "pia_prod.AI.modules       │
                         │       .khonkaen_weapon.service")│
                         │ 4. WeaponService 클래스 반환    │
                         └─────────────────────────────────┘
```

---

## 6. 이벤트 타입 분류

이 시스템에서 AI 감지 이벤트는 **3가지 타입**으로 분류됩니다.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 이벤트 타입 분류                                                            │
├──────────┬────────────────────────┬─────────────────────────────────────────┤
│ 타입      │ 감지 방식               │ 사용 모듈                             │
├──────────┼────────────────────────┼─────────────────────────────────────────┤
│          │ OD/CLS/Tracker/전처리   │ khonkaen_helmet, khonkaen_weapon       │
│ cvEvent  │ 기반 감지               │ samsung_fire, ax_fall, ax_harness      │
│          │                        │ aramco_loitering, vanguard_patient      │
│          │ 카테고리명 접미사: _cv   │ Daegu_intrusion, DaeGu_crowd_people   │
│          │                        │ yeonsei_smoke, yeonsei_falldown         │
│          │ 개별 Param 모델 필요    │ yonsei_tailgate, yonsei_walljump       │
│          │ (stream_params 등록)    │ kumho_pinch, vehicle_reverse           │
├──────────┼────────────────────────┼─────────────────────────────────────────┤
│          │ PE(Perception Encoder) │ perception_encoder (PE 범용)            │
│ retEvent │ 이미지-텍스트 유사도     │ pe_violence (PE 폭력)                 │
│          │ 기반 감지               │ hyundai_esfalldown (현대 에스컬레이터) │
│          │                        │                                         │
│          │ 카테고리명 접미사: _ret  │ ※ RetrievalBase로 일괄 파싱           │
│          │                        │   (stream_params 등록 불필요)           │
│          │                        │   abnormalText, normalText 필드 포함    │
├──────────┼────────────────────────┼─────────────────────────────────────────┤
│          │ InternVL3 VLM          │ gangnam_falldown (강남 낙상)            │
│ vqaEvent │ 질의응답 기반 감지       │ gangnam_violence (강남 폭력)          │
│          │                        │ samsung_internvl3 (삼성 범용)           │
│          │ 카테고리명 접미사: _vqa  │                                       │
│          │                        │ ※ VQABase로 일괄 파싱                   │
│          │                        │   (stream_params 등록 불필요)           │
└──────────┴────────────────────────┴─────────────────────────────────────────┘

※ 참고: perception_encoder 모듈은 특수하게 _ret 카테고리(retEvent 사용)와
  _vqa 카테고리(cvEvent parser에서 VQABase로 파싱)를 모두 가지고 있습니다.
```

---

## 7. 파이프라인 유형별 패턴

동일한 ServiceBase를 상속하지만, 모듈마다 `_detect()` 내부 파이프라인이 다릅니다.

```
[유형 1] OD 단독 ─────────────────────────────────────────────
  대표: ax_fall

  프레임 → ROI 크롭 → Letterbox → OD 추론 → NMS → 이벤트 판정 → 알람

[유형 2] OD + ROI 기반 판정 ───────────────────────────────────
  대표: Daegu_intrusion, kumho_pinch

  프레임 → ROI 크롭 → Letterbox → OD 추론 → NMS → ROI 내부 존재 여부 판정 → 이벤트 → 알람

[유형 3] OD + Classification ──────────────────────────────────
  대표: khonkaen_helmet

  프레임 → ROI 크롭 → OD 추론 → 감지 객체 크롭 → CLS 추론 → 이벤트 판정 → 알람

[유형 3-b] OD + OD 캐스케이드 (2단계 검출) ────────────────────
  대표: khonkaen_weapon

  프레임 → ROI 크롭 → PGIE(사람 OD) → 감지 객체 크롭 → SGIE(무기 OD) → 이벤트 판정 → 알람

[유형 4] OD + Tracker ─────────────────────────────────────────
  대표: aramco_loitering

  프레임 → ROI 크롭 → OD 추론 → NMS → 트래커(OC-SORT) → 배회 판정 → 알람

[유형 5] OD + Tracker + Classification ────────────────────────
  대표: vanguard_patient

  프레임 → ROI 크롭 → OD 추론 → 트래커 → 객체 크롭 → CLS(색상) → 이벤트 → 알람

[유형 6] 전처리 집약형 ────────────────────────────────────────
  대표: yeonsei_smoke, samsung_fire

  프레임 → 색상/모션 필터 → 윤곽 검출 → 후보 크롭 → CLS 추론 → 이벤트 판정 → 알람

[유형 7] VLM(Vision-Language Model) 기반 ──────────────────────
  대표: gangnam_falldown, gangnam_violence, samsung_internvl3

  프레임 → InternVL3 추론 → 텍스트 응답 해석 → 이벤트 판정 → 알람

[유형 8] PE(Perception Encoder / Retrieval) 기반 ──────────────
  대표: perception_encoder, pe_violence, hyundai_esfalldown

  프레임 → 이미지 임베딩 → 텍스트 프롬프트 유사도 계산 → 이벤트 판정 → 알람
```

---

## 8. 인프라 초기화 (`utils/init.py`)

서비스가 시작될 때 `TEAM` 환경변수에 따라 초기화되는 인프라가 달라집니다.

```
                    ┌─────────────┐
                    │ TEAM 환경변수     │
                    └──────┬──────┘
                                        │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌────────────────┐       ┌────────────────┐
     │   TEAM = "ai"  │       │   TEAM = "es"  │
     ├────────────────┤       ├────────────────┤
     │ Logger      ✓  │       │ Logger      ✓  │
     │ DataLake(S3) ✓ │       │ DataLake    ✗  │
     │ Redis        ✓ │       │ Redis       ✗  │
     │ RabbitMQ     ✓ │       │ RabbitMQ    ✗  │
     │ 추론 쓰레드   ✓ │       │ 추론 쓰레드  ✗│
     └────────────────┘       └────────────────┘
```

---

## 9. CI/CD 워크플로우

```
┌─────────────────────────────────────────────────────────────────┐
│ GitHub Actions 워크플로우                                       │
├─────────────────┬───────────────────┬───────────────────────────┤
│ 워크플로우       │ 트리거             │ 동작                    │
├─────────────────┼───────────────────┼───────────────────────────┤
│ create-branch   │ Issue 할당 시      │ Linear ID 기반 브랜치    │
│ .yaml           │                   │ 자동 생성 + Draft PR 생성 │
├─────────────────┼───────────────────┼───────────────────────────┤
│ on-pr.yaml      │ PR 생성 시        │ PR 작성자에게 자동 할당   │
├─────────────────┼───────────────────┼───────────────────────────┤
│ ready-pr.yaml   │ PR Ready 시       │ 리뷰어 자동 지정          │
└─────────────────┴───────────────────┴───────────────────────────┘

이슈 할당 → 브랜치 자동 생성 → 개발 → PR 생성 → 리뷰어 지정 → Merge to dev
```

---

## 10. 현재 등록된 모듈 전체 목록 (21개)

| # | 모듈 디렉토리 | 서비스 클래스 | 프로젝트/고객사 | 감지 대상 | 파이프라인 유형 |
|:---:|:---|:---|:---|:---|:---:|
| 1 | `ax_fall` | FallService | AX | 낙상 | OD |
| 2 | `ax_harness` | HarnessService | AX | 안전벨트 미착용 | OD |
| 3 | `DaeGu_crowd_people` | CPService | 대구 | 군중 밀집 | OD |
| 4 | `Daegu_intrusion` | IntrusionService | 대구 | 침입 | OD+ROI |
| 5 | `gangnam_falldown` | InternVL3TrtFalldownService | 강남 | 낙상 | VLM |
| 6 | `gangnam_violence` | InternVL3TrtViolenceService | 강남 | 폭력 | VLM |
| 7 | `hyundai_esfalldown` | ESFalldownService | 현대 | 에스컬레이터 낙상 | PE(Retrieval) |
| 8 | `khonkaen_helmet` | HelmetService | 콘깬 | 헬멧 미착용 | OD+CLS |
| 9 | `khonkaen_weapon` | WeaponService | 콘깬 | 무기 소지 | OD+OD 캐스케이드 |
| 10 | `kumho_pinch` | PinchService | 금호타이어 | 협착 | OD+ROI |
| 11 | `perception_encoder` | PEService | 공통 | PE 범용 | PE(Retrieval) |
| 12 | `pe_violence` | PVService | 공통 | 폭력 | PE(Retrieval) |
| 13 | `samsung_fire` | FireService | 삼성 | 화재 | 전처리+CLS |
| 14 | `samsung_internvl3` | InternVL3Service | 삼성 | 범용 | VLM |
| 15 | `vehicle_reverse` | VehicleReverseService | 공통 | 차량 역주행 | OD |
| 16 | `yeonsei_falldown` | FalldownService | 연세 | 낙상 | OD |
| 17 | `yeonsei_smoke` | SmokeService | 연세 | 연기 | 전처리+CLS |
| 18 | `yonsei_tailgate` | TailgateService | 연세 | 테일게이트 | OD |
| 19 | `yonsei_walljump` | WalljumpService | 연세 | 담장 넘기 | OD |
| 20 | `vanguard_patient` | PatientService | 뱅가드 | 환자 배회 | OD+Tracker+CLS |
| 21 | `aramco_loitering` | LoiteringService | 아람코 | 배회 | OD+Tracker |

---

## 11. 환경 변수 정리

| 변수명 | 기본값 | 용도 |
|:---|:---:|:---|
| `TEAM` | (없음) | 팀 구분: `ai` 또는 `es` |
| `ENV` | (없음) | 환경 구분: `dev` 또는 `prod` |
| `LIMITED_NUM_OF_CAMERA` | 16 | 동시 처리 가능한 최대 카메라 수 |
| `LIMITED_NUM_OF_PERSON_PER_CAMERA` | 16 | 카메라당 최대 감지 인원 수 |
| `LOGGING_STATE_REMAIN_MINUTE` | 60 | 로깅 활성 상태 유지 시간(분) |
| `IMAGE_SAVE_PATH` | `logs` | 디버그 이미지 저장 경로 |
| 각 모듈별 `MODEL_*_ONNX_PATH` | (모듈별) | 모델 파일 경로 오버라이드 |

---

## 12. 빠른 시작 가이드

```bash
# 1. 깃허브 리파지토리 클론
git clone git@github.com:TeamPIA/Product-AI-mono.git
cd Product-AI-mono

# 2. Conda env 생성
conda create -n Product-AI-mono python=3.11
conda activate Product-AI-mono

# 3. 필요한 패키지 설치
pip install .

# 4. Commit hook 설정
make init

# 5. pre-commit 테스트
pre-commit run --all-files

# 6. 환경변수 설정
export TEAM=ai

# 7. 테스트 실행
pytest .
```
