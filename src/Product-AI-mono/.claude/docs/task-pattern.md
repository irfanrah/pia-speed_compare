# 모듈 구조 패턴

## 디렉터리 구조

모든 추론 모듈은 동일한 구조를 따른다.

```
packages/pia_prod/AI/modules/{module_name}/
├── __init__.py
├── config.py         # 환경변수 기반 설정 (경로, 임계값, 카테고리)
├── service.py        # ServiceBase 상속 — 추론 서비스
├── event.py          # EventBase 상속 — 이벤트 상태 머신
├── param.py          # CategoryBase 상속 — Pydantic 파라미터 모델
├── roi_manager.py    # ROIManagerBase 상속 — ROI 크롭 처리 (선택)
└── prompts.py        # VQA 프롬프트 (VQA 모듈만)
```

---

## config.py 패턴

환경변수로 모든 설정을 관리한다.

```python
import os

# 모델 경로
MODEL_DIR = os.getenv("MODEL_INTERNVL3_PATH", "assets/model/InternVL3-2B")

# 카테고리
FIRE_CATEGORY = ["fire_vqa", "화재_vqa"]
FALLDOWN_CATEGORY = ["falldown_vqa", "쓰러짐_vqa"]
MODEL_OUTPUTS = ["falldown", "fire"]
SUPPORT_CATEGORIES = FALLDOWN_CATEGORY

# 이벤트 판정
QUEUE_SIZE = int(os.environ.get("VQA_FALLDOWN_QUEUE_SIZE", 1))
ALARM_DURATION_THRESHOLD = int(os.environ.get("VQA_FALLDOWN_ALARM_DURATION_THRESHOLD", 1))
```

### 규칙
- 환경변수 접두사로 모듈 구분
- 기본값은 개발 환경 기준으로 설정
- 카테고리, 임계값, 모델 경로를 한 파일에 집중
- 다른 모듈의 config를 import하지 않는다

---

## service.py 패턴

```python
from pia_prod.AI.bases.service_base import ServiceBase
from pia.ai.model import PiaTorchModel

class MyService(ServiceBase):
    def __init__(self, analysis_data_queue: Queue):
        super().__init__(analysis_data_queue)

    def _load_model(self):
        """모델 로딩. PiaTorchModel로 태스크/모델 지정."""
        self.config = ...
        self.model = PiaTorchModel(target_task="VQA", target_model="internvl3trt_llm", config=self.config)

    def _init_values(self):
        """추가 초기화 (선택)."""
        pass

    def _load_event_manager(self):
        """이벤트 매니저 반환."""
        return MyEventManager()

    def _load_roi_manager(self):
        """ROI 매니저 반환 (선택)."""
        return MyRoIManager()

    def _detect(self, **datas):
        """추론 로직. batches, stream_ids, user_params를 받아 알람 반환."""
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        # 추론 → 이벤트 판정 → 알람 반환
        ...
```

### ServiceBase 핵심 메서드 (변경 금지)

| 메서드 | 데코레이터 | 설명 |
|--------|-----------|------|
| `thread_ai_inference()` | `@final` | Queue에서 데이터 꺼내 `_detect()` 호출 |
| `send_alarm()` | `@final` | 알람 결과를 RabbitMQ로 전송 |
| `_load_model()` | `@abstractmethod` | 서브클래스에서 구현 |
| `_detect()` | `@abstractmethod` | 서브클래스에서 구현 |

---

## event.py 패턴

```python
from pia_prod.AI.bases.event_base import EventBase
from collections import deque, defaultdict

class MyEventManager(EventBase):
    def __init__(self):
        super().__init__(alarm_duration=ALARM_DURATION_THRESHOLD)
        self.duration_queue = defaultdict(lambda: defaultdict(lambda: deque(maxlen=QUEUE_SIZE)))
        self.event_status = defaultdict(lambda: defaultdict(int))

    def update(self, responses, categories_per_stream, stream_ids):
        """추론 결과를 받아 알람 상태를 판정한다."""
        # 상태 머신: 0(없음) → 1(시작) → 2(지속) → 3(종료)
        ...
        return self.check_alarm_duration()
```

---

## param.py 패턴

```python
from pia_prod.AI.DTO.param_base import CategoryBase

class MyModel(CategoryBase):
    """stream_params.py의 AddStreamModel에서 참조된다."""
    pass
```

---

## 서비스 등록

새 모듈 추가 시:
1. `modules/{module_name}/` 디렉터리 생성 (위 구조 따름)
2. `packages/pia_prod/AI/__init__.py`의 `__all__`, `_SERVICE_MODULE_OVERRIDE`, `TYPE_CHECKING` 블록에 등록
3. `packages/pia_prod/AI/DTO/stream_params.py`에 파라미터 모델 등록

---

## 외부 연동 아키텍처

| 컴포넌트 | 용도 | 초기화 위치 |
|----------|------|------------|
| RabbitMQ | 알람 메시지 큐 | `utils/init.py` |
| Redis | 로깅 상태, 설정 | `utils/init.py` |
| S3 / DataLake | 썸네일 저장 | `utils/init.py` |
| HuggingFace | 모델 다운로드 | `tests/conftest.py` |
| NAS | 테스트 영상 다운로드 | `tests/conftest.py` |
