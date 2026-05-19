# PR Review Guide

## 리뷰 체크리스트

### 모듈 구조
- config.py → service.py → event.py 패턴을 따르는가
- Service는 `ServiceBase`를 상속하는가
- Event는 `EventBase`를 상속하고 `update()`를 구현하는가
- `_load_model()`, `_detect()` 추상 메서드를 구현했는가
- `ServiceBase`의 `@final` 메서드를 오버라이드하지 않았는가

### config.py
- 환경변수 기반으로 설정을 관리하는가
- 카테고리, 임계값, 모델 경로가 한 파일에 집중되어 있는가
- 다른 모듈의 config를 import하지 않는가 (모듈 간 결합 금지)
- 환경변수 접두사로 모듈이 구분되는가

### 코딩 컨벤션
- 클래스: CamelCase, 함수/변수: snake_case 를 따르는가
- 임포트 순서: 표준 라이브러리 → 서드파티 → pia → pia_prod
- pia / pia_prod 패키지 임포트는 절대 경로를 사용하는가
- 타입 힌트가 함수 시그니처에 포함되어 있는가

### 버그 및 안정성
- 배열/리스트 길이가 zip 순회 시 일치하는가
- parse 함수가 모든 예상 출력을 처리하는가 (예: fire, falldown, normal)
- 리소스(GPU 메모리, 파일 핸들 등) 해제가 되는가
- 선택적 의존성은 try-except로 처리했는가

### 성능
- GPU/CPU 메모리 사용이 적절한가
- 불필요한 데이터 복사나 변환이 없는가
- 배치 처리가 가능한 곳에서 배치로 처리하는가

### 이벤트/알람
- 상태 머신 전이가 올바른가 (0→1→2→3)
- duration_queue의 maxlen이 적절한가
- 알람 중복 발생이 방지되는가

### ROI
- ROI 좌표 처리가 올바른가
- ROI가 없을 때 원본을 그대로 반환하는가
- 카메라별 ROI 캐시가 올바르게 갱신되는가

### 테스트
- 새 기능에 대한 테스트가 포함되어 있는가
- 테스트가 `packages/pia_prod/AI/tests/modules/` 하위에 위치하는가
- 영상 소스를 `nas_downloader`로 받는가 (로컬 파일 의존 금지)
- 서비스 종료 시 `stop_thread()`로 정리하는가

### DTO / 파라미터
- Pydantic 모델이 `CategoryBase` 또는 `VQABase`를 상속하는가
- `stream_params.py`의 AddStreamModel에 등록했는가
- `__init__.py`의 `__all__`에 서비스를 등록했는가
