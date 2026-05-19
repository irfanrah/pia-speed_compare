# CLAUDE.md

## 프로젝트 개요

Product-AI-mono — 고객별 AI 추론 서비스를 통합 관리하는 제품 AI 패키지

## 프로젝트 구조

```
packages/pia_prod/
├── AI/
│   ├── __init__.py          # Lazy import 서비스 레지스트리
│   ├── global_config.py     # 전역 상수 (환경변수 키, 이벤트 키 등)
│   ├── bases/               # 추상 베이스 클래스
│   │   ├── service_base.py  # ServiceBase — 추론 서비스 템플릿
│   │   └── event_base.py    # EventBase — 이벤트 상태 머신
│   ├── modules/             # 고객별 추론 모듈 (23개)
│   │   ├── gangnam_falldown/
│   │   ├── khonkaen_helmet/
│   │   ├── vqa_internvl/
│   │   └── ...
│   ├── DTO/                 # Data Transfer Objects (Pydantic)
│   │   ├── param_base.py    # ROIModel, CategoryBase, VQABase
│   │   ├── stream_params.py # AddStreamModel (스트림 등록 API)
│   │   └── output_handler.py # 알람 메시지 구성 및 라우팅
│   ├── utils/               # 유틸리티
│   │   ├── utils.py         # 모델 변환, 임계값 체크 등
│   │   ├── init.py          # 싱글턴 매니저 초기화 (Redis, RabbitMQ, S3)
│   │   └── log_templates.py # 로그 템플릿
│   ├── validate/            # 검증/스코어링 로직
│   └── tests/               # 테스트 코드
│       ├── conftest.py      # 세션 fixture (HF, NAS, 환경변수)
│       └── modules/         # 모듈별 테스트
```

## 모듈 구조 패턴

각 모듈은 동일한 파일 구성을 따릅니다:
- `config.py` — 환경변수 기반 설정 (모델 경로, 임계값, 카테고리)
- `service.py` — `ServiceBase` 상속, `_load_model()`, `_detect()` 구현
- `event.py` — `EventBase` 상속, 이벤트 상태 머신 (`update()`)
- `param.py` — `CategoryBase` 상속 Pydantic 모델
- `roi_manager.py` — ROI 크롭 처리 (선택)
- `prompts.py` — VQA 프롬프트 (VQA 모듈만)

## 상황별 참조 문서

필요한 상황에서만 해당 문서를 읽으세요:

| 상황 | 참조 문서 |
|------|----------|
| PR 리뷰 수행 시 | `.claude/docs/review-guide.md` |
| 리뷰 피드백 반영하여 코드 수정 시 | `.claude/docs/fix-guide.md` |
| 모듈 구조 확인 시 | `.claude/docs/task-pattern.md` |
| 코딩 스타일/네이밍 확인 시 | `.claude/docs/coding-convention.md` |
| 새 카테고리(모듈) 추가 시 | `.claude/docs/repo/new-category-guide.md` → 체크리스트: `new-category-checklist.md` |
| 프로젝트 전체 구조 파악 시 | `.claude/docs/repo/repo_structure.md` |

## 핵심 규칙

- 모듈 구조 패턴(config → service → event)을 따를 것
- 다른 모듈의 config를 import하지 말 것 (모듈 간 결합 금지)
- 환경변수 설정은 `config.py`에 집중할 것
- 테스트는 `nas_downloader`로 영상을 받을 것 (로컬 파일 의존 금지)

## GitHub Actions

- `review.yaml` — PR에 review request 시 Claude Code가 자동 리뷰 수행
- `review_fix.yaml` — 리뷰 피드백을 Claude가 자동 수정
