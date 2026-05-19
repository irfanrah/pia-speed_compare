# Code Fix Guide (for review_fix.yaml)

이 문서는 Claude가 리뷰 피드백을 반영하여 코드를 수정할 때 따르는 규칙이다.

---

## 수정 원칙

1. **리뷰 내용만 수정한다** — 리뷰에서 지적하지 않은 부분은 건드리지 않는다
2. **최소 변경** — 문제를 해결하는 가장 작은 변경을 택한다
3. **기존 패턴을 따른다** — 주변 코드의 스타일과 패턴을 유지한다
4. **수정 불확실 시 설명만 한다** — 의도가 모호하면 코드를 건드리지 말고 댓글로 설명한다

---

## 수정 전 확인사항

### 반드시 읽을 문서
1. `CLAUDE.md` — 프로젝트 구조와 모듈 패턴 파악
2. `coding-convention.md` — 네이밍, 임포트 규칙
3. `task-pattern.md` — 변경 대상이 모듈 코드일 경우

### 반드시 확인할 코드
- 수정 대상 파일의 **전체 컨텍스트** (해당 파일을 먼저 읽는다)
- 수정 대상이 상속 구조에 있으면 **부모 클래스**도 확인 (`ServiceBase`, `EventBase`)
- Config 변경 시 **해당 Config를 사용하는 곳** 전부 확인
- 다른 모듈에서 같은 상수를 import하고 있는지 확인

---

## 수정 금지 영역

| 영역 | 이유 |
|------|------|
| `ServiceBase`의 `@final` 메서드 | 모든 서비스가 의존하는 템플릿 메서드 |
| `EventBase`의 상태 전이 테이블 | 전체 알람 로직에 영향 |
| `global_config.py`의 키 상수 | 전체 모듈이 참조 |
| 다른 모듈 폴더의 코드 | 리뷰 범위 외 |
| `utils/init.py`의 싱글턴 초기화 | 전체 서비스 인프라에 영향 |

---

## 커밋 규칙

### 메시지 형식

```
{type}({scope}): {설명}
```

| type | 용도 |
|------|------|
| `fix` | 버그 수정 |
| `feat` | 새 기능 추가 |
| `refactor` | 리팩토링 (동작 변경 없음) |
| `docs` | 문서 수정 |
| `test` | 테스트 추가/수정 |
| `chore` | 설정, 의존성 변경 |

### scope
- 모듈명: `vqa_internvl`, `gangnam_falldown`, `khonkaen_helmet` 등
- 공통: `base`, `config`, `DTO`, `utils`, `tests`

### 예시

```
fix(vqa_internvl): parse_prediction에 fire 카테고리 파싱 추가
feat(khonkaen_helmet): 헬멧 미착용 알람 duration 설정 추가
refactor(gangnam_falldown): config import를 자체 모듈로 분리
```

---

## push 전 체크리스트

1. `git fetch origin` 실행
2. `git merge origin/{base_branch}` 로 최신 코드 동기화
3. 충돌 발생 시 수정하지 말고 상황을 댓글로 보고
4. `git push` 실행

---

## 수정 결과 보고 형식

작업 완료 후 출력할 내용:

```
## 수정 요약

### 변경 파일
- `path/to/file.py`: 변경 내용 설명

### 수정 내용
리뷰 지적사항별로 어떻게 수정했는지 설명

### 미반영 사항 (있는 경우)
수정하지 않은 지적사항과 그 이유
```
