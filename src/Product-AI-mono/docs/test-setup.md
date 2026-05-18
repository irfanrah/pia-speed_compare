# 테스트 환경 설정 가이드

## 1. 패키지 설치 (editable 모드)

```bash
pip install -e .
```

> `pip install .` 대신 `-e` 옵션을 사용해야 로컬 코드 변경사항이 즉시 반영됩니다.

---

## 2. 환경변수 영구 설정

`~/.bashrc` (또는 `~/.zshrc`)에 추가하여 영구 적용합니다.

```bash
echo 'export TEAM=ai' >> ~/.bashrc
source ~/.bashrc
```

설정 확인:

```bash
echo $TEAM  # ai 출력되면 정상
```

> `TEAM=ai` 가 설정되지 않으면 `redis_manager`, `alarm_producer` 등이 `None` 으로 초기화되어 테스트가 실패합니다.

---

## 3. HuggingFace 로그인

모델 다운로드를 위해 HuggingFace CLI 로그인이 필요합니다.

```bash
hf auth login
```

> 로그인 후에는 `HF_AUTH_TOKEN` 환경변수 없이도 캐시된 토큰이 자동으로 사용됩니다.

---

## 4. 테스트 실행 전 서비스 상태 확인

| 서비스 | 기본 주소 | 확인 방법 |
|--------|-----------|-----------|
| Redis | `localhost:6379` | `python -c "import redis; print(redis.Redis().ping())"` |
| RabbitMQ | `localhost:5672` | `curl http://localhost:15672` |
| NAS | `172.168.47.36` | `curl http://172.168.47.36` |

---

## 5. 테스트 실행

#### 전체 테스트

```bash
pytest .
```

#### 특정 모듈만

```bash
pytest packages/pia_prod/AI/tests/modules/test_ax_fall.py
```
