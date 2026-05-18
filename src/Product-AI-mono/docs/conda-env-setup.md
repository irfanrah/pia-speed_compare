# Conda 환경변수 설정 가이드

## 왜 필요한가

`pia_prod.AI.utils.init` 모듈은 **임포트 시점**에 `TEAM` 환경변수를 읽어 `redis_manager`, `alarm_producer` 등을 초기화합니다.

```python
# utils/init.py
now_team = os.getenv("TEAM", None)

if now_team == "ai":
    redis_manager = declare_redis_manager()   # 정상 초기화
else:
    redis_manager = None                      # None으로 고정
```

`TEAM=ai`가 설정되지 않은 상태에서 테스트를 실행하면 `redis_manager`가 `None`이 되어 아래 에러가 발생합니다.

```
AttributeError: 'NoneType' object has no attribute 'set'
```

---

## 설정 방법

### conda activate 시 자동 적용 (권장)

conda 환경을 activate할 때 환경변수가 자동으로 세팅되도록 등록합니다.

```bash
# 1. 환경 활성화
conda activate <환경명>   # ex) conda activate pia

# 2. activate.d 디렉토리 생성 (없는 경우)
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# 3. TEAM=ai 등록
echo 'export TEAM=ai' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 4. 재활성화하여 즉시 적용
conda activate <환경명>

# 5. 확인
echo $TEAM   # ai 출력되면 정상
```

> 이후 해당 conda 환경을 activate할 때마다 `TEAM=ai`가 자동으로 적용됩니다.

---

### 이미 등록된 내용 확인

```bash
cat $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

---

### 등록 해제 방법

`env_vars.sh`에서 해당 라인을 직접 삭제합니다.

```bash
# 현재 내용 확인
cat $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 편집기로 export TEAM=ai 라인 삭제
vi $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

---

## TEAM 값에 따른 초기화 동작

| `TEAM` 값 | `redis_manager` | `alarm_producer` | `datalake_manager` |
|---|---|---|---|
| `ai` | RedisManager 객체 | RabbitMQProducer 객체 | DataLakeManager 객체 |
| 미설정 또는 다른 값 | `None` | `None` | `None` |

> `TEAM=ai`로 설정하면 Redis, RabbitMQ, S3(DataLake)에 대한 연결이 초기화됩니다.
> 각 서비스가 실행 중이지 않아도 연결 실패는 예외처리로 흡수되어 테스트 실행에는 영향이 없습니다.
