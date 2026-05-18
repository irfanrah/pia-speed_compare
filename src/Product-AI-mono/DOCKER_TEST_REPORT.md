# Docker 모듈별 격리 테스트 결과 보고서

> **갱신일**: 2026-04-29
> **관련 PR**: #409 (pia 병합 + lazy import) → #410 (docker 환경 + 모듈 deps)
> **환경**: Linux / Docker 27.x / NVIDIA GPU x2 / CUDA 13.0
> **로그 위치**: `.docker_test_logs/v3-pr410/<module>.log`

---

## 1. 작업 흐름

| 단계 | PR | 내용 |
|------|-----|------|
| 1 | **#409** | `pia-source/develop` (b0d7290 — pia-ai-package 의 develop tip) 통합 + `pia.ai.model` / 5개 task factory / `pia.ai.exports` lazy import 전환 |
| 2 | **#410** | 13개 모듈에 격리 Docker 환경 (Dockerfile / docker-compose.yaml / module-specific requirements.txt) + 모듈별 deps 정리 |

**핵심**: #409 의 lazy import 덕에 #410 의 모듈 requirements 가 자기가 실제로 사용하는 task family 의 deps 만 가지면 된다. `pia.ai.model` import 가 더 이상 OD/CP/T2VRet/VQA/tracker 의 합집합 deps 를 끌어오지 않음.

---

## 2. 검증 환경 (`feat/jaeyong/prod-종속성-테스트` 브랜치 = PR #410)

각 모듈을 다음 사이클로 순차 실행 (디스크 폭주 방지):

```bash
cd packages/pia_prod/AI/modules/<module>
HF_NAMESPACE=PIA-SPACE-LAB HF_AUTH_TOKEN=*** \
  docker compose up --build --abort-on-container-exit 2>&1 | tee ../.../<module>.log
docker compose down --rmi local -v
docker builder prune -f        # 필요 시
```

빌드 컨텍스트: Product-AI-mono root (assets/ 는 .dockerignore 로 제외).

---

## 3. 12개 모듈 결과

| # | 모듈 | 결과 | 시간 | 비고 |
|---|------|------|------|------|
| 1 | qwen3_vl_embedding | ⚠️ 1/6 | 18s | 5 fail = HF fire prompt 데이터 mismatch (외부, 4절) |
| 2 | Daegu_intrusion | ✅ 3/3 | 18s | |
| 3 | vanguard_patient | ✅ 3/3 | 133s | yolox@git + `--no-build-isolation` |
| 4 | two_stage_pe_qwen3vle | ⚠️ 1/6 | 30s | qwen-vl-utils + accelerate + compressed-tensors 추가; 5 fail 동일 (4절) |
| 5 | yonsei_tailgate | ✅ 3/3 | 8s | |
| 6 | aramco_loitering | ✅ 3/3 | 20s | yolox + filterpy + Dockerfile `--no-build-isolation` |
| 7 | pe_violence | ✅ 6/6 | 324s | |
| 8 | DaeGu_crowd_people | ✅ 2/2 | 500s | |
| 9 | khonkaen_peoplecounting | ✅ 6/6 | 2379s (~40분) | onnxruntime 추가 |
| 10 | vehicle_reverse | ✅ 3/3 | 100s | gdown + tensorboard 추가 |
| 11 | yonsei_walljump | ✅ 3/3 | 12s | |
| 12 | khonkaen_weapon | ✅ 3/3 | 37s | |

**총합**: 38 passed / 10 failed. 10 fail 은 모두 같은 외부 데이터 mismatch (4절).

---

## 4. 미해결 — HF 모델 데이터 mismatch (PR 변경과 무관)

### 증상

`qwen3_vl_embedding` / `two_stage_pe_qwen3vle` 의 5+5 = 10개 fail 모두 동일:
```
ValueError: torch.cat(): expected a non-empty list of Tensors
  at packages/pia_prod/AI/modules/qwen3_vl_embedding/service.py:93
  in _load_and_stack_tensors
```

### Root cause (HuggingFace 레포 listing 으로 확정)

`config.py` 가 기대하는 prompt 파일과 HF 의 실제 파일명이 다름:

| 카테고리 | config 기대 | HF 실제 | 결과 |
|----------|-----------|---------|------|
| smoke (APOv2.2.1) | `smoke/smoke.pt` | `smoke/smoke.pt` (단일) | ✅ |
| **fire (APOv2.2.1)** | `fire/fire.pt` (단일) | `fire/fire_16.pt`, `fire_17.pt`, ... (인덱스 다수) | ❌ |
| **fire (APOv2.2.1)** | `normal/normal.pt` (단일) | `normal/normal_0.pt`, `normal_19.pt`, `normal_33.pt` (인덱스 다수) | ❌ |

다운로드가 404 로 실패 → `_load_and_stack_tensors` 가 빈 디렉토리에 listdir → `torch.cat([])` ValueError.

### Reproducer 단위 테스트

`packages/pia_prod/AI/tests/modules/test_qwen3vle_hf_prompt_consistency.py` — HuggingFace API listing 과 `config.py` 의 파일명을 비교해 mismatch 를 명시적으로 단언한다. 이 PR 머지 후에도 `test_hf_fire_files_match_config` 는 fail 상태로 유지되어 publisher 측 조치 필요성을 GitHub Actions 등에서 가시화한다.

### 해결 옵션 (PR 범위 밖)

- **A**. publisher (PIA-SPACE-LAB) 가 HF 의 `text_prompt_APOv2.2.1/fire_pred_prompts/{normal,fire}/` 에 단일 `normal.pt` / `fire.pt` 를 업로드 — config 변경 없이 통과.
- **B**. `config.py` 의 `LIST_OF_NORMAL_FIRE_TXT_PROMPTS` / `LIST_OF_TARGET_FIRE_TXT_PROMPTS` 를 인덱스 붙은 실제 파일들로 업데이트.

---

## 5. PR #410 에서 발견된 모듈별 누락 deps (수정 완료)

### 5-1. lazy import 후 처음 드러난 누락
| 모듈 | 추가된 deps | 트레이싱 |
|------|-----------|---------|
| qwen3_vl_embedding | transformers + qwen-vl-utils + accelerate + compressed-tensors | T2VRet/qwen3_vl_embedding model |
| two_stage_pe_qwen3vle | qwen-vl-utils + accelerate + compressed-tensors | qwen3 stage |
| khonkaen_peoplecounting | onnxruntime | model.py 가 ClipEBCOnnx 직접 import |
| vehicle_reverse | gdown + tensorboard | torchreid 가 GDrive 가중치 + torch.utils.tensorboard |
| aramco_loitering | yolox + filterpy + Dockerfile `--no-build-isolation` | service 가 vanguard_patient.tracker (OC-SORT) 사용 |

### 5-2. 버전 핀 (deterministic 결과 보장)
- `transformers==4.57.3`, `compressed-tensors==0.13.0` (qwen3_vl_embedding, two_stage_pe_qwen3vle)
  - latest 버전은 FP8 quantized 모델의 BFloat16 / Half 캐스팅을 다르게 처리해 deterministic 검증이 환경마다 다르게 결과.

### 5-3. yolox@git PEP 517 빌드 격리 회피
- `vanguard_patient`, `aramco_loitering` Dockerfile 에 `pip install --no-build-isolation` 플래그.
- 이유: `yolox @ git+...OC_SORT` 의 `pyproject.toml` 이 빌드 타임에 `import torch` 호출. pip 의 격리된 PEP 517 빌드 venv 에는 torch 가 없어 실패. 베이스 이미지에 이미 torch 가 있으므로 격리를 끄고 그것을 재사용.

---

## 6. Docker 인프라 검증 결과

- ✅ **lazy import 효과**: `pia.ai.model` import 만으로는 5개 task factory 모두 끌려오지 않음. 단일 task 만 사용하는 모듈은 그 task family 의 deps 만 필요.
- ✅ **격리**: 모듈별 requirements 만으로 빌드/테스트 가능. 다른 모듈의 deps 불필요.
- ✅ **빌드 컨텍스트**: `.dockerignore` 로 `assets/` (55GB+) 제외 정상.
- ✅ **`docker compose down --rmi local -v`** 로 모듈 이미지 깨끗이 정리. `pia-test-base` 보존.
- ⚠️ **빌드 캐시 누적**: 12개 모듈 순차 검증 중 build cache 가 30~40GB 까지 누적. 주기적 `docker builder prune -f` 권장 (디스크 100GB 미만 환경에서는 필수).

---

## 7. 결론

**모듈별 격리 Docker 테스트 인프라 자체는 정상 작동**. 12개 모듈 모두 collection 통과 + 1개 import test 통과 (lazy import + deps 격리 검증). 38개 실제 inference 테스트가 통과했고, 남은 10개 실패는 모두 동일한 외부 HF 모델 데이터 mismatch (fire 카테고리) 로, PR 변경과 무관하며 publisher 측 또는 config 측 조치로 해결.

후속 작업:
1. `test_qwen3vle_hf_prompt_consistency.py` 의 `test_hf_fire_files_match_config` 가 통과하도록 HF 의 `fire_pred_prompts/` 정리 (publisher 또는 config 한 쪽).
2. transformers / compressed-tensors / accelerate 의 정확한 최소 버전 핀 검증.
