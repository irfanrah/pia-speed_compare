# Product-AI-mono

AI 상품화 멀티 프로젝트 통합 저장소입니다.
각 프로젝트는 폴더 단위로 관리되며, 커밋 메시지 및 구조에 대한 규칙을 반드시 따릅니다.

---

## 📁 폴더 네이밍 규칙

- 형식: `프로젝트_모델명 또는 프로젝트명_카테고리`
- 예시:
  - `GangnamOpenInnovation_falldown`
  - `GangnamOpenInnovation_InternVL3` → 한 모델로 여러 개 카테고리를 커버하는 경우 + 해당 폴더에 readme 작성 후 카테고리 작성 필요
  - `yeonsei_smoke`

> 💡 한 모델이 여러 카테고리를 커버하는 경우, **모델명 기준**으로 작성하되, 상세 구분은 하위 디렉토리에서 관리할 수 있습니다.

---

## ✅ 커밋 메시지 규칙
### 기본 포맷:  type: Subject line
* type은 아래 타입 목록 확인
* Subject는 최대 50자
* body는 자유
### 예시:
* `feat: Add smoke detection baseline model`
* `fix: Correct keypoint postprocessing`
* `docs: Update folder naming policy`

### 타입 목록:
* `feat`: 새로운 기능 추가
* `fix`: 버그 수정
* `refactor`: 코드 리팩토링
* `style`: 스타일 수정 (세미콜론, 포맷 등)
* `docs`: 문서 수정
* `test`: 테스트 코드 추가/변경
* `chore`: 기타 작업 (빌드, 설정 등)
---

## 🛑 커밋 전에 반드시 확인할 사항

- [ ] 추상 클래스 상속하여 개발 하였는가? ( event, service )
- [ ] 불필요한 파일이 포함되어 있지 않은가?
예: 모델 파일, 개인 노트북에서 생성된 로그/캐시, 임시 스크립트 등
- [ ] 다른 레포의 코드를 포함하는 경우 Submodule 금지
- [ ] `.pre-commit` hook이 정상적으로 동작하는가?
- [ ] 추론/학습 코드에서 GPU/CPU 디바이스 하드코딩 여부
---


## 🛠 초기 설정 가이드

### 1. 리포지토리 클론
```bash
git clone git@github.com:TeamPIA/Product-AI-mono.git
cd Product-AI-mono
```
### 2. Conda env 생성
```bash
conda create -n Product-AI-mono python=3.11
conda activate Product-AI-mono
```

### 3. Conda 환경변수 설정 (TEAM=ai 자동 적용)

conda 환경을 activate할 때 `TEAM=ai`가 자동으로 세팅되도록 등록합니다.

```bash
conda activate <환경명>  # ex) conda activate pia

# activate.d 디렉토리가 없으면 생성
mkdir -p $CONDA_PREFIX/etc/conda/activate.d

# TEAM=ai 등록
echo 'export TEAM=ai' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

# 재활성화하여 적용
conda activate <환경명>

# 확인
echo $TEAM  # ai 출력되면 정상
```

> 자세한 내용은 [docs/conda-env-setup.md](docs/conda-env-setup.md)를 참고하세요.

---

### 4. 필요 package 설치
```bash
pip install .
pip install "yolox@git+https://github.com/noahcao/OC_SORT.git@7d06bffe98b5e57cc696ce56739554483c7e99ca" --no-build-isolation
```
### 5. commit lint 적용
```bash
make init
```
### 6. pre-commit 테스트
```bash
pre-commit run --all-files
```
### 7. 테스트 환경 설정 및 실행


자세한 내용은 [docs/test-setup.md](docs/test-setup.md)를 참고하세요.
