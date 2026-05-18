# How to Install

### InternVL3trt Tutorial
- internvl3trt 모듈을 사용하시려면 반드시 아래 서버를 켜놓은 상태로 실행해야 합니다.
- 서버는 포트 9997에서 HTTP로 동작합니다.
- [internVL3trt server tutorial](packages/pia/ai/tasks/VQA/models/internVL3trt/internvl3_trtserver.md)

### HuggingFace token Tutorial
- [hf token tutorial](docs/hf_token.md)

### Required Packages for Linux
- [onnx version install tutorial](docs/onnx_install.md)

## Editable Mode (to Develop)
### Base Instalation
```bash
conda create -n pia_common python=3.11 -y
conda activate pia_common

pip install -r requirements.txt
pip install -U pip
pip install -e ".[dev]"

# Set up CUDA library paths for the conda environment
bash scripts/bootstrap_pia_env.sh
conda deactivate && conda activate pia_common

```

# API

API documentation is available at [https://pia-ai-package.netlify.app/](https://pia-ai-package.netlify.app/)

# How to Do Pre-commit

```bash
$ pip install pre-commit
$ pre-commit install # make pre-commit config at .git/hook/pre-commit
$ git add .
$ git commit -m "COMMIT MESSAGE"
```
# How to Run Test Code
Go to [`packages/pia/tests/README.md`](packages/pia/tests)
