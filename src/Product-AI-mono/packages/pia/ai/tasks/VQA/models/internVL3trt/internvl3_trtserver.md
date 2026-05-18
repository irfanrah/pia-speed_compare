# InternVL3-2B Tensorrt llm 서버 실행 가이드
## 도커 파일 빌드

- 프로젝트 루트 위치에서 아래 실행
```bash
## InternVL3 - falldown 사용시
cd packages/pia/ai/tasks/VQA/models/internVL3trt/dockerfile/internvl3
## InternVL3 - violence 사용시
cd packages/pia/ai/tasks/VQA/models/internVL3trt/dockerfile/internvl3_violence

docker build \
  --build-arg MAX_INPUT_LEN=4096 \
  --build-arg MAX_SEQ_LEN=4096 \
  --build-arg MAX_NUM_TOKENS=4608 \
  --build-arg MAX_MULTIMODAL_LEN=4608 \
  -t internvl3_server:latest . \
&& docker run -it --rm \
  --gpus all \
  -p 9997:9997 \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
  --name ivl3_2b_server \
  internvl3_server:latest
```
```bash
[START SUCCESS]
>> You can access http://0.0.0.0:9997/ to observe server status.
>> You can check service info by `grpst ps` cmd.
>> Will show logs after 5s, you can stop showing by `Ctrl+C`.
```
- 위에 메세지 확인하면 그냥 컨테이너 띄운상태로 나가도 무방 Ctrl+C 눌러도 알아서 돌아감


## 도커 컨테이너로 동작시 step by step 상세 내용

> **전제조건**: 이미지 pull 당겨서 이미지 띄운뒤 스텝바이 스텝 trt 변환 및 서버 세팅 가이드입니다.

## 0. 환경 준비

```bash
https://github.com/jungseoik/grps_trtllm_v0.0.git
cd grps_trtllm_v0.0
mkdir -p assets
```
## 1. 컨테이너 실행

```bash
# 컨테이너 생성 및 실행
docker run -itd --name ivl3_2b_server --runtime=nvidia --network host \
  --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/grps_dev -v $(pwd)/assets:/assets -w /grps_dev \
  registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12 bash

# 컨테이너 접속
docker exec -it ivl3_2b_server bash
```

## 2. 모델 다운로드 & 의존 패키지
```bash
# 2B 클론
apt update && apt install -y git-lfs
git lfs install
git clone https://huggingface.co/OpenGVLab/InternVL3-2B /assets/InternVL3-2B

pip install -r ./tools/internvl2/requirements.txt   # ViT 변환 스크립트용

```
## 3. 체크포인트 변환
```bash
rm -rf /assets/InternVL3-2B/tllm_checkpoint/
python3 tools/internvl2/convert_qwen2_ckpt.py \
        --model_dir  /assets/InternVL3-2B \
        --output_dir /assets/InternVL3-2B/tllm_checkpoint/ \
        --dtype bfloat16 --load_model_on_cpu

```
## 4. **LLM 엔진** 빌드 (batch size 1)

```bash

  모델 빌드 용량이 달라짐 대략 어느정도 토큰 소모되는지 파악하고 빌드하기 바랍니다.
  현재 세팅은 10GB정도 잡아먹는 빌드
  아래 내용 실행 추천
--------------------------------------
rm -rf /assets/InternVL3-2B/trt_engines/
trtllm-build \
  --checkpoint_dir /assets/InternVL3-2B/tllm_checkpoint/ \
  --output_dir     /assets/InternVL3-2B/trt_engines/ \
  --gemm_plugin bfloat16 \
  --max_batch_size 1 \
  --paged_kv_cache enable \
  --use_paged_context_fmha enable \
  --max_input_len 4096 \
  --max_seq_len 4096 \
  --max_num_tokens 4608 \
  --max_multimodal_len 4608

```
---

## 6. **ViT 엔진** 빌드

- `-maxBS` 는 *이미지 패치* 동시 처리 기준이라 그대로 26 유지.

```bash
python3 tools/internvl2/build_vit_engine.py \
  --pretrainedModelPath /assets/InternVL3-2B \
  --onnxFile            /assets/InternVL3-2B/vision_encoder_bfp16.onnx \
  --trtFile             /assets/InternVL3-2B/vision_encoder_bfp16.trt \
  --imagePath           ./data/frames/frame_0.jpg \
  --dtype bfloat16 \
  --minBS 1 --optBS 13 --maxBS 26
```
--------------------------------------------------------------


## 7. 패키징 & 서버 기동

### 기존 아카이브 파일이 있는 경우
```bash
# 아카이브 파일 확인
ls -la server.mar

# 바로 서버 시작
grpst start ./server.mar --inference_conf=conf/inference_internvl3.yml

# 서버 상태 확인
grpst ps
```

### 아카이브 파일이 없는 경우
```bash
# 새로운 아카이브 생성
grpst archive .

# 서버 시작
grpst start ./server.mar --inference_conf=conf/inference_internvl3.yml

# 서버 상태 확인
grpst ps
```

## 9. 서버 테스트

서버가 정상적으로 실행되었는지 테스트:

```bash
curl --no-buffer http://127.0.0.1:9997/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "InternVL3",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "<image>\nexplain this image"
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "file:///tmp/InternVL3-2B/examples/image1.jpg"
            }
          }
        ]
      }
    ],
    "max_tokens": 256
  }'


```

## 10. 서버 관리 명령어

```bash
# 서버 상태 확인
grpst ps

# 서버 중지
grpst stop

# 서버 재시작
grpst restart

# 로그 확인
grpst logs
```
