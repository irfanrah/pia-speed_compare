FROM registry.cn-hangzhou.aliyuncs.com/opengrps/grps_gpu:grps1.1.0_cuda12.6_cudnn9.6_trtllm0.16.0_py3.12

LABEL maintainer="jungseoik"
LABEL description="InternVL3-2B TensorRT LLM Server (auto-build & run)"

# 0) 루트 작업 디렉토리
WORKDIR /grps_dev

# 1) 레포 클론 + assets 생성 (가이드 그대로)
RUN apt-get update && apt-get install -y git-lfs && \
    git lfs install && \
    git clone https://github.com/jungseoik/grps_trtllm_v0.0.git /grps_dev/grps_trtllm_v0.0 && \
    mkdir -p /grps_dev/grps_trtllm_v0.0/assets && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 이후 모든 경로는 레포 루트 기준(= 가이드의 `cd grps_trtllm_v0.0`)
WORKDIR /grps_dev/grps_trtllm_v0.0

# 2) 모델 다운로드
RUN git clone https://huggingface.co/PIA-SPACE-LAB/InternVL3-2B_gangnam /assets/InternVL3-2B

# 3) 의존 패키지 (ViT 변환 스크립트용)
RUN pip install --no-cache-dir -r tools/internvl2/requirements.txt

# 5) trtllm 빌드 파라미터 (유동적으로 조정)
ARG MAX_INPUT_LEN=4096
ARG MAX_SEQ_LEN=4096
ARG MAX_NUM_TOKENS=4608
ARG MAX_MULTIMODAL_LEN=4608

ENV MAX_INPUT_LEN=${MAX_INPUT_LEN}
ENV MAX_SEQ_LEN=${MAX_SEQ_LEN}
ENV MAX_NUM_TOKENS=${MAX_NUM_TOKENS}
ENV MAX_MULTIMODAL_LEN=${MAX_MULTIMODAL_LEN}

# 6) 컨테이너 뜨면 변환 → 엔진 빌드 → 서버 자동 기동
CMD bash -c '\
    rm -rf /assets/InternVL3-2B/tllm_checkpoint/ && \
    python3 tools/internvl2/convert_qwen2_ckpt.py \
        --model_dir  /assets/InternVL3-2B \
        --output_dir /assets/InternVL3-2B/tllm_checkpoint/ \
        --dtype bfloat16 --load_model_on_cpu && \
    rm -rf /assets/InternVL3-2B/trt_engines/ && \
    trtllm-build \
      --checkpoint_dir /assets/InternVL3-2B/tllm_checkpoint/ \
      --output_dir     /assets/InternVL3-2B/trt_engines/ \
      --gemm_plugin bfloat16 \
      --max_batch_size 1 \
      --paged_kv_cache enable \
      --use_paged_context_fmha enable \
      --max_input_len ${MAX_INPUT_LEN} \
      --max_seq_len ${MAX_SEQ_LEN} \
      --max_num_tokens ${MAX_NUM_TOKENS} \
      --max_multimodal_len ${MAX_MULTIMODAL_LEN} && \
    python3 tools/internvl2/build_vit_engine.py \
      --pretrainedModelPath /assets/InternVL3-2B \
      --onnxFile            /assets/InternVL3-2B/vision_encoder_bfp16.onnx \
      --trtFile             /assets/InternVL3-2B/vision_encoder_bfp16.trt \
      --imagePath           ./data/frames/frame_0.jpg \
      --dtype bfloat16 \
      --minBS 1 --optBS 13 --maxBS 26 && \
    if [ ! -f ./server.mar ]; then grpst archive .; fi && \
    exec grpst start ./server.mar --inference_conf=conf/inference_internvl3.yml \
'
