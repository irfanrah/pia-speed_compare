import os

import tensorrt as trt


def export_trt_engine(
    onnx_file,
    save_file_path,
    min_batch_size=1,
    max_batch_size=32,
    opt_batch_size=8,
    min_frames=1,
    opt_frames=8,
    max_frames=16,
    width=336,
    height=336,
    half_precision=True,
):
    """
    ONNX 모델을 TensorRT 엔진으로 변환하고 저장한다.
    - Temporal dimension (T) 포함: 입력 (B, T, 3, H, W)
    """
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"ONNX file {onnx_file} does not exist.")

    if os.path.exists(save_file_path):
        print(f"[INFO] Engine file {save_file_path} already exists. Skipping export.")
        return

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

    if builder.platform_has_fast_fp16 and half_precision:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    input_shape = input_tensor.shape
    print(f"[INFO] Network input: {input_name} | shape={input_shape}")

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(min_batch_size, min_frames, 3, height, width),
        opt=(opt_batch_size, opt_frames, 3, height, width),
        max=(max_batch_size, max_frames, 3, height, width),
    )
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine.")

    with open(save_file_path, "wb") as f:
        f.write(serialized_engine)

    print(
        f"Engine saved at {save_file_path} "
        f"({os.path.getsize(save_file_path) / 1024 / 1024:.2f} MB)"
    )
