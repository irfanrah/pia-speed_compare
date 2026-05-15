import os

import tensorrt as trt


def export_trt_engine(
    onnx_file: str,
    save_file_path: str,
    input_size=(3, 336, 336),
    min_batch_size: int = 1,
    max_batch_size: int = 32,
    opt_batch_size: int = 16,
    half_precision: bool = True,
):
    if not os.path.exists(onnx_file):
        raise FileNotFoundError(f"ONNX file {onnx_file} does not exist.")

    if os.path.exists(save_file_path):
        print(f"Engine file {save_file_path} already exists. Skipping export.")
        return

    os.makedirs(os.path.dirname(save_file_path) or ".", exist_ok=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    if builder.platform_has_fast_fp16 and half_precision:
        config.set_flag(trt.BuilderFlag.FP16)

    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(min_batch_size, *input_size),
        opt=(opt_batch_size, *input_size),
        max=(max_batch_size, *input_size),
    )
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build the TensorRT engine")

    with open(save_file_path, "wb") as f:
        f.write(serialized_engine)

    print(
        f"Engine saved at {save_file_path} "
        f"({os.path.getsize(save_file_path) / 1024 / 1024:.2f} MB)"
    )
