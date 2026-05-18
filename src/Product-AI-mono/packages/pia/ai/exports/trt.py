import subprocess
from pathlib import Path
from typing import List, Tuple, Union


def onnx2trt(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path] | None = None,
    device: int = 0,
    fp16: bool = False,
    overwrite: bool = False,
    input_shape: Union[Tuple[int, ...], List[int]] = (3, 640, 640),
    max_batch: int = 8,
    opt_batch: int = 4,
    min_batch: int = 1,
) -> str:
    """
    Convert an ONNX model to a TensorRT engine.

    The function first tries the TensorRT Python API. If that fails it falls
    back to the ``trtexec`` CLI tool. The ONNX input tensor name is detected
    automatically so callers do not need to provide it manually.
    """
    if not isinstance(input_shape, (tuple, list)):
        raise ValueError(f"input_shape must be a tuple or list, got {type(input_shape)}")

    dims = tuple(map(int, input_shape))
    if len(dims) < 1:
        raise ValueError(f"input_shape must have at least 1 dimension, got {dims}")
    if any(d <= 0 for d in dims):
        raise ValueError(f"All dimensions must be positive: {dims}")

    onnx_path = Path(onnx_path)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    if engine_path is None:
        engine_path = onnx_path.with_suffix(".engine")
    engine_path = Path(engine_path)

    if engine_path.exists() and not overwrite:
        print(f"TensorRT engine already exists: {engine_path}")
        return str(engine_path)

    input_names = _extract_onnx_model_inputs(onnx_path)
    if not input_names:
        raise RuntimeError("Failed to detect ONNX model inputs.")
    input_name = input_names[0]
    if len(input_names) > 1:
        print(f"[Warn] Multiple ONNX inputs detected: {input_names}. Using '{input_name}'.")
    else:
        print(f"[Info] Detected ONNX input tensor: '{input_name}'")

    python_success = False
    try:
        onnx2trt_python(
            onnx_path=onnx_path,
            engine_path=engine_path,
            device=device,
            fp16=fp16,
            overwrite=True,
            input_shape=dims,
            max_batch=max_batch,
            opt_batch=opt_batch,
            min_batch=min_batch,
            input_name=input_name,
        )
        python_success = engine_path.is_file()
    except Exception as err:
        print(f"[Warn] TensorRT Python API build failed ({err}). Falling back to trtexec.")

    if not python_success:
        onnx2trt_trtexec(
            onnx_path=onnx_path,
            engine_path=engine_path,
            device=device,
            fp16=fp16,
            overwrite=True,
            input_shape=dims,
            max_batch=max_batch,
            opt_batch=opt_batch,
            min_batch=min_batch,
            input_name=input_name,
        )

    if engine_path.is_file():
        return str(engine_path)
    raise RuntimeError(f"TensorRT engine build failed: {engine_path}")


def onnx2trt_python(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path] | None = None,
    device: int = 0,
    fp16: bool = False,
    overwrite: bool = False,
    input_shape: Union[Tuple[int, ...], List[int]] = (3, 640, 640),
    max_batch: int = 8,
    opt_batch: int = 4,
    min_batch: int = 1,
    input_name: str | None = None,
) -> str:
    """
    Build a TensorRT engine from an ONNX model using the TensorRT Python API.

    When ``input_name`` is None the method will infer the first input tensor
    from the parsed network.
    """
    import tensorrt as trt

    dims = tuple(map(int, input_shape))

    onnx_path = Path(onnx_path)
    if engine_path is None:
        engine_path = onnx_path.with_suffix(".engine")
    engine_path = Path(engine_path)

    if engine_path.exists() and not overwrite:
        print(f"TensorRT engine already exists: {engine_path}")
        return str(engine_path)

    try:
        from cuda import cudart

        err = cudart.cudaSetDevice(device)[0]
        if err != 0:
            print(f"[Warn] cudaSetDevice({device}) returned error code {err}, continuing anyway.")
    except Exception as exc:
        print(f"[Info] Could not set CUDA device via cuda-python: {exc}")

    logger = trt.Logger(trt.Logger.INFO)
    try:
        flags = trt.NetworkDefinitionCreationFlags.EXPLICIT_BATCH
    except AttributeError:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    with trt.Builder(logger) as builder, \
         builder.create_network(flags) as network, \
         trt.OnnxParser(network, logger) as parser, \
         builder.create_builder_config() as config:

        onnx_bytes = onnx_path.read_bytes()
        if not parser.parse(onnx_bytes):
            print("[Error] Failed to parse ONNX. Parser messages:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parsing failed.")

        if network.num_inputs == 0:
            raise RuntimeError("No input tensors found in ONNX model.")

        detected_input_name = network.get_input(0).name
        if input_name and input_name != detected_input_name:
            print(
                f"[Warn] Provided input name '{input_name}' differs from network input "
                f"'{detected_input_name}'. Using the network input name."
            )
        input_name = detected_input_name
        print(f"[Info] Using ONNX input tensor: '{input_name}'")

        if network.num_inputs > 1:
            all_inputs = [network.get_input(i).name for i in range(network.num_inputs)]
            print(f"[Warn] Multiple inputs detected: {all_inputs}. Using first input '{input_name}'.")

        try:
            nb_dims = network.get_input(0).shape.nbDims
            expected_rank = len(dims) + 1  # +1 for batch dimension
            if nb_dims not in (-1, expected_rank):
                print(
                    f"[Warn] Network input rank ({nb_dims}) does not match provided shape "
                    f"(expected {expected_rank}). Proceeding with profile setup."
                )
        except Exception as e:
            print(f"[Warn] Could not inspect network input shape: {e}")

        min_shape = (min_batch,) + dims
        opt_shape = (opt_batch,) + dims
        max_shape = (max_batch,) + dims

        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        if fp16:
            try:
                config.set_flag(trt.BuilderFlag.FP16)
            except AttributeError:
                print("[Warn] FP16 flag not supported in this TensorRT version.")

        engine_bytes = builder.build_serialized_network(network, config)
        if engine_bytes is None:
            raise RuntimeError("Failed to build TensorRT engine.")

        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(bytes(engine_bytes))
        print(f"[Success] TensorRT engine saved at: {engine_path}")
        return str(engine_path)


def onnx2trt_trtexec(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path] | None = None,
    device: int = 0,
    fp16: bool = False,
    overwrite: bool = False,
    input_shape: Union[Tuple[int, ...], List[int]] = (3, 640, 640),
    max_batch: int = 8,
    opt_batch: int = 4,
    min_batch: int = 1,
    input_name: str | None = None,
) -> str:
    """Convert an ONNX model to a TensorRT engine using ``trtexec``."""
    dims = tuple(map(int, input_shape))
    if len(dims) < 1:
        raise ValueError(f"input_shape must have at least 1 dimension, got {dims}")

    onnx_path = Path(onnx_path)
    if engine_path is None:
        engine_path = onnx_path.with_suffix(".engine")
    engine_path = Path(engine_path)

    if engine_path.exists() and not overwrite:
        print(f"TensorRT engine already exists: {engine_path}")
        return str(engine_path)

    if not input_name:
        raise ValueError("input_name must be provided when invoking trtexec.")

    shape_suffix = "x".join(map(str, dims))
    min_shape = f"{min_batch}x{shape_suffix}"
    opt_shape = f"{opt_batch}x{shape_suffix}"
    max_shape = f"{max_batch}x{shape_suffix}"

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--device={device}",
        f"--minShapes={input_name}:{min_shape}",
        f"--optShapes={input_name}:{opt_shape}",
        f"--maxShapes={input_name}:{max_shape}",
    ]
    if fp16:
        cmd.append("--fp16")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return str(engine_path)


def _extract_onnx_model_inputs(onnx_path: Path) -> List[str]:
    """Return logical model input tensors, excluding constant initializers."""
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("The 'onnx' Python package is required to read model inputs.") from exc

    model = onnx.load(str(onnx_path))
    graph = model.graph
    initializer_names = {init.name for init in graph.initializer}
    return [value.name for value in graph.input if value.name not in initializer_names]
