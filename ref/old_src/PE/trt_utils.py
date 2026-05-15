import tensorrt as trt
import torch


def trt_to_torch_dtype(dt):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }[dt]


def run_v3(context):
    if hasattr(context, "execute_v3"):
        return context.execute_v3()
    elif hasattr(context, "enqueue_v3"):
        return context.enqueue_v3(0)
    elif hasattr(context, "execute_async_v3"):
        return context.execute_async_v3(stream_handle=0)
    else:
        raise RuntimeError("TensorRT v3 execution API not found in this build.")
