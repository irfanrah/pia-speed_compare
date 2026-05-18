"""
TensorRT engine wrappers for loading and running .engine files at inference time.

Classes:
    VisionEngine      — loads Vision.engine, runs vision encoder on pixel patches
    TransformerEngine  — loads Transformer.engine, runs the LLM on merged embeddings

Both accept torch CUDA tensors as input and return torch CUDA tensors as output,
so the full pipeline stays GPU-resident (no per-frame GPU→CPU→GPU round-trips).
"""

import os
import threading
from typing import Dict

import tensorrt as trt
import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _trt_to_torch_dtype(dt):
    return {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT8: torch.int8,
        trt.DataType.INT32: torch.int32,
        trt.DataType.BOOL: torch.bool,
    }[dt]


def _run_v3(context) -> bool:
    """Pick whichever v3 execution API the installed TensorRT exposes."""
    if hasattr(context, "execute_v3"):
        return context.execute_v3()
    if hasattr(context, "enqueue_v3"):
        return context.enqueue_v3(0)
    if hasattr(context, "execute_async_v3"):
        return context.execute_async_v3(stream_handle=0)
    raise RuntimeError("TensorRT v3 execution API not found in this build.")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class _BaseEngine:
    """Deserialises a .engine file and discovers its input/output tensor names."""

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f"Failed to load TensorRT engine: {engine_path}\n"
                f"  Local TensorRT version: {trt.__version__}\n"
                f"  Engines must be rebuilt when TensorRT is upgraded.\n"
                f"  Run:  rm {os.path.dirname(engine_path)}/*.engine && "
                f"python -m pia_prod.AI.modules.qwen3vle_trt.export.c_export_onnx_to_trt "
                f"--onnx-dir {os.path.dirname(engine_path)}"
            )
        self.context = self.engine.create_execution_context()

        # TensorRT IExecutionContext is NOT thread-safe (per NVIDIA docs).
        # Concurrent set_input_shape / set_tensor_address / execute_v3 calls
        # on the same context cause heap corruption ("double free or
        # corruption (out)" abort, observed under MAX_CONCURRENT_INFERENCES>1).
        # This lock serialises engine-level access; throughput becomes
        # equivalent to a 1-wide semaphore but the server stops crashing.
        self._lock = threading.Lock()

        self.num_io = self.engine.num_io_tensors
        self.tensor_names = [self.engine.get_tensor_name(i) for i in range(self.num_io)]
        self.input_names = [
            n for n in self.tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            n for n in self.tensor_names
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]


# ---------------------------------------------------------------------------
# Vision engine
# ---------------------------------------------------------------------------
class VisionEngine(_BaseEngine):
    """
    Runs the vision encoder (Vision.engine).

    Input:  pixel_values — torch.Tensor [num_patches, 3, temporal_patch, patch, patch] on CUDA
    Output: dict of torch.Tensor on CUDA (float32) — deepstack_feature_*, vision_hidden_states
    """

    def __init__(self, engine_path: str):
        super().__init__(engine_path)
        assert len(self.input_names) == 1, f"Expected 1 input, got {self.input_names}"
        self.input_name = self.input_names[0]

    def __call__(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        inp_buf = pixel_values
        if inp_buf.device.type != "cuda":
            inp_buf = inp_buf.to("cuda", non_blocking=True)
        if not inp_buf.is_contiguous():
            inp_buf = inp_buf.contiguous()

        # Hold the engine lock for the full set_shape → set_address → run →
        # synchronize sequence. Releasing earlier would let another caller
        # mutate context state while this run is still resolving on the GPU.
        with self._lock:
            self.context.set_input_shape(self.input_name, tuple(inp_buf.shape))
            self.context.set_tensor_address(self.input_name, int(inp_buf.data_ptr()))

            out_bufs: Dict[str, torch.Tensor] = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = _trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
                buf = torch.empty(shape, dtype=dtype, device="cuda")
                out_bufs[name] = buf
                self.context.set_tensor_address(name, int(buf.data_ptr()))

            if not _run_v3(self.context):
                raise RuntimeError("TensorRT vision inference failed.")
            torch.cuda.synchronize()

        return {n: out_bufs[n].detach().float() for n in self.output_names}


# ---------------------------------------------------------------------------
# Transformer engine
# ---------------------------------------------------------------------------
class TransformerEngine(_BaseEngine):
    """
    Runs the LLM transformer (Transformer.engine).

    Inputs (dict of torch.Tensor on CUDA, float32):
        hidden_states           — [1, seq_len, hidden_size]
        deepstack_features_{i}  — [1, seq_len, hidden_size]  (one per deepstack layer)
        rotary_cos              — [1, seq_len, 1, 1, head_dim]
        rotary_sin              — [1, seq_len, 1, 1, head_dim]
        attention_mask          — [1, 1, 1, seq_len, seq_len]

    Output: last_hidden_state torch.Tensor [1, seq_len, hidden_size] on CUDA (float32).
    """

    def __init__(self, engine_path: str):
        super().__init__(engine_path)
        self.deepstack_input_names = sorted(
            [n for n in self.input_names if n.startswith("deepstack_features_")],
            key=lambda s: int(s.split("_")[-1]),
        )

    @property
    def deepstack_count(self) -> int:
        return len(self.deepstack_input_names)

    def __call__(self, feeds: Dict[str, torch.Tensor]) -> torch.Tensor:
        missing = [n for n in self.input_names if n not in feeds]
        if missing:
            raise ValueError(f"Missing inputs: {missing}")

        # Pre-process inputs (host→device + dtype/contiguous fixup) outside
        # the lock so we don't serialise that pure tensor work too. The lock
        # only needs to cover the context-mutating calls.
        inp_bufs: Dict[str, torch.Tensor] = {}
        for name in self.input_names:
            t = feeds[name]
            if t.device.type != "cuda":
                t = t.to("cuda", non_blocking=True)
            if t.dtype != torch.float32:
                t = t.to(torch.float32)
            if not t.is_contiguous():
                t = t.contiguous()
            inp_bufs[name] = t

        with self._lock:
            for name in self.input_names:
                self.context.set_input_shape(name, tuple(inp_bufs[name].shape))
                self.context.set_tensor_address(name, int(inp_bufs[name].data_ptr()))

            out_bufs: Dict[str, torch.Tensor] = {}
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                dtype = _trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
                buf = torch.empty(shape, dtype=dtype, device="cuda")
                out_bufs[name] = buf
                self.context.set_tensor_address(name, int(buf.data_ptr()))

            if not _run_v3(self.context):
                raise RuntimeError("TensorRT transformer inference failed.")
            torch.cuda.synchronize()

        return out_bufs[self.output_names[0]].detach().float()
