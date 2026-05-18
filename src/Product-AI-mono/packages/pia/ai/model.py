import json
import os
from collections import OrderedDict, namedtuple
from typing import List, Literal, Union

import numpy as np
import torch
import torch.nn as nn
from pia.vision.preprocessing import video_preprocess
from torchvision.transforms import Compose, ToPILImage, ToTensor

from .base import PiaConfigBase

# Lazy task-factory resolver.
# Eager-importing every factory at module load time pulls in the union of all
# task-family deps (onnxruntime, ultralytics, transformers, einops, filterpy, ...)
# regardless of which task a caller actually uses, defeating module isolation.
# We resolve factories on first use instead — so a caller using only T2VRet
# never triggers OD/CP/VQA/tracker imports.
_TASK_KEY = {
    0: "OD",        "OD": "OD",
    1: "T2VRet",    "RET": "T2VRet",
    2: "CP",        "CP": "CP",
    3: "VQA",       "VQA": "VQA",
    4: "tracker",   "trackers": "tracker",
}


def _load_factory(key: str):
    if key == "OD":
        from .tasks.OD.factory import ODFactory
        return ODFactory
    if key == "CP":
        from .tasks.CP.factory import CpFactory
        return CpFactory
    if key == "T2VRet":
        from .tasks.T2VRet.factory import T2VRetFactory
        return T2VRetFactory
    if key == "VQA":
        from .tasks.VQA.factory import VQAFactory
        return VQAFactory
    if key == "tracker":
        from .tasks.tracker.factory import TrackerFactory
        return TrackerFactory
    raise KeyError(f"Unknown task key: {key!r}")


# Backward-compatible alias. Existing callers that introspect SUPPORTED_TASKS
# (e.g. for membership/keys) keep working; values are factory keys instead of
# the factory class itself, matching how PiaTorchModel uses them below.
SUPPORTED_TASKS = _TASK_KEY


class PiaTorchModel:
    """
    Initialize PiaTorchModel class with task, model, and configuration settings.

    Args:
        target_task (Union[str, int]): 0 or "OD" for object detection, 1 or "RET" for text-to-video retrieval
        target_model (Union[str, int]): When target_task is "OD", 0 or "yolov8" for YOLOv8 model. When target_task is "RET", 0 or "clip4clip" for Clip4Clip model and 1 or "clip-vip" for CLIPVIP model.
        config (PiaConfigBase): ODConfig or T2VRetConfig class instance

    Raises:
        KeyError: If the target task is not recognized or supported.

    Returns:
        YOLOv8 or Clip4Clip or CLIPVIP: The object detection or text-to-video retrieval model instance.

    Examples:
        Infer object detection using object detection settings:

        >>> config = ODConfig(model_path="model.pth")
        >>> model = PiaTorchModel(target_task="OD", target_model="yolov8", config=config)
        >>>
        >>> img = cv2.imread("image.jpg")
        >>>
        >>> ret = model.forward(img)

        Batch infer object detection using object detection settings with tracker:

        >>> tracker_config = TrackerConfig(tracker="sort", max_age=3, min_hits=1, threshold=200)
        >>> config = ODConfig(model_path="model.pth", tracker_config=tracker_config)
        >>> model = PiaTorchModel(target_task="OD", target_model="yolov8", config=config)
        >>>
        >>> img1 = cv2.imread("image1.jpg")
        >>> img2 = cv2.imread("image2.jpg")
        >>>
        >>> img1 = model.preprocess(img1)
        >>> img2 = model.preprocess(img2)
        >>>
        >>> imgs = torch.vstack((img1, img2))
        >>> ret = model.forward(imgs)

        Infer text-to-video retrieval using text-to-video retrieval settings:

        >>> config = T2VRetConfig(model_path="model.pth")
        >>> model = PiaTorchModel(target_task="RET", target_model="clip4clip", config=config)
        >>>
        >>> video = np.random.rand(3, 12, 224, 224, 3)
        >>> text = "a photo of a cat"
        >>>
        >>> visual_vector = model.forward(video)
        >>> text_vector = model.forward(text)
        >>> similiarity = model.forward(visual_vector, text_vector)
    """

    def __new__(
        cls,
        target_task: Union[str, int],
        target_model: Union[str, int],
        config: PiaConfigBase,
        **kwargs,
    ):
        # TODO : -> PiaModelBase :  type hint 적용 방법 조사 필요:
        # TODO : doc string 추가

        # model 없는건지 체크
        if target_task not in _TASK_KEY:
            raise KeyError(
                f"Unknown task {target_task}, available tasks are {list(_TASK_KEY.keys())}"
            )

        # model load (factory imported lazily)
        factory_cls = _load_factory(_TASK_KEY[target_task])
        return factory_cls(
            config=config, target_model=target_model, **kwargs
        ).load()


class PiaONNXTensorRTModel(nn.Module):
    """
    for load onnx or trt model
    """

    def __init__(
        self, model_path: str, device: Literal["cuda", "mps", "cpu"], half: bool = False
    ) -> None:
        super().__init__()
        assert os.path.isfile(model_path), f"can't load model file at {model_path}"
        self.model_allowed_exts = [".engine", ".trt", ".onnx"]
        self.model_framework = None
        self.device = device
        self.half = half
        self.load_model(model_path)
        self.non_normalized_transform = Compose([ToPILImage(), ToTensor()])

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        """
        Executes a forward pass through the model using the given input data `d`.

        Parameters:
            d (torch.Tensor): The input tensor to be processed by the model.
        Raises:
            AssertionError: If the input tensor shape does not match the expected shape when
                            using a static TensorRT model.
            RuntimeError: If the model framework is not recognized or supported.
        Returns:
            torch.Tensor or list[torch.Tensor]: The model's output(s) as PyTorch tensor(s).
            If the model returns multiple outputs, a list of tensors is returned; otherwise,
            a single tensor is returned.
        """
        if self.model_framework == ".trt" or self.model_framework == ".engine":
            if self.dynamic and d.shape != self.bindings[self.input_names[0]].shape:
                self.context.set_input_shape(self.input_names[0], d.shape)
                self.bindings[self.input_names[0]] = self.bindings[self.input_names[0]]._replace(shape=d.shape)
                for name in self.output_names:
                    self.bindings[name].data.resize_(tuple(self.context.get_tensor_shape(name)))
            s = self.bindings[self.input_names[0]].shape
            assert (
                d.shape == s
            ), f"input size {d.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs[self.input_names[0]] = int(d.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            pred = [self.bindings[x].data for x in sorted(self.output_names)]

        elif self.model_framework == ".onnx":
            if len(self.self.session.get_inputs()) != 1:
                inputs = {}
                for i in range(len(self.self.session.get_inputs())):
                    inputs[self.session.get_inputs()[i].name] = d[i].cpu().numpy()
            else:
                d = d.cpu().numpy()
                inputs = {self.session.get_inputs()[0].name: d}
            pred = self.session.run(self.output_names, inputs)
        else:
            raise AttributeError("Supported type : onnx, trt(engine)")

        if isinstance(pred, (list, tuple)):
            return self.from_numpy(pred[0]) if len(pred) == 1 else [self.from_numpy(x) for x in pred]
        else:
            return self.from_numpy(pred)

    def load_model(self, path):
        ext = os.path.splitext(path)[-1]
        if ext == ".engine" or ext == ".trt":
            self.load_tensorrt_model(path)
        elif ext == ".onnx":
            self.load_onnx_model(path)
        else:
            raise f"check model's extention, is on this list? : {str(self.model_allowed_exts)}"

        self.model_framework = ext

    def load_tensorrt_model(self, path):
        import tensorrt as trt

        logger = trt.Logger(trt.Logger.ERROR)
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))

        with open(path, "rb") as f, trt.Runtime(logger) as runtime:
            # 1차 시도: 순수 엔진 파일 (trtexec output)
            engine_data = f.read()
            model = runtime.deserialize_cuda_engine(engine_data)

            # 2차 시도: metadata가 포함된 엔진 파일
            if model is None:
                print("INFO : reload - with meta data")
                f.seek(0)
                try:
                    meta_len = int.from_bytes(f.read(4), byteorder="little")
                    metadata = json.loads(f.read(meta_len).decode("utf-8"))
                except Exception:
                    f.seek(0)  # 실패 시 그냥 처음부터 다시
                model = runtime.deserialize_cuda_engine(f.read())

            if model is None:
                raise RuntimeError(f"Failed to load TensorRT engine from {path}")

        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        input_names = []

        for i in range(model.num_io_tensors):
            name = model.get_tensor_name(i)
            dtype = trt.nptype(model.get_tensor_dtype(name))
            if model.get_tensor_mode(name) == trt.TensorIOMode.INPUT :
                input_names.append(name)
            if model.get_tensor_mode(name).value == 1:
                if -1 in tuple(model.get_tensor_shape(name)):  # dynamic
                    dynamic = True
                    context.set_input_shape(
                        name,
                        tuple(model.get_tensor_profile_shape(name, 0)[2]),
                    )
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_tensor_shape(name))
            d = torch.from_numpy(np.empty(shape, dtype=dtype)).to(torch.device("cuda"))
            bindings[name] = Binding(name, dtype, shape, d, int(d.data_ptr()))
        binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings[input_names[0]].shape[0]  # if dynamic, this is instead max batch size

        self.__dict__.update(locals())  # assign all variables to self

    def load_onnx_model(self, path):
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(path, providers=providers)
        output_names = [x.name for x in self.session.get_outputs()]
        # input_names = [x.name for x in session.get_inputs()]

        self.__dict__.update(locals())  # assign all variables to self

    def model_warmup(self, d_size=(3, 3, 224, 224)):
        if self.model_framework == ".onnx" and self.session.get_inputs()[0].type.find("int") > 0:
            dtype = torch.int32
        else:
            dtype = torch.half if self.half else torch.float
        d = torch.empty(*d_size, dtype=dtype, device=self.device)  # input
        for _ in range(2):  #
            self.forward(d)  # warmup

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def get_c4c_onnx_vis_vector(
        self,
        frame: np.ndarray,
        tile_size: Literal["L", "M", "S"] = None,
        img_size: List[int] = [224, 224],
    ) -> torch.Tensor:
        """get visual vector from onnx model"""
        tensor_frame, tensor_mask = video_preprocess(
            video=frame,
            device=self.device,
            tile_size=tile_size,
            img_size=img_size,
            transform=self.non_normalized_transform,
        )

        B, T, F, C, W, H = tensor_frame.shape
        viewed_frame = tensor_frame.view(B * T * F, C, W, H)

        ort_inputs = {self.session.get_inputs()[0].name: viewed_frame.cpu().numpy()}
        ort_outs = self.session.run(None, ort_inputs)[0]  # [1,1,512] 리스트
        ort_outs = torch.from_numpy(ort_outs).to(self.device)

        B_T_F_D_ort_outs = ort_outs.view(B, T, F, -1)
        pooled_ort_outs = B_T_F_D_ort_outs.mean(dim=2)  # frame축에 대한 mean pooling
        norm_pooled_ort_outs = pooled_ort_outs / pooled_ort_outs.norm(
            dim=-1, keepdim=True
        )  # L2 normalization

        return norm_pooled_ort_outs
