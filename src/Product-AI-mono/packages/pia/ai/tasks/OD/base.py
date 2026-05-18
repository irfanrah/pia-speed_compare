from abc import abstractmethod
from typing import List, Union

import numpy as np
import torch
from pia.ai.base import PiaConfigBase, PiaModelBase
from pia.ai.tasks.tracker.base import TrackerConfig


class ODConfig(PiaConfigBase):
    """
    Initializes the ODConfig class with configuration for object detection models.

    Args:
        model_path (str): The path of the model file.
        device (str, optional): The device to run the model. Defaults to "cpu".
        use_half_precision (bool, optional): If True, the model will run in half precision. Defaults to False.
        tracker_config (TrackerConfig, optional): The configuration of the tracker. Defaults to TrackerConfig().
        img_size (List[int], optional): The size of the input image. Defaults to [640, 640].
        scalefill (bool, optional): If True, the image will be scaled to fill the input size. Defaults to False.
        scaleup (bool, optional): If True, the image will be scaled up to the input size. Defaults to False.
        conf_thres_for_nms (float, optional): The confidence threshold for non-maximum suppression. Defaults to 0.25.
        iou_thres_for_nms (float, optional): The IoU threshold for non-maximum suppression. Defaults to 0.45.
        classes (List[int], optional): The classes to detect. Defaults to None.
        agnostic (bool, optional): If True, the model will be agnostic. Defaults to True.
        max_det (int, optional): The maximum number of detections. Defaults to 300.
        max_wh (int, optional): The maximum width and height of the image. Defaults to 7680.
        do_bbox_resize (bool, optional): If True, the bounding boxes will be resized. Defaults to False.
        do_merge_bboxes (bool, optional): If True, the bounding boxes will be merged. Defaults to False.
        iou_thres_for_merge (float, optional): The IoU threshold for merging bounding boxes. Defaults to 0.1.
        distance_thres_for_merge (float, optional): The distance threshold for merging bounding boxes. Defaults to 100.0.

    Returns:
        None

    Examples:
        Initialize ODConfig with default settings:

        >>> config = ODConfig(model_path="model.pth")
        >>> model = PiaTorchModel(target_task="OD", target_model="yolov8", config=config)
        >>> img = cv2.imread("image.jpg")
        >>>
        >>> ret = model.forward(img)

        Initialize ODConfig with custom settings:

        >>> config = ODConfig(
        ...     model_path="model.pth",
        ...     device="cuda",
        ...     tracker_config=TrackerConfig(tracker="sort"),
        ...     img_size=[1280, 720],
        ...     conf_thres_for_nms=0.5,
        ...     iou_thres_for_nms=0.3,
        ...     classes=[0, 1, 2],
        ...     agnostic=False
        ... )
        >>> model = PiaTorchModel(target_task="OD", target_model="yolov8", config=config)
        >>> img = cv2.imread("image.jpg")
        >>>
        >>> ret = model.forward(img)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        use_half_precision: bool = False,
        tracker_config: TrackerConfig = TrackerConfig(),
        img_size: List[int] = [640, 640],
        scalefill: bool = False,
        scaleup: bool = False,
        conf_thres_for_nms: float = 0.25,
        iou_thres_for_nms: float = 0.45,
        classes: List[int] = None,
        agnostic: bool = True,
        max_det: int = 300,
        max_wh: int = 7680,
        do_bbox_resize: bool = False,
        do_merge_bboxes: bool = False,
        iou_thres_for_merge: float = 0.1,
        distance_thres_for_merge: float = 100.0,
    ) -> None:
        super().__init__(model_path, device, use_half_precision)
        self.img_size = img_size
        self.scalefill = scalefill
        self.scaleup = scaleup
        self.conf_thres_for_nms = conf_thres_for_nms
        self.iou_thres_for_nms = iou_thres_for_nms
        self.classes = classes
        self.agnostic = agnostic
        self.max_det = max_det
        self.max_wh = max_wh
        self.tracker_config = tracker_config
        self.do_bbox_resize = do_bbox_resize
        self.do_merge_bboxes = do_merge_bboxes
        self.iou_thres_for_merge = iou_thres_for_merge
        self.distance_thres_for_merge = distance_thres_for_merge


class ODONNXConfig:
    pass


class ODModelBase(PiaModelBase):
    @abstractmethod
    def _load_model(self, model_path: str):
        pass

    @abstractmethod
    def preprocess(self, img: Union[torch.Tensor, np.ndarray]):
        pass

    @abstractmethod
    @torch.no_grad
    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        return x

    @abstractmethod
    def export2onnx(self, model):
        pass
