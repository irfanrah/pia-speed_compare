from typing import Union

import numpy as np
import torch
from pia.ai.tasks.OD.base import ODConfig, ODModelBase, ODONNXConfig
from pia.ai.tasks.tracker.models.sort.main import CallTracker
from pia.utils.exception.model_handler import raise_exception_decorator
from pia.vision.postprocessing.nms import non_max_suppression
from pia.vision.roi import BoundingBoxManager

from .coordinate_utils import LetterBox
from .load_models import AutoBackend


class YOLOv8(ODModelBase):
    def __init__(self, config: ODConfig):
        self.config = config
        self.model = self.__load_model__()
        self.transform = LetterBox(
            new_shape=self.config.img_size,
            auto=False,
            scaleFill=self.config.scalefill,
            scaleup=self.config.scaleup,
        )
        self.tracker = CallTracker(config.tracker_config)
        self.roi_manager = BoundingBoxManager(
            iou_threshold=config.iou_thres_for_merge,
            distance_threshold=config.distance_thres_for_merge,
        )

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.forward(x)

    @raise_exception_decorator(FileNotFoundError)
    def _load_model(self):
        model = AutoBackend(
            weights=self.config.model_path,
            device=torch.device(self.config.device),
            fp16=self.config.use_half_precision,
            verbose=False,
        )  # 외부 라이브러리 그대로 임포트 해서 사용
        model.warmup(imgsz=(1, 3, *self.config.img_size))
        model.eval()
        return model

    def preprocess(self, im: np.ndarray) -> torch.Tensor:
        # 나중에 config에서 설정 가져와서 전처리
        im = self.transform(image=im)
        not_tensor = not isinstance(im, torch.Tensor)

        if not_tensor:
            im = im[None, :]
            im = im.transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.config.device)
        img = img.half() if self.config.use_half_precision else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    @torch.no_grad
    def forward(self, batch: Union[torch.Tensor, np.ndarray]) -> list:
        """
        Processes the input tensor or array and returns a tensor.

        This function is designed to work with various types of input, including
        original images, preprocessed images, and batch images.

        Args:
            batch (Union[torch.Tensor, np.ndarray]): Input data which can be a PyTorch
                tensor or a NumPy array.

        Returns:
            list : The processed output as a list.
        """
        state, batch = self.check_inputs(batch)
        if state:
            batch = self.preprocess(batch)

        model_results = self.model(batch)

        nms_results = non_max_suppression(
            prediction=model_results,
            conf_thres=self.config.conf_thres_for_nms,
            iou_thres=self.config.iou_thres_for_nms,
            classes=self.config.classes,
            agnostic=self.config.agnostic,
            max_det=self.config.max_det,
            max_wh=self.config.max_wh,
        )
        ret = []
        for x in nms_results:
            if self.config.device == "cuda":
                x = x.detach().cpu().numpy()
            else:
                x = x.detach().numpy()

            if self.config.do_merge_bboxes:
                x = self.roi_manager.update_boxes(x)

            x = self.tracker(x)  # Batch inference is only possible for one video.

            if self.config.do_bbox_resize:
                x = self.transform.get_origin_size_bbox(x)
                # Values can be negative or larger than the original image, so use clip to limit them.
                x[:, [0, 2]] = x[:, [0, 2]].clip(0, self.transform.origin_shape[1])
                x[:, [1, 3]] = x[:, [1, 3]].clip(0, self.transform.origin_shape[0])
            ret.append(x)

        return ret

    def export2onnx(self, params: ODONNXConfig):
        pass

    def check_inputs(self, im):
        if isinstance(im, np.ndarray):
            return True, im
        elif isinstance(im, torch.Tensor):
            b, c, w, h = im.shape
            if w != self.config.img_size[0] or h != self.config.img_size[1]:
                return True, np.array(im)
            else:
                return False, im
        else:
            raise "return type must be torch.Tensor or np.ndarray"
