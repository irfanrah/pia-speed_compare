from pia_prod.AI.modules.khonkaen_helmet.service import HelmetService
from typing import List, Dict, Tuple
import numpy as np
from pia_prod.AI.modules.khonkaen_helmet.config import (
    OD_TARGET_CLASSES,
    OD_INPUT_SIZE,
    TARGET_CATEGORY_INDEX,
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    OD_NMS_THRESHOLD,
    OD_CONFIDENCE_THRESHOLD,
)
import time
from pia.vision.postprocessing.nms import torch_non_max_suppression
from pia_prod.AI.utils.utils import threshold_check
from pia_prod.AI.modules.khonkaen_helmet.debug_utils import save_snapshot_for_odcls
from pia.vision.postprocessing.bbox import modified2origin_coordinate

# import pandas as pd


class ProfileHelmetService(HelmetService):
    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        self._init_values()
        # ======= 결과 정리 =======
        self.stages = [
            "total",
            "model1_preprocess",
            "model1_inference",
            "model1_nms",
            "model2_batch_make",
            "model2_preprocess",
            "model2_inference",
            "postprocess_logic",
            "send_alarm",
        ]

    def is_sink(self) -> bool:
        ts = [self.t0, self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7, self.t8]
        for i in range(1, len(ts)):
            if ts[i] < ts[i - 1]:
                # print(f"⚠️ Timestamp order reversed: t{i-1}={ts[i-1]:.6f}, t{i}={ts[i]:.6f}")
                return False
        return True

    def _detect(self, **datas):
        self.t0 = time.perf_counter()  # start
        batches = datas["batches"]
        stream_ids = datas["stream_ids"]
        user_params = datas["user_params"]
        if "rest" in datas:
            rest = datas["rest"]  # noqa

        """AI inference를 수행하고 결과를 rabbitMQ에 전송합니다.

        Args:
            batches (List[np.array]): RTSP로부터 받아온 len(batches)개의 이미지
            stream_ids (List[str]): 카메라의 stream_id
            user_params (List[AddStreamModel]): AddStreamModel 데이터클래스에 정의된 파라미터들

        Returns:
            None
        """
        (cropped_images, matched_info, total_bbox_info, cropped_batches, categories_batches) = self.get_pgie_results(batches=batches, stream_ids=stream_ids, user_params=user_params)

        # cls는 병렬처리이므로 for문 밖에서 처리
        cls_results, raw_cls_results = self.get_classification_result(cropped_images, matched_info)  # Classification

        if self.logging_flag:
            save_snapshot_for_odcls(
                images=cropped_batches,
                stream_ids=stream_ids,
                bboxes=total_bbox_info,
                raw_cls_results=raw_cls_results,
                category_index=TARGET_CATEGORY_INDEX,
                classify_matched_info=matched_info,
                video_mode=self.save_video,
                video_instance=self.video_manager,
            )

        alarm = {}
        for total_idx, origin_batch_idx in enumerate(categories_batches):  # Send Alarm
            state = self.alarm_event_manager.event_state[stream_ids[origin_batch_idx]]
            if state in [0, 2]:  # 알람이 없거나 계속 진행중일 땐 메세지를 보내지 않음
                continue
            is_start = True if state == 1 else False  # 알람이 처음 생긴 경우 thumbnail을 보냄
            alarm[stream_ids[origin_batch_idx]] = [is_start, None]

        self.t7 = time.perf_counter()  # postprocess logic
        self.get_alarm_helmet(cls_results, matched_info)  # Event Manager, Get Alarm
        self.t8 = time.perf_counter()  # end
        times = [
            (self.t8 - self.t0) * 1000,
            (self.t1 - self.t0) * 1000,
            (self.t2 - self.t1) * 1000,
            (self.t3 - self.t2) * 1000,
            (self.t4 - self.t3) * 1000,
            (self.t5 - self.t4) * 1000,
            (self.t6 - self.t5) * 1000,
            (self.t7 - self.t6) * 1000,
            (self.t8 - self.t7) * 1000,
        ]

        if self.is_sink():
            self.time_dict["total"].append(times[0])
            self.time_dict["model1_preprocess"].append(times[1])
            self.time_dict["model1_inference"].append(times[2])
            self.time_dict["model1_nms"].append(times[3])
            self.time_dict["model2_batch_make"].append(times[4])
            self.time_dict["model2_preprocess"].append(times[5])
            self.time_dict["model2_inference"].append(times[6])
            self.time_dict["postprocess_logic"].append(times[7])
            self.time_dict["send_alarm"].append(times[8])

            # ======= 표로 출력 -> average 로 대처=======
            # df = pd.DataFrame([times], columns=self.stages)
            # print(df.round(2).to_string(index=False))
            self.frame_cnt += 1
        # else:
        #     print("⚠️ Inference timing error detected; skipping time logging.")
        return alarm, batches, stream_ids, user_params, self.is_needed_cvt_color

    def get_pgie_results(
        self,
        batches: List[np.array],
        stream_ids: List[str],
        user_params: List,
    ) -> Tuple[List[np.array], Dict[str, int]]:
        """N개로 들어오는 이미지를 받아 object detection task를 수행하고, 결과를 반환합니다.

        Args:
            batches (List[np.array]): len(batches)개만큼 들어오는 이미지
            stream_ids (list[str]): 카메라의 stream_ids
            user_params (dict): AddStreamModel 데이터클래스에 정의된 파라미터들. ROI 정의시 사용됨

        Returns:
            crop_list(List[np.array]): detected된 object를 crop한 이미지 리스트
            matched_info(Dict[str, int]): stream_ids 당 몇개씩 detection 되었는지 카운트를 알 수 있는 변수

        NOTE:
            categories_batches
                >>> [0, 1, 0, 0]  # 0은 월담, 1은 따라들어가기 순으로 배치 들어온것임. len(batches)보다 클 수 있음
            divided_roi_info[0][2]
                >>> array([[456, 411],
                    [458, 411],
                    [633, 419],
                    [612, 205],
                    [465, 175]], dtype=int32)
            # 0번째 categories_batches의 2번째 ROI 좌표. 이안에 bbox가 검출되었는지 확인하는 용도로 쓰인다.
        """
        # Preprocess 1 - Crop with RoI
        cropped_batches, categories_batches = self.roi_manager.process_batches_with_roi(batches=batches, user_params=user_params)
        self.t1 = time.perf_counter()  # start - model1 crop and preprocess
        # Preprocess 2 - Letterbox & Normalize
        resized_images = self.od_letterbox_instance(imgs=cropped_batches)
        preprocessed_images = resized_images / 255.0

        # Inference model
        raw_od_results = self.object_detection_instance(preprocessed_images)  # len() = N
        self.t2 = time.perf_counter()  # model1 crop and preprocess - forward
        # Postprocess
        # TODO: conf_thres, iou_thres 리스트로 유저에 의해 변경될 수 있도록 변경
        nms_od_result = torch_non_max_suppression(
            prediction=raw_od_results,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=OD_TARGET_CLASSES,
            agnostic=True,
            max_det_per_img=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )
        self.t3 = time.perf_counter()  # forward - model1 nms
        matched_info = {}
        cropped_images = []
        total_bbox_info = []

        for cropped_batch_idx, batch_idx_info in enumerate(categories_batches):
            cnt = 0
            now_bbox_info = []

            for pred in nms_od_result[cropped_batch_idx]:
                x1, y1, x2, y2, conf, cls = pred
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    continue  # 너무 작은 박스는 무시

                origin_bbox = modified2origin_coordinate(
                    xyxy=(x1, y1, x2, y2),
                    now_shape=OD_INPUT_SIZE,
                    original_shape=cropped_batches[batch_idx_info].shape[1:3],
                )
                origin_bbox.extend([conf.item(), int(cls.item())])
                now_bbox_info.append(origin_bbox)
                x1, y1, x2, y2 = origin_bbox[:4]
                cropped_image = cropped_batches[batch_idx_info][:, y1:y2, x1:x2]
                cropped_images.append(cropped_image)
                cnt += 1
            matched_info[stream_ids[batch_idx_info]] = cnt
            total_bbox_info.append(now_bbox_info)
        self.t4 = time.perf_counter()  # model2 batch_make
        return cropped_images, matched_info, total_bbox_info, cropped_batches, categories_batches

    def get_classification_result(self, cropped_image: List[np.array], matched_info) -> List[bool]:
        """np.array type의 이미지를 받아 classification task를 수행하고, 결과를 반환합니다.

        Args:
            cropped_image (List[np.array]): Cropped된 이미지 리스트

        Returns:
            List[bool]: 모든 crooped_image에 대해 분류된 결과가 이상상황이면 True, 아니면 False를 반환받은 리스트
        """
        results = []
        raw_cls_results = []

        if len(cropped_image) == 0:
            return [self.ZERO_TENSOR] * len(matched_info), raw_cls_results

        resized_images = self.cls_letterbox_instance(imgs=cropped_image)
        preprocessed_images = resized_images / 255.0
        self.t5 = time.perf_counter()  # model2 preprocess
        raw_cls_results = self.classification_instance(preprocessed_images)  # len() = N
        self.t6 = time.perf_counter()  # model2 forward
        results = threshold_check(
            raw_cls_results,
            matched_info,
            TARGET_CATEGORY_INDEX,
            self.CLS_CONFIDENCE_THRESHOLD_TENSOR,
        )

        return results, raw_cls_results

    def get_alarm_helmet(self, cls_results: List[bool], matched_info: Dict[str, int]) -> None:
        """
        Classification 결과를 받아 이벤트 관리를 수행합니다.
        alarm_event_manager 클래스의 get_alarm 메소드를 호출하여 지정된 카메라들에 대해 이벤트를 관리합니다.

        Args:
            cls_results (List[bool]): Classification 결과
            matched_info (Dict[str, int]): stream_id 당 몇개씩 detection 되었는지 카운트를 알 수 있는 변수

        Returns:
            None
        """
        for idx, (camera_id, event_count) in enumerate(matched_info.items()):
            self.alarm_event_manager.update(cls_results[idx], camera_id)

        self.alarm_event_manager.get_alarm(matched_info, self.logging_flag)
