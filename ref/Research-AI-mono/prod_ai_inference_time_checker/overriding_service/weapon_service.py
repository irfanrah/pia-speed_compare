from pia_prod.AI.modules.khonkaen_weapon.service import WeaponService
from typing import List, Dict
import numpy as np
from pia.vision.postprocessing.nms import torch_non_max_suppression
from pia_prod.AI.modules.khonkaen_weapon.config import (
    LIMITED_NUM_OF_PERSON_PER_CAMERA,
    OD_NMS_THRESHOLD,
    OD_INPUT_SIZE,
    OD_TARGET_CLASSES,
    WEAPON_DETECTION_CONFIDENCE_THRESHOLD,
    TARGET_CATEGORY_INDEX,
    TOP_MARGIN_RATIO,
    LEFT_MARGIN_RATIO,
    RIGHT_MARGIN_RATIO,
    BOTTOM_MARGIN_RATIO,
    OD_CONFIDENCE_THRESHOLD,
)
from pia.vision.postprocessing.bbox import expand_bbox
from pia.vision.postprocessing.bbox import modified2origin_coordinate
import time


class ProfileWeaponService(WeaponService):
    def __init__(self, analysis_data_queue):
        super().__init__(analysis_data_queue)
        # ======= 결과 정리 =======
        self.stages = [
            "total",
            "model1_preprocess",
            "model1_inference",
            "model1_nms",
            "model2_batch_make",
            "model2_preprocess",
            "model2_inference",
            "model2_nms",
            "postprocess_logic",
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

        (
            cropped_images,
            matched_info,
            self.total_bbox_info,  # for draw snapshot
            cropped_batches,  # for debug
        ) = self.get_pgie_results(batches=batches, stream_ids=stream_ids, user_params=user_params)

        sgie_results = self.get_sgie_results(
            cropped_images=cropped_images,
            matched_info=matched_info,
            stream_ids=stream_ids,
            user_params=user_params,
        )

        self.alarm_event_manager.update(sgie_results, stream_ids)
        alarms = self.alarm_event_manager.get_alarms(stream_ids)
        self.t8 = time.perf_counter()  # postprocess logic

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
            self.time_dict["model2_nms"].append(times[7])
            self.time_dict["postprocess_logic"].append(times[8])
            self.time_dict["send_alarm"].append(times[9])

            # ======= 표로 출력 -> average 로 대처=======
            # df = pd.DataFrame([times], columns=self.stages)
            # print(df.round(2).to_string(index=False))
        self.frame_cnt += 1
        
        return alarms, batches, stream_ids, user_params, self.is_needed_cvt_color

    def get_pgie_results(
        self,
        batches: List[np.array],
        stream_ids: List[str],
        user_params: List[Dict],
    ):
        # Preprocess 1 - Crop with RoI
        cropped_batches = self.roi_manager.process_batches_with_roi(batches=batches, user_params=user_params)
        self.t1 = time.perf_counter()  # start - model1 crop and preprocess
        # Preprocess 2 - Letterbox & Normalize
        resized_images = self.pgie_letterbox_instance(cropped_batches)
        preprocess_images = resized_images / 255.0

        # Inference model
        raw_od_results = self.pgie_detection_instance(preprocess_images)
        self.t2 = time.perf_counter()  # model1 crop and preprocess - forward

        # Postprocess
        # TODO: conf_thres, iou_thres 리스트로 유저에 의해 변경될 수 있도록 변경
        nms_od_results = torch_non_max_suppression(
            prediction=raw_od_results,
            conf_thres=OD_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=OD_TARGET_CLASSES,
            agnostic=True,
            max_det_per_img=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )
        self.t3 = time.perf_counter()  # forward - model1 nms

        total_bbox_info = []
        cropped_images = []
        matched_info = {}
        for cropped_batch_index, batch_index_info in enumerate(range(len(batches))):
            cnt = 0
            now_bbox_info = []
            max_w, max_h = (
                cropped_batches[cropped_batch_index].shape[2],
                cropped_batches[cropped_batch_index].shape[1],
            )
            for pred in nms_od_results[cropped_batch_index]:
                x1, y1, x2, y2, conf, cls = pred
                x1, y1, x2, y2 = expand_bbox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    max_w=max_w,
                    max_h=max_h,
                    top_ratio=TOP_MARGIN_RATIO,
                    left_ratio=LEFT_MARGIN_RATIO,
                    right_ratio=RIGHT_MARGIN_RATIO,
                    bottom_ratio=BOTTOM_MARGIN_RATIO,
                )
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                if (x2 - x1) < 2 or (y2 - y1) < 2:
                    continue  # crop 불가능한 bbox 제외

                origin_bbox = modified2origin_coordinate(
                    xyxy=(x1, y1, x2, y2),
                    now_shape=OD_INPUT_SIZE,
                    original_shape=cropped_batches[batch_index_info].shape[1:3],
                )
                origin_bbox.extend([conf.item(), int(cls.item())])
                now_bbox_info.append(origin_bbox)
                x1, y1, x2, y2 = origin_bbox[:4]
                cropped_image = cropped_batches[batch_index_info][:, y1:y2, x1:x2]
                cropped_images.append(cropped_image)
                cnt += 1
            matched_info[stream_ids[batch_index_info]] = cnt
            total_bbox_info.append(now_bbox_info)

        self.t4 = time.perf_counter()  # model2 batch_make
        return (cropped_images, matched_info, total_bbox_info, cropped_batches)

    def get_sgie_results(
        self,
        cropped_images: List[np.array],
        matched_info: Dict,
        stream_ids: List[str],
        user_params: List[Dict],
    ) -> List[set]:
        result = []
        if len(cropped_images) == 0:
            return [{0}] * len(matched_info)

        resized_images = self.sgie_letterbox_instance(cropped_images)
        resized_images = resized_images / 255.0

        self.t5 = time.perf_counter()  # model2 preprocess
        raw_sgie_results = self.sgie_detection_instance(resized_images)
        self.t6 = time.perf_counter()  # model2 forward
        nms_od_results = torch_non_max_suppression(
            prediction=raw_sgie_results,
            conf_thres=WEAPON_DETECTION_CONFIDENCE_THRESHOLD,
            iou_thres=OD_NMS_THRESHOLD,
            classes=TARGET_CATEGORY_INDEX,
            agnostic=True,
            max_det_per_img=LIMITED_NUM_OF_PERSON_PER_CAMERA,
        )
        self.t7 = time.perf_counter()  # model2 nms
        if self.logging_flag:
            cnt = 0
            for stream_index in range(len(user_params)):  # 0,1,2...
                now_key = stream_ids[stream_index]
                now_num_batches = matched_info[now_key]
                target_od_results = nms_od_results[cnt : cnt + now_num_batches]
                for target_idx, target in enumerate(target_od_results):
                    if len(target) != 0:
                        for pred in target:
                            x1, y1, x2, y2 = map(int, pred[:4].tolist())
                            conf = pred[4].item()
                            cls = int(pred[5].item())
                            related_coordinates = modified2origin_coordinate(
                                xyxy=(x1, y1, x2, y2),
                                now_shape=OD_INPUT_SIZE,
                                original_shape=cropped_images[cnt + target_idx].shape[1:3],
                            )
                            real_x1 = self.total_bbox_info[stream_index][target_idx][0] + related_coordinates[0]
                            real_y1 = self.total_bbox_info[stream_index][target_idx][1] + related_coordinates[1]
                            real_x2 = self.total_bbox_info[stream_index][target_idx][0] + related_coordinates[2]
                            real_y2 = self.total_bbox_info[stream_index][target_idx][1] + related_coordinates[3]

                            self.total_bbox_info[stream_index].append([real_x1, real_y1, real_x2, real_y2, conf, cls])
                cnt += now_num_batches

        cnt = 0
        for stream_index in range(len(user_params)):  # 0,1,2...
            now_key = stream_ids[stream_index]
            now_num_batches = matched_info[now_key]
            if now_num_batches == 0:
                result.append({0})
                continue

            detected_class_idx = set()
            target_od_results = nms_od_results[cnt : cnt + now_num_batches]

            for t in target_od_results:  # 카테고리 정보만 set 자료형으로 추출하여 배치마다 저장
                cats = t[:, 5].tolist()
                detected_class_idx.update(int(c) for c in cats)

            result.append(detected_class_idx)
            cnt += now_num_batches

        return result
