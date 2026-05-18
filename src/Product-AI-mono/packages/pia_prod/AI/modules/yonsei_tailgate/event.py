from collections import defaultdict
import numpy as np
from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.yonsei_tailgate.config import (
    VARIANCE_THRESHOLD_FOR_TAILGATE,
    TAIL_QUEUE_WINDOW_SIZE,
)

from pia_prod.AI.modules.yonsei_tailgate.debug_utils import save_snapshot_for_tailgate


class TailgateEventManager(EventBase):
    """
    카메라당 1개의 TailEventManager 인스턴스를 생성한다.
    하나의 TailEventManager 인스턴스는 N개의 camera 키를가진 dict를 가진다.
    각 Dict는 M개의 나눠진 ROI에 대해 각각을 관리하게 된다.

    patience는 각 카메라에 대한 dict 마지막 리스트의 값이 patience이다.
    patience가 0이면 해당 카메라는 분석할 필요가 없다.
    ROI내에 2명이 들어올 떄 patience는 5로 초기화되며, 사람이 없을 때 1씩 감소한다.
    patience가 존재할 때 각 카메라가 update되며 분산값을 통해 사람이 카드를 찍지않고 지나가는지에 대해 검사한다.
    """

    PATIENCE = 3  # 감지되는 Patience프레임간 사람 미검출 시, event_potision_dict 초기화
    ZERO_QUEUE_LIST = [0] * (TAIL_QUEUE_WINDOW_SIZE + 5)  #

    CHECK_NOW_QUEUE_IDX = -5
    OUT_FIRST_PERSON_IDX = -4
    NUM_DETECTIED_PERSON_IDX = -3
    NUM_KEEP_FRAME_IDX = -2
    PATIENCE_IDX = -1

    def __init__(self):
        super().__init__()
        self.category_name = "tailgate"
        self.alert_message = "ALERT: Tailgating detected!!"
        self.event_position_dict = defaultdict(
            lambda: defaultdict(lambda: self.ZERO_QUEUE_LIST.copy())
        )
        self.event_state = defaultdict(lambda: defaultdict(int))

        # For draw snapshot
        self.bbox_location = None
        self.letter_im = None
        self.roi = None

    def _get_people_position_list(
        self, vertical_info: tuple[int, int], bbox_infos: list = []
    ) -> list:
        """
        사람의 위치를 수직 좌표값을 기준으로 변환한다.

        Args:
            vertical_info (tuple[int, int]): 관심 영역(ROI)의 상단 및 하단 좌표값.
            bbox_infos (list): 감지된 사람의 바운딩 박스 정보.

        Returns:
            list: 변환된 좌표 리스트.
        """
        results = list()

        # error handling
        if len(bbox_infos) == 0 or vertical_info is None:
            return results

        min_y, max_y = vertical_info

        for bbox_info in bbox_infos:
            x0, y0, x1, y1 = bbox_info
            center_y = (y0 + y1) / 2
            relative_position = min(1, max(0, (center_y - min_y) / (max_y - min_y)))
            mapped_index = int(relative_position * TAIL_QUEUE_WINDOW_SIZE)
            results.append(mapped_index)

        return results

    def update(self, camera_id: int, roi_idx: int, position_list: list) -> None:
        (
            check_now_queue,
            out_first_person,
            num_detected_people_before,
            num_detected_frame,
            patience_score,
        ) = self.event_position_dict[camera_id][roi_idx][-5:]
        num_detected_people = len(position_list)

        # 사람이 1명이하임과 동시에 대기가 아닌 경우 넘기기
        if patience_score == 0 and num_detected_people <= 1:
            return

        now_queue = self.event_position_dict[camera_id][roi_idx]

        # 맨 첫번째 사람이 나간 경우의 큐
        if out_first_person:
            if check_now_queue:  # 이전프레임이 체크되었을 경우
                now_queue[-5] = 0  # check_now_queue를 False 처리
                now_queue[:-5] = [0] * TAIL_QUEUE_WINDOW_SIZE  # 체크되었다면 큐를 0으로 초기화

            if (num_detected_people_before - num_detected_people) >= 1:  # 사람이 1명 이상 나간 경우
                now_queue[-4] = 1  # out_first_person is True
                now_queue[-5] = 1  # 사람이 1명 나간경우 체크해야함

        # 빈 큐에 한쌍의 사람일 경우. 이때 사람이 더 추가될 일은 없으며 빠질일만 있다. 빠지면 out_first를 True로 바꾼다.
        elif not out_first_person:
            if num_detected_people == 2:
                now_queue[-1] = self.PATIENCE

            if (num_detected_people_before - num_detected_people) >= 1:  # 사람이 1명 이상 나간 경우
                now_queue[-4] = 1  # out_first_person is True

        if num_detected_people != 0:  # 사람 1명이라도 있으면
            now_queue[-1] = self.PATIENCE  # patience 초기화
        else:
            now_queue[-1] -= 1  # 사람없으면 patience 감소
            if now_queue[-1] == 0:
                self.event_position_dict[camera_id][
                    roi_idx
                ] = self.ZERO_QUEUE_LIST.copy()  # patience가 0이되면 del이 아니라 초기화
                return

        self.event_position_dict[camera_id][roi_idx][-2] += 1  # num_detected_frame
        self.event_position_dict[camera_id][roi_idx][
            -3
        ] = num_detected_people  # num_detected_people

        # 사람에 대한 처리
        for position in position_list:
            self.event_position_dict[camera_id][roi_idx][position] += 1

    def get_alarm(self, target_camaras: list, logging_flag: bool) -> None:
        """
        꼬리물기 감지를 수행하고 이벤트가 발생하면 알람을 생성한다.

        Args:
            target_cameras (list): 분석할 대상 카메라 리스트.
        """
        for camera_idx, camera_id in enumerate(target_camaras):
            for roi_idx, each_roi_list in self.event_position_dict[camera_id].items():
                now_variance = np.var(each_roi_list[:-5])  # 뒤에 5개는 제외
                check_now_queue = each_roi_list[-5]
                out_first_person = each_roi_list[-4]
                now_people_count = each_roi_list[-3]
                now_frame_number = each_roi_list[-2]
                patience_score = each_roi_list[-1]

                before_status = self.event_state[camera_id][roi_idx]

                if patience_score:
                    # 임의의 분산값보다 분산값이 작으면 now_status를 True로. 이상상황 감지
                    flag_fast_out_2_people = (now_people_count == 0) and (now_frame_number < 5)
                    flag_low_variance = (
                        (now_variance < VARIANCE_THRESHOLD_FOR_TAILGATE)
                        and (now_frame_number >= 5)
                        and (now_people_count == 0)
                        and (check_now_queue == 1)
                        and (out_first_person == 1)
                    )
                    # flag_3_people_over_detected = now_people_count >= 3  # 3명 들어갈일이 없음

                    if before_status in [
                        1,
                        2,
                    ]:  # patience가 있고, 이전 상태에서 진행중일 경우 바로 True 반환
                        now_status = True
                    elif flag_low_variance or flag_fast_out_2_people:  # 로직에 해당할 시 True 반환
                        now_status = True
                    else:  # 이전 상태가 진행중이지 않고, 로직에 해당하지 않을 시 False 반환
                        now_status = False

                    if logging_flag:
                        pass  # 임시 주석 처리
                        # print(
                        #     f"{camera_id}카메라의 {roi_idx}의 분산값: {now_variance}"
                        #     f"분산: {self.event_position_dict[camera_id][roi_idx]}"
                        #     f"사람수: {now_people_count}, patience: {patience_score}"
                        # )

                else:
                    now_status = False
                self.event_state[camera_id][roi_idx] = self.STATUS_TRANSITION[before_status][
                    now_status
                ]

            # Draw snapshot for tailgate
            # camera_id  # ['1_pia', ...]
            if logging_flag:
                now_origin_bboxs = self.bbox_location[camera_id]
                now_image = self.letter_im[camera_idx]  # 1장 이미지
                now_roi = self.roi[camera_idx]

                save_snapshot_for_tailgate(
                    now_image,
                    camera_id,
                    now_origin_bboxs,
                    now_roi,
                    category_name=self.category_name,
                )
