from collections import defaultdict, deque

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.khonkaen_weapon.config import (
    WEAPON_DETECTION_QUEUE_SIZE,
    WEAPON_DETECTION_QUEUE_THRESHOLD,
    TARGET_CATEGORY_INDEX,
    # CATEGORY_DICT,
)


class WeaponEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.alert_message = "ALERT: Weapon detected!!"
        self.event_dict = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=WEAPON_DETECTION_QUEUE_SIZE))
        )
        self.event_state = defaultdict(int)

    def update(self, sgie_results: list, stream_ids) -> None:
        """
        특정 카메라에서 감지된 이상 상황 발생 횟수를 업데이트한다.

        Args:
            sgie_results (list): SGIE(Secondary General Inference Engine) 결과 리스트.
        """
        for idx, now_weapon_set in enumerate(sgie_results):
            camera_id = stream_ids[idx]
            for target_category in TARGET_CATEGORY_INDEX:
                weapon_count = 1 if target_category in now_weapon_set else 0
                self.event_dict[camera_id][str(target_category)].append(weapon_count)

    def get_alarms(self, stream_ids) -> dict:
        """
        이벤트 임계값을 초과한 카메라에 대해 알람을 트리거한다.

        Returns:
            dict: 알람이 발생한 카메라와 해당 알람 상태를 포함하는 딕셔너리.
        """
        result_dict = {}
        for stream_id in stream_ids:
            before_status = self.event_state[stream_id]
            now_status = self._get_now_status(stream_id)
            state = self.STATUS_TRANSITION[before_status][now_status]
            self.event_state[stream_id] = state

            if state == 1:  # 알람 첫 시작
                result_dict[stream_id] = (True, None)
            elif state == 3:  # 알람 종료
                result_dict[stream_id] = (False, None)

        return result_dict

    def _get_now_status(self, camera_id: str) -> bool:
        is_detected = False
        for target_category in TARGET_CATEGORY_INDEX:
            if (
                sum(self.event_dict[camera_id][str(target_category)])
                >= WEAPON_DETECTION_QUEUE_THRESHOLD
            ):
                is_detected = True
                # print(
                #     f"target_category: {CATEGORY_DICT[target_category]}"
                # )
                break
        return is_detected
