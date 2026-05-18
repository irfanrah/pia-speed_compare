from collections import defaultdict, deque

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.glenc_harness.config import HARNESS_WITH_CLS_QUEUE_SIZE, HARNESS_WITH_CLS_QUEUE_THRESHOLD


class HarnessWithClsEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.alert_message = "ALERT: Not Wearing Harness detected!!"
        self.event_dict = defaultdict(lambda: deque(maxlen=HARNESS_WITH_CLS_QUEUE_SIZE))
        self.event_state = defaultdict(int)

    def update(self, cnt: int, camera_id: str) -> None:
        """
        특정 카메라에서 감지된 이상 상황 발생 횟수를 업데이트한다.

        Args:
            cnt (int): 감지된 이벤트 횟수.
            camera_id (str): 감지된 카메라의 ID.
        """
        self.event_dict[camera_id].append(cnt)

    def get_alarm(self, target) -> None:
        """
        이벤트 임계값을 초과한 카메라에 대해 알람을 트리거한다.

        Args:
            target (Iterable[str]): 분석할 대상 카메라 ID iterable. dict가 들어오면 key(camera_id)를 순회한다.
        """
        for camera_id in target:
            now_status = bool(sum(self.event_dict[camera_id]) >= HARNESS_WITH_CLS_QUEUE_THRESHOLD)  # False or True
            before_status = self.event_state[camera_id]  # EVENT_STATUS_DICT 내부의 상태

            self.event_state[camera_id] = self.STATUS_TRANSITION[before_status][
                now_status
            ]  # STATUS_TRANSITION을 통해 상태를 업데이트
