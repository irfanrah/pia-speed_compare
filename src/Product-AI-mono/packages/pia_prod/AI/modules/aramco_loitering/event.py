from collections import defaultdict
from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.aramco_loitering.config import (
    LOITERING_INTERVAL_SECOND,
    LOITERING_THRESHOLD_SECOND,
)


class LoiteringEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "loitering_cv"

        # {Stream ID: {Track ID: Integer Score}}
        self.duration_queue = defaultdict(lambda: defaultdict(int))

        self.event_status = defaultdict(int)
        self.threshold = LOITERING_THRESHOLD_SECOND / LOITERING_INTERVAL_SECOND
        self.max_score = self.threshold * 1.5  # 여유 버퍼 상단 선언

    def update(self, results: bool, stream_id: str, track_ids: list):
        """
        고유 Track ID를 직접 Key로 사용하여 업데이트하며,
        화면에서 사라져 카운트가 0이 된 객체는 딕셔너리에서 즉시 삭제하여 메모리를 관리합니다.
        """
        # 1. 이번 프레임에 존재하는 객체들의 점수 업데이트
        for t_id in track_ids:
            # results가 모든 객체에 동일하게 적용된다는 전제하의 로직 (수정 필요 시 변경)
            delta = 1 if results else -1
            self.duration_queue[stream_id][t_id] += delta

            # Clamp 적용
            self.duration_queue[stream_id][t_id] = clamp(
                self.duration_queue[stream_id][t_id], 0, self.max_score
            )

        # 2. 화면에서 사라진 객체들(Decay Logic) 처리
        # RuntimeError(dictionary changed size during iteration) 방지를 위해 list()로 키 복사
        active_track_ids = list(self.duration_queue[stream_id].keys())

        current_frame_set = set(track_ids)

        for t_id in active_track_ids:
            if t_id not in current_frame_set:
                # 화면에서 사라진 경우 카운트 차감
                self.duration_queue[stream_id][t_id] -= 1
                self.duration_queue[stream_id][t_id] = clamp(
                    self.duration_queue[stream_id][t_id], 0, self.max_score
                )

                # [메모리 관리] 점수가 0이 되면 추적할 필요가 없으므로 딕셔너리에서 삭제
                if self.duration_queue[stream_id][t_id] == 0:
                    del self.duration_queue[stream_id][t_id]

    def get_alarm(self) -> dict:
        # 기존과 동일하며, slot_idx 대신 t_id로 순회하는 의미가 됩니다.
        alarms = {}
        for s_id, tracks_dict in self.duration_queue.items():
            is_trigger = False

            # 특정 카메라(s_id) 내의 어떤 객체라도 임계치를 넘었는지 확인
            for count in tracks_dict.values():
                if count >= self.threshold:
                    is_trigger = True
                    break

            before_status = self.event_status[s_id]
            now_status = self.STATUS_TRANSITION[before_status][int(is_trigger)]
            self.event_status[s_id] = now_status

            if now_status in [1, 3]:
                alarms[s_id] = [self.EVENT_STATUS_DICT[now_status], None]

        return alarms


def clamp(n, min_n, max_n):
    return max(min_n, min(n, max_n))
