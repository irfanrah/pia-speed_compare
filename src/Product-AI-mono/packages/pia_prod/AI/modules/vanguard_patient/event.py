from collections import defaultdict, deque

from pia_prod.AI.bases.event_base import EventBase
from pia_prod.AI.modules.vanguard_patient.config import (
    PATIENT_QUEUE_SIZE,
    PATIENT_QUEUE_THRESHOLD,
    LIMITED_NUM_OF_TRACKED_OBJECTS,
)


class PatientEventManager(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "patient_cv"
        self.alarm_duration = PATIENT_QUEUE_THRESHOLD

        # [수정 1] Queue 저장소: {Stream ID: {Slot Index(0~15): Deque}}
        self.duration_queue = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=PATIENT_QUEUE_SIZE))
        )

        # [신규] 슬롯 소유자 추적: {Stream ID: {Slot Index: Real Track ID}}
        # 어떤 실제 ID가 현재 슬롯을 점유하고 있는지 확인용
        self.slot_owners = defaultdict(dict)

        # 관리할 최대 객체 수 (Slot 크기)
        self.max_slots = LIMITED_NUM_OF_TRACKED_OBJECTS

        self.event_status = defaultdict(int)

    def update(self, results: list, stream_id: str, track_ids: list):
        """
        Modulo 연산을 이용해 고정된 슬롯(Queue)만 사용하여 데이터를 갱신합니다.
        """

        # 이번 프레임에서 업데이트된 슬롯들을 기록 (나중에 사라진 객체 처리용)
        updated_slots = set()

        # 1. 현재 프레임의 감지된 객체 처리
        for t_id, is_abnormal in zip(track_ids, results):
            # 모듈러 연산으로 슬롯 인덱스 결정 (0 ~ 15)
            slot_idx = t_id % self.max_slots
            updated_slots.add(slot_idx)

            # [중요] 슬롯 주인이 바뀌었는지 확인 (ID 충돌 또는 새로운 객체 진입)
            current_owner = self.slot_owners[stream_id].get(slot_idx)

            if current_owner != t_id:
                # 주인이 다르다면(새로운 사람이 이 슬롯을 차지함), 기존 큐 초기화
                self.duration_queue[stream_id][slot_idx].clear()
                # 주인 변경
                self.slot_owners[stream_id][slot_idx] = t_id

            # 데이터 추가
            self.duration_queue[stream_id][slot_idx].append(int(is_abnormal))

        # 2. 화면에서 사라진 객체 처리 (Decay Logic)
        # 현재 점유 중인 모든 슬롯을 확인
        active_slots = list(self.slot_owners[stream_id].keys())

        for slot_idx in active_slots:
            # 이번 프레임에 업데이트되지 않은 슬롯 (=화면에서 사라진 객체)
            if slot_idx not in updated_slots:
                # 연속성을 끊기 위해 0(False) 추가
                self.duration_queue[stream_id][slot_idx].append(0)

                # [메모리/슬롯 관리]
                # 큐가 가득 찼는데 전부 0이라면(오랫동안 이상행동 없음/사라짐), 슬롯 점유 해제
                dq = self.duration_queue[stream_id][slot_idx]
                if len(dq) == dq.maxlen and sum(dq) == 0:
                    del self.slot_owners[stream_id][slot_idx]
                    dq.clear()

    def get_alarm(self) -> dict:
        """
        모든 스트림의 상태를 체크하여 이벤트가 발생한 카메라만 반환
        """
        alarms = {}
        all_streams = list(self.duration_queue.keys())

        for s_id in all_streams:
            slots_dict = self.duration_queue[s_id]

            is_trigger = False

            # 모든 활성 슬롯 검사
            if slots_dict:
                for slot_idx, queue in slots_dict.items():
                    # 큐의 합이 임계치를 넘으면 알람
                    if sum(queue) >= PATIENT_QUEUE_THRESHOLD:
                        is_trigger = True
                        break

            # 상태 전이 로직
            before_status = self.event_status[s_id]
            is_over_queue = int(is_trigger)

            now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
            self.event_status[s_id] = now_status

            if now_status in [1, 3]:
                alarms[s_id] = [self.EVENT_STATUS_DICT[now_status], None]

        return alarms
