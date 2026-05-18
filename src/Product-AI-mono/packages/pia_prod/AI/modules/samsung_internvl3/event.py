from pia_prod.AI.bases.event_base import EventBase
from collections import deque, defaultdict
from typing import List
import json
import re
from pia.utils.devtools.debug_tools import print_only_debug_mode
from pia_prod.AI.utils.init import logger
from pia_prod.AI.modules.samsung_internvl3.config import (
    QUEUE_SIZE,
    ALARM_DURATION_THRESHOLD,
    MODEL_OUTPUTS,
    FALLDOWN_CATEGORY,
    FIRE_CATEGORY,
)


class VQAEventManager(EventBase):
    def __init__(self):
        super().__init__(
            # queue_size=QUEUE_SIZE,
            alarm_duration=ALARM_DURATION_THRESHOLD
        )
        self.duration_queue = defaultdict(lambda: defaultdict(lambda: deque(maxlen=(QUEUE_SIZE))))
        self.event_status = defaultdict(lambda: defaultdict(int))

    def update(self, responses: List[str], categories_per_stream: List[str], stream_ids: List[str]):
        # categories_per_stream : [[cat1, cat2, cat3], [cat1, cat2, cat3], [cat1, cat2, cat3]]
        # response : [res1, res2, res3, res4, res5, res6, res7, res8, res9]
        # stream_ids : [stream_id1, stream_id2, stream_id3]
        i = 0
        for categories, stream_id in zip(categories_per_stream, stream_ids):
            for cat in categories:
                res = responses[i]
                i += 1
                predict = self.extract_info(res)
                print_only_debug_mode(
                    f"stream_id : {stream_id}\tcategory : {cat}\tpredict : {predict}"
                )
                if predict in MODEL_OUTPUTS[0] and cat in FALLDOWN_CATEGORY:  # falldown
                    self.duration_queue[stream_id][cat].append(1)
                elif predict in MODEL_OUTPUTS[1] and cat in FIRE_CATEGORY:  # fire
                    self.duration_queue[stream_id][cat].append(1)
                else:
                    self.duration_queue[stream_id][cat].append(0)
        return self.check_alarm_duration()

    def check_alarm_duration(self):
        alarms = {}
        for stream_id, cat_dict in self.duration_queue.items():
            for category_id, value in cat_dict.items():
                before_status = self.event_status[stream_id][category_id]
                is_over_queue = int(sum(value) >= ALARM_DURATION_THRESHOLD)
                now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
                self.event_status[stream_id][category_id] = now_status
                if now_status in [1, 3]:
                    alarms[stream_id] = [self.EVENT_STATUS_DICT[now_status], category_id]
        return alarms

    @staticmethod
    def extract_info(s):
        try:
            # 1. 문자열 앞뒤 공백 제거
            s = s.strip()

            # 2. ```json ... ``` 제거
            s = re.sub(r"^```json\s*", "", s)
            s = re.sub(r"\s*```$", "", s)

            # 3. 다시 공백 정리
            s = s.strip()

            # ✅ 3.5 json.loads 전에 JSON형식 보정
            # 완전한 JSON이 아니면 category 키를 포함한 JSON 문자열로 감쌈
            if not (s.startswith("{") and s.endswith("}")):
                # category 값만 추출 (fallback: normal)
                m = re.search(r'"?category"?\s*[:=]\s*"?([a-zA-Z_ -]+)"?', s)
                if m:
                    cat = m.group(1).strip().lower()
                else:
                    # 문장 안에서 normal/falldown/fire 키워드 탐색
                    extracted_category = s.lower()
                    if "falldown" in extracted_category:
                        cat = "falldown"
                    elif "fire" in extracted_category:
                        cat = "fire"
                    elif "normal" in extracted_category:
                        cat = "normal"
                    else:
                        raise KeyError(f"Unrecognized category : {s}")
                s = json.dumps({"category": cat})

            # 4. JSON 파싱
            data = json.loads(s)

            category = data.get("category")
            return category

        except Exception as e:
            logger.info(f"JSON 추출 실패: {e}\n원본 문자열:\n{s}")
            return "normal"
