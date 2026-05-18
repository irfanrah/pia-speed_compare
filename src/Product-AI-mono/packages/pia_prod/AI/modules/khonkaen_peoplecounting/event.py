from pia_prod.AI.bases.event_base import EventBase

# from collections import defaultdict, deque
# from pia_prod.AI.global_config import USER_PARAM_KEY, CV_EVENT_KEY


class PeoplecountingEvent(EventBase):
    def __init__(self):
        super().__init__()
        self.category_name = "peoplecounting_cv"
        self.frame_cnt = 0
        # self.event_dict = defaultdict(lambda: deque(maxlen=CROWD_PEOPLE_QUEUE_SIZE))
        # self.event_status = defaultdict(int)

    def update(self, batches_pred, stream_ids, user_params):
        # for result, stream_id, user_param in zip(batches_pred, stream_ids, user_params):
        #     is_crowd = 0
        #     category_name = list(user_param[USER_PARAM_KEY][CV_EVENT_KEY].keys())[0]
        #     if result > user_param[USER_PARAM_KEY][CV_EVENT_KEY][category_name]["people_threshold"]:
        #         is_crowd = 1
        #     self.event_dict[stream_id].append(is_crowd)
        # return self.check_alarm_duration()
        pass

    def check_alarm_duration(self):
        # alarms = {}
        # for stream_id, value in self.event_dict.items():
        #     before_status = self.event_status[stream_id]
        #     is_over_queue = int(sum(value) >= ALARM_THRESHOLD)
        #     now_status = self.STATUS_TRANSITION[before_status][is_over_queue]
        #     self.event_status[stream_id] = now_status
        #     if now_status in [1, 3]:
        #         alarms[stream_id] = [self.EVENT_STATUS_DICT[now_status], None]
        # return alarms
        pass
