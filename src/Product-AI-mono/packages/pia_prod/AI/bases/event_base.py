from abc import ABC, abstractmethod
from typing import final


class EventBase(ABC):
    EVENT_STATUS_DICT = {
        0: "no_event",  # No event
        1: True,  # Event started now
        2: "continue",  # Event is continuing currently
        3: False,  # Event is ended now
    }

    STATUS_TRANSITION = {
        0: {0: 0, 1: 1},  # false -> false의 경우 `false`  # false -> true인 경우 `true`
        1: {0: 3, 1: 2},  # true -> false 의 경우 `end`  # true -> true 의 경우 `continue`
        2: {0: 3, 1: 2},  # continue -> false의 경우 `end`  # continue -> true의 경우 `continue`
        3: {0: 0, 1: 1},  # end -> false의 경우 `false`  # end -> true의 경우 `true`
    }

    def __init__(self, alarm_duration=None):
        self.alret_message = ""
        self.alarm_duration = alarm_duration

    @final
    def __call__(self, *args, **kwds):
        return self.update(*args, **kwds)

    @abstractmethod
    def update(self, *args, **kwds) -> list:
        """
        알람이 발생한 스트림만을 리턴
        시작 또는 종료 시에만 return에 값 추가
        그 외는 빈 배열 리턴 (return [ ] )
        """
        return [["stream_id", "is_start"]]
