from pia.ai.tasks.tracker.base import TrackerBase, TrackerConfig

from .sort import Sort


class CallTracker:
    AVAILABLE_TRACKER = {
        None: TrackerBase,
        0: Sort,
        "sort": Sort,
    }

    def __new__(cls, config: TrackerConfig) -> TrackerBase:
        return cls.AVAILABLE_TRACKER[config.tracker](**config.__dict__)
