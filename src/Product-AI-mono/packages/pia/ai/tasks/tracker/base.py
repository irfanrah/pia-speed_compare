from abc import abstractmethod
from typing import Union


class TrackerBase:
    """
    This class is not set as an abstract class
    because it needs to function when a tracker is not selected.
    """

    @abstractmethod
    def __init__(self, *args, **keyword) -> None:
        pass

    def __call__(self, *args, **keyword):
        return self.update(*args, **keyword)

    @abstractmethod
    def update(self, *args, **keyword):
        return args[0]  # To return the box information as is when there is no tracker.


class TrackerConfig:
    """
    Initialize TrackerConfig class with configuration for object tracking.

    Args:
        tracker (Union[str, int], optional): 0 or "sort" for the SORT tracker. Defaults to None.
        match_type (str, optional): "euc" for Euclidean distance, "cos" for cosine similarity. Defaults to "euc".
        threshold (Union[int, float], optional): The threshold for matching. Defaults to 100.
        max_age (int, optional): The maximum age of the tracker. Defaults to 1.
        min_hits (int, optional): The minimum number of hits. Defaults to 3.

    Returns:
        None

    Examples:
        Initialize TrackerConfig with default settings:

        >>> tracker_config = TrackerConfig()
        >>> config = ODConfig(model_path="model.pth", tracker_config=tracker_config)
        >>> model = PiaTorchModel(target_task="OD", target_model="yolov8", config=config)
        >>> img = cv2.imread("image.jpg")
        >>>
        >>> ret = model.forward(img)

        Initialize TrackerConfig with custom settings:

        >>> tracker_config = TrackerConfig(tracker="sort", max_age=3, min_hits=1, threshold=200)
        >>> config = ODConfig(model_path="model.pth", tracker_config=tracker_config)
        >>> model = PiaTorchModel(target_task="OD", target_model="yolov8", config=config)
        >>> img = cv2.imread("image.jpg")
        >>>
        >>> ret = model.forward(img)
    """

    def __init__(
        self,
        tracker: Union[str, int] = None,
        match_type: str = "euc",
        threshold: Union[int, float] = 100,
        max_age: int = 1,
        min_hits: int = 3,
    ) -> None:
        self.tracker = tracker
        self.match_type = match_type
        self.threshold = threshold
        self.max_age = max_age
        self.min_hits = min_hits
