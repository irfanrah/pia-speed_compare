from typing import Union

import numpy as np
from filterpy.kalman import KalmanFilter
from pia.ai.tasks.tracker.base import TrackerBase


def linear_assignment(cost_matrix):
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment

        x, y = linear_sum_assignment(cost_matrix, maximize=False)
        return np.array(list(zip(x, y)))


def distance_batch(bb_pred, bb_track):
    """
    Computes pixel euclidean distance between two bboxes in the form [tl_x, tl_y, br_x, br_y] -> bbox center point
    """
    pred_x_center = np.expand_dims(bb_pred[:, 0] + (bb_pred[:, 2] - bb_pred[:, 0]) / 2, 1)
    pred_y_center = np.expand_dims(bb_pred[:, 1] + (bb_pred[:, 3] - bb_pred[:, 1]) / 2, 1)

    track_x_center = np.expand_dims(bb_track[:, 0] + (bb_track[:, 2] - bb_track[:, 0]) / 2, 0)

    track_y_center = np.expand_dims(bb_track[:, 1] + (bb_track[:, 3] - bb_track[:, 1]) / 2, 0)

    distance = np.sqrt(
        np.square(pred_x_center - track_x_center) + np.square(pred_y_center - track_y_center)
    )

    return distance


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
    )
    return o


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if score is None:
        return np.array([x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0]).reshape(
            (1, 4)
        )
    else:
        return np.array(
            [x[0] - w / 2.0, x[1] - h / 2.0, x[0] + w / 2.0, x[1] + h / 2.0, score]
        ).reshape((1, 5))


class KalmanBoxTracker:
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """

    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 10.0  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        # self.kf.x[5] = bbox[5]
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        self.class_name = bbox[-1]
        KalmanBoxTracker.count += 1
        # if KalmanBoxTracker.count > MANY_WAY_LINE_BUDGET:
        #   KalmanBoxTracker.count = 0

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.class_name = bbox[-1]
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, threshold=100, match_type="euc"):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )
    if match_type == "euc":
        matrix = distance_batch(detections, trackers)
    elif match_type == "iou":
        matrix = iou_batch(detections, trackers)

    if min(matrix.shape) > 0:
        a = (matrix <= threshold).astype(np.int32)  # threshold 이상 표시
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:  #
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            if match_type == "euc":
                matched_indices = linear_assignment(matrix)
            elif match_type == "iou":
                matched_indices = linear_assignment(-matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # filter out matched with far distance
    matches = []
    for m in matched_indices:
        if match_type == "euc":
            condition = matrix[m[0], m[1]] > threshold
        elif match_type == "iou":
            condition = matrix[m[0], m[1]] < threshold

        if condition:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(TrackerBase):
    def __init__(
        self,
        max_age: int = 1,
        min_hits: int = 3,
        threshold: Union[int, float] = 100,
        match_type="euc",
        **keyword
    ):
        """
        Sets key parameters for SORT

        params :
            match_type : euc or iou
        """
        self.max_age = max_age  # 최대 검출 안되고 인덱스가 유지되는 기간
        self.min_hits = min_hits  # 몇번 이상 검출되야 추적박스라고 볼지
        self.threshold = threshold  # 박스 매칭시  임계치
        self.trackers = []  # 현재 트레킹 되고있는 박스 정보
        self.frame_count = 0
        self.match_type = match_type
        KalmanBoxTracker.count = 0

    def update(self, dets=np.empty((0, 6))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score,category],[x1,y1,x2,y2,score,category],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 6))  # 예측값을 초기화 현재는 x,y,w,h, index, category
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [
                pos[0],
                pos[1],
                pos[2],
                pos[3],
                0,
                self.trackers[t].class_name,
            ]  # 인덱스 0부터 시작 , class name 값 추가
            if np.any(np.isnan(pos)):  # 계산 오류가 있으면(Nan값) 삭제해야할 리스트에 저장한다
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets, trks, self.threshold, match_type=self.match_type
        )

        # update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id + 1, trk.class_name])).reshape(1, -1)
                )  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))
