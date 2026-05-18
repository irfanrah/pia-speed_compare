from trackers.ocsort_tracker.kalmanfilter import KalmanFilterNew
from trackers.ocsort_tracker.ocsort import KalmanBoxTracker
from copy import deepcopy
import numpy as np
import torch
from trackers.ocsort_tracker.association import (
    iou_batch,
    giou_batch,
    diou_batch,
    ciou_batch,
    ct_dist,
    linear_assignment,
    associate,
)


def k_previous_obs(observations, cur_age, k):
    if len(observations) == 0:
        return [-1, -1, -1, -1, -1]
    for i in range(k):
        dt = k - i
        if cur_age - dt in observations:
            return observations[cur_age - dt]
    max_age = max(observations.keys())
    return observations[max_age]


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
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r]).reshape((4, 1))


class PiaKalmanFilterNew(KalmanFilterNew):
    # 수정: dim_x와 dim_z를 인자로 받도록 변경
    def __init__(self, dim_x=7, dim_z=4):
        super().__init__(dim_x, dim_z)

    def unfreeze(self):
        if self.attr_saved is not None:
            new_history = deepcopy(self.history_obs)
            self.__dict__ = self.attr_saved
            # self.history_obs = new_history
            self.history_obs = self.history_obs[:-1]
            occur = [int(d is None) for d in new_history]
            indices = np.where(np.array(occur) == 0)[0]
            index1 = indices[-2]
            index2 = indices[-1]
            # box1 = new_history[index1]
            box1 = np.asarray(new_history[index1]).reshape(-1)
            x1, y1, s1, r1 = box1
            w1 = np.sqrt(s1 * r1)
            h1 = np.sqrt(s1 / r1)
            # box2 = new_history[index2]
            box2 = np.asarray(new_history[index2]).reshape(-1)
            x2, y2, s2, r2 = box2
            w2 = np.sqrt(s2 * r2)
            h2 = np.sqrt(s2 / r2)
            time_gap = index2 - index1
            dx = (x2 - x1) / time_gap
            dy = (y2 - y1) / time_gap
            dw = (w2 - w1) / time_gap
            dh = (h2 - h1) / time_gap
            for i in range(index2 - index1):
                """
                The default virtual trajectory generation is by linear
                motion (constant speed hypothesis), you could modify this
                part to implement your own.
                """
                x = x1 + (i + 1) * dx
                y = y1 + (i + 1) * dy
                w = w1 + (i + 1) * dw
                h = h1 + (i + 1) * dh
                s = w * h
                r = w / float(h)
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                """
                    I still use predict-update loop here to refresh the parameters,
                    but this can be faster by directly modifying the internal parameters
                    as suggested in the paper. I keep this naive but slow way for
                    easy read and understanding
                """
                self.update(new_box)
                if not i == (index2 - index1 - 1):
                    self.predict()


class PiaKalmanBoxTracker(KalmanBoxTracker):
    # 수정 1: track_id를 인자로 받도록 변경 (전역 변수 count 제거)
    def __init__(self, bbox, track_id, delta_t=3, orig=False):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        if not orig:
            KalmanFilter = PiaKalmanFilterNew
            self.kf = KalmanFilter(dim_x=7, dim_z=4)
        else:
            from filterpy.kalman import KalmanFilter

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
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0

        # 수정 1 반영: 외부에서 주입받은 ID 사용
        self.id = track_id

        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.last_observation = np.array([-1, -1, -1, -1, -1])
        self.observations = dict()
        self.history_observations = []
        self.velocity = None
        self.delta_t = delta_t
        self.cate = 0  # public update 사용 시 필요


ASSO_FUNCS = {
    "iou": iou_batch,
    "giou": giou_batch,
    "ciou": ciou_batch,
    "diou": diou_batch,
    "ct_dist": ct_dist,
}


class PiaOCSort(object):
    def __init__(
        self,
        det_thresh,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
        max_objs=None,  # 수정 2: 최대 객체 수 제한 인자 추가
    ):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.delta_t = delta_t
        self.asso_func = ASSO_FUNCS[asso_func]
        self.inertia = inertia
        self.use_byte = use_byte

        # 수정 2: 최대 객체 수 저장
        self.max_objs = max_objs

        # 수정 1: 인스턴스별 ID 카운터 관리 (전역 변수 대체)
        self.track_id_count = 0

    def update(self, output_results, img_info, img_size):
        if output_results is None or len(output_results) == 0:
            return np.empty((0, 5))

        self.frame_count += 1

        if isinstance(output_results, torch.Tensor):
            output_results = output_results.cpu().numpy()

        # post_process detections
        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            # scores = output_results[:, 4] * output_results[:, 5]
            # bboxes = output_results[:, :4]
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]  # x1y1x2y2

        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale
        dets = np.concatenate((bboxes, np.expand_dims(scores, axis=-1)), axis=1)

        inds_low = scores > 0.1
        inds_high = scores < self.det_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = dets[inds_second]
        remain_inds = scores > self.det_thresh
        dets = dets[remain_inds]

        # get predicted locations
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        velocities = np.array(
            [
                trk.velocity if trk.velocity is not None else np.array((0, 0))
                for trk in self.trackers
            ]
        )
        last_boxes = np.array([trk.last_observation for trk in self.trackers])
        k_observations = np.array(
            [k_previous_obs(trk.observations, trk.age, self.delta_t) for trk in self.trackers]
        )

        """ First round of association """
        matched, unmatched_dets, unmatched_trks = associate(
            dets, trks, self.iou_threshold, velocities, k_observations, self.inertia
        )
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        """ Second round of association by OCR (BYTE) """
        if self.use_byte and len(dets_second) > 0 and unmatched_trks.shape[0] > 0:
            u_trks = trks[unmatched_trks]
            iou_left = self.asso_func(dets_second, u_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                matched_indices = linear_assignment(-iou_left)
                to_remove_trk_indices = []
                for m in matched_indices:
                    det_ind, trk_ind = m[0], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets_second[det_ind, :])
                    to_remove_trk_indices.append(trk_ind)
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        if unmatched_dets.shape[0] > 0 and unmatched_trks.shape[0] > 0:
            left_dets = dets[unmatched_dets]
            left_trks = last_boxes[unmatched_trks]
            iou_left = self.asso_func(left_dets, left_trks)
            iou_left = np.array(iou_left)
            if iou_left.max() > self.iou_threshold:
                rematched_indices = linear_assignment(-iou_left)
                to_remove_det_indices = []
                to_remove_trk_indices = []
                for m in rematched_indices:
                    det_ind, trk_ind = unmatched_dets[m[0]], unmatched_trks[m[1]]
                    if iou_left[m[0], m[1]] < self.iou_threshold:
                        continue
                    self.trackers[trk_ind].update(dets[det_ind, :])
                    to_remove_det_indices.append(det_ind)
                    to_remove_trk_indices.append(trk_ind)
                unmatched_dets = np.setdiff1d(unmatched_dets, np.array(to_remove_det_indices))
                unmatched_trks = np.setdiff1d(unmatched_trks, np.array(to_remove_trk_indices))

        for m in unmatched_trks:
            self.trackers[m].update(None)

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            # 수정 2 적용: 카메라당 최대 인원 수 제한 체크
            # 현재 트래커 수가 제한(max_objs) 이상이면 새로운 트래커 생성을 중단합니다.
            if self.max_objs is not None and len(self.trackers) >= self.max_objs:
                continue

            # 수정 1 적용: 인스턴스 내부 카운터 사용
            self.track_id_count += 1
            new_id = self.track_id_count

            trk = PiaKalmanBoxTracker(dets[i, :], track_id=new_id, delta_t=self.delta_t)
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            if trk.last_observation.sum() < 0:
                d = trk.get_state()[0]
            else:
                d = trk.last_observation[:4]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(np.concatenate((d, [trk.id])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))

    # update_public 메서드 생략 (위의 update와 동일한 로직으로 수정 필요)


# (이전의 PiaOCSort, PiaKalmanBoxTracker 등의 코드는 그대로 유지)


class MultiStreamPiaOCSort:
    def __init__(
        self,
        det_thresh,
        max_objs=16,
        max_age=30,
        min_hits=3,
        iou_threshold=0.3,
        delta_t=3,
        asso_func="iou",
        inertia=0.2,
        use_byte=False,
    ):

        # 트래커 생성에 필요한 인자들 저장
        self.tracker_args = {
            "det_thresh": det_thresh,
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_threshold,
            "delta_t": delta_t,
            "asso_func": asso_func,
            "inertia": inertia,
            "use_byte": use_byte,
            "max_objs": max_objs,
        }

        # 스트림별 트래커 관리 딕셔너리
        self.trackers = {}

    def update(self, results, stream_ids, img_infos, img_sizes=None):
        """
        Returns:
            tuple: (batch_outputs, track_info)
                - batch_outputs (list): [Result_Array_1, Result_Array_2, ...]
                - track_info (dict): {'stream_id': [id1, id2, ...], ...}
        """

        batch_size = len(stream_ids)
        if len(results) != batch_size or len(img_infos) != batch_size:
            raise ValueError("Input lists lengths must match.")

        if img_sizes is None:
            img_sizes = img_infos

        batch_outputs = []
        track_info = {}  # ID 정보를 담을 딕셔너리 초기화

        # 배치 내부 순회 (0 ~ Batch-1)
        for i in range(batch_size):
            s_id = stream_ids[i]
            dets = results[i]
            info = img_infos[i]
            size = img_sizes[i]

            # 1. 트래커 인스턴스 확보
            if s_id not in self.trackers:
                self.trackers[s_id] = PiaOCSort(**self.tracker_args)

            tracker = self.trackers[s_id]

            # 2. Tensor -> Numpy 변환
            if isinstance(dets, torch.Tensor):
                dets = dets.cpu().numpy()

            # 3. 업데이트 수행
            ret = tracker.update(dets, info, size)
            batch_outputs.append(ret)

            # 4. [추가된 로직] track_info 생성 (Numpy 연산 즉시 처리)
            # ret 형식: [[x1, y1, x2, y2, id], ...]
            current_ids = []
            if ret is not None and len(ret) > 0:
                # 마지막 컬럼(ID)만 추출 -> 정수 변환 -> 리스트 변환
                current_ids = ret[:, 4].astype(int).tolist()

            # 딕셔너리에 추가 (동일 stream_id가 여러 번 나올 경우 확장)
            if s_id not in track_info:
                track_info[s_id] = current_ids
            else:
                track_info[s_id].extend(current_ids)

        return batch_outputs, track_info
