import numpy as np
from collections import deque, Counter


def smooth_velocity(points: deque, win: int = 5) -> np.ndarray:
    if len(points) < 2:
        return np.array([0.0, 0.0])
    k = min(win, len(points) - 1)
    return (np.asarray(points[-1]) - np.asarray(points[-1 - k])) / max(k, 1)


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    if not np.array_equal(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    inside = False
    x0, y0 = x, y
    for i in range(len(poly) - 1):
        (x1, y1), (x2, y2) = poly[i], poly[i + 1]
        cond = ((y1 > y0) != (y2 > y0)) and (x0 < (x2 - x1) * (y0 - y1) / (y2 - y1 + 1e-12) + x1)
        if cond:
            inside = not inside
    return inside


class LanePolygon:
    def __init__(self, polygon_points, dir_vec):
        arr = np.asarray(polygon_points, float)
        if not np.array_equal(arr[0], arr[-1]):
            arr = np.vstack([arr, arr[0]])
        self.poly = arr
        dv = np.asarray(dir_vec, float)
        self.dir_unit = dv / (np.linalg.norm(dv) + 1e-9)

    def contains(self, pt):
        # accept pt as (x,y) tuple/list/ndarray and guard against strange shapes
        if isinstance(pt, (list, tuple)):
            px, py = pt[0], pt[1]
        else:
            arr = np.asarray(pt)
            if arr.ndim == 1 and arr.size >= 2:
                px, py = float(arr[0]), float(arr[1])
            else:
                raise ValueError("pt must be a sequence of length >=2")
        return point_in_polygon(px, py, self.poly)

    def cosine_with(self, v):
        n = np.linalg.norm(v)
        if n < 1e-6:
            return None
        return float(np.dot(v, self.dir_unit) / (n + 1e-9))


class WrongWayParams:
    def __init__(
        self,
        fps=25,
        vel_win=5,
        speed_min=2.0,
        cos_on=-0.55,
        cos_off=-0.35,
        consec_on=8,
        grace_off=6,
        avg_win=7,
        lane_stable=5,
        max_idle=15,
        homography=None,
    ):
        self.fps = fps
        self.vel_win = vel_win
        self.speed_min = speed_min
        self.cos_on = cos_on
        self.cos_off = cos_off
        self.consec_on = consec_on
        self.grace_off = grace_off
        self.avg_win = avg_win
        self.lane_stable = lane_stable
        self.max_idle = max_idle
        self.homography = homography


class TrackState:
    def __init__(self):
        self.centers = deque(maxlen=240)
        self.last_frame = None
        self.neg_count = 0
        self.grace_left = 0
        self.active = False
        self.event_start = None
        self.lane_idx = None
        self.lane_hist = deque(maxlen=20)
        self.cos_hist = deque(maxlen=50)
        self.event_log = []


class WrongWayDetectorPolygon:
    def __init__(self, lanes: list[LanePolygon], params: WrongWayParams):
        self.lanes: list[LanePolygon] = lanes
        self.p: WrongWayParams = params
        self.tracks = {}
        self.events = []
        self.lane_centroids = [self._poly_centroid(line.poly) for line in self.lanes]

    @staticmethod
    def _poly_centroid(poly):
        P = poly[:-1] if np.array_equal(poly[0], poly[-1]) else poly
        x = P[:, 0]
        y = P[:, 1]
        a = 0.0
        cx = 0.0
        cy = 0.0
        for i in range(len(P)):
            j = (i + 1) % len(P)
            cross = x[i] * y[j] - x[j] * y[i]
            a += cross
            cx += (x[i] + x[j]) * cross
            cy += (y[i] + y[j]) * cross
        a *= 0.5
        if abs(a) < 1e-6:
            return np.array([x.mean(), y.mean()], float)
        cx /= 6.0 * a
        cy /= 6.0 * a
        return np.array([cx, cy], float)

    def _point_to_poly_dist(self, C, poly):
        if point_in_polygon(C[0], C[1], poly):
            return 0.0
        P = poly[:-1] if np.array_equal(poly[0], poly[-1]) else poly
        mind = 1e18
        for i in range(len(P)):
            a = P[i]
            b = P[(i + 1) % len(P)]
            ab = b - a
            t = np.clip(np.dot(C - a, ab) / (np.dot(ab, ab) + 1e-9), 0, 1)
            proj = a + t * ab
            d = np.linalg.norm(C - proj)
            if d < mind:
                mind = d
        return float(mind)

    def _choose_lane_with_hysteresis(self, C, prev_idx=None, switch_thresh=20.0):
        for i, lane in enumerate(self.lanes):
            if lane.contains((C[0], C[1])):
                return i
        dists = [self._point_to_poly_dist(C, line.poly) for line in self.lanes]
        best = int(np.argmin(dists))
        if prev_idx is None:
            return best
        return best if dists[best] + 1e-6 < dists[prev_idx] - switch_thresh else prev_idx

    def update(self, frame_idx: int, track_id: int, cx: float, cy: float) -> None:
        st = self.tracks.get(track_id)
        if st is None:
            st = TrackState()
            self.tracks[track_id] = st

        if st.last_frame is not None and (frame_idx - st.last_frame) > self.p.max_idle:
            st = TrackState()
            self.tracks[track_id] = st

        C = np.array([cx, cy], float)
        st.centers.append(C)
        st.last_frame = frame_idx

        v = smooth_velocity(st.centers, win=self.p.vel_win)
        speed_px_per_frame = np.linalg.norm(v)
        speed_px_per_sec = speed_px_per_frame * self.p.fps

        if not st.event_log:
            tracking_start_frame = frame_idx
            st.event_log.append(
                {
                    "id": track_id,
                    "tracking_start_frame": tracking_start_frame,
                    "lane_idx": st.lane_idx,
                    "start_frame": st.event_start,
                    "end_frame": frame_idx,
                    "speed_px_s": float(speed_px_per_sec),
                    "active": bool(st.active),
                    "velocity_vec": v.tolist(),
                    "path_length": 0.0,
                }
            )
        else:
            cur_evt = st.event_log[-1]
            cur_evt["end_frame"] = frame_idx

        if speed_px_per_sec < self.p.speed_min:
            if st.active:
                st.grace_left = max(st.grace_left, self.p.grace_off)
            st.neg_count = 0
            st.cos_hist.append(0.0)
            cur_evt = st.event_log[-1]
            # vectorized path length computation
            centers_arr = np.asarray(st.centers)
            if centers_arr.shape[0] >= 2:
                path_len = float(np.linalg.norm(np.diff(centers_arr, axis=0), axis=1).sum())
            else:
                path_len = 0.0
            cur_evt.update(
                {
                    "speed_px_s": float(speed_px_per_sec),
                    "active": bool(st.active),
                    "velocity_vec": v.tolist(),
                    "path_length": path_len,
                }
            )
            return None

        prev_lane = st.lane_idx
        st.lane_idx = self._choose_lane_with_hysteresis(C, prev_lane)
        st.lane_hist.append(st.lane_idx)

        if (
            len(st.lane_hist) < self.p.lane_stable
            or sum(1 for i in st.lane_hist if i == st.lane_idx) < self.p.lane_stable
        ):
            st.neg_count = 0
            if st.active:
                st.grace_left = max(st.grace_left, 1)
            st.cos_hist.append(0.0)
            cur_evt = st.event_log[-1]
            cur_evt.update(
                {
                    "speed_px_s": float(speed_px_per_sec),
                    "active": bool(st.active),
                    "lane_idx": st.lane_idx,
                    "velocity_vec": v.tolist(),
                    "path_length": float(
                        sum(
                            np.linalg.norm(st.centers[i] - st.centers[i - 1])
                            for i in range(1, len(st.centers))
                        )
                    ),
                }
            )
            return None

        lane = self.lanes[st.lane_idx]
        cos_sim_inst = lane.cosine_with(v)
        if cos_sim_inst is None:
            if st.active:
                st.grace_left = max(st.grace_left, 1)
            st.neg_count = 0
            st.cos_hist.append(0.0)
            cur_evt = st.event_log[-1]
            cur_evt.update(
                {
                    "speed_px_s": float(speed_px_per_sec),
                    "active": bool(st.active),
                    "lane_idx": st.lane_idx,
                    "velocity_vec": v.tolist(),
                    "path_length": float(
                        sum(
                            np.linalg.norm(st.centers[i] - st.centers[i - 1])
                            for i in range(1, len(st.centers))
                        )
                    ),
                }
            )
            return None

        st.cos_hist.append(float(cos_sim_inst))
        cos_window = list(st.cos_hist)[-self.p.avg_win :]
        cos_med = float(np.median(cos_window))
        alpha = 0.3
        prev_val = st.cos_hist[-2] if len(st.cos_hist) >= 2 else cos_med
        cos_avg = float(alpha * cos_med + (1 - alpha) * prev_val)

        if not st.active:
            if cos_avg < self.p.cos_on:
                st.neg_count += 1
                if st.neg_count >= self.p.consec_on:
                    st.active = True
                    st.event_start = frame_idx - self.p.consec_on + 1
                    st.grace_left = self.p.grace_off
            else:
                st.neg_count = max(0, st.neg_count - 1)
        else:
            if cos_avg > self.p.cos_off:
                st.grace_left = max(0, st.grace_left - 1)
                if st.grace_left == 0:
                    st.active = False
                    st.event_start = None
                    st.neg_count = 0
            else:
                st.grace_left = self.p.grace_off

        path_length = (
            float(np.linalg.norm(np.diff(np.asarray(st.centers), axis=0), axis=1).sum())
            if len(st.centers) >= 2
            else 0.0
        )

        cur_evt = st.event_log[-1]
        cur_evt.update(
            {
                "lane_idx": st.lane_idx,
                "cos_inst": float(cos_sim_inst),
                "cos_avg": float(cos_avg),
                "speed_px_s": float(speed_px_per_sec),
                "active": bool(st.active),
                "start_frame": st.event_start,
                "velocity_vec": v.tolist(),
                "path_length": path_length,
            }
        )

        return None

    def is_wrong_way(self, track_id, cx=None, cy=None):
        st = self.tracks.get(track_id)
        if st is None or len(st.event_log) == 0:
            return False, 0.0, {}

        last = st.event_log[-1]
        active = bool(st.active)
        cos_avg = last.get("cos_avg", last.get("cos_inst", 0.0))
        speed_px_s = last.get("speed_px_s", 0.0)

        if active:
            denom = max(1e-6, 1.0 - self.p.cos_on)
            raw = (self.p.cos_on - cos_avg) / denom
            conf = float(np.clip(raw, 0.0, 1.0))
            if cos_avg > self.p.cos_on and getattr(st, "grace_left", 0) > 0:
                conf *= min(1.0, st.grace_left / max(1, self.p.grace_off))
        else:
            conf = 0.0

        meta = {
            "lane_idx": st.lane_idx,
            "event_start": st.event_start,
            "grace_left": getattr(st, "grace_left", 0),
            "neg_count": getattr(st, "neg_count", 0),
            "cos_inst": last.get("cos_inst", None),
            "cos_avg": cos_avg,
            "speed_px_s": speed_px_s,
            "active": active,
        }

        return active, conf, meta

    def statistic_summary(self, lanes: list[LanePolygon], min_frame=10):
        lane_len = len(lanes)
        lane_counter = Counter()
        wrong_counter = Counter()
        is_wrong_fn = self.is_wrong_way
        for track_id, st in self.tracks.items():
            if st is None or len(st.event_log) == 0:
                continue
            last_event = st.event_log[-1]
            total_frames = last_event["end_frame"] - (last_event["tracking_start_frame"] or 0) + 1
            if total_frames < min_frame:
                continue
            if not st:
                continue
            lane_idx = st.lane_idx
            if lane_idx is None or lane_idx < 0 or lane_idx >= lane_len:
                continue
            lane_counter[lane_idx] += 1
            if is_wrong_fn(track_id)[0]:
                wrong_counter[lane_idx] += 1

        sum_of_passed_by_lane = [lane_counter.get(i, 0) for i in range(lane_len)]
        sum_of_wrong_by_lane = [wrong_counter.get(i, 0) for i in range(lane_len)]
        return sum_of_passed_by_lane, sum_of_wrong_by_lane
