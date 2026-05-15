from statistics import median


def _percentile(values, p):
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


def _make_latency_fn(lat_map: dict) -> "callable[[int], float]":
    pts = sorted(lat_map.keys())

    def lat_ms(b: int) -> float:
        if b <= 0:
            return 0.0
        if b in lat_map:
            return float(lat_map[b])
        if b <= pts[0]:
            return lat_map[pts[0]] * b / pts[0]
        if b >= pts[-1]:
            return lat_map[pts[-1]] * b / pts[-1]
        lo = max(p for p in pts if p <= b)
        hi = min(p for p in pts if p >= b)
        if lo == hi:
            return float(lat_map[lo])
        frac = (b - lo) / (hi - lo)
        return lat_map[lo] + frac * (lat_map[hi] - lat_map[lo])

    return lat_ms


def analytic_stream_sim(
    lat_map: dict,
    num_channels: int,
    channel_fps: float,
    duration_s: float = 30.0,
    max_batch: int = 32,
    budget_ms: float = 500.0,
    cap_batch_to_budget: bool = True,
) -> dict:
    """Discrete-event simulation of a realtime CCTV scenario.

    Each of `num_channels` channels emits a frame every 1/channel_fps seconds.
    A single consumer always drains the queue into a batch (up to max_batch)
    and runs one inference; the per-batch latency is taken from `lat_map`
    (measured by the sync sweep), linearly interpolated for batch sizes that
    weren't measured. We then compute the wall-clock latency from emit to
    inference completion for every emitted frame.

    When `cap_batch_to_budget=True`, the consumer caps batch size at the
    largest size whose latency fits the budget — this models a real scheduler
    that would not pick a batch known to blow the 500 ms deadline.
    """
    if num_channels < 1 or channel_fps <= 0:
        return {"error": "invalid arguments"}

    lat_ms = _make_latency_fn(lat_map)
    if cap_batch_to_budget:
        cap = best_sustainable_capacity(lat_map, budget_ms=budget_ms)
        effective_batch = min(max_batch, cap["best_batch"])
    else:
        effective_batch = max_batch
    period = 1.0 / channel_fps

    queue: list[float] = []
    latencies: list[float] = []
    t = 0.0
    next_emit_t = 0.0

    while t < duration_s or queue:
        while next_emit_t <= t and next_emit_t < duration_s:
            for _ in range(num_channels):
                queue.append(next_emit_t)
            next_emit_t += period

        if not queue:
            t = max(t, next_emit_t)
            continue

        b = min(len(queue), effective_batch)
        done_t = t + lat_ms(b) / 1000.0
        for _ in range(b):
            emit_t = queue.pop(0)
            latencies.append((done_t - emit_t) * 1000.0)
        t = done_t

        if t > duration_s + 60.0:
            break

    expected = int(num_channels * channel_fps * duration_s)
    if not latencies:
        return {
            "num_channels": num_channels,
            "channel_fps": channel_fps,
            "aggregate_target_fps": round(num_channels * channel_fps, 3),
            "duration_s": duration_s,
            "frames_processed": 0,
            "expected_frames_approx": expected,
            "backlog_remaining": len(queue),
            "throughput_fps": 0.0,
            "meets_realtime_500ms": False,
        }

    throughput = len(latencies) / max(t, duration_s)
    return {
        "num_channels": num_channels,
        "channel_fps": channel_fps,
        "aggregate_target_fps": round(num_channels * channel_fps, 3),
        "effective_batch": effective_batch,
        "duration_s": round(t, 3),
        "frames_processed": len(latencies),
        "expected_frames_approx": expected,
        "backlog_remaining": len(queue),
        "throughput_fps": round(throughput, 2),
        "latency_p50_ms": round(median(latencies), 2),
        "latency_p95_ms": round(_percentile(latencies, 95), 2),
        "latency_p99_ms": round(_percentile(latencies, 99), 2),
        "latency_max_ms": round(max(latencies), 2),
        "meets_realtime_500ms": bool(
            _percentile(latencies, 95) <= budget_ms
            and len(queue) <= num_channels
        ),
    }


def best_sustainable_capacity(lat_map: dict, budget_ms: float = 500.0) -> dict:
    """Across measured batch sizes, find the configuration with the highest
    aggregate throughput whose per-batch latency fits inside `budget_ms`.
    Returns the chosen batch + the implied max aggregate fps (= channels × fps).
    """
    candidates = []
    for b, ms in sorted(lat_map.items()):
        fps = (b * 1000.0) / ms
        candidates.append((b, ms, fps, ms <= budget_ms))
    fits = [c for c in candidates if c[3]]
    if not fits:
        b, ms, fps, _ = min(candidates, key=lambda c: c[1])
        return {
            "best_batch": b,
            "best_batch_latency_ms": round(ms, 2),
            "max_sustainable_fps": round(fps, 2),
            "note": "no batch size fits the budget; smallest-latency option shown",
        }
    b, ms, fps, _ = max(fits, key=lambda c: c[2])
    return {
        "best_batch": b,
        "best_batch_latency_ms": round(ms, 2),
        "max_sustainable_fps": round(fps, 2),
        "headroom_ms": round(budget_ms - ms, 2),
    }
