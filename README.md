# pia-speed_compare

Latency benchmarks for the **PE** (`PE-Core-L14-336`) and **FT_PE**
(`FT_PE-Core-L14-336`) vision pipelines, run through the real production
service objects (`PEService`, `FTPEService`) — not synthetic forward-only
harnesses — so the measured numbers reflect what each tick of the production
loop actually costs.

```
scripts/speed_calculate_PE.sh     → src/speed_calculate_PE.py
scripts/speed_calculate_FTPE.sh   → src/speed_calculate_FTPE.py
scripts/aggregate_results.sh      → src/aggregate_results.py   (results/*.json → summary.txt)
scripts/plot_results.sh           → src/plot_results.py        (results/*.json → results/*.png)
```

## Stage convention

Each script reports four stages per run. The dividing principle is **video
side vs text side**, with `three_quarters_cycle` factoring out the disk-read
cost:

| Stage | Scope | Output | Disk read? | Text side? |
|---|---|---|---|---|
| `inference` | pure encoder model compute on a pre-cooked CUDA tensor | per-frame img emb | no | no |
| `half_cycle` | everything **up to the latest video embedding** | video emb `(B, 1024)` | yes | **no** |
| `three_quarters_cycle` | one production tick **minus disk read** | alarm decision | **no** | yes |
| `full_cycle` | one end-to-end production tick | alarm decision | yes | yes |

The rule of thumb: **the moment any text embedding (cos-sim vs text features,
top-K, alarm event manager) is involved, the timing is no longer `half_cycle`
— it's `full_cycle` / `three_quarters_cycle`.** This boundary is what makes
the PE and FT_PE half_cycle numbers semantically comparable.

`three_quarters_cycle` answers "how fast is one production tick when frames
are already in RAM" (e.g. an RTSP grabber hands you decoded ndarrays, no
local disk I/O). `full_cycle - three_quarters_cycle` is therefore the
isolated disk-read cost.

### What "video embedding" means in each pipeline

- **PE** is a per-image encoder. `TEMPORAL_SIZE = 1`, so the per-stream
  video embedding *is* the per-image visual vector. `half_cycle` =
  preprocess + model → `(B, 1024)`.
- **FT_PE** is a temporal encoder. The TRT engine
  (`..._vision_no_mean_pooling.engine`) outputs `(B, T, 1024)` — one
  embedding per frame in the window, with mean-pooling intentionally
  stripped. The per-stream video embedding is the mean over a
  `TEMPORAL_SIZE = 8` buffer of these per-frame embeddings. `half_cycle` =
  preprocess + temporal model + per-stream `torch.stack(buf).mean(dim=0)` →
  `(B, 1024)`.

Concretely, `_postprocess_stage` does two things in sequence per tick:

```
(1) video side : append latest embeddings to per-stream buffer
                 → torch.stack(buf).mean(dim=0)
                 → L2 normalize
                 ──────────────────────────────────────  ← half_cycle stops here
(2) text side  : cos-sim vs per-class text features
                 → predict / duration queue
                 → alarm event manager
                 ──────────────────────────────────────  ← full_cycle stops here
```

`half_cycle` for both PE and FT_PE corresponds to step (1) only.
`full_cycle` covers (1) + (2).

## Stage detail

### PE (`src/speed_calculate_PE.py`)

```
full_cycle           : disk read     → cv_bgr2rgb + ROI + preprocess_image
                                     → TRT model        → (B, 1024)
                                     → deque append → cos-sim vs text → top-K → alarm
three_quarters_cycle : in-mem ndarray → cv_bgr2rgb + ROI + preprocess_image
                                     → TRT model        → (B, 1024)
                                     → deque append → cos-sim vs text → top-K → alarm
half_cycle           : disk read     → cv_bgr2rgb + ROI + preprocess_image
                                     → TRT model        → (B, 1024)                  ← stops here
inference            : pre-cooked (B, 3, H, W) tensor → TRT model → (B, 1024)
```

### FT_PE (`src/speed_calculate_FTPE.py`)

`--frames T` drives the temporal-window dimension. The bench forces
`stride = 1` (sliding_window_size = T-1, prediction_size = 1) so every tick
triggers an encode.

```
full_cycle           : disk read     → _preprocess_stage(B)
                                     → gather buffer (window_size=T)
                                     → TRT model           → (B, T, 1024)
                                     → frame_buffer (TEMPORAL_SIZE=8)
                                     → mean-pool           → (B, 1024)
                                     → L2 norm → per-class cos-sim → alarm
three_quarters_cycle : in-mem ndarray → _preprocess_stage(B)
                                     → gather buffer
                                     → TRT model           → (B, T, 1024)
                                     → frame_buffer
                                     → mean-pool           → (B, 1024)
                                     → L2 norm → per-class cos-sim → alarm
half_cycle           : disk read     → _preprocess_stage(B)
                                     → gather buffer
                                     → TRT model           → (B, T, 1024)
                                     → frame_buffer
                                     → mean-pool           → (B, 1024)         ← stops here
inference            : pre-cooked (B, T, 3, H, W) tensor → TRT model → (B, T, 1024)
```

`full_cycle`, `three_quarters_cycle`, and `half_cycle` are all **one
production tick** — they share the same `service.gather_frame_buffers` and
`service.frame_buffers` state, which is primed once during warmup and
advanced by one tick per timed call. The differences between the three
stages are only the disk-read step and/or the text-side block at the end,
so `half_cycle <= full_cycle` and `three_quarters_cycle <= full_cycle` hold
**by construction**.
The earlier "synthetic B×T bulk preprocess" definition was scrapped: it
inflated half_cycle's preprocess work T× relative to a real tick and produced
the nonsensical `half > full` ordering at `T > 1`.

## Throughput unit

All three stages report `*_imgs_per_s` in the same unit: **frames encoded
per second** = `B * T * 1000 / mean_ms`.

- For PE there's no T dimension (T = 1 implicit), so this equals
  streams per second.
- For FT_PE at `stride = 1`, every tick re-encodes a full `(B, T)` window
  through the temporal model, so the work unit per tick is `B * T` frames
  — same as half_cycle / inference. The numbers are directly comparable.

The **production input rate** (unique frames ingested per second) is
reported separately as `throughput.full_cycle_streams_per_s` and
`throughput.half_cycle_streams_per_s` (= `B * 1000 / mean_ms`) for FT_PE
only. PE has no separate streams-per-second key because it equals
`*_imgs_per_s`.

## ROI / retEvent shape

Both `make_user_params` helpers emit `retEvent` as a **dict** keyed by
category id with empty `polygonCoordinates`, matching what
`AddStreamModel2dict` produces in production:

```python
retEvent = {
    "fire_ret":     {"roi": {"polygonCoordinates": []}},
    "falldown_ret": {"roi": {"polygonCoordinates": []}},
    "smoke_ret":    {"roi": {"polygonCoordinates": []}},
}
```

This exercises the dict-lookup path in both `PERoIManager` and
`FTPERoIManager` (the list-of-strings shape that earlier revisions used
side-stepped that path and skipped per-category processing). Empty
`polygonCoordinates` falls back to a whole-frame ROI, and all three event
categories (fire / falldown / smoke) are evaluated by the alarm event
manager per tick.

## Warmup

PE warms up the full `_preprocess + _inference + _postprocess` chain so the
first measured `full_cycle` iter doesn't pay the alarm-path cold-cache hit.

FT_PE additionally primes `FTPEService`'s temporal buffer for
`TEMPORAL_SIZE + window_size + 2` ticks so every measured `full_cycle` tick
actually produces a video embedding (otherwise the first ~9 ticks return
`None` while the buffer fills). The warmup loop then covers all three timed
paths (`_detect`, bulk `_preprocess_stage`, inference + stack+mean).

## A note on the existing results

The JSONs currently under `results/` were produced before:

- both scripts switched to the dict-shape `retEvent` (now 3 categories
  evaluated, not 1);
- FT_PE `half_cycle` was extended through the mean-pool;
- FT_PE throughput unit was unified.

So they're not directly comparable to runs from the current code. Re-run
both scripts before drawing PE-vs-FT_PE comparisons.

GPU thermal throttling on the A4000 also dominates tail-end iters in the
existing runs (560 ms head → 1600 ms tail as the GPU climbs 74 °C → 97 °C).
`min_ms` is the closest proxy to unthrottled speed; means/p95s are pulled
down by the throttle.
