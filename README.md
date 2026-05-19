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

Each script reports five stages per run. The dividing principle is **video
side vs text side**, with input-prep cost reported as its own dedicated
stage so the encoder + service work in `full_cycle` / `half_cycle` is
isolated from input materialization.

| Stage | Scope | Output | Includes input prep? | Includes text side? |
|---|---|---|---|---|
| `inference` | pure encoder model compute on this iter's preprocessed CUDA tensor | per-frame img emb | no | no |
| `half_cycle` | `_preprocess_stage` → … → latest video embedding | video emb `(B, 1024)` | no | **no** |
| `full_cycle` | `_preprocess_stage` → end-to-end production tick | alarm decision | no | yes |
| `input_gen_and_load` | isolated cost of producing B `(1080, 1920, 3)` uint8 ndarrays | B ndarrays | yes | no |
| `cos_sim` | isolated cos-sim **dot product** vs text features | sim tensors | no | yes |

`full_cycle`, `half_cycle`, `inference`, and `cos_sim` all share the
**same** `fresh_batches()` call within each measure iter — one shared
preprocess+model+mean run feeds the full/half snapshots (taken at the
half boundary and again after the text-side block), and `inference` /
`cos_sim` then re-time their respective sub-pieces on that run's
intermediate tensors (`x` for inference, `vis_vectors` / populated
`stream_vector_queues` for cos_sim). So per iter, the four GPU stages
operate on the same pixel data. The batches themselves come from a
**pre-generated input pool** — a one-time-allocated list of unique random
`(1080, 1920, 3)` uint8 ndarrays sized to cover every B-batch consumer
in the bench (warmup + buffer_warmup + 1 × measure_iters). Across iters,
no two iters see the same draw.

The rule of thumb: **the moment any text embedding (cos-sim vs text
features, top-K, alarm event manager) is involved, the timing is no longer
`half_cycle` — it's `full_cycle`.** Inputs are randomly generated, so there
is no disk I/O at any point in the bench.

**Pool sizing** (printed at the start of every run as `[pool] generated N
frames = K batches × B  (X GB)`):

```
PE        : n_batches =     warmup_iters +     measure_iters
FT_PE     : n_batches =     warmup_iters +     measure_iters + (TEMPORAL_SIZE + T + 2)
FT_PE_INT8: n_batches = 2 + warmup_iters + 2 × measure_iters + (TEMPORAL_SIZE + T + 2)
pool_bytes        = n_batches × B × (1080 × 1920 × 3) ≈ n_batches × B × 6.2 MB
```

(FT_PE_INT8 hasn't been moved onto the shared-batch construction yet, so it
still pulls two batches per measure iter — one for `full_cycle`, one for
`half_cycle` — plus two setup ticks.)

Typical: B=16, T=3, warmup=5, iters=25 → ~1 GB. At the upper end (B=128
on the dynamic FT_PE_INT8 engine) it climbs into the multi-GB range — the
pool-size banner lets you see the cost before activations land on GPU.

Subtraction games:

```
full_cycle  − half_cycle   ≈ cos_sim                    (text-side cost)
input_gen_and_load          = isolated input-prep cost  (random ndarray gen)
```

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

Step-by-step view of what each timed stage actually does. Rows are the
ordered work units; `✓` = included in that stage's timed region, `—` = not.

### PE (`src/speed_calculate_PE.py`)

```
┌──────────────────────────────────────────────────────────────────┬──────┬──────┬───────────┬───────────┬─────────┐
│                              Step                                │ full │ half │ inference │ input_gen │ cos_sim │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ gen_random_frame × B  (RNG → uint8 (1080,1920,3), fresh per call)│  —   │  —   │     —     │     ✓     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ fresh_batches()  (next B from pre-generated pool, .copy()-d)     │  ✓   │  ✓   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ x = preprocessed (B, 3, H, W) tensor reused from this iter's run │  —   │  —   │     ✓     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ _preprocess_stage  (cv_bgr2rgb + ROI + resize / normalize)       │  ✓   │  ✓   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ _inference_stage → TRT model → (B, 1024)                         │  ✓   │  ✓ ⤴ │    ✓ ⤴    │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ deque append per stream  (stream_vector_queues, maxlen=1)        │  ✓   │  —   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ mean-pool + cos-sim  (mean @ self.gpu_vectors.T)                 │  ✓   │  —   │     —     │     —     │  ✓ ⤴    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ _decide_top_category_opt  (TOP_CANDIDATE = 13 ranking)           │  ✓   │  —   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ process_category  (append 1/0 to duration_queue per retEvent)    │  ✓   │  —   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ check_alarm_duration  (STATUS_TRANSITION → alarms dict)          │  ✓   │  —   │     —     │     —     │    —    │
└──────────────────────────────────────────────────────────────────┴──────┴──────┴───────────┴───────────┴─────────┘
```

Where each stage stops (the `⤴` arrow above):
- `full_cycle` and `half_cycle` are measured from a **single** shared run
  per iter: timer snapshotted at the half boundary, the run continues into
  `_postprocess_stage`, timer snapshotted again for full. So per iter
  `full_cycle ≥ half_cycle` by construction.
- `half_cycle` returns the `(B, 1024)` visual vector straight out of
  `_inference_stage`. PE has `TEMPORAL_SIZE = 1`, so the per-image visual
  vector *is* the per-stream video embedding.
- `inference` re-times `_inference_stage` on the same `x` tensor that
  full/half just produced — so its input data is from the same
  `fresh_batches()` call, but the preprocess cost is NOT in inference's
  timed region.
- `cos_sim` reads `service.stream_vector_queues` (just populated by the
  shared full+half run earlier in this iter — and by warmup before iter 0)
  and times **only** the per-stream
  `mean-pool + (mean @ self.gpu_vectors.T)`. The downstream
  `_decide_top_category_opt` / `process_category` / `check_alarm_duration`
  are NOT in the timed call.
- `input_gen_and_load` runs only the random-frame generation, nothing else.

### FT_PE (`src/speed_calculate_FTPE.py`)

`--frames T` drives the temporal-window dimension. The bench forces
`stride = 1` (`sliding_window_size = T-1`, `prediction_size = 1`) so every
tick triggers an encode.

```
┌──────────────────────────────────────────────────────────────────┬──────┬──────┬───────────┬───────────┬─────────┐
│                              Step                                │ full │ half │ inference │ input_gen │ cos_sim │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ gen_random_frame × B  (RNG → uint8 (1080,1920,3), fresh per call)│  —   │  —   │     —     │     ✓     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ fresh_batches()  (next B from pre-generated pool, .copy()-d)     │  ✓   │  ✓   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ (B, T, 3, H, W) tensor: x.unsqueeze(1).expand(...).contiguous()  │  —   │  —   │     ✓     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ _preprocess_stage  (cv_bgr2rgb + ROI + resize / normalize)       │  ✓   │  ✓   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ gather_frame_buffers append + _unconsumed_frames++               │  ✓   │  ✓   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ window check + torch.stack(gather) → (B_enc, T, 3, H, W)         │  ✓   │  ✓   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ _inference_stage → TRT model → (B_enc, T, 1024) + L2 norm        │  ✓   │  ✓   │    ✓ ⤴    │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ frame_buffers extend + rotate (del [1] if > TEMPORAL_SIZE)       │  ✓   │  ✓   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ per-stream torch.stack(buf).mean(dim=0) → video_emb (1024,)      │  ✓   │  ✓   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ torch.stack video_emb → (B_ready, 1024)                          │  ✓   │  ✓ ⤴ │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ L2 norm of (B, 1024) video embeddings                            │  ✓   │  —   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ per-class cos-sim  (vis @ cat_txt[c] and cat_normal[c]).max(1)   │  ✓   │  —   │     —     │     —     │  ✓ ⤴    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ (sim_abn_max > sim_nrm_max).cpu().tolist() per category          │  ✓   │  —   │     —     │     —     │    —    │
├──────────────────────────────────────────────────────────────────┼──────┼──────┼───────────┼───────────┼─────────┤
│ alarm_event_manager.update  (duration_queue + status transition) │  ✓   │  —   │     —     │     —     │    —    │
└──────────────────────────────────────────────────────────────────┴──────┴──────┴───────────┴───────────┴─────────┘
```

Where each stage stops (the `⤴` arrow above):
- `full_cycle` and `half_cycle` are measured from a **single** shared run
  per iter: preprocess + gather/frame buffer + model + per-stream mean →
  snapshot for half → text-side block (L2 + cos-sim + alarm) → snapshot
  for full. So per iter `full_cycle ≥ half_cycle` by construction.
- `half_cycle` runs the full vision side via `_postprocess_to_video_emb`
  (mirror of `_postprocess_stage` truncated right after the per-stream
  mean-pool), and returns the stacked `(B, 1024)` video embeddings.
- `inference` re-times `_inference_stage` on a `(B, T, 3, H, W)` tensor
  built per-iter from the shared run's `x` via
  `x.unsqueeze(1).expand(-1, T, -1, -1, -1).contiguous()`. The
  `.contiguous()` copy runs outside `time_call`, so only the model call
  is timed — no preprocess, no gather buffer, no frame buffer, no
  mean-pool.
- `cos_sim` reads the same `(B, 1024)` `vis_vectors` tensor the shared
  run just produced, so it skips the entire vision side and times **only**
  the per-class `(vis @ cat_txt[c]).max(1)` /
  `(vis @ cat_normal[c]).max(1)` dot-product loop. The L2 norm of
  video_embeddings, the `> sim_normal` comparison + `.cpu().tolist()`
  sync, and `alarm_event_manager.update` are NOT in the timed call.
- `input_gen_and_load` runs only the random-frame generation, nothing else.

`full_cycle` and `half_cycle` are sampled from the same **one production
tick** — they share `service.gather_frame_buffers` and
`service.frame_buffers` state (primed once during warmup, advanced by one
tick per measure iter). By construction `half_cycle ≤ full_cycle` (the
only difference is the text-side block at the tail).

## Diff equations (subtraction games)

```
full        − half             ≈ cos_sim                (text-side cost)
input_gen_and_load              = isolated input-prep cost  (random ndarray gen)
```

By construction: `half_cycle ≤ full_cycle` **per iter** (they're sampled
from the same shared run — see `iterations.full_cycle_ms[i]` vs
`iterations.half_cycle_ms[i]` in the result JSON). The only differing
work is the text-side block; input prep is excluded from both.

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
`None` while the buffer fills). The warmup loop then runs `_detect`,
which internally calls `_inference_stage` on the same `(B_enc, T, ...)`
shape the measure loop hits — so every timed stage's hot path is warmed
by one common loop.

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
