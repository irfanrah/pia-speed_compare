# `perception_encoder` vs `ft_pe` — Algorithm & Design Comparison

This document explains, in detail, how the two PE-family detection services in
`src/Product-AI-mono/packages/pia_prod/AI/modules/` differ:

- [`perception_encoder/`](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/) (referred to as **PE** below)
- [`ft_pe/`](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/) (referred to as **FT_PE** below)

Both services share the same upstream goal — turn a stream of frames into
per-stream, per-category alarms by comparing visual embeddings against
text embeddings from the PE-Core-L14-336 family. The differences are in
**temporal aggregation**, **prompt-vector layout**, and **how the final
class decision is made**.

---

## 1. High-level summary

| Aspect | `perception_encoder` (PE) | `ft_pe` (FT_PE) |
| --- | --- | --- |
| Model checkpoint | `PE-Core-L14-336` (vanilla) | `FT_PE-Core-L14-336_*` (fine-tuned) |
| TRT engine input | `(B, 3, 336, 336)` | `(B, W, 3, 336, 336)` — temporal dim |
| TRT engine output | `[B, 1024]` (per-image embedding) | `[B, W, 1024]` (per-frame embedding) |
| Per-tick model call | Every frame, every stream | Only when a stream's sliding window fills |
| Temporal aggregation | `TEMPORAL_SIZE = 1` (none) | `TEMPORAL_SIZE = 8` mean-pooled frame buffer |
| Sliding window | n/a | `WINDOW_SIZE` / `SLIDING_WINDOW_SIZE` / `PREDICTION_SIZE` driven by `FT_PE_MODE` (`3fps` or `8fps`) |
| Prompt store layout | Flat JSON: list of `{ID, class, prompt, feature}` | Nested JSON: `{category: {text_features: {normal: [...], <category>: [...]}}}` |
| Normal-class pool | One row in the flat list (id=`normal`) | **Per-class** normal pool (no union) |
| Decision rule | Top-K candidate voting across all prompts, dedupe by ID, `bincount` → winning class | Per-category Top-1 binary: `max(sim_abnormal_c) > max(sim_normal_c)` |
| Categories | `normal / falldown / fire / smoke / smoking` | `violence / falldown / fire / smoke` |
| ROI gating | Applies only to the falldown category | Applies to **any** registered abnormal category (first match wins) |
| Event smoothing | `QUEUE_SIZE` (10) / `ALARM_DURATION_THRESHOLD` (5) | `ALARM_QUEUE_SIZE` (3) / `ALARM_THRESHOLD` (2) |
| Alarm key format | `stream_id` | `f"{stream_id}__{category_id}"` (composite key prevents multi-category overwrite) |

---

## 2. Input pipeline

Both services inherit from `ServiceBase` and expose the same `_detect()` flow:

```
batches ─► _preprocess_stage ─► _inference_stage ─► _postprocess_stage ─► alarms
```

### 2.1 Preprocessing (almost identical)

Both use [`perception_encoder/trt_utils.py:preprocess_image`](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/trt_utils.py)
to resize to 336×336, normalize with `mean = std = [0.5, 0.5, 0.5]`, cast to fp32,
and move to CUDA. FT_PE imports the same function rather than duplicating it
([service.py:33](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/service.py#L33)).

The single difference is in **ROI selection**:

- **PE** ([roi_manager.py:32](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/roi_manager.py#L32))
  hard-codes `roi_category_list = FALLDOWN_CATEGORY` — ROI cropping
  only activates when the user requests a falldown event.
- **FT_PE** ([roi_manager.py:28](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/roi_manager.py#L28))
  uses `roi_category_list = ALL_CATEGORIES` — the first abnormal
  ret_event in the user's payload supplies the ROI.

### 2.2 Inference (the key structural difference)

**PE — stateless per-frame**
[service.py:89](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/service.py#L89)

```python
def _inference_stage(self, image_cuda):
    return self.model(image_cuda)         # [B, 1024]
```

Every tick produces one embedding per stream. The TRT engine is built with
a 2D channel layout `(B, 3, 336, 336)` (see
[trt_export.py:51](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/trt_export.py#L51)).
A per-stream `deque(maxlen=TEMPORAL_SIZE=1)` is kept but it really only holds
the latest embedding, plus a `zero_mask_vec` warm-up entry
([service.py:46-53](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/service.py#L46-L53)).

**FT_PE — temporal sliding window**
[service.py:165-191](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/service.py#L165-L191)

Two queues per stream:

1. `gather_frame_buffers[sid]` — `deque(maxlen=WINDOW_SIZE)` of preprocessed
   frames waiting to fill a window.
2. `frame_buffers[sid]` — `deque(maxlen=TEMPORAL_SIZE+1)` of per-frame
   embeddings produced by the model.

A window is encoded only when **both** conditions hold:

```python
len(gbuf) == WINDOW_SIZE and unconsumed_frames[sid] >= stride
```

The TRT engine output `[B_enc, W, 1024]` is sliced to the latest
`PREDICTION_SIZE` embeddings per stream, then appended to `frame_buffers`.
If the embedding buffer grows past `TEMPORAL_SIZE`, **index 1** is deleted
(not index 0) so the oldest frame is preserved for longer temporal context
([service.py:189-191](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/service.py#L189-L191)).

Mode presets (`FT_PE_MODE` env var):

| Mode | WINDOW_SIZE | SLIDING_WINDOW_SIZE | PREDICTION_SIZE | TIME_INTERVAL |
| --- | --- | --- | --- | --- |
| `3fps` | 3 | 2 | 1 | 0.333 s |
| `8fps` | 1 | 0 | 1 | 0.125 s |

The `8fps` mode degenerates to a per-frame encode (no overlap) but still keeps
the 8-frame embedding buffer for temporal smoothing — i.e. it still gets the
benefit of mean-pooled video embeddings even without sliding.

### 2.3 Text features (prompt embeddings)

**PE** — flat list. Each row is one prompt with its precomputed feature
vector and a categorical `ID`/`class` tag
([prompts.py:6-58](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/prompts.py#L6-L58)):

```json
[ {"ID": <id>, "class": <int>, "prompt": "...", "feature": [..1024..]}, ... ]
```

Loaded once at boot into a single `[N_prompts, 1024]` matrix.

**FT_PE** — per-category nested dict
([service.py:91-133](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/service.py#L91-L133)):

```json
{
  "fire":   { "text_features": { "normal": [[..1024..], ...],
                                 "fire":   [[..1024..], ...] } },
  "smoke":  { "text_features": { "normal": [...], "smoke":  [...] } },
  ...
}
```

The comment at [service.py:104-109](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/service.py#L104-L109)
calls out *why* normals are kept per-class instead of unioned: a normal
sample under `smoke` was visually close to fire content and swallowed the
fire margin when normals were merged. The fix is to compare each class
against **its own** normal pool only.

Each per-class matrix is stored transposed (`[1024, N]`) and L2-normalized
on load so detection is a single matmul.

---

## 3. Decision rules

### 3.1 PE — Top-K candidate voting

[event.py:72-116](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/event.py#L72-L116)

```
sim                = vis_vec @ all_prompts.T              # [N_prompts]
sorted_sim, idx    = topk(sim, k = TOP_CANDIDATE * 10)
# Walk sorted candidates, keep `normal` always, dedupe other classes by ID,
# stop once TOP_CANDIDATE survivors collected.
counts             = bincount(surviving_classes)
winners            = argmax(counts)                       # may be ties
```

So PE's class decision is a *plurality vote* over the strongest matching
prompts, after deduping by prompt-group ID. `normal` (id=0) is treated
specially — it is allowed to appear multiple times in the survivor list,
which biases the vote toward normal when nothing stands out.

The visual vector is the mean of `vis_vectors[stream_id]` (length-1 deque,
so effectively the latest frame).

### 3.2 FT_PE — per-category Top-1 abnormal vs. normal

[service.py:208-225](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/service.py#L208-L225)

```
vid_emb        = mean(frame_buffer[-TEMPORAL_SIZE:])      # [1024]
vid_emb        = L2_norm(vid_emb)

for c in ABNORMAL_CLASS_NAMES:                            # violence, falldown, fire, smoke
    s_abn_max  = (vid_emb @ A_c).max()                    # A_c = [1024, N_c]
    s_nrm_max  = (vid_emb @ N_c).max()                    # N_c = [1024, M_c] (per-class)
    pred[c]    = (s_abn_max > s_nrm_max)                  # bool — independent per class
```

Each category is judged independently — multiple categories can fire on the
same frame. There is no `argmax` across categories; there is no `normal`
class in the output, just a boolean per abnormal class.

### 3.3 Event smoothing (similar shapes, different sizes)

Both event managers inherit from `EventBase` and use the same
`STATUS_TRANSITION` finite-state machine (no_event / start / continue / end).
Both maintain a per-stream × per-category 0/1 ring buffer and trigger when
`sum(queue) >= threshold`.

The differences:

| | PE | FT_PE |
| --- | --- | --- |
| Queue size | `QUEUE_SIZE = 10` ([config.py:37](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/config.py#L37)) | `ALARM_QUEUE_SIZE = 3` ([config.py:76](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/config.py#L76)) |
| Trigger count | `ALARM_DURATION_THRESHOLD = 5` | `ALARM_THRESHOLD = 2` |
| Alarm key | `alarms[stream_id]` | `alarms[f"{stream_id}__{category_id}"]` |

The composite alarm key in FT_PE
([event.py:58](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/event.py#L58)) is
required because FT_PE can raise multiple categories per stream in the same
tick — a single `stream_id` key would let the second category overwrite the
first.

---

## 4. Generalized pseudocode (covers both PE and FT_PE)

The two services collapse into one parametric algorithm. The PE pipeline is
recovered by setting `WINDOW_SIZE = 1`, `TEMPORAL_SIZE = 1`,
`PREDICTION_SIZE = 1`, `SLIDING_WINDOW_SIZE = 0`, and switching the decision
rule to *topk-vote*.

```text
# --- Configuration -----------------------------------------------------------
# Common
DEVICE                : "cuda"
IMG_SIZE              : (336, 336)
TEMPORAL_SIZE         : int        # 1  for PE,  8  for FT_PE
ALARM_QUEUE_SIZE      : int        # 10 for PE,  3  for FT_PE
ALARM_THRESHOLD       : int        # 5  for PE,  2  for FT_PE

# Temporal-encoder only (PE uses defaults that disable it)
WINDOW_SIZE           : int        # 1   for PE, {1,3} for FT_PE
SLIDING_WINDOW_SIZE   : int        # 0   for PE, {0,2} for FT_PE
PREDICTION_SIZE       : int        # 1   for PE, 1     for FT_PE
STRIDE                = max(1, WINDOW_SIZE - SLIDING_WINDOW_SIZE)

DECISION_MODE         : "topk_vote" | "per_class_top1"


# --- State (per service instance) -------------------------------------------
gather_buf[sid]       : deque(maxlen=WINDOW_SIZE)     of preprocessed frames
embed_buf[sid]        : deque(maxlen=TEMPORAL_SIZE+1) of per-frame embeddings
unconsumed[sid]       : int = 0                       # frames not yet covered by a window
duration_queue[sid][cat] : deque(maxlen=ALARM_QUEUE_SIZE)  of 0/1
event_status[sid][cat]   : int (FSM state)


# --- Text-feature loading ---------------------------------------------------
# DECISION_MODE = "topk_vote":
#   text_db = (ids, class_ids, prompts, features)        # flat
#   features : [N_prompts, 1024]  L2-normalized
#
# DECISION_MODE = "per_class_top1":
#   abn[c]    : [1024, N_c]  L2-normalized   # per-class abnormal prompts
#   nrm[c]    : [1024, M_c]  L2-normalized   # per-class normal prompts


# --- Per-tick: detect(batches, stream_ids, user_params) ---------------------
def detect(batches, stream_ids, user_params):

    # 1. Preprocess
    frames_chw = preprocess(batches, user_params)        # ROI crop + resize + norm
                                                         # -> [B, 3, 336, 336]

    # 2. Gather → encode (temporal window)
    encode_sids, encode_inputs = [], []
    for sid, frame in zip(stream_ids, frames_chw):
        gather_buf[sid].append(frame)
        unconsumed[sid] += 1
        if len(gather_buf[sid]) == WINDOW_SIZE and unconsumed[sid] >= STRIDE:
            encode_inputs.append(stack(gather_buf[sid]))   # [W, C, H, W]
            unconsumed[sid] -= STRIDE
            encode_sids.append(sid)

    if encode_sids:
        if WINDOW_SIZE == 1:
            # PE-style: model takes [B, C, H, W] and returns [B, 1024]
            embs = model(squeeze_W(stack(encode_inputs)))      # [B_enc, 1024]
            embs = unsqueeze_W(embs)                           # [B_enc, 1, 1024]
        else:
            # FT_PE-style: model takes [B, W, C, H, W] and returns [B, W, 1024]
            embs = model(stack(encode_inputs))                 # [B_enc, W, 1024]
        embs = L2_norm(embs, axis=-1)

        for sid, emb in zip(encode_sids, embs):
            for v in emb[-PREDICTION_SIZE:]:                   # keep last P embeddings
                embed_buf[sid].append(v)
            if len(embed_buf[sid]) > TEMPORAL_SIZE:
                del embed_buf[sid][1]    # keep oldest for long context (FT_PE choice)

    # 3. Build per-stream video embedding (only for streams whose buffer is full)
    ready_sids, video_embs = [], []
    for sid in stream_ids:
        if len(embed_buf[sid]) >= TEMPORAL_SIZE:
            v = mean(stack(embed_buf[sid]))                   # [1024]
            video_embs.append(L2_norm(v))
            ready_sids.append(sid)
    if not ready_sids:
        return None

    V = stack(video_embs)                                     # [B_ready, 1024]

    # 4. Decision rule
    if DECISION_MODE == "topk_vote":
        # PE: dedupe-by-ID Top-K vote across a flat prompt pool
        preds_per_stream = []
        for v in V:
            sim = v @ features.T                              # [N_prompts]
            cand_scores, cand_idx = topk(sim, k=TOP_CANDIDATE * 10)
            survivors = []
            seen_ids = set()
            for i in cand_idx:
                cid = ids[i]
                if cid == 0 or cid not in seen_ids:           # normal always; abnormal dedup
                    seen_ids.add(cid)
                    survivors.append(class_ids[i])
                if len(survivors) >= TOP_CANDIDATE: break
            counts  = bincount(survivors, minlength=num_classes)
            winners = where(counts == counts.max())           # set of class ids
            preds_per_stream.append(winners)

    elif DECISION_MODE == "per_class_top1":
        # FT_PE: per-category binary, max-abn > max-own-normal
        preds_per_stream = []
        for v in V:
            pred = {}
            for c in ABNORMAL_CLASSES:
                s_abn = (v @ abn[c]).max()
                s_nrm = (v @ nrm[c]).max()
                pred[c] = bool(s_abn > s_nrm)
            preds_per_stream.append(pred)

    # 5. Smoothing + FSM (identical shape in both services)
    alarms = {}
    for sid, pred, up in zip(ready_sids, preds_per_stream, user_params):
        for ret_event_key in up.requested_events():
            cat = map_ret_event_to_class(ret_event_key)
            triggered = is_triggered(pred, cat)               # see (*) below
            duration_queue[sid][ret_event_key].append(int(triggered))

            before = event_status[sid][ret_event_key]
            over   = int(sum(duration_queue[sid][ret_event_key]) >= ALARM_THRESHOLD)
            after  = STATUS_TRANSITION[before][over]
            event_status[sid][ret_event_key] = after

            if after in (STARTED, ENDED):
                alarms[alarm_key(sid, ret_event_key)] = (after, ret_event_key)

    return alarms or None


# (*) is_triggered:
#   topk_vote        : ret_event_key’s class id ∈ winners
#   per_class_top1   : pred[map_ret_event_to_class(ret_event_key)] is True
#
# alarm_key:
#   PE   : sid                       (single category per stream)
#   FT_PE: f"{sid}__{ret_event_key}" (multi-category capable)
```

### 4.1 Parameter cheat sheet — recovering each service from the generic algorithm

| Parameter | PE | FT_PE (`3fps`) | FT_PE (`8fps`) |
| --- | --- | --- | --- |
| `WINDOW_SIZE` | 1 | 3 | 1 |
| `SLIDING_WINDOW_SIZE` | 0 | 2 | 0 |
| `STRIDE` | 1 | 1 | 1 |
| `PREDICTION_SIZE` | 1 | 1 | 1 |
| `TEMPORAL_SIZE` | 1 | 8 | 8 |
| `ALARM_QUEUE_SIZE` | 10 | 3 | 3 |
| `ALARM_THRESHOLD` | 5 | 2 | 2 |
| `DECISION_MODE` | `topk_vote` | `per_class_top1` | `per_class_top1` |
| Model output shape | `[B, 1024]` | `[B, 3, 1024]` | `[B, 1, 1024]` |
| Normal pool layout | unified flat (`class=0`) | per-class | per-class |
| Alarm key | `sid` | `sid__cat` | `sid__cat` |

---

## 5. Practical consequences

1. **Latency vs. accuracy.** PE pays one model call per frame per stream and
   decides instantly; FT_PE only calls the model every `STRIDE` frames per
   stream (cheaper), but holds back alarms until the embedding buffer fills
   (≈ `TEMPORAL_SIZE × TIME_INTERVAL` of warm-up per stream).

2. **Failure modes.** PE's `bincount` vote is robust when prompts of the same
   class outnumber distractors but is sensitive to prompt-set imbalance — if
   one class has many more prompts than another, it can win on prompt count
   alone. FT_PE sidesteps this by comparing only inside a category, but is
   sensitive to the per-class normal pool: a noisy normal embedding will
   suppress that class.

3. **Multi-category alarms.** PE picks one winner per tick (an `argmax`
   across classes via `bincount`). FT_PE produces an *independent* boolean
   per class, so a single stream can simultaneously alarm on, e.g., `fire`
   and `smoke`. The composite alarm key
   ([event.py:58](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/event.py#L58))
   exists specifically to keep those alarms separate downstream.

4. **Model export.** PE's TRT engine has a 4D input
   ([trt_export.py](../src/Product-AI-mono/packages/pia_prod/AI/modules/perception_encoder/trt_export.py));
   FT_PE's has a 5D input with an additional temporal dim
   ([trt_export.py](../src/Product-AI-mono/packages/pia_prod/AI/modules/ft_pe/trt_export.py)).
   They are not interchangeable engines even though they share a base
   architecture.
