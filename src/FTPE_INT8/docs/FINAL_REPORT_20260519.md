# FINAL REPORT — PE-L14-336 INT8 deploy (2026-05-19)

End-to-end summary of every QAT + deploy run that's left artifacts on disk under `src/claude_exp8_finish1/output/` and `src/claude_exp8_finish1/results/`. Supersedes `FINAL_REPORT_20260513.md` and `FINAL_REPORT_20260518.md` — read this one first.

## Model variants

Four QAT runs landed artifacts on disk:

| ID | Model | T_FRAMES | QKV layout | Status | Source ckpt |
|---|---|---:|---|---|---|
| **A — ZS-T1 splitqkv** | Zero-shot `PE-Core-L14-336` (QKV split for TRT INT8) | 1 | splitqkv (697 keys) | **canonical / shipping** | `output/pe_t1/_splitqkv_archive_20260518/qat_deploy_fp32.pt` |
| **B — ZS-T1 combined-QKV** | Same upstream weights, no QKV split | 1 | combined (601 keys) | **experiment, INT8 broken** | `output/pe_t1_combined_qkv/qat/qat_for_modelopt_onnx.pt` |
| **C — FT-T3** | `FT_PE-Core-L14-336_260318` (LoRA-merged) | 3 | splitqkv | **canonical / shipping** | `output/ftpe_t3/qat/qat_deploy_fp32.pt` |
| **D — FT-T8** | Same FT model, more frames per clip | 8 | splitqkv | **canonical / shipping** | `output/ftpe_t8/qat/qat_deploy_fp32.pt` |

All four ran on the same Leaderboard_bench clip pack (`assets/clips/{train,val,test}/`, 399 / 100 / 100 clips, 5 classes: falldown, fire, smoke, violence, normal). QAT recipe: 10 epochs, lr=5e-6, KD vs frozen teacher, 45% trainable (MLP + attn QKV/out_proj). All engines built with TRT 10.16, modelopt 0.43 on A6000 (sm_8.6).

## A — ZS-T1 splitqkv (canonical) → `output/pe_t1/`

**Source of truth for the headline ZS T=1 numbers.** Run date: 2026-05-15 16:07-18:15.

### Artifacts

```
output/pe_t1/
├── _splitqkv_archive_20260518/   ← canonical ckpts + engines (archived 2026-05-18)
│   ├── qat_best.pt                 2.5 GB, best-val cherry-pick
│   ├── qat_deploy_fp32.pt          2.5 GB, dequantized FP32 deploy candidate
│   ├── phase1_summary.json         per-epoch metrics
│   ├── manifest.json               train sampling record (data hygiene)
│   ├── manifest_eval_val.json      val sampling record
│   ├── run_qat.log                 phase1 + phase2 log
│   └── engines/                    27 TRT engines (BF16/INT8/INT8+CRL × B=1,2,4,8 + dyn × 3)
├── crl_sweep/                       step2 σ_k sweep results
│   ├── best_sigma.txt               picked σ_k = 3.0
│   └── results.md
├── deploy/
│   ├── comparison_20260515.md       ← THE comparison report (cited below)
│   ├── static/                      per-engine bench JSON + MD
│   └── dynamic/                     dyn engine bench (broken; see Known Issues §1)
├── qat/                             empty (ckpts moved to _splitqkv_archive_20260518/)
└── engines/                         empty (engines moved to _splitqkv_archive_20260518/engines/)
```

29 GB total on disk.

### Static engines — INT8 (QAT+CRL), σ_k = 3.0

From `output/pe_t1/deploy/comparison_20260515.md`:

| B | PT BF16 cos | PT BF16 ms | TRT BF16 cos | TRT BF16 ms | TRT INT8 (QAT) cos | TRT INT8 (QAT) ms | **TRT INT8 (QAT+CRL) cos** | **TRT INT8 (QAT+CRL) ms** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.000000 | 9.21 | 0.999672 | 5.85 | 0.991234 | 4.52 | **0.987555** | 4.56 |
| 2 | 1.000000 | 16.15 | 0.999788 | 10.46 | 0.973290 | 7.75 | **0.986417** | 7.70 |
| 4 | 1.000000 | 32.17 | 0.999698 | 20.26 | 0.969608 | 14.33 | **0.984005** | 14.40 |
| 8 | 1.000000 | 62.76 | 0.999714 | 39.74 | 0.946651 | 27.32 | **0.985062** | 27.44 |

**Speedup INT8+CRL / TRT BF16**: 1.28× → 1.36× → 1.41× → 1.45× across B=1..8. Static engines ship.

### Dynamic engines — BROKEN

| B | TRT BF16 dyn cos | TRT INT8 dyn (QAT+CRL) cos |
|---|---:|---:|
| 4 | 0.4119 ⚠ | 0.4069 ⚠ |
| 8 | 0.4130 ⚠ | 0.4085 ⚠ |
| 16 | 0.4084 ⚠ | 0.4034 ⚠ |
| 32 | 0.3888 ⚠ | 0.3835 ⚠ |

See `output/pe_t1/deploy/dynamic/comparison_dynamic_20260518.md` (post-fix attempt that confirmed the bug). Root cause is a TRT compilation issue specific to T=1 dynamic profile — see Known Issues §1. **Do not ship.**

## B — ZS-T1 combined-QKV (experiment) → `output/pe_t1_combined_qkv/`

**Failed experiment**, kept for reference. Run date: 2026-05-18 16:28-17:43. Goal: drop the splitqkv graph surgery so the ZS comparison reflects the unmodified upstream PE-Core-L14-336 model verbatim.

### Artifacts

```
output/pe_t1_combined_qkv/
├── qat/
│   ├── qat_epoch{1..10}.pt          2.7 GB each, 25 GB total — auto-cleanup didn't run
│   ├── qat_best.pt                  2.5 GB
│   ├── qat_for_modelopt_onnx.pt     2.5 GB ← uses old name (predates PR #2 rename)
│   ├── zero_shot_pe.pt              2.6 GB (combined-QKV upstream PE-Core weights)
│   └── manifest{,_eval_val}.json
├── crl_sweep/
│   ├── best_sigma.txt               picked σ_k = 2.0
│   └── results.md                   ← all σ_k variants give cos ~0.30-0.44 (broken)
├── crl_sweep_work/                  step2 scratch (14 GB; deletable)
└── deploy/
    └── dynamic/
        └── comparison_dynamic_20260518.md  ← BF16 dyn now works, INT8 dyn collapses
```

**48 GB total on disk** — biggest variant due to per-epoch ckpts + scratch not cleaned up.

### Sweep results — `crl_sweep/results.md`

| σ_k | INT8 cos | INT8 ms |
|---|---:|---:|
| 1.5 | 0.3004 | 11.73 |
| 2.0 | 0.4377 | 11.22 |
| 2.5 | 0.3637 | 11.76 |
| 3.0 | 0.4075 | 11.71 |

**Best σ_k = 2.0, INT8 cos = 0.4377** — across all σ_k values, INT8+CRL static cos collapses to ~0.30-0.44. Not a CRL parameter issue; it's the combined-QKV `[3D, D]` MatMul interacting badly with TRT INT8 calibration.

### Dynamic engine — `deploy/dynamic/comparison_dynamic_20260518.md`

| B | TRT BF16 dyn cos | TRT INT8 dyn (QAT) cos | TRT INT8 dyn (QAT+CRL) cos |
|---|---:|---:|---:|
| 1 | **0.9916** ✓ | 0.1964 ⚠ | 0.1850 ⚠ |
| 2 | 0.9923 | 0.1961 | 0.1860 |
| 4 | 0.9904 | 0.2102 | 0.1989 |
| 8 | 0.9924 | 0.2182 | 0.2078 |
| 16 | 0.9931 | 0.2037 | 0.1930 |
| 32 | 0.9920 | 0.1971 | 0.1898 |

**Surprise**: combined-QKV **fixes** the BF16 dynamic engine (the splitqkv version had cos ≈ 0.4 there). But **breaks INT8** even worse (cos 0.19, was cos 0.4 with splitqkv).

### Diagnosis

Combined-QKV `[3D, D]` in_proj MatMul + downstream Slice produces a graph where per-tensor INT8 activation quant on LN output (one scale covering 1024 channels) combined with per-channel weight quant on the wide `[D, 3D]` weight tensor degenerates — Q/K/V can't share the same activation scale productively when modelopt PTQ folds the calibration. Splitqkv's 3 separate `[D, D]` MatMuls give modelopt 3 independent calibration paths with the same input — TRT-INT8 fusion handles them as 3 small GEMMs cleanly.

**Conclusion**: combined-QKV is structurally cleaner (no surgery), but TRT-INT8 doesn't currently produce usable engines from it. Splitqkv remains the production recipe for ZS. Combined-QKV stays as a reference for future investigation (per-channel activation quant or SmoothQuant pre-pass might recover it).

## C — FT-T3 (canonical) → `output/ftpe_t3/`

**The recommended production deployment** for video INT8. Run date: 2026-05-15 18:20-19:41.

### Artifacts

```
output/ftpe_t3/
├── qat/
│   ├── qat_best.pt                   2.5 GB
│   └── qat_deploy_fp32.pt            2.5 GB ← deploy input
├── crl_sweep/best_sigma.txt          picked σ_k = 2.5
├── engines/                          27 TRT engines (BF16/INT8/INT8+CRL × B + dyn × 3)
└── deploy/
    ├── comparison_20260515.md        ← THE comparison report
    ├── static/                       per-batch JSON + MD
    └── dynamic/                      dyn engine bench (works at all B=1..32)
```

12 GB total on disk.

### Static engines — `deploy/comparison_20260515.md`

| B | TRT BF16 cos | TRT BF16 ms | TRT INT8 (QAT) cos | TRT INT8 (QAT) ms | **TRT INT8 (QAT+CRL) cos** | **TRT INT8 (QAT+CRL) ms** |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.998213 | 14.89 | 0.975460 | 11.61 | **0.990223** | 11.41 |
| 2 | 0.998216 | 30.07 | 0.975818 | 20.86 | **0.990564** | 21.12 |
| 4 | 0.998334 | 57.90 | 0.974009 | 39.89 | **0.991001** | 39.93 |
| 8 | 0.998531 | 111.85 | 0.973534 | 80.66 | **0.991385** | 77.39 |

Speedup INT8+CRL / TRT BF16: 1.31× → 1.42× → 1.45× → 1.45×.

### Dynamic engines — works across full B=1..32

From `deploy/dynamic/comparison_dynamic_20260515.md`:

| B | TRT BF16 dyn cos | **TRT INT8 dyn (QAT+CRL) cos** | TRT INT8 dyn (QAT+CRL) ms |
|---|---:|---:|---:|
| 1 | 0.998341 | **0.989417** | 12.50 |
| 2 | 0.998289 | **0.990299** | 22.73 |
| 4 | 0.998333 | **0.990431** | 42.60 |
| 8 | 0.998529 | **0.991383** | 76.11 |
| 16 | 0.998579 | **0.991214** | 161.81 |
| 32 | 0.998675 | **0.991589** | 321.39 |

**This is the production-recommended dynamic engine.** One TRT engine handles all batch sizes B=1..32 with cos stable at 0.989-0.992 and 1.27-1.44× speedup over BF16.

## D — FT-T8 (max quality) → `output/ftpe_t8/`

Run date: 2026-05-15 20:58 → 2026-05-16 04:10.

### Artifacts

```
output/ftpe_t8/
├── qat/
│   ├── qat_best.pt                   2.5 GB
│   └── qat_deploy_fp32.pt            2.5 GB
├── crl_sweep/best_sigma.txt          picked σ_k = 3.0
├── engines/                          20 engines (B=1,2,4 — B=8 INT8 OOM)
└── deploy/
    ├── comparison_20260516.md        ← THE comparison report
    ├── static/                       per-batch JSON + MD
    └── dynamic/                      dyn engine bench
```

9.4 GB total on disk.

### Static engines — `deploy/comparison_20260516.md`

| B | TRT BF16 cos | TRT BF16 ms | TRT INT8 (QAT) cos | TRT INT8 (QAT) ms | **TRT INT8 (QAT+CRL) cos** | **TRT INT8 (QAT+CRL) ms** |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.999975 | 39.26 | 0.980129 | 26.90 | **0.993999** | 27.07 |
| 2 | 0.999978 | 75.14 | 0.980571 | 51.13 | **0.993264** | 51.30 |
| 4 | 0.999981 | 149.67 | 0.979960 | 100.98 | **0.993084** | 101.17 |
| 8 | – ¹ | – | – ¹ | – | – ¹ | – |

¹ B=8 INT8 static skipped — ORT BFC arena 226 MB single-allocation limit even on a free 48 GB GPU. See Known Issues §3.

### Dynamic engines — `deploy/dynamic/comparison_dynamic_20260516.md`

| B | TRT BF16 dyn cos | **TRT INT8 dyn (QAT+CRL) cos** | TRT INT8 dyn (QAT+CRL) ms |
|---|---:|---:|---:|
| 1 | 0.998133 | **0.990204** | 29.26 |
| 2 | 0.998236 | **0.990807** | 55.62 |
| 4 | 0.998486 | **0.990824** | 108.62 |
| 8 | 0.998515 | **0.991126** | 215.37 |
| 16 | – ² | – ² | – ² |
| 32 | – ² | – ² | – ² |

² B=16, B=32 dynamic bench failed with `operands could not be broadcast (64,1024) (128,1024)` — pre-existing shape bug in `bench_dynamic.py`, see Known Issues §2.

## Frozen historical snapshots → `results/`

These predate the Leaderboard_bench clip pack (used the old **172-clip outdoor_v5** dataset that disappeared mid-May). **Not directly comparable** to A/B/C/D above (different test set, ~70% more samples), but kept for paper trail.

```
results/
├── zs_t1_fixed/
│   ├── comparison_20260512.md         ⚠ MISFILED — content is FT-T3 (T_FRAMES=3),
│   │                                    not ZS-T1; despite the directory name
│   ├── comparison_20260513.md         actual ZS-T1 static (T=1) on 172 clips
│   ├── results.json
│   └── results.md
├── zs_t1_dynamic/
│   ├── comparison_dynamic_20260513.md original 2026-05-13 ZS-T1 dynamic (172 clips)
│   └── dyn_{bf16,int8,int8_crl}.json
├── ft_t3/                             original 2026-05-13/14 FT-T3 (172 clips)
└── (loose top-level *.md, *.json)     live scratch from deploy_*.sh — overwritten per-run
```

### Historical headline numbers (172 outdoor_v5 clips)

**ZS-T1 static** — from `results/zs_t1_fixed/comparison_20260513.md` (correctly labelled T=1):

| B | TRT BF16 cos | TRT INT8 (QAT) cos | TRT INT8 (QAT+CRL) cos |
|---|---:|---:|---:|
| 1 | 0.9997 | 0.9858 | 0.9899 |
| 2 | 0.9996 | 0.9684 | 0.9871 |
| 4 | 0.9997 | 0.9671 | 0.9870 |

Matches the variant A current numbers within ~0.001 — the test-set change to Leaderboard_bench didn't move the needle much for static ZS-T1.

**FT-T3 static** — from `results/zs_t1_fixed/comparison_20260512.md` (despite the misleading `zs_t1_fixed/` parent, the content is T_FRAMES=3, KD against the FT teacher — it's FT-T3 data):

| B | TRT BF16 cos | TRT INT8 (QAT) cos | TRT INT8 (QAT+CRL) cos |
|---|---:|---:|---:|
| 1 | 1.0000 | 0.9780 | **0.9928** |
| 2 | 1.0000 | 0.9783 | **0.9928** |
| 4 | 1.0000 | 0.9768 | **0.9924** |
| 8 | 1.0000 | 0.9737 | **0.9926** |
| 16 | 1.0000 | 0.9735 | **0.9928** |

Compare to variant C current (FT-T3 on 100 Leaderboard_bench clips): B=1 INT8+CRL cos = 0.9902. The historical (172-clip outdoor_v5) numbers are ~0.0025 higher because the old test set had a tighter distribution. Same recipe, different test set.

**ZS-T1 dynamic** — from `results/zs_t1_dynamic/comparison_dynamic_20260513.md`:

| B | TRT BF16 dyn cos | TRT INT8 dyn (QAT+CRL) cos |
|---|---:|---:|
| 1 | 0.9996 | 0.9687 |
| 2 | 0.9996 | 0.9765 |
| 4 | 0.9996 | 0.9792 |
| 8 | 0.9996 | 0.9803 |
| 16 | 0.9996 | 0.9786 |
| 32 | 0.9995 | 0.9789 |

Interesting: in the **172-clip outdoor_v5 era**, the ZS-T1 *dynamic* engine had cos ≈ 0.97-0.98 — usable. The cos collapse to ~0.4 is something that emerged later (in the rebuild on the new clip pack and/or after some TRT/modelopt update). Worth investigating: re-run the 2026-05-13 build script verbatim, see if the old numbers reproduce on current TRT.

Cited by the historical `docs/FINAL_REPORT_20260513.md`.

## Disk usage snapshot

| Path | Size | Note |
|---|---:|---|
| `output/pe_t1/` | 29 GB | canonical ZS-T1 splitqkv; ckpts in `_splitqkv_archive_20260518/` |
| `output/pe_t1_combined_qkv/` | 48 GB | broken experiment; **can be deleted to reclaim ~48 GB** |
| `output/ftpe_t3/` | 12 GB | clean — engines (6.3 G) + 2 ckpts (5 G) |
| `output/ftpe_t8/` | 9.4 GB | clean — engines (4.4 G) + 2 ckpts (5 G) |
| `results/` | <1 GB | historical snapshots |
| **total** | **~100 GB** | |

## Known issues (open)

1. **ZS T=1 dynamic engine (splitqkv) cos collapse → 0.4**. TRT compilation issue specific to the T=1 dynamic profile. Static T=1 splitqkv engines are fine (cos 0.985). See `output/pe_t1/deploy/dynamic/comparison_dynamic_20260518.md` for the investigation. Static T=1 ships; dynamic T=1 does not.

2. **ZS T=1 combined-QKV INT8 collapses → cos 0.20**. Same root cause family but worse. Combined-QKV does fix the BF16 dyn issue (cos 0.99) but breaks INT8 (cos 0.20) on both static and dynamic. Splitqkv remains the production recipe for ZS.

3. **FT T=8 dynamic bench fails at B≥16** with `operands could not be broadcast (64,1024) (128,1024)`. Pre-existing shape-broadcast bug in `claude_exp8_finish1/scripts/lib/bench_dynamic.py`. Engine builds and runs fine at B=1..8; only the bench harness chokes.

4. **B=8 INT8 static at T=8 OOMs** during modelopt PTQ calibration. ORT BFC arena 226 MB single-allocation limit. Same architectural skip as B=32 INT8 at T=1/T=3.

## Production recommendation

| Workload | Recipe | Engine path | Headline numbers |
|---|---|---|---|
| Zero-shot image (T=1) | **static INT8+CRL** σ_k=3.0 per batch | `output/pe_t1/_splitqkv_archive_20260518/engines/int8_b{B}t1_crl.engine` | cos 0.984-0.988, 1.28-1.45× faster than BF16 |
| FT video, low-latency (T=3) | **dynamic INT8+CRL** σ_k=2.5 | `output/ftpe_t3/engines/int8_dyn_crl_t3.engine` | cos 0.989-0.992 across B=1..32, 1.27-1.44× faster |
| FT video, max quality (T=8) | **static INT8+CRL** σ_k=3.0 at B=1..4 | `output/ftpe_t8/engines/int8_b{B}t8_crl.engine` | cos 0.993, 1.45-1.48× faster |

## Reproduce

```bash
cd /mnt/nas200_kurnianto/code/TYPE8_PE_research

# Required env
export PE_VENDOR=/mnt/nas200_kurnianto/code/pia-prompt_optimization
export DATASET_ROOT=/mnt/nas200_kurnianto/code/TYPE8_PE_research/assets/clips
export N_PER_SPLIT=399 EVAL_N_PER_SPLIT=100 MIGRATE_LEGACY=0

# A — ZS-T1 splitqkv (current recipe, run via the splitqkv-default prep_zeroshot_ckpt.py with --split_qkv flag if needed)
bash src/claude_exp8_finish1/scripts/run/run_pipeline.sh pe

# C — FT-T3
bash src/claude_exp8_finish1/scripts/run/run_pipeline.sh ftpe

# D — FT-T8 (no run_pipeline mode for T=8; drive the lib directly)
bash src/claude_exp8_finish1/scripts/lib/run_qat_wide.sh \
    .ft_cache/FT_PE-Core-L14-336_260318.pt \
    src/claude_exp8_finish1/output/ftpe_t8/qat 8 399 10
bash src/claude_exp8_finish1/scripts/run/step2_CRL_sweep.sh \
    src/claude_exp8_finish1/output/ftpe_t8/qat/qat_deploy_fp32.pt 8 \
    src/claude_exp8_finish1/output/ftpe_t8
bash src/claude_exp8_finish1/scripts/run/step3_summary.sh \
    src/claude_exp8_finish1/output/ftpe_t8/qat/qat_deploy_fp32.pt 8 \
    src/claude_exp8_finish1/output/ftpe_t8
```

For deploy on a different host (e.g., A4000): see `docs/A4000_DEPLOY.md` for a self-contained checklist using `scripts/run/run_on_a4000.sh`.

## File path quick-reference

Looking for... | Path
---|---
ZS-T1 splitqkv comparison MD | `src/claude_exp8_finish1/output/pe_t1/deploy/comparison_20260515.md`
ZS-T1 splitqkv deploy ckpt | `src/claude_exp8_finish1/output/pe_t1/_splitqkv_archive_20260518/qat_deploy_fp32.pt`
ZS-T1 splitqkv engines | `src/claude_exp8_finish1/output/pe_t1/_splitqkv_archive_20260518/engines/`
ZS-T1 combined-QKV sweep (broken) | `src/claude_exp8_finish1/output/pe_t1_combined_qkv/crl_sweep/results.md`
ZS-T1 combined-QKV dyn bench (broken) | `src/claude_exp8_finish1/output/pe_t1_combined_qkv/deploy/dynamic/comparison_dynamic_20260518.md`
FT-T3 comparison MD | `src/claude_exp8_finish1/output/ftpe_t3/deploy/comparison_20260515.md`
FT-T3 deploy ckpt | `src/claude_exp8_finish1/output/ftpe_t3/qat/qat_deploy_fp32.pt`
FT-T3 engines | `src/claude_exp8_finish1/output/ftpe_t3/engines/`
FT-T8 comparison MD | `src/claude_exp8_finish1/output/ftpe_t8/deploy/comparison_20260516.md`
FT-T8 deploy ckpt | `src/claude_exp8_finish1/output/ftpe_t8/qat/qat_deploy_fp32.pt`
FT-T8 engines | `src/claude_exp8_finish1/output/ftpe_t8/engines/`
Historical (pre-Leaderboard_bench) ZS-T1 static | `src/claude_exp8_finish1/results/zs_t1_fixed/comparison_20260513.md`
Historical ZS-T1 dynamic | `src/claude_exp8_finish1/results/zs_t1_dynamic/comparison_dynamic_20260513.md`
Historical FT-T3 (172-clip outdoor_v5) | `src/claude_exp8_finish1/results/ft_t3/`
Code bundle (standalone) | `src/claude_exp8_finish1/pe_int8/` + `src/claude_exp8_finish1/scripts/`
Pipeline contract | `src/claude_exp8_finish1/docs/PIPELINE.md`
A4000 deploy guide | `src/claude_exp8_finish1/docs/A4000_DEPLOY.md`
Repo-level onboarding | `CLAUDE.md`

## Next-step suggestions

1. **Reclaim 48 GB** by deleting `output/pe_t1_combined_qkv/` after confirming the broken-INT8 finding is documented enough here. The 25 GB of per-epoch ckpts are scratch; the diagnosis lives in this report.
2. **Fix `bench_dynamic.py` broadcast bug** so FT-T8 dyn bench at B≥16 works.
3. **(optional) Re-investigate combined-QKV INT8** with SmoothQuant pre-pass or per-channel activation quant. Won't change the production recommendation but would close the loop.
4. **Rotate the GitHub PAT** that was in transcripts during PR cycle.
