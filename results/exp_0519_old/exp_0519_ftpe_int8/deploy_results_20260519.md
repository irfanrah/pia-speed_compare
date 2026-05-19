# claude_exp3_phase2 -- FT BF16 / TRT BF16 / TRT INT8 video sweep (patch-embed exclusion + simplify + optional autotune)

- model: `PE-Core-L14-336` + FT weights from `qat_deploy_fp32.pt`
- video clips: 100 train + 100 val (seed=20260508); T=3
- engine BT = B*T; iters: 30 after 5 warmup, CUDA events
- video accuracy gate: cos >= 0.999, mse <= 1.0e-05
- INT8 deployment target: >= 1.5x vs TRT BF16

| Mode | B | BT | Precision | Size | cos (frame) | MSE (frame) | cos (video) | MSE (video) | ms | vid/s | img/s | vs TRT BF16 | vs PT BF16 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pt_bf16 | 4 | 12 | bf16 | - | 1.000000 | 0.00e+00 | 1.000000 | 0.00e+00 | 254.64 | 15.7 | 47.1 | 2.32x | 1.00x |
| bf16_b4t3.engine | 4 | 12 | bf16 | 613.3 MiB | 0.999918 | 1.62e-07 | 0.999962 | 7.41e-08 | 590.72 | 6.8 | 20.3 | 1.00x | 0.43x |
| int8_b4t3.engine | 4 | 12 | int8 | 340.7 MiB | 0.977661 | 4.36e-05 | 0.979566 | 3.96e-05 | 427.81 | 9.3 | 28.0 | 1.38x | 0.60x |
| int8_b4t3_crl.engine | 4 | 12 | int8 | 340.7 MiB | 0.992352 | 1.49e-05 | 0.993094 | 1.34e-05 | 431.08 | 9.3 | 27.8 | 1.37x | 0.59x |

**B=4 verdict:** int8/bf16 = 1.38x; video accuracy FAIL; deploy=False; _INT8 only marginally faster: re-run surgery.py to clear residual Q/DQ around layout-only ops_

