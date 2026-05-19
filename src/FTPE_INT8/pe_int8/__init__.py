"""pe_int8 — self-contained INT8 deploy bundle for PE-Core-L14-336.

Bundled helpers (originally scattered across claude_exp3_phase2 / claude_exp5 /
claude_exp7_mixqvit):

    ft_loader.py             — load a PE.CLIP checkpoint, auto-detect QKV layout
    video_utils.py           — clip sampling + calibration tensor helpers
    export_onnx.py           — trace PE.CLIP to FP32 ONNX
    quantize_onnx.py         — modelopt-onnx PTQ wrapper
    surgery.py               — strip layout-only Q/DQ from a quantized ONNX
    build_engine_py.py       — TRT 10 engine build (fixed shape)
    build_dynamic_engine.py  — TRT 10 engine build (dynamic profile)
    bench_trt.py             — bench engines vs PT BF16 (static)
    bench_dynamic.py         — bench dynamic engines (in scripts/lib/)
    aggregate_results.py     — emit static-bench comparison markdown
    aggregate_dynamic.py     — emit dynamic-bench comparison markdown
    crl_pass.py              — Clipped LayerNorm Reparam pre-PTQ pass
    phase1_qat_modelopt_wide.py — wide QAT (training-time)
    phase2_dequantize_qat.py    — dequantize QAT state_dict to plain FP32

Required external dependencies:

    * pia-prompt_optimization (PE vendor) — set $PE_VENDOR or place at
      <exp_root>/vendor/pia-prompt_optimization
    * Python env: torch, tensorrt 10.16, nvidia-modelopt 0.43, onnx, onnxruntime-gpu
    * For loading raw LoRA-PEFT FT checkpoints (not the deploy path):
      peft, plus the TYPE8_PE_research src/PE_FPINT8/ helpers on PYTHONPATH.

Typical usage from a script:

    import os, sys
    PE_INT8 = "/path/to/claude_exp8_finish1/pe_int8"
    os.environ["PE_VENDOR"] = "/path/to/pia-prompt_optimization"
    sys.path.insert(0, PE_INT8)
    from ft_loader import load_ft_clip
    model = load_ft_clip("/path/to/qat_deploy_fp32.pt").cuda().eval()

Or end-to-end, just run the standalone script:

    bash <exp_root>/scripts/run/run_on_a4000.sh
"""
