"""Clipped LayerNorm Reparameterization (CRL) — ONNX pre-quantization pass.

Background
----------
Post-LayerNorm activations in ViT-style models have heavy per-channel
outliers. A per-tensor INT8 quantizer (what TRT IMMA needs) has to use
the global maximum as its scale, wasting precision on the bulk channels.

The CRL trick from Mix-QViT (arXiv 2501.06357) fixes this with no
deploy-side cost:

  1. Calibrate per-channel max|activation| s_c at the LN output.
  2. Clip outlier channels: r_c = min(threshold, s_c) / s_c
     where threshold = mean(s) + K * std(s)  (default K=2).
  3. Fold the clip ratio r_c into the preceding LayerNorm:
        γ̂_c = γ_c · r_c
        β̂_c = β_c · r_c
     so the LN now emits y'_c = r_c · y_c.
  4. Compensate the next MatMul / Gemm by inverse-rescaling the
     weight along its input-channel axis:
        W̃[c, :] = W[c, :] / r_c   (MatMul)
        W̃[:, c] = W[:, c] / r_c   (Gemm with trans_B=1)

Mathematically the network output is unchanged (modulo the clipping
on outlier channels themselves, which by construction we accept).
The benefit: the post-LN activation distribution that modelopt-onnx
calibrates becomes well-conditioned, the per-tensor scale uses the
bulk of the channels' range, and INT8 round-off error shrinks.

This pass writes a modified ONNX that downstream
`claude_exp3_phase2/quantize_onnx.py` can ingest as if it were the
original FP32 export.

Usage
-----
    python crl_pass.py \
        --in_onnx  src/claude_exp7_mixqvit/onnx/fp32_b1t3.onnx \
        --out_onnx src/claude_exp7_mixqvit/onnx/fp32_b1t3_crl.onnx \
        --calib_npy /tmp/exp6.../calib/calibration.npy \
        --sigma_k 2.0
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_onnx", required=True)
    p.add_argument("--out_onnx", required=True)
    p.add_argument("--calib_npy", required=True,
                   help="(N, 3, H, W) calibration tensor.")
    p.add_argument("--sigma_k", type=float, default=2.0,
                   help="Clip threshold = mean(s_c) + sigma_k * std(s_c).")
    p.add_argument("--batch_size", type=int, default=3,
                   help="BT axis 0 of the ONNX input.")
    p.add_argument("--max_calib_batches", type=int, default=32,
                   help="Cap calibration batches (32*BT=96 frames default).")
    return p.parse_args()


# ---------- Graph traversal ----------

def _initializer_map(model) -> Dict[str, np.ndarray]:
    """Initializer-name -> writable ndarray view."""
    out = {}
    for init in model.graph.initializer:
        out[init.name] = numpy_helper.to_array(init).copy()
    return out


def _consumers_index(model) -> Dict[str, List[onnx.NodeProto]]:
    """Tensor-name -> list of consumer nodes."""
    idx: Dict[str, List[onnx.NodeProto]] = {}
    for n in model.graph.node:
        for t in n.input:
            idx.setdefault(t, []).append(n)
    return idx


def _trace_to_linear(start_tensor: str, consumers: Dict[str, List[onnx.NodeProto]],
                     max_hops: int = 3) -> List[onnx.NodeProto]:
    """Walk forward from `start_tensor` skipping layout-only ops
    (Cast, Identity, Reshape, Transpose) until we hit MatMul/Gemm
    consumers. Returns the list of MatMul/Gemm nodes that consume
    (transitively) `start_tensor`.

    Multiple branches are supported (MHA QKV fan-out)."""
    seen_tensors = set()
    matmul_nodes: List[onnx.NodeProto] = []
    frontier = [(start_tensor, 0)]
    while frontier:
        t, hop = frontier.pop()
        if t in seen_tensors or hop > max_hops:
            continue
        seen_tensors.add(t)
        for n in consumers.get(t, []):
            if n.op_type in ("MatMul", "Gemm"):
                matmul_nodes.append(n)
            elif n.op_type in ("Cast", "Identity", "Reshape", "Transpose"):
                # transparent: keep walking
                for out in n.output:
                    frontier.append((out, hop + 1))
            # other ops (Add, Mul, Softmax, Tanh, etc.) terminate this branch
    return matmul_nodes


def find_ln_to_linear_pairs(model) -> List[Tuple[onnx.NodeProto, List[onnx.NodeProto]]]:
    """Return (ln_node, [consumer_matmul_or_gemm_nodes]) for every
    LayerNormalization that reaches at least one MatMul / Gemm."""
    consumers = _consumers_index(model)
    pairs = []
    for n in model.graph.node:
        if n.op_type != "LayerNormalization":
            continue
        ln_out = n.output[0]
        matmuls = _trace_to_linear(ln_out, consumers)
        if matmuls:
            pairs.append((n, matmuls))
    return pairs


# ---------- Calibration ----------

def _add_outputs(model, tensor_names: List[str]):
    """Add tensor_names to graph outputs for inspection via ORT."""
    existing = {o.name for o in model.graph.output}
    for tname in tensor_names:
        if tname in existing:
            continue
        # We don't know shape/dtype precisely; use empty TypeProto and
        # let onnxruntime infer it at session-init time.
        v = onnx.helper.make_tensor_value_info(tname, onnx.TensorProto.FLOAT, None)
        model.graph.output.append(v)


def _strip_added_outputs(model, original_outputs: List[onnx.ValueInfoProto]):
    """Reset graph.output to the original list."""
    del model.graph.output[:]
    model.graph.output.extend(original_outputs)


def calibrate_per_channel_max(model, init_map: Dict[str, np.ndarray],
                              ln_output_names: List[str],
                              calib: np.ndarray, batch_size: int,
                              max_batches: int) -> Dict[str, np.ndarray]:
    """Run the model on calib slices, collect per-channel max|abs|
    for each LN output tensor.

    Returns: dict tensor_name -> ndarray of shape (D,)."""
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime not installed; cannot calibrate")

    original_outputs = list(model.graph.output)
    _add_outputs(model, ln_output_names)

    # Serialize the modified model and feed it to ORT via bytes.
    # Use external data so 2 GB+ models work.
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    # Save to a temp file so onnxruntime can resolve external-data
    # references relative to the original onnx file location.
    print(f"[crl] persisting CRL-instrumented onnx for ORT session")
    tmp_path = os.environ.get(
        "CRL_TMP_ONNX",
        f"/tmp/exp7_crl_work_{os.getuid()}/onnx/_crl_calib.onnx",
    )
    os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
    onnx.save(model, tmp_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=os.path.basename(tmp_path) + ".data",
              size_threshold=1024, convert_attribute=False)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(tmp_path, sess_options=sess_opts, providers=providers)
    input_name = sess.get_inputs()[0].name
    print(f"[crl] ORT providers: {sess.get_providers()}  input: {input_name}")

    # Restore the graph outputs to the original list on the in-memory model
    # so save-out at the end doesn't carry the calibration outputs.
    _strip_added_outputs(model, original_outputs)

    # Stream calibration
    per_ch_max: Dict[str, np.ndarray] = {}
    n_imgs = calib.shape[0]
    n_used = 0
    for b_idx in range(min(max_batches, (n_imgs + batch_size - 1) // batch_size)):
        s = b_idx * batch_size
        e = min(s + batch_size, n_imgs)
        if e - s < batch_size:
            break
        x = calib[s:e].astype(np.float32)
        outs = sess.run(ln_output_names, {input_name: x})
        for name, arr in zip(ln_output_names, outs):
            # arr shape is (..., D); reduce along all axes except last
            abs_arr = np.abs(arr)
            ch_max = abs_arr.reshape(-1, abs_arr.shape[-1]).max(axis=0)
            if name not in per_ch_max:
                per_ch_max[name] = ch_max
            else:
                per_ch_max[name] = np.maximum(per_ch_max[name], ch_max)
        n_used += (e - s)
    print(f"[crl] calibration: {n_used} images across {b_idx+1} batches")

    # Drop the on-disk temp.
    try:
        os.remove(tmp_path)
        os.remove(tmp_path + ".data")
    except OSError:
        pass

    # onnx.save(..., save_as_external_data=True) mutated the in-memory
    # model so its TensorProtos now reference the just-deleted .data file.
    # Re-internalize every initializer from init_map (which holds the
    # original data we captured before the temp save) so the subsequent
    # final save can re-externalize cleanly.
    for init in list(model.graph.initializer):
        # data_location field exists in newer onnx; check by checking
        # external_data presence (proto2 field).
        if init.external_data:
            model.graph.initializer.remove(init)
    # init_map values are deep copies; rebuild all initializers from it.
    # Any initializer NOT in init_map (added after init_map was built) is
    # left alone. We rebuild the list ordering matches the original.
    existing_names = {i.name for i in model.graph.initializer}
    for name, arr in init_map.items():
        if name in existing_names:
            continue  # already present (internal); will be updated by _set_initializer if needed
        model.graph.initializer.append(
            numpy_helper.from_array(arr, name=name)
        )
    return per_ch_max


# ---------- Clip ratio ----------

def compute_clip_ratios(per_channel_max: np.ndarray, sigma_k: float
                        ) -> Tuple[np.ndarray, dict]:
    """r_c = min(threshold, s_c) / s_c. Returns (r, stats)."""
    s = per_channel_max.astype(np.float64)
    mu, sigma = s.mean(), s.std()
    threshold = mu + sigma_k * sigma
    # Avoid divide-by-zero on dead channels
    s_safe = np.maximum(s, 1e-12)
    r = np.minimum(threshold, s_safe) / s_safe
    r = np.clip(r, 1e-3, 1.0)  # never go below 1e-3 (numerical safety)
    stats = {
        "channels": int(s.shape[0]),
        "s_mean": float(mu), "s_std": float(sigma),
        "s_max": float(s.max()), "s_min": float(s.min()),
        "threshold": float(threshold),
        "n_clipped": int((s > threshold).sum()),
        "min_r": float(r.min()),
    }
    return r.astype(np.float32), stats


# ---------- Reparameterization ----------

def _set_initializer(model, name: str, arr: np.ndarray) -> bool:
    for i, init in enumerate(model.graph.initializer):
        if init.name == name:
            new = numpy_helper.from_array(arr, name=name)
            model.graph.initializer.remove(init)
            model.graph.initializer.append(new)
            return True
    return False


def _gemm_trans_b(node) -> int:
    for a in node.attribute:
        if a.name == "transB":
            return int(a.i)
    return 0


def _gemm_trans_a(node) -> int:
    for a in node.attribute:
        if a.name == "transA":
            return int(a.i)
    return 0


def apply_crl_to_pair(model, init_map: Dict[str, np.ndarray],
                     ln_node, consumers: List[onnx.NodeProto], r: np.ndarray
                     ) -> int:
    """Rescale γ, β of `ln_node` and weights of every consumer MatMul/Gemm.
    Returns number of weight initializers modified."""
    # LN inputs: [X, Scale, Bias]
    if len(ln_node.input) < 3:
        print(f"[crl] WARN: LN node {ln_node.name} has fewer than 3 inputs; skipping")
        return 0
    scale_name, bias_name = ln_node.input[1], ln_node.input[2]
    if scale_name not in init_map or bias_name not in init_map:
        print(f"[crl] WARN: LN {ln_node.name}: scale or bias not in initializers; skipping")
        return 0
    gamma = init_map[scale_name]
    beta = init_map[bias_name]
    D = gamma.shape[0]
    if r.shape[0] != D:
        print(f"[crl] WARN: r shape {r.shape} != D {D} for LN {ln_node.name}; skipping")
        return 0
    gamma_new = (gamma * r).astype(gamma.dtype)
    beta_new = (beta * r).astype(beta.dtype)
    _set_initializer(model, scale_name, gamma_new)
    _set_initializer(model, bias_name, beta_new)

    n_modified = 0
    inv_r = (1.0 / np.maximum(r, 1e-12)).astype(np.float32)
    for c in consumers:
        # Weight is input[1] (MatMul or Gemm)
        if len(c.input) < 2:
            continue
        w_name = c.input[1]
        if w_name not in init_map:
            # Not an initializer (e.g., dynamic weight or activation-weight product).
            print(f"[crl] skip {c.op_type} consumer {c.name}: weight '{w_name}' not an initializer")
            continue
        W = init_map[w_name]
        if c.op_type == "MatMul":
            # ONNX MatMul: y = x @ W, W shape last two = [..., in, out]
            # Rescale along axis -2 (input channels)
            if W.ndim != 2 or W.shape[0] != D:
                print(f"[crl] skip MatMul {c.name}: weight shape {W.shape} doesn't have axis-0 == D ({D})")
                continue
            W_new = W * inv_r[:, None]
        elif c.op_type == "Gemm":
            transB = _gemm_trans_b(c)
            if transB:
                # W shape [out, in], rescale axis 1
                if W.ndim != 2 or W.shape[1] != D:
                    print(f"[crl] skip Gemm {c.name} (transB=1): weight shape {W.shape} doesn't have axis-1 == D ({D})")
                    continue
                W_new = W * inv_r[None, :]
            else:
                if W.ndim != 2 or W.shape[0] != D:
                    print(f"[crl] skip Gemm {c.name} (transB=0): weight shape {W.shape} doesn't have axis-0 == D ({D})")
                    continue
                W_new = W * inv_r[:, None]
        else:
            continue

        _set_initializer(model, w_name, W_new.astype(W.dtype))
        n_modified += 1
    return n_modified


# ---------- Main ----------

def main():
    args = parse_args()
    print(f"[crl] loading {args.in_onnx}")
    model = onnx.load(args.in_onnx)
    init_map = _initializer_map(model)

    pairs = find_ln_to_linear_pairs(model)
    print(f"[crl] found {len(pairs)} LayerNorm -> MatMul/Gemm pairs")
    if not pairs:
        print(f"[crl] FATAL: no LN -> linear pairs detected.", file=sys.stderr)
        return 1
    # Sanity: count consumers
    consumer_counts = [len(c) for _, c in pairs]
    print(f"[crl] consumers per LN: min={min(consumer_counts)} max={max(consumer_counts)} "
          f"mean={sum(consumer_counts)/len(consumer_counts):.1f}")

    ln_output_names = [ln.output[0] for ln, _ in pairs]
    print(f"[crl] loading calibration tensor: {args.calib_npy}")
    calib = np.load(args.calib_npy)
    print(f"[crl] calib shape: {calib.shape}  dtype: {calib.dtype}")

    per_ch_max = calibrate_per_channel_max(
        model, init_map, ln_output_names, calib,
        batch_size=args.batch_size, max_batches=args.max_calib_batches,
    )
    print(f"[crl] got per-channel max for {len(per_ch_max)} LN outputs")

    # Reparam each pair
    total_modified_weights = 0
    for i, (ln, consumers) in enumerate(pairs):
        ln_out = ln.output[0]
        if ln_out not in per_ch_max:
            print(f"[crl] WARN: no calib stats for {ln_out}; skipping")
            continue
        r, stats = compute_clip_ratios(per_ch_max[ln_out], args.sigma_k)
        n_mod = apply_crl_to_pair(model, init_map, ln, consumers, r)
        total_modified_weights += n_mod
        if i % 8 == 0 or i == len(pairs) - 1:
            print(f"[crl] [{i+1:2d}/{len(pairs)}] LN {ln.name}: "
                  f"clipped {stats['n_clipped']}/{stats['channels']} channels "
                  f"(s∈[{stats['s_min']:.3f}, {stats['s_max']:.3f}], "
                  f"thr={stats['threshold']:.3f}, min r={stats['min_r']:.4f}); "
                  f"modified {n_mod} weight(s)")
    print(f"[crl] total weight initializers modified: {total_modified_weights}")

    print(f"[crl] saving {args.out_onnx}")
    onnx.save(model, args.out_onnx,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=os.path.basename(args.out_onnx) + ".data",
              size_threshold=1024, convert_attribute=False)
    print(f"[crl] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
