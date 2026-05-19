"""Strip stray Q/DQ pairs around layout-only ops and residual Adds.

Step 4 of the ViT recipe. Even with `--op_types_to_exclude=Add`, modelopt can
leave Q/DQ around `Transpose`/`Reshape`/`Concat`/`LayerNormalization`/`Softmax`
in the patch-embedding stem and around the non-skip input of residual Adds.

For each forbidden Q -> DQ pair feeding a banned consumer, this script:
  1. rewires the consumer's input from the DQ output back to the Q input
  2. removes the now-orphaned Q and DQ nodes

After surgery:
  - `onnx.checker.check_model` should pass
  - re-inspect in Netron and confirm no Q/DQ around the banned op types

Run from src/claude_exp2:
    python surgery.py --onnx onnx/int8_model.onnx --out onnx/int8_model_clean.onnx
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Set

import onnx
from onnx import NodeProto


# Layout-only / non-arithmetic ops that should never sit downstream of Q/DQ.
_BANNED_LAYOUT_OPS = {
    "Transpose",
    "Reshape",
    "Concat",
    "LayerNormalization",
    "Softmax",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True,
                   help="Input ONNX from quantize_onnx.py.")
    p.add_argument("--out", default=None,
                   help="Output path. Defaults to <input_stem>_clean.onnx.")
    p.add_argument("--strip_residual_add", action="store_true", default=True,
                   help="Strip Q/DQ feeding the B input of residual-pattern Add nodes.")
    p.add_argument("--no_check", action="store_true",
                   help="Skip onnx.checker.check_model after surgery.")
    p.add_argument("--report_only", action="store_true",
                   help="Print findings and exit without modifying the graph.")
    return p.parse_args()


def _build_consumer_index(graph) -> Dict[str, List[NodeProto]]:
    consumers: Dict[str, List[NodeProto]] = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            if inp:
                consumers[inp].append(node)
    return consumers


def _is_q(node: NodeProto) -> bool:
    return node.op_type == "QuantizeLinear"


def _is_dq(node: NodeProto) -> bool:
    return node.op_type == "DequantizeLinear"


def _producer_index(graph) -> Dict[str, NodeProto]:
    out: Dict[str, NodeProto] = {}
    for node in graph.node:
        for o in node.output:
            if o:
                out[o] = node
    return out


def _residual_b_inputs(graph) -> Set[str]:
    """Return the tensor names that feed the *non-skip* (B) input of residual Adds.

    Heuristic: in ViT residual blocks an Add consumes (skip, MLP-or-attn-output).
    The 'skip' branch typically traces back through a chain of layout-only ops
    to the prior block's output and contains no MatMul/Gemm/LayerNorm. The 'B'
    branch is the one that *does* contain compute -- so we mark inputs whose
    producer (or recent ancestor within ~6 hops) is a Q/DQ guarding a MatMul,
    Gemm, or LayerNorm output as 'residual B'.
    """
    producer = _producer_index(graph)
    compute_ops = {"MatMul", "Gemm", "LayerNormalization"}
    flagged: Set[str] = set()
    for node in graph.node:
        if node.op_type != "Add" or len(node.input) < 2:
            continue
        for inp in node.input:
            tensor = inp
            for _ in range(6):
                src = producer.get(tensor)
                if src is None:
                    break
                if src.op_type in compute_ops:
                    flagged.add(inp)
                    break
                if src.op_type in {"DequantizeLinear", "QuantizeLinear",
                                   "Transpose", "Reshape", "Cast"}:
                    tensor = src.input[0]
                    continue
                break
    return flagged


def _qdq_pair_above(tensor: str, producer: Dict[str, NodeProto]):
    """If `tensor` is produced by DQ <- Q <- src, return (q_node, dq_node, src_tensor)."""
    dq = producer.get(tensor)
    if dq is None or not _is_dq(dq):
        return None
    q = producer.get(dq.input[0])
    if q is None or not _is_q(q):
        return None
    return q, dq, q.input[0]


def strip_qdq(model: onnx.ModelProto, *, strip_residual_add: bool,
              report_only: bool) -> onnx.ModelProto:
    graph = model.graph
    producer = _producer_index(graph)
    residual_b = _residual_b_inputs(graph) if strip_residual_add else set()

    # Inputs (consumer node, input slot index, current tensor name) we want to rewire.
    rewires: List = []
    nodes_to_drop: Set[int] = set()

    # node -> ordinal index in graph.node, for stable removal
    node_index = {id(n): i for i, n in enumerate(graph.node)}

    def schedule_strip(consumer: NodeProto, slot: int):
        tensor = consumer.input[slot]
        pair = _qdq_pair_above(tensor, producer)
        if pair is None:
            return False
        q, dq, src = pair
        # Only drop if DQ's *only* consumer is this consumer (otherwise we'd
        # break siblings that legitimately want the dequantized value).
        dq_consumers = sum(1 for n in graph.node
                           for inp in n.input if inp == dq.output[0])
        if dq_consumers != 1:
            return False
        rewires.append((consumer, slot, src))
        nodes_to_drop.add(node_index[id(q)])
        nodes_to_drop.add(node_index[id(dq)])
        return True

    layout_hits = 0
    residual_hits = 0
    for node in graph.node:
        if node.op_type in _BANNED_LAYOUT_OPS:
            for slot in range(len(node.input)):
                if schedule_strip(node, slot):
                    layout_hits += 1
        if node.op_type == "Add" and strip_residual_add:
            for slot, inp in enumerate(node.input):
                if inp in residual_b and schedule_strip(node, slot):
                    residual_hits += 1

    print(f"[surgery] layout-op Q/DQ to strip:  {layout_hits}")
    print(f"[surgery] residual-Add Q/DQ to strip: {residual_hits}")

    if report_only:
        return model

    for consumer, slot, src in rewires:
        consumer.input[slot] = src

    if nodes_to_drop:
        kept = [n for i, n in enumerate(graph.node) if i not in nodes_to_drop]
        del graph.node[:]
        graph.node.extend(kept)

    return model


def main() -> int:
    args = parse_args()
    out = args.out or os.path.splitext(args.onnx)[0] + "_clean.onnx"

    print(f"[surgery] loading {args.onnx}")
    model = onnx.load(args.onnx, load_external_data=True)

    model = strip_qdq(model, strip_residual_add=args.strip_residual_add,
                      report_only=args.report_only)

    if args.report_only:
        return 0

    if not args.no_check:
        onnx.checker.check_model(model, full_check=False)

    base = os.path.splitext(os.path.basename(out))[0]
    data_fname = f"{base}.data"
    onnx.save_model(
        model, out,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_fname,
        size_threshold=0,
    )
    print(f"[surgery] DONE -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
