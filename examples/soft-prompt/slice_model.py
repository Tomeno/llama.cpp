#!/usr/bin/env python3
"""Writes a smaller copy of a GGUF model that keeps a subset of its layers (for fast tests and iteration).

The embeddings, the output norm and head and all hyperparameters stay as they are; the kept blocks are renumbered
and per-layer metadata arrays are sliced. The default keeps the first and the last 4 blocks, which for EdgeRock /
Qwen3.5 (one attention layer every 4) keeps the layer pattern: 3 delta-net layers + 1 attention (+ RMA) layer.

    python slice_model.py EdgeRock-4B-Q4_K_M.gguf EdgeRock-slice8.gguf --layers 0-3,28-31
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "gguf-py"))
import gguf  # noqa: E402


def parse_layers(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--layers", default=None, help="kept blocks, e.g. 0-3,28-31 (default: first and last 4)")
    args = ap.parse_args()

    r = gguf.GGUFReader(args.input)
    arch = r.fields["general.architecture"].contents()
    n_layer = int(r.fields[f"{arch}.block_count"].contents())
    keep = parse_layers(args.layers) if args.layers else list(range(4)) + list(range(n_layer - 4, n_layer))
    if any(i < 0 or i >= n_layer for i in keep) or len(set(keep)) != len(keep):
        sys.exit(f"invalid --layers for {n_layer} blocks")
    remap = {old: new for new, old in enumerate(keep)}
    print(f"{arch}: keeping {len(keep)} of {n_layer} blocks: {keep}", file=sys.stderr)

    w = gguf.GGUFWriter(args.output, arch)
    for f in r.fields.values():
        if f.name == "general.architecture" or f.name.startswith("GGUF."):
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        val = f.contents()
        if f.name.endswith(".block_count"):
            val = len(keep)
        elif vtype == gguf.GGUFValueType.ARRAY and isinstance(val, list) and len(val) == n_layer and not f.name.startswith("tokenizer."):
            # per-layer arrays (recurrent_layers, per-layer head counts, ...)
            val = [val[i] for i in keep]
        w.add_key_value(f.name, val, vtype, sub_type=sub)

    tensors = []
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", t.name)
        if m:
            i = int(m.group(1))
            if i not in remap:
                continue
            name = f"blk.{remap[i]}.{m.group(2)}"
        else:
            name = t.name
        tensors.append((name, t))
        w.add_tensor_info(name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _, t in tensors:
        w.write_tensor_data(t.data, tensor_endianess=r.endianess)
    w.close()
    print(f"wrote {args.output} ({sum(int(np.prod(t.data.shape)) * t.data.dtype.itemsize for _, t in tensors) / 2**20:.0f} MiB of tensors)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
