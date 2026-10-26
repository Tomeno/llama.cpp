#!/usr/bin/env python3
"""Downloads the EdgeRock reference implementation and, optionally, the RMA weights of some layers.

    python fetch_reference.py ref/                    # modeling code, config, tokenizer
    python fetch_reference.py ref/ --memory 3 7       # + ref/memory_weights.npz (BF16 -> F32) of layers 3 and 7

The memory weights are read with HTTP range requests from the safetensors shards, so the 9 GB checkpoint is not
downloaded. Source: https://huggingface.co/rockmatrix-ai/EdgeRock
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np

REPO = "https://huggingface.co/rockmatrix-ai/EdgeRock/resolve/main/"
FILES = ["modeling_edgerock.py", "configuration_edgerock.py", "config.json", "generation_config.json",
         "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json", "runtime/chat_template.jinja"]


def get(url: str, first: int | None = None, last: int | None = None) -> bytes:
    req = urllib.request.Request(url)
    if first is not None:
        req.add_header("Range", f"bytes={first}-{last}")
    with urllib.request.urlopen(req) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--memory", type=int, nargs="*", default=[], help="layers whose RMA weights to fetch")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for f in FILES:
        dst = args.out / Path(f).name
        if not dst.exists():
            dst.write_bytes(get(REPO + f))
            print(f"fetched {f}")

    if args.memory:
        index = json.loads((args.out / "model.safetensors.index.json").read_text())["weight_map"]
        headers: dict[str, tuple[int, dict]] = {}
        out = {}
        for name, shard in index.items():
            if not any(f"model.layers.{l}.memory." in name for l in args.memory):
                continue
            if shard not in headers:
                n = struct.unpack("<Q", get(REPO + shard, 0, 7))[0]
                headers[shard] = (n, json.loads(get(REPO + shard, 8, 7 + n)))
            n, h = headers[shard]
            e = h[name]
            a, b = e["data_offsets"]
            assert e["dtype"] == "BF16", e
            raw = np.frombuffer(get(REPO + shard, 8 + n + a, 8 + n + b - 1), dtype=np.uint16)
            out[name] = (raw.astype(np.uint32) << 16).view(np.float32).reshape(e["shape"])
            print(f"fetched {name} {e['shape']}")
        np.savez(args.out / "memory_weights.npz", **out)


if __name__ == "__main__":
    main()
