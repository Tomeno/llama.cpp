#!/usr/bin/env python3
"""Compares llama.cpp's RMA (memory) output of one layer with the reference IndependentGQAMemory.

The reference module is executed verbatim from modeling_edgerock.py (only its two transformers helpers are stubbed),
on the exact layer inputs llama.cpp saw:

    llama-edgerock-dump EdgeRock-4B-BF16.gguf text.txt dump/ 1736 300 1800 l_out-2 mem_out-3 l_out-6 mem_out-7
    python ref_memory.py --ref ref/ --dump dump/ --layer 3 --chunk 300 --bf16-act

Use the BF16 GGUF so both sides have the same weights. --bf16-act emulates llama.cpp's BF16 matmuls (activations
rounded to bf16, except admission_down, which is F32 in the GGUF); expected: mean relative error ~3e-4, max < 2e-3.
Without it the error is ~3e-3 (precision only). An implementation error shows up as O(1) errors after token 256.
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn


class Qwen3_5RMSNorm(nn.Module):
    """same definition as transformers' Qwen3.5 RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w), in float32"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (out * (1.0 + self.weight.float())).type_as(x)


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def load_memory_class(modeling: Path):
    m = types.ModuleType("transformers.models.qwen3_5.modeling_qwen3_5")
    m.Qwen3_5RMSNorm, m.rotate_half = Qwen3_5RMSNorm, rotate_half
    for name in ["transformers", "transformers.models", "transformers.models.qwen3_5"]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["transformers.models.qwen3_5.modeling_qwen3_5"] = m
    cfg = types.ModuleType("configuration_edgerock")
    cfg.EdgeRockConfig = object
    sys.modules["configuration_edgerock"] = cfg
    src = modeling.read_text()
    src = src[: src.index("from typing import Any")]  # the memory module and its helpers only
    ns = {"__name__": "edgerock_ref"}
    exec(compile(src, str(modeling), "exec"), ns)
    return ns["IndependentGQAMemory"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True, help="directory from fetch_reference.py (with memory_weights.npz)")
    ap.add_argument("--dump", type=Path, required=True, help="directory from llama-edgerock-dump")
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=300, help="prefill chunk of the stateful reference pass")
    ap.add_argument("--decode-tail", type=int, default=64, help="tokens decoded one by one at the end (stateful pass)")
    ap.add_argument("--bf16-act", action="store_true", help="emulate llama.cpp BF16 matmuls")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    import json
    conf = json.loads((args.ref / "config.json").read_text())
    cfg = types.SimpleNamespace(backend="torch", hidden_size=conf["hidden_size"], num_attention_heads=conf["num_attention_heads"],
                                num_key_value_heads=conf["num_key_value_heads"], head_dim=conf["head_dim"],
                                independent_memory=conf["independent_memory"], fixed_memory_swa_window=conf["fixed_memory_swa_window"],
                                rms_norm_eps=conf["rms_norm_eps"], initializer_range=0.02)
    mem = load_memory_class(args.ref / "modeling_edgerock.py")(cfg)
    W = np.load(args.ref / "memory_weights.npz")
    prefix = f"model.layers.{args.layer}.memory."
    print(mem.load_state_dict({k[len(prefix):]: torch.from_numpy(W[k]) for k in W.files if k.startswith(prefix)}, strict=True))
    mem.eval()
    if args.bf16_act:
        fwd = nn.Linear.forward
        nn.Linear.forward = lambda self, x: fwd(self, x if self is mem.admission_down else x.to(torch.bfloat16).float())

    H = cfg.hidden_size
    x = np.fromfile(args.dump / f"l_out-{args.layer - 1}.bin", dtype=np.float32).reshape(-1, H)
    y_cpp = np.fromfile(args.dump / f"mem_out-{args.layer}.bin", dtype=np.float32).reshape(-1, H)
    N = x.shape[0]

    # Qwen3.5 text RoPE: partial rotary dims, interleaved MRoPE reduces to plain RoPE for text positions
    rp = conf["rope_parameters"]
    rot = int(conf["head_dim"] * rp.get("partial_rotary_factor", conf.get("partial_rotary_factor", 0.25)))
    inv = 1.0 / (rp["rope_theta"] ** (torch.arange(0, rot, 2, dtype=torch.float32) / rot))
    freqs = torch.arange(N, dtype=torch.float32)[:, None] * inv[None]
    emb = torch.cat((freqs, freqs), -1)
    cos, sin = emb.cos()[None], emb.sin()[None]

    h = torch.from_numpy(x)[None]
    with torch.no_grad():
        y_ref = mem(h, (cos, sin))[0].numpy()
        cache, outs, i = types.SimpleNamespace(memory_state=None), [], 0
        while i < N:
            n = min(args.chunk if i < N - args.decode_tail else 1, N - i)
            outs.append(mem(h[:, i:i + n], (cos[:, i:i + n], sin[:, i:i + n]), cache_layer=cache)[0])
            i += n
        y_ref_st = torch.cat(outs, 0).numpy()

    def rel(a, b):
        return np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(b, axis=-1) + 1e-12)

    print(f"tokens {N}")
    print(f"reference stateless vs stateful: max rel {rel(y_ref_st, y_ref).max():.3e}")
    r = rel(y_cpp, y_ref)
    print(f"llama.cpp vs reference: mean rel {r.mean():.3e}, max rel {r.max():.3e} (token {r.argmax()})")
    chunk = conf["independent_memory"]["max_commit_chunk"]
    for a in range(0, N, 4 * chunk):
        b = min(N, a + 4 * chunk)
        print(f"  tokens [{a:5d}, {b:5d}): mean rel {r[a:b].mean():.3e}, max {r[a:b].max():.3e}")
    print(f"tokens before the first commit ({chunk}) read nothing: max |y| llama.cpp {np.abs(y_cpp[:chunk]).max():.2e},"
          f" reference {np.abs(y_ref[:chunk]).max():.2e}")


if __name__ == "__main__":
    main()
