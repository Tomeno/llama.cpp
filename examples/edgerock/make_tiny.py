#!/usr/bin/env python3
"""Writes a tiny random EdgeRock checkpoint (HF format) for end-to-end tests against the reference.

8 layers (attention + RMA at 3 and 7), hidden 64, window 16, memory capacity 8, commit chunk 4: a 120-token text
goes through ~30 commits and many window slides. The tokenizer is the real one.

    python fetch_reference.py ref/
    python make_tiny.py ref/ tiny/
    python ../../convert_hf_to_gguf.py tiny/ --outtype f32 --outfile tiny/tiny-f32.gguf
    llama-edgerock-logits tiny/tiny-f32.gguf text.txt 120 90 7 tiny/logits_90_7.bin   # also 120 120 512 and 120 1 1
    python ref_full.py tiny/ tiny/logits_*.bin

Note: llama.cpp builds one RMA read per commit chunk a ubatch can touch, so with a commit chunk of 4 keep ubatches
small (-ub 64 or less): a 2048-token ubatch needs 513 reads per layer and runs out of graph allocator blocks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", type=Path, help="directory from fetch_reference.py")
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "generation_config.json", "configuration_edgerock.py", "modeling_edgerock.py"]:
        shutil.copy(args.ref / f, args.out / f)

    c = json.loads((args.ref / "config.json").read_text())
    L, H, FF = 8, 64, 96
    # linear head dim 16: the smallest the CUDA gated delta net supports
    LK, LKH, LV, LVH = 16, 2, 16, 4
    QKV, Z = 2*LK*LKH + LV*LVH, LV*LVH
    c.update(hidden_size=H, num_attention_heads=4, num_key_value_heads=2, head_dim=16, intermediate_size=FF, num_hidden_layers=L,
             linear_key_head_dim=LK, linear_num_key_heads=LKH, linear_value_head_dim=LV, linear_num_value_heads=LVH,
             fixed_memory_swa_layers=[3, 7], fixed_memory_swa_window=16,
             independent_memory={"admission_rank": 8, "capacity": 8, "max_commit_chunk": 4, "writer_intermediate_size": 32},
             layer_types=(["linear_attention"] * 3 + ["full_attention"]) * 2)
    c["rope_parameters"]["mrope_section"] = [1, 1, 0]  # rotary dims 16 * 0.25 = 4 -> 2 frequencies
    (args.out / "config.json").write_text(json.dumps(c, indent=2))

    g = torch.Generator().manual_seed(args.seed)

    def r(*s, std=0.05):
        return (torch.randn(*s, generator=g) * std).to(torch.bfloat16)

    V = c["vocab_size"]
    w = {"model.embed_tokens.weight": r(V, H, std=0.5), "model.norm.weight": r(H, std=0.1)}
    for i in range(L):
        p = f"model.layers.{i}."
        w[p + "input_layernorm.weight"] = r(H, std=0.1)
        w[p + "post_attention_layernorm.weight"] = r(H, std=0.1)
        w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"], w[p + "mlp.down_proj.weight"] = r(FF, H), r(FF, H), r(H, FF)
        if c["layer_types"][i] == "linear_attention":
            q = p + "linear_attn."
            w[q + "in_proj_qkv.weight"], w[q + "in_proj_z.weight"] = r(QKV, H), r(Z, H)
            w[q + "in_proj_b.weight"], w[q + "in_proj_a.weight"] = r(LVH, H), r(LVH, H)
            w[q + "conv1d.weight"], w[q + "dt_bias"], w[q + "A_log"] = r(QKV, 1, 4, std=0.3), r(LVH, std=1), r(LVH, std=1)
            w[q + "norm.weight"] = (1 + r(LV, std=0.1).float()).to(torch.bfloat16)
            w[q + "out_proj.weight"] = r(H, Z)
        else:
            q = p + "self_attn."
            w[q + "q_proj.weight"], w[q + "k_proj.weight"], w[q + "v_proj.weight"], w[q + "o_proj.weight"] = r(128, H), r(32, H), r(32, H), r(H, 64)
            w[q + "q_norm.weight"], w[q + "k_norm.weight"] = r(16, std=0.1), r(16, std=0.1)
            m = p + "memory."
            for n, s in [("q_proj", (64, H)), ("k_proj", (32, H)), ("v_proj", (32, H)), ("o_proj", (H, 64)), ("write_gate_proj", (32, H)),
                         ("write_up_proj", (32, H)), ("write_down_proj", (H, 32)), ("admission_down", (8, H)), ("admission_up", (2, 8))]:
                w[m + n + ".weight"] = r(*s, std=0.2)
            w[m + "input_norm.weight"], w[m + "q_norm.weight"], w[m + "k_norm.weight"] = r(H, std=0.1), r(16, std=0.1), r(16, std=0.1)
            w[m + "read_scale"], w[m + "read_gate_weight"], w[m + "read_gate_bias"] = r(4, std=0.5), r(4, 16, std=0.5), r(4, std=0.5)
    save_file(w, str(args.out / "model.safetensors"), metadata={"format": "pt"})
    print(f"wrote {len(w)} tensors to {args.out}")


if __name__ == "__main__":
    main()
