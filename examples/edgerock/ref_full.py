#!/usr/bin/env python3
"""Compares llama.cpp logits (from llama-edgerock-logits) with the reference EdgeRockForCausalLM.

    python ref_full.py tiny/ tiny/logits_120_512.bin tiny/logits_90_7.bin tiny/logits_1_1.bin

Needs torch and transformers==5.7.0 (the version the reference was written for; newer versions changed the cache API).
Runs the model directory's modeling_edgerock.py in float32, once without cache and once like generation (prefill of
64 tokens, then one token at a time). Expected on the tiny model: max |dlogit| ~2e-3 on logits of scale ~34 and all
argmaxes equal; the largest error is in the first tokens, before any commit (plain Qwen3.5 parity).
"""

from __future__ import annotations

import sys

import numpy as np
import torch


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    model_dir, files = sys.argv[1], sys.argv[2:]
    sys.path.insert(0, model_dir)
    import modeling_edgerock as me  # noqa: E402

    model = me.load_model(model_dir, torch_dtype=torch.float32).eval()
    V = model.config.vocab_size
    for lf in files:
        raw = np.fromfile(lf, dtype=np.uint8)
        n = len(raw) // (4 + 4 * V)
        toks = raw[:4 * n].view(np.int32)
        lc = raw[4 * n:].view(np.float32).reshape(n, V)
        with torch.no_grad():
            ref = model(torch.tensor(toks[None], dtype=torch.long), use_cache=False).logits[0].float().numpy()
            out = model(torch.tensor(toks[None, :64], dtype=torch.long), use_cache=True)
            pkv, steps = out.past_key_values, [out.logits[0]]
            for i in range(64, n):
                o = model(torch.tensor(toks[None, i:i + 1], dtype=torch.long), past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values
                steps.append(o.logits[0])
            ref_cache = torch.cat(steps, 0).float().numpy()
        d = np.abs(lc - ref).max(-1)
        print(f"{lf}: max |dlogit| {d.max():.2e} (logit scale {np.abs(ref).max():.2f}), argmax agree {(lc.argmax(-1) == ref.argmax(-1)).sum()}/{n},"
              f" reference cached vs stateless {np.abs(ref_cache - ref).max():.2e}")
        print("   per 16 tokens:", " ".join(f"{d[i:i + 16].max():.1e}" for i in range(0, n, 16)))


if __name__ == "__main__":
    main()
