#!/usr/bin/env python3
"""Samples a response with a soft prompt (or a text prompt via --init-text)."""

from __future__ import annotations

import argparse

from softprompt import add_common_args, setup


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-n", "--gen-max", type=int, default=2048)
    ap.add_argument("--temp", type=float, default=0.6, help="<= 0: greedy")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    eng, theta, tpl = setup(args, n_seq=1)
    toks, finished = eng.generate(theta, [tpl.post(eng, args.prompt)], args.gen_max, args.temp, args.top_p, args.top_k, args.seed)[0]
    print(eng.detokenize(toks[:-1] if finished else toks))


if __name__ == "__main__":
    main()
