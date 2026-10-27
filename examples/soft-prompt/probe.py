#!/usr/bin/env python3
"""Samples responses with a soft (or text) prompt and reports the reward terms.

Use it before GRPO: the share of groups with a reward spread is what GRPO can learn from.

    # the text prompt as the starting point
    python probe.py -m EdgeRock-4B-Q4_K_M.gguf --prompts simple.jsonl --init-text "..." --gen-max 256
    # no system turn at all
    python probe.py -m EdgeRock-4B-Q4_K_M.gguf --prompts simple.jsonl --init-text "" --before "" --after ""
"""

from __future__ import annotations

import argparse
import sys
import time

import rewards
import rollout
from softprompt import add_common_args, load_state, read_jsonl, setup, write_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    rollout.add_sampling_args(ap, gen_max=256)
    rewards.add_reward_args(ap)
    ap.add_argument("--prompts", required=True, help="JSONL, {\"prompt\", optional \"answer\"} per line")
    ap.add_argument("-n", "--n-prompts", type=int, default=0, help="prompts used, 0 = all")
    args = ap.parse_args()

    eng, theta, tpl = setup(args)
    state = load_state(args.init_file)
    if state is not None:
        print(f"with the recurrent state of {args.init_file}", file=sys.stderr)
    cfg = rewards.config_from_args(args)
    rows = read_jsonl(args.prompts)
    if args.n_prompts > 0:
        rows = rows[:args.n_prompts]
    prompts = rollout.Prompts(eng, tpl, rows)

    t0 = time.time()
    ids = [i for i in range(len(prompts)) for _ in range(args.group)]
    samples = rollout.rollout(eng, theta, prompts, ids, args.gen_max, cfg, rollout.sampling_from_args(args), args.seed, args.reward_cmd, state)
    if args.samples_out:
        write_jsonl(args.samples_out, (rollout.sample_row(s, prompts) for s in samples))

    spread = 0
    for i in range(len(prompts)):
        r = [s["reward"] for s in samples if s["prompt"] == i]
        spread += max(r) - min(r) > 1e-9
    print(f"probe: {len(prompts)} prompts x {args.group} samples, {time.time() - t0:.1f} s", file=sys.stderr)
    print(f"probe: {rollout.summary(samples)}", file=sys.stderr)
    print(f"probe: {spread} of {len(prompts)} groups have a reward spread (usable by GRPO)", file=sys.stderr)


if __name__ == "__main__":
    main()
