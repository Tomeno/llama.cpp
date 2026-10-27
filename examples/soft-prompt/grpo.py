#!/usr/bin/env python3
"""GRPO-style soft prompt tuning without backpropagation.

Every step samples G responses for B prompts with the current soft prompt, rewards them, and computes Dr. GRPO
advantages (reward minus the group mean, no std division). Groups without a reward spread are dropped (DAPO dynamic
sampling). The clipped surrogate (DAPO clip-higher) with a k3 KL to the initial soft prompt is a weighted sum of
token log-probs of the sampled responses, so it is evaluated with teacher forcing for every member of an antithetic
ES population, and Adam takes the ES gradient. The samples are reused for --es-steps updates.

Prompts: JSONL, {"prompt", optional "answer"} per line. Rewards: see rewards.py.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

import es as es_mod
import rewards
import rollout
from softprompt import add_common_args, read_jsonl, save_soft_prompt, setup, write_jsonl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    es_mod.add_es_args(ap, steps=100)
    rollout.add_sampling_args(ap, gen_max=512)
    rewards.add_reward_args(ap)
    g = ap.add_argument_group("grpo")
    g.add_argument("--train", required=True, help="training prompts (JSONL)")
    g.add_argument("--val", help="validation prompts (JSONL), sampled with a fixed seed")
    g.add_argument("--n-val", type=int, default=0, help="validation prompts used, 0 = all")
    g.add_argument("--prompts", type=int, default=16, help="prompts per step")
    g.add_argument("--gen-min", type=int, default=0, help="draw the token budget from [gen-min, gen-max] every step (0: gen-max)")
    g.add_argument("--es-steps", type=int, default=2, help="ES updates per sampled batch")
    g.add_argument("--clip-low", type=float, default=0.2)
    g.add_argument("--clip-high", type=float, default=0.28)
    g.add_argument("--kl", type=float, default=0.01, help="k3 KL to the initial soft prompt")
    g.add_argument("--norm", choices=["const", "token"], default="const",
                   help="Dr. GRPO constant normalizer (samples x budget) or DAPO token-level mean")
    g.add_argument("--no-dynamic", action="store_true", help="keep groups whose rewards are all equal")
    args = ap.parse_args()

    eng, theta_ref, tpl = setup(args, n_seq=max(args.par, 2 * args.pairs))
    if len(theta_ref) == 0:
        sys.exit("nothing to optimize: the soft prompt is empty")
    cfg = rewards.config_from_args(args)
    sampling = rollout.sampling_from_args(args)
    train = rollout.Prompts(eng, tpl, read_jsonl(args.train))
    val_rows = read_jsonl(args.val) if args.val else []
    if args.n_val > 0:
        val_rows = val_rows[:args.n_val]
    val = rollout.Prompts(eng, tpl, val_rows)

    es, rms = es_mod.from_args(args, theta_ref)
    rng = np.random.default_rng(args.seed)

    def evaluate(step: int) -> float:
        if not len(val):
            return float("nan")
        ids = [i for i in range(len(val)) for _ in range(args.group)]
        samples = rollout.rollout(eng, es.theta, val, ids, args.gen_max, cfg, sampling, args.seed + 7, args.reward_cmd)
        print(f"step {step:5d}: val {rollout.summary(samples)}", file=sys.stderr)
        return float(np.mean([s["reward"] for s in samples]))

    best = evaluate(0)
    save_soft_prompt(args.out, es.theta, tpl, {"step": 0, "val_reward": best})

    order, cursor = rng.permutation(len(train)), 0
    for step in range(1, args.steps + 1):
        t0 = time.time()
        max_new = int(rng.integers(args.gen_min, args.gen_max + 1)) if 0 < args.gen_min < args.gen_max else args.gen_max
        ids = []
        for _ in range(args.prompts):
            if cursor >= len(order):
                order, cursor = rng.permutation(len(train)), 0
            ids += [int(order[cursor])] * args.group
            cursor += 1
        samples = rollout.rollout(eng, es.theta, train, ids, max_new, cfg, sampling, args.seed + 1_000_003 * step, args.reward_cmd)
        if args.samples_out:
            write_jsonl(args.samples_out, ({"step": step, **rollout.sample_row(s, train)} for s in samples), append=True)
        t_gen = time.time() - t0

        # Dr. GRPO advantages; DAPO dynamic sampling drops groups without a reward spread
        groups = defaultdict(list)
        for s in samples:
            groups[s["prompt"]].append(s)
        kept, adv = [], []
        for grp in groups.values():
            r = np.array([s["reward"] for s in grp])
            if not args.no_dynamic and r.max() - r.min() < 1e-9:
                continue
            kept += grp
            adv += list(r - r.mean())
        if not kept:
            print(f"step {step:5d}: {rollout.summary(samples)} | no group with a reward spread, {t_gen:.1f} s", file=sys.stderr)
            continue
        adv = np.array(adv)
        posts = [train.posts[s["prompt"]] for s in kept]
        resps = [s["toks"] for s in kept]
        n_tok = sum(len(r) for r in resps)
        Z = n_tok if args.norm == "token" else len(kept) * max_new

        # log-probs under the sampling soft prompt and under the initial one
        zeros = [0] * len(kept)
        lp_old = eng.score(es.theta[None], zeros, posts, resps, args.temp)
        lp_ref = eng.score(theta_ref[None], zeros, posts, resps, args.temp) if args.kl > 0 else None
        kl = 0.0
        if lp_ref is not None:
            d = np.concatenate(lp_ref) - np.concatenate(lp_old)
            kl = float(np.mean(np.exp(d) - d - 1))

        obj = d0 = 0.0
        for _ in range(args.es_steps):
            pop = es.population()
            n_m = len(pop)
            idx = [m for m in range(n_m) for _ in kept]
            lp = es_mod.score_population(eng, pop, idx, posts * n_m, resps * n_m, args.temp)
            J = np.zeros(n_m)
            for k, cur in enumerate(lp):
                m, i = divmod(k, len(kept))
                ratio = np.exp(cur.astype(np.float64) - lp_old[i])
                clip = np.clip(ratio, 1 - args.clip_low, 1 + args.clip_high)
                J[m] += np.minimum(ratio * adv[i], clip * adv[i]).sum()
                if lp_ref is not None:
                    dd = lp_ref[i] - cur.astype(np.float64)
                    J[m] -= args.kl * (np.exp(dd) - dd - 1).sum()
            obj = float(J.mean() / Z)
            d0 = es.update(-J / Z)

        drift = float(np.sqrt(np.mean((es.theta - theta_ref) ** 2)) / rms)
        print(f"step {step:5d}: {rollout.summary(samples)} | groups {len(set(s['prompt'] for s in kept))}/{len(groups)},"
              f" max_new {max_new}, objective {obj:.3e}, |dL| {d0:.2e}, KL {kl:.4f}, drift {drift:.3f} RMS,"
              f" gen {t_gen:.1f} s, total {time.time() - t0:.1f} s", file=sys.stderr)

        if step % args.eval_every == 0 or step == args.steps:
            vr = evaluate(step)
            if not len(val) or vr > best:
                best = vr
                save_soft_prompt(args.out, es.theta, tpl, {"step": step, "val_reward": vr})
            save_soft_prompt(args.out + ".last", es.theta, tpl, {"step": step, "val_reward": vr})


if __name__ == "__main__":
    main()
