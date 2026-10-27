#!/usr/bin/env python3
"""Tunes a soft prompt with a contrastive objective on pairs of responses (DPO + NLL on the chosen response),
with antithetic ES + Adam.

Data: JSONL, {"prompt", "chosen", "rejected"} per line; both responses are scored as is (teacher forcing).
Per pair, with the initial soft prompt as the frozen reference:

    m = (log p(chosen) - log p_ref(chosen)) - (log p(rejected) - log p_ref(rejected))      (sums over tokens)
    L = -log sigmoid(beta * m) + alpha * (mean NLL per token of chosen)

The NLL term keeps the chosen responses likely (plain DPO can lower both). With --val-prompts, the best checkpoint is
the one with the highest mean reward on those prompts (as train_nll.py), else the lowest validation loss.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import es as es_mod
import rewards
import rollout
from softprompt import add_common_args, read_jsonl, save_soft_prompt, setup


def load(eng, tpl, path: str, max_resp: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for r in read_jsonl(path):
        post = tpl.post(eng, r["prompt"])
        c, j = eng.tokenize(r["chosen"])[:max_resp], eng.tokenize(r["rejected"])[:max_resp]
        if len(post) and len(c) and len(j):
            out.append((post, c, j))
    print(f"{path}: {len(out)} pairs, {sum(len(c) + len(j) for _, c, j in out)} scored tokens", file=sys.stderr)
    return out


def logps(eng, thetas: np.ndarray, jobs: list[tuple[int, tuple]], population: bool = False) -> list[tuple[float, float, int]]:
    """(sum log p chosen, sum log p rejected, chosen length) per job; jobs are (theta index, pair) tuples;
    population: thetas is an ES population (mirrored slots, see es.score_population)."""
    idx, posts, resps = [], [], []
    for j, (post, c, r) in jobs:
        idx += [j, j]
        posts += [post, post]
        resps += [c, r]
    lp = (es_mod.score_population if population else type(eng).score)(eng, thetas, idx, posts, resps)
    return [(float(lp[2 * k].sum()), float(lp[2 * k + 1].sum()), len(lp[2 * k])) for k in range(len(jobs))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    es_mod.add_es_args(ap)
    ap.add_argument("--train", required=True, help="pairs (JSONL): prompt, chosen, rejected")
    ap.add_argument("--val", help="validation pairs (JSONL)")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=1.0, help="weight of the chosen NLL term (0: plain DPO)")
    ap.add_argument("--max-resp", type=int, default=1024, help="response tokens scored per response")
    ap.add_argument("--examples", type=int, default=8, help="pairs per ES pair and step")
    ap.add_argument("--val-prompts", help="prompts (JSONL) sampled at every evaluation; selects the best checkpoint by reward")
    ap.add_argument("--n-val-prompts", type=int, default=0, help="validation prompts sampled, 0 = all")
    rollout.add_sampling_args(ap)
    rewards.add_reward_args(ap)
    args = ap.parse_args()

    eng, theta0, tpl = setup(args, n_seq=max(args.par, 4 * args.pairs))
    if len(theta0) == 0:
        sys.exit("nothing to optimize: the soft prompt is empty")
    train = load(eng, tpl, args.train, args.max_resp)
    val = load(eng, tpl, args.val, args.max_resp) if args.val else []
    vp_rows = read_jsonl(args.val_prompts) if args.val_prompts else []
    if args.n_val_prompts > 0:
        vp_rows = vp_rows[:args.n_val_prompts]
    vp = rollout.Prompts(eng, tpl, vp_rows)
    cfg = rewards.config_from_args(args)
    sampling = rollout.sampling_from_args(args)

    # reference log-probs: the initial soft prompt, once for every pair
    t0 = time.time()
    ref_train = logps(eng, theta0[None], [(0, e) for e in train])
    ref_val = logps(eng, theta0[None], [(0, e) for e in val]) if val else []
    print(f"reference log-probs: {time.time() - t0:.1f} s", file=sys.stderr)

    def loss(cur, ref) -> tuple[float, float]:
        """(loss, margin) of one pair."""
        m = (cur[0] - ref[0]) - (cur[1] - ref[1])
        return float(np.logaddexp(0.0, -args.beta * m) + args.alpha * (-cur[0] / max(cur[2], 1))), m

    def val_loss(theta) -> tuple[float, float]:
        if not val:
            return float("nan"), float("nan")
        res = [loss(c, r) for c, r in zip(logps(eng, theta[None], [(0, e) for e in val]), ref_val)]
        return float(np.mean([l for l, _ in res])), float(np.mean([m > 0 for _, m in res]))

    def val_reward(theta, step: int) -> float:
        ids = [i for i in range(len(vp)) for _ in range(args.group)]
        samples = rollout.rollout(eng, theta, vp, ids, args.gen_max, cfg, sampling, args.seed + 7, args.reward_cmd)
        print(f"step {step:5d}: val sampled {rollout.summary(samples)}", file=sys.stderr)
        return float(np.mean([s["reward"] for s in samples]))

    def evaluate(theta, step: int) -> tuple[float, float, float]:
        """(selection score, lower is better; validation loss; share of validation pairs with a positive margin)"""
        vl, acc = val_loss(theta)
        return (-val_reward(theta, step) if len(vp) else vl), vl, acc

    es, rms = es_mod.from_args(args, theta0)
    rng = np.random.default_rng(args.seed)
    order, cursor = rng.permutation(len(train)), 0

    def draw() -> int:
        nonlocal order, cursor
        if cursor >= len(order):
            order, cursor = rng.permutation(len(train)), 0
        cursor += 1
        return int(order[cursor - 1])

    best, vl, acc = evaluate(es.theta, 0)
    print(f"step {0:5d}: val loss {vl:.5f}, margin > 0 {100 * acc:.1f}% (init)", file=sys.stderr)
    save_soft_prompt(args.out, es.theta, tpl, {"step": 0, "val_loss": vl, "val_score": best})

    for step in range(1, args.steps + 1):
        t0 = time.time()
        pop = es.population()
        # both members of a pair see the same examples, so their difference cancels the example difficulty
        exs = [draw() for _ in range(args.examples)]
        jobs = [(m, train[i]) for m in range(len(pop)) for i in exs]
        res = logps(eng, pop, jobs, population=True)
        L = np.zeros(len(pop))
        margins = []
        for (m, _), i, cur in zip(jobs, exs * len(pop), res):
            l, mg = loss(cur, ref_train[i])
            L[m] += l / len(exs)
            margins.append(mg)
        d0 = es.update(L)
        drift = float(np.sqrt(np.mean((es.theta - theta0) ** 2)) / rms)
        print(f"step {step:5d}: train {L.mean():.5f} (population mean), margin {np.mean(margins):+.3f}, |dL| {d0:.2e},"
              f" drift {drift:.3f} RMS, {time.time() - t0:.1f} s", file=sys.stderr)
        if step % args.eval_every == 0 or step == args.steps:
            score, vl, acc = evaluate(es.theta, step)
            meta = {"step": step, "val_loss": vl, "val_score": score}
            if score < best:
                best = score
                save_soft_prompt(args.out, es.theta, tpl, meta)
            save_soft_prompt(args.out + ".last", es.theta, tpl, meta)
            print(f"step {step:5d}: val loss {vl:.5f}, margin > 0 {100 * acc:.1f}%, score {score:.5f} (best {best:.5f})", file=sys.stderr)


if __name__ == "__main__":
    main()
