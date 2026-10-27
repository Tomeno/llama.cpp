#!/usr/bin/env python3
"""Tunes a soft prompt to minimize the NLL of given responses (teacher forcing), with antithetic ES + Adam.

Data: JSONL, {"prompt", "response"} per line; the response is scored as is (for a thinking model it starts inside
<think>). With --eval-only, prints the validation NLL with and without the soft prompt.

With --val-prompts, every evaluation also samples those prompts and scores them with the reward options (as grpo.py);
the best checkpoint is then the one with the highest mean reward instead of the lowest validation NLL.
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


def load(eng, tpl, path: str, max_resp: int) -> list[tuple[np.ndarray, np.ndarray]]:
    out = []
    for r in read_jsonl(path):
        post, resp = tpl.post(eng, r["prompt"]), eng.tokenize(r["response"])[:max_resp]
        if len(post) and len(resp):
            out.append((post, resp))
    print(f"{path}: {len(out)} examples, {sum(len(r) for _, r in out)} scored tokens", file=sys.stderr)
    return out


def nll(eng, thetas: np.ndarray, jobs: list[tuple[int, tuple]], states: np.ndarray | None = None) -> np.ndarray:
    """Mean NLL per token for every soft prompt (or state); jobs are (theta index, example) pairs.
    Several thetas (or states) are an ES population."""
    n = len(states) if states is not None else len(thetas)
    score = es_mod.score_population if n > 1 else type(eng).score
    lp = score(eng, thetas, [j for j, _ in jobs], [e[0] for _, e in jobs], [e[1] for _, e in jobs], states=states)
    tot = np.zeros(n)
    cnt = np.zeros(n)
    for (j, _), x in zip(jobs, lp):
        tot[j] -= x.sum()
        cnt[j] += len(x)
    return tot / np.maximum(cnt, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_common_args(ap)
    es_mod.add_es_args(ap)
    ap.add_argument("--train")
    ap.add_argument("--val")
    ap.add_argument("--n-val", type=int, default=0, help="validation examples used, 0 = all")
    ap.add_argument("--max-resp", type=int, default=1024, help="response tokens scored per example")
    ap.add_argument("--examples", type=int, default=4, help="examples per pair and step")
    ap.add_argument("--assign", choices=["shared", "split"], default="shared",
                    help="all pairs on the same examples, or every pair on its own examples")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--val-prompts", help="prompts (JSONL) sampled at every evaluation; selects the best checkpoint by reward")
    ap.add_argument("--n-val-prompts", type=int, default=0, help="validation prompts sampled, 0 = all")
    rollout.add_sampling_args(ap)
    rewards.add_reward_args(ap)
    args = ap.parse_args()

    eng, theta0, tpl = setup(args, n_seq=max(args.par, 2 * args.pairs))
    val = load(eng, tpl, args.val, args.max_resp) if args.val else []
    if args.n_val > 0:
        val = val[:args.n_val]
    vp_rows = read_jsonl(args.val_prompts) if args.val_prompts else []
    if args.n_val_prompts > 0:
        vp_rows = vp_rows[:args.n_val_prompts]
    vp = rollout.Prompts(eng, tpl, vp_rows)
    cfg = rewards.config_from_args(args)
    sampling = rollout.sampling_from_args(args)

    def val_nll(theta) -> float:
        return float(nll(eng, theta[None], [(0, e) for e in val])[0]) if val else float("nan")

    def val_reward(theta, step: int) -> float:
        ids = [i for i in range(len(vp)) for _ in range(args.group)]
        samples = rollout.rollout(eng, theta, vp, ids, args.gen_max, cfg, sampling, args.seed + 7, args.reward_cmd)
        print(f"step {step:5d}: val sampled {rollout.summary(samples)}", file=sys.stderr)
        return float(np.mean([s["reward"] for s in samples]))

    def evaluate(theta, step: int) -> tuple[float, float]:
        """(selection score, lower is better; validation NLL)"""
        vl = val_nll(theta)
        return (-val_reward(theta, step) if len(vp) else vl), vl

    if args.eval_only:
        empty = np.zeros((0, eng.n_embd), np.float32)
        print(f"val loss: soft prompt {val_nll(theta0):.5f}, no soft prompt {val_nll(empty):.5f} (over {len(val)} examples)", file=sys.stderr)
        return

    train = load(eng, tpl, args.train, args.max_resp)
    es, rms = es_mod.from_args(args, theta0)
    rng = np.random.default_rng(args.seed)
    order, cursor = rng.permutation(len(train)), 0

    def draw():
        nonlocal order, cursor
        if cursor >= len(order):
            order, cursor = rng.permutation(len(train)), 0
        cursor += 1
        return train[order[cursor - 1]]

    best, vl = evaluate(es.theta, 0)
    print(f"step {0:5d}: val {vl:.5f} (init)", file=sys.stderr)
    save_soft_prompt(args.out, es.theta, tpl, {"step": 0, "val_loss": vl, "val_score": best})

    for step in range(1, args.steps + 1):
        t0 = time.time()
        pop = es.population()
        # both members of a pair see the same examples, so their difference cancels the example difficulty
        shared = [draw() for _ in range(args.examples)]
        jobs = []
        for i in range(args.pairs):
            exs = shared if args.assign == "shared" else [draw() for _ in range(args.examples)]
            jobs += [(2 * i, e) for e in exs] + [(2 * i + 1, e) for e in exs]
        L = nll(eng, pop, jobs)
        d0 = es.update(L)
        drift = float(np.sqrt(np.mean((es.theta - theta0) ** 2)) / rms)
        print(f"step {step:5d}: train {L.mean():.5f} (population mean), |dL| {d0:.2e}, drift {drift:.3f} RMS, {time.time() - t0:.1f} s",
              file=sys.stderr)
        if step % args.eval_every == 0 or step == args.steps:
            score, vl = evaluate(es.theta, step)
            meta = {"step": step, "val_loss": vl, "val_score": score}
            if not (val or len(vp)) or score < best:
                best = score
                save_soft_prompt(args.out, es.theta, tpl, meta)
            save_soft_prompt(args.out + ".last", es.theta, tpl, meta)
            print(f"step {step:5d}: val {vl:.5f}, score {score:.5f} (best {best:.5f})", file=sys.stderr)


if __name__ == "__main__":
    main()
