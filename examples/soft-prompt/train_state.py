#!/usr/bin/env python3
"""Initial state tuning: tunes the recurrent state after the prompt prefix (EdgeRock / Qwen3.5: the delta-net matrices,
24 layers x 32 heads x 128 x 128 = 12.6M values) on the NLL of target responses, with antithetic ES + Adam.

The soft prompt (or text) prefix stays fixed and is processed as usual; its recurrent state is then replaced by
    state = s0 + scale * z
where s0 is the state the prefix leaves, scale the RMS of each head's matrix in s0 and z the tuned parameters
(init 0, or the state of the --init-file, so --sigma / --lr are relative to each head's own scale). The noise is low rank per matrix (--rank).
The window KV, the conv states and the RMA records stay as the prefix leaves them.

Data and validation as train_nll.py. With --forks (dpo.py pairs that share a prefix, e.g. the thinking before the
point where instruction talk starts), the loss adds w_fork * -log sigmoid(beta * m) per pair, m the DPO margin against
the starting state; only the tokens after the shared prefix are scored. The result is a soft prompt file that also holds the state (probe.py and
generate.py use it with --init-file).

With --no-soft-prompt there is no prefix at all (no soft prompt, no before / after text): the state is the initial state
of the model, before the first token of the prompt; s0 is zero then, so the scales come from the state of the --init-file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import es as es_mod
import rewards
import rollout
from softprompt import add_common_args, load_state, read_jsonl, save_state, setup
import dpo
from train_nll import load, nll

HEAD = 128  # head_v_dim: the state of a head is HEAD x HEAD


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    es_mod.add_es_args(ap)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("--n-val", type=int, default=0, help="validation examples used, 0 = all")
    ap.add_argument("--max-resp", type=int, default=1024, help="response tokens scored per example")
    ap.add_argument("--examples", type=int, default=8, help="examples per pair and step")
    ap.add_argument("--val-prompts", help="prompts (JSONL) sampled at every evaluation; selects the best checkpoint by reward")
    ap.add_argument("--n-val-prompts", type=int, default=0, help="validation prompts sampled, 0 = all")
    ap.add_argument("--forks", help="DPO pairs (JSONL) whose chosen and rejected responses share a prefix")
    ap.add_argument("--fork-examples", type=int, default=8, help="fork pairs per member and step")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--w-fork", type=float, default=1.0, help="weight of the fork DPO term")
    ap.add_argument("--scale-floor", type=float, default=0.1, help="minimum head scale, relative to the median")
    ap.add_argument("--no-soft-prompt", action="store_true",
                    help="no prefix: tune the initial state before the prompt (from the state of --init-file, if any)")
    rollout.add_sampling_args(ap)
    rewards.add_reward_args(ap)
    args = ap.parse_args()

    init_file = args.init_file
    if args.no_soft_prompt:
        args.init_file, args.init_text, args.before, args.after = None, "", "", ""
    eng, theta, tpl = setup(args, n_seq=max(args.par, 2 * args.pairs))
    args.init_file = init_file
    if eng.n_state == 0 or eng.n_state % (HEAD * HEAD):
        sys.exit(f"the model has {eng.n_state} recurrent state values, not a stack of {HEAD} x {HEAD} matrices")
    s0 = eng.get_state(theta).reshape(-1, HEAD, HEAD)
    st0 = load_state(args.init_file)
    if args.no_soft_prompt and st0 is None:
        sys.exit("--no-soft-prompt needs an --init-file with a state (the scales come from it)")
    # without a prefix s0 is zero: the scales of the starting state instead
    ref = st0.reshape(s0.shape) if args.no_soft_prompt and st0 is not None else s0
    scale = np.sqrt(np.mean(np.square(ref, dtype=np.float64), axis=(1, 2))).astype(np.float32)
    scale = np.maximum(scale, args.scale_floor * float(np.median(scale)))
    print(f"state: {len(s0)} matrices, head scale median {np.median(scale):.2e}, min {scale.min():.2e}, max {scale.max():.2e}",
          file=sys.stderr)

    def states(z: np.ndarray) -> np.ndarray:
        """[n, n_state] from parameters [n, M, HEAD, HEAD]"""
        return (s0 + scale[:, None, None] * z).reshape(len(z), -1)

    train = load(eng, tpl, args.train, args.max_resp)
    val = load(eng, tpl, args.val, args.max_resp) if args.val else []
    if args.n_val > 0:
        val = val[:args.n_val]
    forks = []
    for post, c, j in (dpo.load(eng, tpl, args.forks, args.max_resp) if args.forks else []):
        n = 0
        while n < min(len(c), len(j)) - 1 and c[n] == j[n]:
            n += 1
        forks.append((np.concatenate([post, c[:n]]), c[n:], j[n:]))
    if forks:
        print(f"forks: {len(forks)} pairs, {np.mean([len(c) + len(j) for _, c, j in forks]):.1f} scored tokens per pair",
              file=sys.stderr)

    def fork_logps(st: np.ndarray, jobs: list[tuple[int, int]]) -> np.ndarray:
        """[n_jobs, 2] (log p chosen, log p rejected) of fork pairs; jobs are (state index, pair index)"""
        idx = [m for m, _ in jobs for _ in (0, 1)]
        posts = [forks[k][0] for _, k in jobs for _ in (0, 1)]
        resps = [x for _, k in jobs for x in forks[k][1:]]
        score = es_mod.score_population if len(st) > 1 else type(eng).score
        lp = score(eng, theta[None], idx, posts, resps, states=st)
        return np.array([x.sum() for x in lp]).reshape(-1, 2)

    vp_rows = read_jsonl(args.val_prompts) if args.val_prompts else []
    if args.n_val_prompts > 0:
        vp_rows = vp_rows[:args.n_val_prompts]
    vp = rollout.Prompts(eng, tpl, vp_rows)
    cfg = rewards.config_from_args(args)
    sampling = rollout.sampling_from_args(args)

    def evaluate(z, step: int) -> tuple[float, float]:
        """(selection score, lower is better; validation NLL)"""
        st = states(z[None])
        vl = float(nll(eng, theta[None], [(0, e) for e in val], st)[0]) if val else float("nan")
        if not len(vp):
            return vl, vl
        ids = [i for i in range(len(vp)) for _ in range(args.group)]
        samples = rollout.rollout(eng, theta, vp, ids, args.gen_max, cfg, sampling, args.seed + 7, args.reward_cmd, st[0])
        print(f"step {step:5d}: val sampled {rollout.summary(samples)}", file=sys.stderr)
        if args.samples_out:  # appended at every evaluation, tagged with the step
            with open(args.samples_out, "a") as f:
                f.writelines(json.dumps({"step": step, **rollout.sample_row(s, vp)}, ensure_ascii=False) + "\n" for s in samples)
        return -float(np.mean([s["reward"] for s in samples])), vl

    def save(path: str, z, meta: dict):
        save_state(path, theta, states(z[None])[0], tpl, meta)

    # an --init-file that holds a state (a previous run) continues from it
    z0 = np.zeros_like(s0)
    if st0 is not None:
        z0 = (st0.reshape(s0.shape) - s0) / scale[:, None, None]
        print(f"continuing from the state in {args.init_file}, drift {np.sqrt(np.mean(z0 ** 2)):.3f} head RMS", file=sys.stderr)
    ref = fork_logps(states(z0[None]), [(0, k) for k in range(len(forks))]) if forks else None
    rng_f = np.random.default_rng(args.seed + 3)
    fork_order, fork_cursor = rng_f.permutation(len(forks)), 0
    es = es_mod.ES(z0, args.sigma, args.lr, args.pairs, args.rank, args.shaping, args.seed + 1)
    rng = np.random.default_rng(args.seed)
    order, cursor = rng.permutation(len(train)), 0

    def draw():
        nonlocal order, cursor
        if cursor >= len(order):
            order, cursor = rng.permutation(len(train)), 0
        cursor += 1
        return train[order[cursor - 1]]

    def draw_fork() -> int:
        nonlocal fork_order, fork_cursor
        if fork_cursor >= len(fork_order):
            fork_order, fork_cursor = rng_f.permutation(len(forks)), 0
        fork_cursor += 1
        return int(fork_order[fork_cursor - 1])

    best, vl = evaluate(es.theta, 0)
    print(f"step {0:5d}: val {vl:.5f} (init)", file=sys.stderr)
    save(args.out, es.theta, {"step": 0, "val_loss": vl, "val_score": best})

    for step in range(1, args.steps + 1):
        t0 = time.time()
        # the members' states straight from the noise: s0 + scale * (z +- sigma * eps)
        d = es.noise() * (args.sigma * scale[:, None, None])
        base = s0 + scale[:, None, None] * es.theta
        st = np.empty((2 * args.pairs,) + s0.shape, dtype=np.float32)
        np.add(base, d, out=st[0::2])
        np.subtract(base, d, out=st[1::2])
        # both members of a pair see the same examples, so their difference cancels the example difficulty
        exs = [draw() for _ in range(args.examples)]
        jobs = [(m, e) for m in range(len(st)) for e in exs]
        L = nll(eng, theta[None], jobs, st.reshape(len(st), -1))
        fork_info = ""
        if forks:
            ks = [draw_fork() for _ in range(args.fork_examples)]
            lp = fork_logps(st.reshape(len(st), -1), [(m, k) for m in range(len(st)) for k in ks])
            r = ref[np.tile(ks, len(st))]
            mg = ((lp[:, 0] - r[:, 0]) - (lp[:, 1] - r[:, 1])).reshape(len(st), len(ks))
            F = np.logaddexp(0.0, -args.beta * mg).mean(axis=1)
            fork_info = f", fork {F.mean():.4f} (margin {mg.mean():+.2f})"
            L = L + args.w_fork * F
        d0 = es.update(L)
        drift = float(np.sqrt(np.mean(es.theta ** 2)))
        print(f"step {step:5d}: train {L.mean():.5f} (population mean){fork_info}, |dL| {d0:.2e},"
              f" mean |dL| {np.mean(np.abs(L[0::2] - L[1::2])):.2e}, drift {drift:.3f} head RMS, {time.time() - t0:.1f} s",
              file=sys.stderr)
        if step % args.eval_every == 0 or step == args.steps:
            score, vl = evaluate(es.theta, step)
            meta = {"step": step, "val_loss": vl, "val_score": score}
            if score < best:
                best = score
                save(args.out, es.theta, meta)
            save(args.out + ".last", es.theta, meta)
            print(f"step {step:5d}: val {vl:.5f}, score {score:.5f} (best {best:.5f})", file=sys.stderr)


if __name__ == "__main__":
    main()
