#!/usr/bin/env python3
"""Initial state tuning: tunes the recurrent state after the prompt prefix (EdgeRock / Qwen3.5: the delta-net matrices,
24 layers x 32 heads x 128 x 128 = 12.6M values; RWKV7 2.9B: the wkv matrices, 32 layers x 40 heads x 64 x 64 = 5.2M)
on the NLL of target responses, with antithetic ES + Adam.

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
of the model, before the first token of the prompt; s0 is zero then, so the scales come from the state of the --init-file
or, without one (start from the model's own zero state), from the state the model reaches after --scale-text.

--opt grad (with --no-soft-prompt) replaces ES by the exact gradient (backpropagation through the model, llama.cpp's
state-gradient mode) and Adam on the same parameters and loss: every step is one backward pass per example and per fork
response, each sequence (prompt + response, at most --max-len tokens; longer responses are cut) in one ubatch, so -b must
be at least --max-len. With gradient checkpointing (--ckpt layers per segment) a 900-token sequence takes ~6 GB of GPU
memory besides the model, ~10% more time. The ES options (--pairs, --sigma, --rank, --shaping) are not used; --lr is the
Adam step.
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
from softprompt import add_common_args, load_state, read_jsonl, save_state, setup, state_head_dim
import dpo
from train_nll import load, nll

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
    ap.add_argument("--keep-evals", action="store_true", help="also save the state of every evaluation (OUT.stepN)")
    ap.add_argument("--forks", help="DPO pairs (JSONL) whose chosen and rejected responses share a prefix")
    ap.add_argument("--fork-examples", type=int, default=8, help="fork pairs per member and step")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--w-fork", type=float, default=1.0, help="weight of the fork DPO term")
    ap.add_argument("--scale-floor", type=float, default=0.1, help="minimum head scale, relative to the median")
    ap.add_argument("--scale-text", help="--no-soft-prompt without an --init-file state: the head scales are those of the "
                    "state after this text (the tuning starts from the zero state)")
    ap.add_argument("--no-soft-prompt", action="store_true",
                    help="no prefix: tune the initial state before the prompt (from the state of --init-file, if any)")
    ap.add_argument("--opt", choices=["es", "grad"], default="es",
                    help="es: antithetic ES; grad: exact gradients (needs --no-soft-prompt) and Adam")
    ap.add_argument("--max-len", type=int, default=1024,
                    help="--opt grad: prompt + response tokens per sequence, longer responses are cut (GPU memory)")
    ap.add_argument("--ckpt", type=int, default=4,
                    help="--opt grad: gradient checkpointing, layers per segment (0: off, ~16 MB of GPU memory per token)")
    ap.add_argument("--grad-seqs", type=int, default=1,
                    help="--opt grad: sequences per backward pass, padded to the longest (-b must hold them together)")
    rollout.add_sampling_args(ap)
    rewards.add_reward_args(ap)
    args = ap.parse_args()

    if args.opt == "grad" and not args.no_soft_prompt:
        sys.exit("--opt grad tunes the initial state: it needs --no-soft-prompt")
    if args.opt == "grad" and args.batch < args.max_len:
        sys.exit("--opt grad: the batch (-b) must hold a whole sequence (--max-len)")
    init_file = args.init_file
    if args.no_soft_prompt:
        args.init_file, args.init_text, args.before, args.after = None, "", "", ""
    eng, theta, tpl = setup(args, n_seq=args.par if args.opt == "grad" else max(args.par, 2 * args.pairs))
    if args.opt == "grad":
        eng.set_grad_ckpt(args.ckpt)
        eng.set_grad_seqs(args.grad_seqs)
    args.init_file = init_file
    head = state_head_dim(args.model)
    if eng.n_state == 0 or eng.n_state % (head * head):
        sys.exit(f"the model has {eng.n_state} recurrent state values, not a stack of {head} x {head} matrices")
    s0 = eng.get_state(theta).reshape(-1, head, head)
    st0 = load_state(args.init_file)
    ref = s0
    if args.no_soft_prompt:
        # without a prefix s0 is zero: the scales of the starting state, or of the state after --scale-text
        if st0 is not None:
            ref = st0.reshape(s0.shape)
        elif args.scale_text:
            eng.set_before(eng.tokenize(args.scale_text))
            ref = eng.get_state(theta).reshape(s0.shape)
            eng.set_before([])
        else:
            sys.exit("--no-soft-prompt needs an --init-file with a state or --scale-text (the scales come from it)")
    scale = np.sqrt(np.mean(np.square(ref, dtype=np.float64), axis=(1, 2))).astype(np.float32)
    scale = np.maximum(scale, args.scale_floor * float(np.median(scale)))
    print(f"state: {len(s0)} matrices, head scale median {np.median(scale):.2e}, min {scale.min():.2e}, max {scale.max():.2e}",
          file=sys.stderr)

    def states(z: np.ndarray) -> np.ndarray:
        """[n, n_state] from parameters [n, M, head, head]"""
        return (s0 + scale[:, None, None] * z).reshape(len(z), -1)

    train = load(eng, tpl, args.train, args.max_resp)
    if args.opt == "grad":
        train = [(p, r[:args.max_len - len(p)]) for p, r in train if len(p) < args.max_len]
    val = load(eng, tpl, args.val, args.max_resp) if args.val else []
    if args.n_val > 0:
        val = val[:args.n_val]
    forks = []
    for post, c, j in (dpo.load(eng, tpl, args.forks, args.max_resp) if args.forks else []):
        n = 0
        while n < min(len(c), len(j)) - 1 and c[n] == j[n]:
            n += 1
        post = np.concatenate([post, c[:n]])
        if args.opt == "grad":
            if len(post) >= args.max_len:
                continue
            c, j = c[:n + args.max_len - len(post)], j[:n + args.max_len - len(post)]
        forks.append((post, c[n:], j[n:]))
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

    opt = es_mod.Adam(z0, args.lr) if args.opt == "grad" else None

    def cur() -> np.ndarray:
        return opt.theta if opt else es.theta

    def grad_step() -> str:
        """one Adam step on the exact gradient of the loss (NLL + w_fork * fork DPO)"""
        st = states(opt.theta[None])
        exs = [draw() for _ in range(args.examples)]
        n_tok = sum(len(r) for _, r in exs)
        posts = [p for p, _ in exs]
        resps = [r for _, r in exs]
        ws = [np.full(len(r), -1.0 / n_tok, np.float32) for r in resps]
        info = ""
        F = 0.0
        if forks:
            # the DPO slope depends on the margin: the log-probs of the current state first
            ks = [draw_fork() for _ in range(args.fork_examples)]
            lp = fork_logps(st, [(0, k) for k in ks])
            mg = (lp[:, 0] - ref[ks, 0]) - (lp[:, 1] - ref[ks, 1])
            F = float(np.logaddexp(0.0, -args.beta * mg).mean())
            # dF/dm = -beta sigmoid(-beta m) / n, m = log p(chosen) - log p(rejected) - ref
            dm = -args.beta / (1.0 + np.exp(args.beta * mg)) / len(ks)
            for k, d in zip(ks, dm):
                post, c, j = forks[k]
                posts += [post, post]
                resps += [c, j]
                ws += [np.full(len(c), args.w_fork * d, np.float32), np.full(len(j), -args.w_fork * d, np.float32)]
            info = f", fork {F:.4f} (margin {mg.mean():+.2f})"
        lps, g = eng.state_grad(st, [0] * len(posts), posts, resps, ws)
        L = -sum(float(x.sum()) for x in lps[:len(exs)]) / n_tok
        gz = g[0].reshape(s0.shape) * scale[:, None, None]
        opt.update(gz)
        return f"train {L + args.w_fork * F:.5f} (NLL {L:.5f}{info}), |g| {np.linalg.norm(gz):.3e}"

    best, vl = evaluate(cur(), 0)
    print(f"step {0:5d}: val {vl:.5f} (init)", file=sys.stderr)
    save(args.out, cur(), {"step": 0, "val_loss": vl, "val_score": best})

    for step in range(1, args.steps + 1):
        t0 = time.time()
        if opt:
            info = grad_step()
        else:
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
            info = (f"train {L.mean():.5f} (population mean){fork_info}, |dL| {d0:.2e},"
                    f" mean |dL| {np.mean(np.abs(L[0::2] - L[1::2])):.2e}")
        drift = float(np.sqrt(np.mean(cur() ** 2)))
        print(f"step {step:5d}: {info}, drift {drift:.3f} head RMS, {time.time() - t0:.1f} s", file=sys.stderr)
        if step % args.eval_every == 0 or step == args.steps:
            score, vl = evaluate(cur(), step)
            meta = {"step": step, "val_loss": vl, "val_score": score}
            if score < best:
                best = score
                save(args.out, cur(), meta)
            save(args.out + ".last", cur(), meta)
            if args.keep_evals:
                save(f"{args.out}.step{step}", cur(), meta)
            print(f"step {step:5d}: val {vl:.5f}, score {score:.5f} (best {best:.5f})", file=sys.stderr)

if __name__ == "__main__":
    main()
