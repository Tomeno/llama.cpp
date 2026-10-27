"""Sampling groups of responses with a soft prompt and scoring them."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from collections import defaultdict

import numpy as np

import rewards
from softprompt import Engine, Template


class Prompts:
    """Prompts ({"prompt", optional "answer"}) with their tokenized post part."""

    def __init__(self, eng: Engine, tpl: Template, rows: list[dict]):
        self.rows = rows
        self.posts = [tpl.post(eng, r["prompt"]) for r in rows]

    def __len__(self):
        return len(self.rows)


def think_tokens(eng: Engine, toks: np.ndarray) -> int:
    """Tokens before </think> (all tokens when it never closes)."""
    close = eng.tokenize("</think>")
    if len(close) == 1:
        hit = np.flatnonzero(toks == close[0])
        return int(hit[0]) if len(hit) else len(toks)
    text = eng.detokenize(toks)
    i = text.find("</think>")
    return len(toks) if i < 0 else int(round(len(toks) * i / max(1, len(text))))


def rollout(eng: Engine, theta: np.ndarray, prompts: Prompts, ids: list[int], max_new: int, cfg: rewards.RewardConfig,
            sampling: dict, seed: int, reward_cmd: str | None = None, state: np.ndarray | None = None) -> list[dict]:
    """One sample per entry of ids (indices into prompts), with rewards; samples of the same prompt form a group.
    state: the recurrent state after the prefix (initial state tuning), None: the one the soft prompt leaves."""
    gen = eng.generate(theta, [prompts.posts[i] for i in ids], max_new, seed=seed, state=state, **sampling)
    samples = []
    for pi, (toks, fin) in zip(ids, gen):
        text = eng.detokenize(toks)
        n_think = think_tokens(eng, toks)
        s = {"prompt": pi, "toks": toks, "text": text, "finished": fin, "tokens": len(toks), "think_tokens": n_think}
        s.update(rewards.score(text, len(toks), max_new, prompts.rows[pi].get("answer"), cfg, toks[:n_think]))
        samples.append(s)
    if cfg.w_len:
        groups = defaultdict(list)
        for s in samples:
            groups[s["prompt"]].append(s)
        for g in groups.values():
            rewards.group_length_reward(g, [s["tokens"] for s in g])
    extra = external_reward(samples, prompts, reward_cmd) if reward_cmd else [0.0] * len(samples)
    for s, e in zip(samples, extra):
        s["r_cmd"] = e
        rewards.finish(s, cfg, e)
    return samples


def external_reward(samples: list[dict], prompts: Prompts, cmd: str) -> list[float]:
    """Runs CMD IN.jsonl OUT.txt: one JSON sample per line in, one float per line out."""
    with tempfile.TemporaryDirectory() as d:
        fin, fout = os.path.join(d, "in.jsonl"), os.path.join(d, "out.txt")
        with open(fin, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(sample_row(s, prompts), ensure_ascii=False) + "\n")
        subprocess.run(f"{cmd} {fin} {fout}", shell=True, check=True)
        with open(fout) as f:
            vals = [float(x) for x in f.read().split()]
    if len(vals) != len(samples):
        raise RuntimeError(f"--reward-cmd wrote {len(vals)} values for {len(samples)} samples")
    return vals


def sample_row(s: dict, prompts: Prompts) -> dict:
    row = prompts.rows[s["prompt"]]
    return {"prompt": row["prompt"], "answer": row.get("answer"), "response": s["text"], "finished": s["finished"],
            "tokens": s["tokens"], "think_tokens": s["think_tokens"], "meta_frac": s["meta_frac"], "meta_any": s["meta_any"],
            "rep_frac": s["rep_frac"], "loop_frac": s["loop_frac"], "r_short": s["r_short"], "correct": s["correct"],
            "reward": s.get("reward")}


def summary(samples: list[dict]) -> str:
    n = max(1, len(samples))
    m = lambda k: sum(float(s[k]) for s in samples) / n  # noqa: E731
    out = (f"reward {m('reward'):.4f} | meta-talk {100 * m('meta_any'):.1f}% ({m('meta_frac'):.3f} of sentences)"
           f" | repetition {100 * sum(s['rep_frac'] > 0 for s in samples) / n:.1f}% | loops {m('loop_frac'):.3f}"
           f" | think {m('think_tokens'):.0f} / {m('tokens'):.0f} tokens | finished {100 * m('finished'):.1f}%"
           f" | short thinking {100 * sum(s['r_short'] < 0 for s in samples) / n:.1f}%")
    ref = [s for s in samples if s["correct"] is not None]
    if ref:
        out += f" | correct {100 * sum(s['correct'] for s in ref) / len(ref):.1f}% ({len(ref)})"
    return out


def add_sampling_args(ap, gen_max: int = 512):
    g = ap.add_argument_group("sampling")
    g.add_argument("--group", type=int, default=8, help="samples per prompt")
    g.add_argument("--gen-max", type=int, default=gen_max, help="max new tokens")
    g.add_argument("--temp", type=float, default=0.6)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--reward-cmd", help="external reward, run as CMD IN.jsonl OUT.txt (one float per line), added to the reward")
    g.add_argument("--samples-out", help="write the samples as JSONL")


def sampling_from_args(args) -> dict:
    return {"temp": args.temp, "top_p": args.top_p, "top_k": args.top_k}
