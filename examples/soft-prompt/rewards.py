"""Rewards for sampled responses: meta-talk and repetition in the thinking, answer correctness, length."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# recitation of a training-time instruction and checking compliance with it, as seen in EdgeRock-4B outputs
DEFAULT_PATTERNS = [
    r"meta[- ]?instruction",
    r"the user (also )?(gave|provided|included|added|wants|wrote) (a |an |the |some )?(meta|instruction|constraint|format|requirement)",
    r"treat every explicit",
    r"logically rigorous",
    r"logical leaps?",
    r"(we|i) (need to|must|should) (ensure|make sure|check) (that )?(we|i) (don't|do not|didn't|did not)",
    r"must not (repeat|introduce|reject|add|mention)",
    r"no (repetition|repeated reasoning|contradictions?|extraneous)",
    r"\b(constraints?|instructions?) (are|is|as) absolute",
    r"\\boxed",
    r"put (the |your )?final answer (in|within)",
    r"the (user'?s? )?(instructions?|constraints?|requirements?) (say|says|require|requires|demand|want)",
]


@dataclass
class RewardConfig:
    patterns: list[str] = field(default_factory=lambda: list(DEFAULT_PATTERNS))
    w_meta: float = 1.0      # fraction of thinking sentences matching a pattern
    w_any: float = 0.5       # any match at all
    w_rep: float = 1.0       # fraction of thinking sentences repeated 3+ times
    w_loop: float = 1.0      # fraction of repeated 12-token n-grams in the thinking (token-level loops)
    w_correct: float = 1.0   # final answer matches the reference
    w_len: float = 0.0       # Kimi-style group length reward, correct answers only
    overlong: int = 0        # DAPO soft overlong penalty over the last N tokens before the budget (0: off)
    answer_mode: str = "final"  # final: answer after </think>; prefix: the response starts with the answer (continuation)

    def __post_init__(self):
        self._re = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    @staticmethod
    def load_patterns(path: str) -> list[str]:
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?\n])", text) if re.search(r"\w", s)]


def thinking(text: str) -> tuple[str, str | None]:
    """(thinking, answer part) - the answer part is None when </think> never came."""
    i = text.find("</think>")
    return (text, None) if i < 0 else (text[:i], text[i + len("</think>"):])


def extract_answer(text: str) -> str:
    """Last \\boxed{...}, else "answer is X" / "Answer: X", else the last non-empty line."""
    i = text.rfind("\\boxed{")
    if i >= 0:
        depth, out = 0, []
        for ch in text[i + 7:]:
            if ch == "{":
                depth += 1
            elif ch == "}":
                if depth == 0:
                    return "".join(out)
                depth -= 1
            out.append(ch)
    m = re.findall(r"(?:answer is|answer:)\s*\**\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        return m[-1]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())


def answer_matches(got: str, ref: str) -> bool:
    g, r = _norm(got), _norm(ref)
    if not r:
        return False
    if g == r:
        return True
    # a single-letter reference (multiple choice) matches a leading letter: "B", "B.", "(B) ..."
    if len(r) == 1 and r.isalpha():
        m = re.match(r"^\W*\(?([A-Za-z])\)?(\W|$)", got.strip())
        return bool(m) and m.group(1).lower() == r
    return False


def loop_fraction(toks, n: int = 12) -> float:
    """Share of token n-grams that already occurred: ~0 for normal text, close to 1 for a loop."""
    toks = list(toks)
    if len(toks) < 2 * n:
        return 0.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def starts_with_answer(text: str, ref: str) -> bool:
    """Continuation: the text starts with the reference, followed by a non-alphanumeric character or the end."""
    t, r = text.lstrip(), ref.strip()
    return t.startswith(r) and (len(t) == len(r) or not t[len(r)].isalnum())


def score(text: str, n_tokens: int, max_new: int, answer: str | None, cfg: RewardConfig, think_toks=None) -> dict:
    """Reward terms of one sample; the group length term and the total are added by finish()."""
    think, rest = thinking(text)
    sents = split_sentences(think)
    n_meta = sum(1 for s in sents if any(r.search(s) for r in cfg._re))
    counts = Counter(k for k in (_norm(s) for s in sents) if len(k) >= 16)
    n_rep = sum(c for c in counts.values() if c >= 3)
    out = {
        "meta_frac": n_meta / len(sents) if sents else 0.0,
        "meta_any": n_meta > 0,
        "rep_frac": n_rep / len(sents) if sents else 0.0,
        "loop_frac": loop_fraction(think_toks) if think_toks is not None else 0.0,
        "closed": rest is not None,
        "correct": None,
        "r_len": 0.0,
        "r_overlong": 0.0,
    }
    if answer:
        if cfg.answer_mode == "prefix":
            out["correct"] = starts_with_answer(text, answer)
        else:
            out["correct"] = rest is not None and answer_matches(extract_answer(rest), answer)
    if cfg.overlong > 0 and n_tokens > max_new - cfg.overlong:
        out["r_overlong"] = (max_new - cfg.overlong - n_tokens) / cfg.overlong
    return out


def group_length_reward(group: list[dict], lengths: list[int]):
    """Kimi-style: shorter is better within the group; wrong answers never gain from it."""
    lo, hi = min(lengths), max(lengths)
    for r, n in zip(group, lengths):
        lam = 0.5 - (n - lo) / (hi - lo) if hi > lo else 0.0
        r["r_len"] = lam if r["correct"] else min(0.0, lam)


def finish(r: dict, cfg: RewardConfig, extra: float = 0.0) -> float:
    r["reward"] = (-cfg.w_meta * r["meta_frac"] - cfg.w_any * float(r["meta_any"]) - cfg.w_rep * r["rep_frac"] - cfg.w_loop * r["loop_frac"]
                   + (cfg.w_correct * float(r["correct"]) if r["correct"] is not None else 0.0)
                   + cfg.w_len * r["r_len"] + r["r_overlong"] + extra)
    return r["reward"]


def add_reward_args(ap):
    g = ap.add_argument_group("reward")
    g.add_argument("--patterns", help="meta-talk regexes, one per line, case-insensitive (default: built-in list)")
    g.add_argument("--w-meta", type=float, default=1.0)
    g.add_argument("--w-any", type=float, default=0.5)
    g.add_argument("--w-rep", type=float, default=1.0)
    g.add_argument("--w-loop", type=float, default=1.0, help="token-level loops (repeated 12-grams) in the thinking")
    g.add_argument("--w-correct", type=float, default=1.0)
    g.add_argument("--w-len", type=float, default=0.0, help="group length reward for correct answers")
    g.add_argument("--overlong", type=int, default=0, help="DAPO soft overlong penalty over the last N tokens (0: off)")
    g.add_argument("--answer-mode", choices=["final", "prefix"], default="final",
                   help="final: the answer after </think>; prefix: the response starts with the answer (continuation)")


def config_from_args(args) -> RewardConfig:
    pats = RewardConfig.load_patterns(args.patterns) if args.patterns else list(DEFAULT_PATTERNS)
    return RewardConfig(pats, args.w_meta, args.w_any, args.w_rep, args.w_loop, args.w_correct, args.w_len, args.overlong,
                        args.answer_mode)
