"""Rewards for sampled responses: meta-talk and repetition in the thinking, answer correctness, length."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction

import ifeval

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
    answer_mode: str = "final"  # final: extracted answer after </think> equals the reference; contains: the reference appears
                                # as a word after </think>; prefix: the response starts with the reference (continuation)
    min_think: int = 0       # thinking shorter than this many tokens is penalized, linearly (0: off)
    w_short: float = 1.0     # empty thinking scores 0 on every other term, so this keeps "stop thinking" from winning
    task_patterns: dict[str, list[str]] = field(default_factory=dict)  # task family -> patterns, instead of patterns

    def __post_init__(self):
        self._re = [re.compile(p, re.IGNORECASE) for p in self.patterns]
        self._task_re = {t: [re.compile(p, re.IGNORECASE) for p in ps] for t, ps in self.task_patterns.items()}

    def regexes(self, task: str | None) -> list:
        """The meta-talk regexes of a task ("gsm8k", "ifeval:validate_end": its family before the colon)"""
        return self._task_re.get((task or "").split(":")[0], self._re)

    @staticmethod
    def load_patterns(path: str) -> list[str]:
        with open(path, encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip() and not line.startswith("#")]


def split_sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?\n])", text) if re.search(r"\w", s)]


def thinking(text: str) -> tuple[str, str | None]:
    """(thinking, answer part without the end of turn token) - the answer part is None when </think> never came."""
    i = text.find("</think>")
    return (text, None) if i < 0 else (text[:i], re.sub(r"<\|im_end\|>\s*$", "", text[i + len("</think>"):]))


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


# numbers, as in answers: 1,250.00 / -3 / \frac{3}{4} / \dfrac34 / 3/4
_NUM = re.compile(r"(-)?\s*\\[dt]?frac\s*(?:\{\s*(-?\d+)\s*\}|(\d))\s*(?:\{\s*(-?\d+)\s*\}|(\d))"
                  r"|(-?\d[\d,]*(?:\.\d+)?)(?:\s*/\s*(\d+)(?![\d.]))?")


def _numbers(s: str) -> list[Fraction]:
    out = []
    for m in _NUM.finditer(s):
        neg, fa, fa1, fb, fb1, x, den = m.groups()
        try:
            if x is None:
                v = Fraction(int(fa or fa1), int(fb or fb1))
                out.append(-v if neg else v)
            else:
                v = Fraction(x.replace(",", ""))
                out.append(v / int(den) if den else v)
        except (ValueError, ZeroDivisionError):
            pass
    return out


def answer_matches(got: str, ref: str) -> bool:
    # a numeric reference matches the last number of the answer: "$1,250.00", "x = 18", "\frac{3}{4}"
    r_num = _numbers(ref)
    if len(r_num) == 1 and re.fullmatch(r"[\s$\\{}()dfrac0-9.,/-]*", ref):
        nums = _numbers(got)
        return bool(nums) and nums[-1] == r_num[0]
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


def contains_answer(text: str, ref: str) -> bool:
    """The reference appears as a whole word (case-insensitive) in the text."""
    return re.search(r"(?<![0-9A-Za-z])" + re.escape(ref.strip()) + r"(?![0-9A-Za-z])", text, re.IGNORECASE) is not None


def score(text: str, n_tokens: int, max_new: int, answer: str | None, cfg: RewardConfig, think_toks=None,
          task: str | None = None) -> dict:
    """Reward terms of one sample; the group length term and the total are added by finish()."""
    think, rest = thinking(text)
    sents = split_sentences(think)
    rx = cfg.regexes(task)
    n_meta = sum(1 for s in sents if any(r.search(s) for r in rx))
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
        "r_short": 0.0,
    }
    if cfg.min_think > 0 and think_toks is not None and len(think_toks) < cfg.min_think:
        out["r_short"] = (len(think_toks) - cfg.min_think) / cfg.min_think
    if isinstance(answer, (dict, list)):  # IFEval constraints (all must hold), checked on the response after the thinking
        out["correct"] = rest is not None and all(ifeval.check(rest, a) for a in (answer if isinstance(answer, list) else [answer]))
    elif answer:
        if cfg.answer_mode == "prefix":
            out["correct"] = starts_with_answer(text, answer)
        elif cfg.answer_mode == "contains":
            out["correct"] = rest is not None and contains_answer(rest, answer)
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
                   + cfg.w_len * r["r_len"] + r["r_overlong"] + cfg.w_short * r["r_short"] + extra)
    return r["reward"]


def add_reward_args(ap):
    g = ap.add_argument_group("reward")
    g.add_argument("--patterns", help="meta-talk regexes, one per line, case-insensitive (default: built-in list)")
    g.add_argument("--task-patterns", action="append", default=[], metavar="TASK=FILE[,FILE]",
                   help="regexes for the prompts of a task family (the \"task\" field before the colon), instead of --patterns")
    g.add_argument("--w-meta", type=float, default=1.0)
    g.add_argument("--w-any", type=float, default=0.5)
    g.add_argument("--w-rep", type=float, default=1.0)
    g.add_argument("--w-loop", type=float, default=1.0, help="token-level loops (repeated 12-grams) in the thinking")
    g.add_argument("--w-correct", type=float, default=1.0)
    g.add_argument("--w-len", type=float, default=0.0, help="group length reward for correct answers")
    g.add_argument("--overlong", type=int, default=0, help="DAPO soft overlong penalty over the last N tokens (0: off)")
    g.add_argument("--min-think", type=int, default=0, help="penalize thinking shorter than N tokens, linearly (0: off)")
    g.add_argument("--w-short", type=float, default=1.0, help="weight of the --min-think penalty (-1 for empty thinking)")
    g.add_argument("--answer-mode", choices=["final", "contains", "prefix"], default="final",
                   help="final: the extracted answer after </think> equals the reference; contains: the reference appears as a"
                        " word after </think>; prefix: the response starts with the reference (continuation)")


def config_from_args(args) -> RewardConfig:
    pats = RewardConfig.load_patterns(args.patterns) if args.patterns else list(DEFAULT_PATTERNS)
    by_task = {}
    for spec in args.task_patterns:
        t, files = spec.split("=", 1)
        by_task[t] = [p for f in files.split(",") for p in RewardConfig.load_patterns(f)]
    return RewardConfig(pats, args.w_meta, args.w_any, args.w_rep, args.w_loop, args.w_correct, args.w_len, args.overlong,
                        args.answer_mode, args.min_think, args.w_short, by_task)
