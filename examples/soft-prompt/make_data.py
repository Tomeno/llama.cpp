#!/usr/bin/env python3
"""Generates small synthetic datasets for the soft prompt scripts.

    # continuation (no chat template): {"prompt": "37*48=", "answer": "1776", "response": "1776"}
    python make_data.py arith --n 256 -o arith_train.jsonl --seed 0
    python make_data.py arith --n 32  -o arith_val.jsonl   --seed 1 --exclude arith_train.jsonl

    # simple chat questions with short checkable answers, for the meta-talk stage (use --answer-mode contains)
    python make_data.py simple --n 256 -o simple_train.jsonl --seed 0
    python make_data.py simple --n 32  -o simple_val.jsonl   --seed 1 --exclude simple_train.jsonl

Use the continuation data with --before "" --after "" --example "{prompt}" and --answer-mode prefix.
"""

from __future__ import annotations

import argparse
import json
import random

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid", "Japan": "Tokyo", "Canada": "Ottawa",
    "Australia": "Canberra", "Egypt": "Cairo", "Kenya": "Nairobi", "Peru": "Lima", "Norway": "Oslo", "Poland": "Warsaw",
    "Greece": "Athens", "Portugal": "Lisbon", "Austria": "Vienna", "Ireland": "Dublin", "Chile": "Santiago",
    "Argentina": "Buenos Aires", "Thailand": "Bangkok", "Vietnam": "Hanoi", "Sweden": "Stockholm", "Finland": "Helsinki",
    "Hungary": "Budapest", "Czechia": "Prague", "Turkey": "Ankara", "South Korea": "Seoul", "Mexico": "Mexico City",
    "Brazil": "Brasilia", "India": "New Delhi", "China": "Beijing", "Russia": "Moscow", "Cuba": "Havana",
}
WORDS = ["apple", "river", "garden", "window", "planet", "silver", "pencil", "rocket", "forest", "candle", "bridge",
         "orange", "monkey", "letter", "castle", "winter", "button", "dragon", "island", "marble", "tunnel", "violin"]
COLORS = {"banana": "yellow", "grass": "green", "snow": "white", "coal": "black", "a ripe tomato": "red", "the clear sky": "blue"}


def arith(rng: random.Random) -> dict:
    op = rng.choice(["*", "*", "+", "-"])
    if op == "*":
        a, b = rng.randint(12, 99), rng.randint(12, 99)
        v = a * b
    elif op == "+":
        a, b = rng.randint(100, 999), rng.randint(100, 999)
        v = a + b
    else:
        a, b = rng.randint(100, 999), rng.randint(10, 99)
        v = a - b
    p = f"{a}{op}{b}="
    return {"prompt": p, "answer": str(v), "response": str(v)}


def simple(rng: random.Random) -> dict:
    k = rng.randrange(6)
    if k == 0:
        a, b = rng.randint(2, 50), rng.randint(2, 50)
        return {"prompt": f"What is {a} + {b}?", "answer": str(a + b)}
    if k == 1:
        a, b = rng.randint(2, 12), rng.randint(2, 12)
        return {"prompt": f"What is {a} times {b}?", "answer": str(a * b)}
    if k == 2:
        c = rng.choice(sorted(CAPITALS))
        return {"prompt": f"What is the capital of {c}?", "answer": CAPITALS[c]}
    if k == 3:
        w = rng.choice(WORDS)
        return {"prompt": f"How many letters are in the word \"{w}\"?", "answer": str(len(w))}
    if k == 4:
        w = rng.choice(WORDS)
        return {"prompt": f"Write the word \"{w}\" backwards.", "answer": w[::-1]}
    t = rng.choice(sorted(COLORS))
    return {"prompt": f"What color is {t} usually? Answer with one word.", "answer": COLORS[t]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["arith", "simple"])
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude", help="JSONL whose prompts are not repeated (e.g. the training set)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seen = set()
    if args.exclude:
        with open(args.exclude) as f:
            seen = {json.loads(line)["prompt"] for line in f if line.strip()}
    gen = arith if args.kind == "arith" else simple
    rows = []
    for _ in range(args.n * 100):
        if len(rows) == args.n:
            break
        r = gen(rng)
        if args.kind == "simple" or r["prompt"] not in seen:  # simple prompts repeat by design (few templates)
            seen.add(r["prompt"])
            rows.append(r)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
