"""Antithetic evolution strategies with Adam, on a soft prompt [K, n_embd] (or a stack of matrices [..., K, D])."""

from __future__ import annotations

import numpy as np


class ES:
    def __init__(self, theta: np.ndarray, sigma: float, lr: float, pairs: int, rank: int = 0, shaping: str = "none",
                 seed: int = 0, beta1: float = 0.9, beta2: float = 0.999):
        """sigma and lr are absolute; scale them by the RMS of the init embeddings."""
        self.theta = np.array(theta, dtype=np.float32)
        self.sigma, self.lr, self.pairs, self.rank, self.shaping = sigma, lr, pairs, rank, shaping
        self.beta1, self.beta2 = beta1, beta2
        self.m = np.zeros_like(self.theta)
        self.v = np.zeros_like(self.theta)
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self.eps = None

    def population(self) -> np.ndarray:
        """[2*pairs, ..., K, D]: theta + sigma*eps_i, theta - sigma*eps_i for every pair."""
        shape = self.theta.shape
        if self.rank <= 0:
            self.eps = self.rng.standard_normal((self.pairs,) + shape, dtype=np.float32)
        else:
            # EGGROLL-style low-rank noise A B^T / sqrt(r) per matrix, unit variance per element
            a = self.rng.standard_normal((self.pairs,) + shape[:-1] + (self.rank,), dtype=np.float32)
            b = self.rng.standard_normal((self.pairs,) + shape[:-2] + (shape[-1], self.rank), dtype=np.float32)
            self.eps = np.einsum("p...kr,p...dr->p...kd", a, b) / np.float32(np.sqrt(self.rank))
        pop = np.empty((2 * self.pairs,) + shape, dtype=np.float32)
        pop[0::2] = self.theta + self.sigma * self.eps
        pop[1::2] = self.theta - self.sigma * self.eps
        return pop

    def update(self, losses: np.ndarray) -> float:
        """Gradient from the losses of population() (lower is better), then an Adam step; returns |dL| of pair 0."""
        L = np.asarray(losses, dtype=np.float64)
        w = L[0::2] - L[1::2]
        d0 = abs(float(w[0]))
        if self.shaping == "rank" and self.pairs > 1:
            # centered ranks of the differences, robust to outliers
            w = np.argsort(np.argsort(w)) / (self.pairs - 1) - 0.5
        # Adam makes the overall scale irrelevant, only the weights of the pairs matter
        g = np.tensordot(w / (2 * self.sigma * self.pairs), self.eps, axes=1).astype(np.float32)
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g * g
        mhat = self.m / (1 - self.beta1 ** self.t)
        vhat = self.v / (1 - self.beta2 ** self.t)
        self.theta -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)
        return d0


def score_population(eng, pop: np.ndarray, idx, posts, resps, *args, **kw) -> list[np.ndarray]:
    """eng.score for the members of population(): job k runs member idx[k], members 2p / 2p+1 must have the same jobs
    in the same order. Every round of n_seq jobs of + members is followed by the - members in the same sequence slots:
    with quantized weights a job's result depends on its slot (batch layout), and this way that cancels within a pair."""
    idx = np.asarray(idx)
    plus, minus = np.flatnonzero(idx % 2 == 0), np.flatnonzero(idx % 2 == 1)
    assert len(plus) == len(minus)
    order = np.concatenate([np.concatenate([plus[k:k + eng.n_seq], minus[k:k + eng.n_seq]])
                            for k in range(0, len(plus), eng.n_seq)]) if len(plus) else plus
    lp = eng.score(pop, idx[order], [posts[k] for k in order], [resps[k] for k in order], *args, **kw)
    out = [None] * len(idx)
    for k, x in zip(order, lp):
        out[k] = x
    return out


def add_es_args(ap, steps: int = 200):
    g = ap.add_argument_group("ES + Adam")
    g.add_argument("--pairs", type=int, default=8, help="antithetic pairs, population = 2N")
    g.add_argument("--rank", type=int, default=0, help="low-rank noise rank, 0 = full rank")
    g.add_argument("--sigma", type=float, default=0.05, help="noise scale relative to the init embedding RMS")
    g.add_argument("--lr", type=float, default=0.01, help="Adam step relative to the init embedding RMS")
    g.add_argument("--shaping", choices=["none", "rank"], default="none")
    g.add_argument("--steps", type=int, default=steps)
    g.add_argument("--eval-every", type=int, default=25)
    g.add_argument("-o", "--out", default="soft-prompt.gguf")


def from_args(args, theta: np.ndarray) -> tuple[ES, float]:
    rms = float(np.sqrt(np.mean(np.square(theta, dtype=np.float64))))
    return ES(theta, args.sigma * rms, args.lr * rms, args.pairs, args.rank, args.shaping, args.seed + 1), rms
