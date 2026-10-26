"""Python side of soft prompt tuning: ctypes bindings of libllama-soft-prompt, the prompt template and soft prompt files.

A soft prompt is K free input embeddings placed in front of every request:

    [before] [K soft embeddings] [after + formatted prompt] [response]

The library does the performance-critical parts (batched sampling and teacher-forced scoring, one soft prompt per
sequence); everything else (rewards, ES, training loops) is plain Python in this directory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO / "gguf-py") not in sys.path:
    sys.path.insert(0, str(_REPO / "gguf-py"))
import gguf  # noqa: E402

_i32p = ctypes.POINTER(ctypes.c_int32)
_i64p = ctypes.POINTER(ctypes.c_int64)
_f32p = ctypes.POINTER(ctypes.c_float)


def _find_lib() -> str:
    if "SOFTPROMPT_LIB" in os.environ:
        return os.environ["SOFTPROMPT_LIB"]
    name = {"darwin": "libllama-soft-prompt.dylib", "win32": "llama-soft-prompt.dll"}.get(sys.platform, "libllama-soft-prompt.so")
    for d in (_REPO / "build" / "bin", _REPO / "build" / "lib", Path.cwd()):
        if (d / name).exists():
            return str(d / name)
    raise FileNotFoundError(f"{name} not found; build the llama-soft-prompt target or set SOFTPROMPT_LIB")


def _load_lib(path: str) -> ctypes.CDLL:
    lib = ctypes.CDLL(path)
    lib.sp_last_error.restype = ctypes.c_char_p
    lib.sp_init.argtypes = [ctypes.c_char_p] + [ctypes.c_int32] * 6
    lib.sp_init.restype = ctypes.c_void_p
    lib.sp_free.argtypes = [ctypes.c_void_p]
    for f in ("sp_n_embd", "sp_n_vocab", "sp_n_seq"):
        getattr(lib, f).argtypes = [ctypes.c_void_p]
        getattr(lib, f).restype = ctypes.c_int32
    lib.sp_is_eog.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.sp_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, _i32p, ctypes.c_int32]
    lib.sp_detokenize.argtypes = [ctypes.c_void_p, _i32p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
    lib.sp_set_before.argtypes = [ctypes.c_void_p, _i32p, ctypes.c_int32]
    lib.sp_token_embeddings.argtypes = [ctypes.c_void_p, _i32p, ctypes.c_int32, _f32p]
    lib.sp_score.argtypes = [ctypes.c_void_p, _f32p, ctypes.c_int32, _i32p, ctypes.c_int32,
                             _i32p, _i64p, _i32p, _i32p, _i64p, _i32p, ctypes.c_float, _f32p]
    lib.sp_generate.argtypes = [ctypes.c_void_p, _f32p, ctypes.c_int32, ctypes.c_int32,
                                _i32p, _i64p, _i32p, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
                                ctypes.c_uint64, _i32p, _i32p, _i32p]
    for f in ("sp_score", "sp_generate", "sp_token_embeddings"):
        getattr(lib, f).restype = ctypes.c_int32
    return lib


def _ptr(a: np.ndarray, t):
    return a.ctypes.data_as(t)


def _pack(seqs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lens = np.array([len(s) for s in seqs], dtype=np.int32)
    offs = np.zeros(len(seqs), dtype=np.int64)
    if len(seqs) > 1:
        offs[1:] = np.cumsum(lens[:-1], dtype=np.int64)
    flat = np.concatenate([np.asarray(s, dtype=np.int32) for s in seqs]) if seqs else np.zeros(0, np.int32)
    return np.ascontiguousarray(flat, dtype=np.int32), offs, lens


class Engine:
    """One model and one context with n_seq parallel sequences (each holds a full sequence state)."""

    def __init__(self, model: str, n_gpu_layers: int = 999, n_ctx: int = 32768, n_batch: int = 2048, n_seq: int = 32,
                 n_threads: int = 0, max_outputs: int = 512, lib: str | None = None):
        self.lib = _load_lib(lib or _find_lib())
        self.ctx = self.lib.sp_init(model.encode(), n_gpu_layers, n_ctx, n_batch, n_seq, n_threads, max_outputs)
        if not self.ctx:
            raise RuntimeError(self.lib.sp_last_error().decode())
        self.n_embd  = self.lib.sp_n_embd(self.ctx)
        self.n_vocab = self.lib.sp_n_vocab(self.ctx)
        self.n_seq   = self.lib.sp_n_seq(self.ctx)

    def close(self):
        if self.ctx:
            self.lib.sp_free(self.ctx)
            self.ctx = None

    def __del__(self):
        self.close()

    def _check(self, ret: int):
        if ret != 0:
            raise RuntimeError(self.lib.sp_last_error().decode())

    def tokenize(self, text: str) -> np.ndarray:
        b = text.encode()
        out = np.zeros(len(b) + 16, dtype=np.int32)
        n = self.lib.sp_tokenize(self.ctx, b, len(b), _ptr(out, _i32p), len(out))
        if n < 0:
            out = np.zeros(-n, dtype=np.int32)
            n = self.lib.sp_tokenize(self.ctx, b, len(b), _ptr(out, _i32p), len(out))
        return out[:n].copy()

    def detokenize(self, toks: Sequence[int]) -> str:
        t = np.ascontiguousarray(toks, dtype=np.int32)
        cap = 64 * len(t) + 16
        buf = ctypes.create_string_buffer(cap)
        n = self.lib.sp_detokenize(self.ctx, _ptr(t, _i32p), len(t), buf, cap)
        if n < 0:
            buf = ctypes.create_string_buffer(-n)
            n = self.lib.sp_detokenize(self.ctx, _ptr(t, _i32p), len(t), buf, -n)
        return buf.raw[:n].decode(errors="replace")

    def is_eog(self, tok: int) -> bool:
        return bool(self.lib.sp_is_eog(self.ctx, int(tok)))

    def set_before(self, toks: Sequence[int]):
        t = np.ascontiguousarray(toks, dtype=np.int32)
        self.lib.sp_set_before(self.ctx, _ptr(t, _i32p), len(t))

    def token_embeddings(self, ids: Sequence[int]) -> np.ndarray:
        t = np.ascontiguousarray(ids, dtype=np.int32)
        out = np.zeros((len(t), self.n_embd), dtype=np.float32)
        self._check(self.lib.sp_token_embeddings(self.ctx, _ptr(t, _i32p), len(t), _ptr(out, _f32p)))
        return out

    def score(self, thetas: np.ndarray, theta_idx: Sequence[int], posts: Sequence[np.ndarray], resps: Sequence[np.ndarray],
              temp: float = 1.0) -> list[np.ndarray]:
        """Teacher-forced log-probs of resps[j] after posts[j] with the soft prompt thetas[theta_idx[j]].

        thetas: [n_theta, K, n_embd] float32; returns one array of len(resps[j]) log-probs per job."""
        th = np.ascontiguousarray(thetas, dtype=np.float32)
        if th.ndim == 2:
            th = th[None]
        n_soft = th.shape[1]
        idx = np.ascontiguousarray(theta_idx, dtype=np.int32)
        pf, po, pl = _pack(posts)
        rf, ro, rl = _pack(resps)
        out = np.zeros(len(rf), dtype=np.float32)
        self._check(self.lib.sp_score(self.ctx, _ptr(th, _f32p), n_soft, _ptr(idx, _i32p), len(idx),
                                      _ptr(pf, _i32p), _ptr(po, _i64p), _ptr(pl, _i32p),
                                      _ptr(rf, _i32p), _ptr(ro, _i64p), _ptr(rl, _i32p), float(temp), _ptr(out, _f32p)))
        return [out[o:o + n] for o, n in zip(ro, rl)]

    def generate(self, theta: np.ndarray, posts: Sequence[np.ndarray], max_new: int, temp: float = 0.6, top_p: float = 0.95,
                 top_k: int = 20, seed: int = 0) -> list[tuple[np.ndarray, bool]]:
        """Samples a response for every prompt; returns (tokens, finished) pairs (temp <= 0: greedy)."""
        th = np.ascontiguousarray(theta, dtype=np.float32)
        n_soft = th.shape[0] if th.size else 0
        pf, po, pl = _pack(posts)
        n = len(posts)
        toks = np.zeros((n, max_new), dtype=np.int32)
        lens = np.zeros(n, dtype=np.int32)
        fin = np.zeros(n, dtype=np.int32)
        self._check(self.lib.sp_generate(self.ctx, _ptr(th, _f32p), n_soft, n, _ptr(pf, _i32p), _ptr(po, _i64p), _ptr(pl, _i32p),
                                         max_new, float(temp), float(top_p), int(top_k), int(seed) & (2**64 - 1),
                                         _ptr(toks, _i32p), _ptr(lens, _i32p), _ptr(fin, _i32p)))
        return [(toks[i, :lens[i]].copy(), bool(fin[i])) for i in range(n)]


#
# template and soft prompt files
#

@dataclass
class Template:
    # the defaults make the soft prompt a ChatML system turn and open the assistant turn with <think> (EdgeRock)
    before:  str = "<|im_start|>system\n"
    after:   str = "<|im_end|>\n"
    example: str = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n"

    def post(self, eng: Engine, prompt: str) -> np.ndarray:
        return eng.tokenize(self.after + self.example.replace("{prompt}", prompt))


def save_soft_prompt(path: str, theta: np.ndarray, tpl: Template, meta: dict | None = None):
    w = gguf.GGUFWriter(path, "soft_prompt")
    w.add_string("soft_prompt.before", tpl.before)
    w.add_string("soft_prompt.after", tpl.after)
    w.add_string("soft_prompt.example", tpl.example)
    w.add_string("soft_prompt.meta", json.dumps(meta or {}))
    w.add_tensor("soft_prompt.embd", np.ascontiguousarray(theta, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def load_soft_prompt(path: str) -> tuple[np.ndarray, Template, dict]:
    r = gguf.GGUFReader(path)

    def s(key: str) -> str:
        f = r.fields[key]
        return bytes(f.parts[f.data[0]]).decode()

    tpl = Template(s("soft_prompt.before"), s("soft_prompt.after"), s("soft_prompt.example"))
    meta = json.loads(s("soft_prompt.meta")) if "soft_prompt.meta" in r.fields else {}
    t = next(t for t in r.tensors if t.name == "soft_prompt.embd")
    theta = np.array(t.data, dtype=np.float32).reshape(int(t.shape[1]), int(t.shape[0]))
    return theta, tpl, meta


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict], append: bool = False):
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


#
# shared command line handling
#

def add_common_args(ap: argparse.ArgumentParser):
    g = ap.add_argument_group("model and runtime")
    g.add_argument("-m", "--model", required=True)
    g.add_argument("-ngl", "--n-gpu-layers", type=int, default=999)
    g.add_argument("-c", "--ctx", type=int, default=32768, help="context size (per sequence for hybrid/recurrent models)")
    g.add_argument("-b", "--batch", type=int, default=2048)
    g.add_argument("--par", type=int, default=32, help="parallel sequences; each holds a full sequence state")
    g.add_argument("-t", "--threads", type=int, default=0)
    g.add_argument("--max-outputs", type=int, default=512, help="logits rows per decode call (rows x n_vocab x 4 bytes)")
    g = ap.add_argument_group("soft prompt")
    g.add_argument("--init-text", default="You are a helpful assistant.", help="start from the token embeddings of this text")
    g.add_argument("--init-file", help="start from a saved soft prompt (overrides --init-text and the template)")
    g.add_argument("--n-soft", type=int, default=0, help="soft tokens; the init rows repeat cyclically (default: init length)")
    g.add_argument("--before", default=Template.before)
    g.add_argument("--after", default=Template.after)
    g.add_argument("--example", default=Template.example, help="prompt format, {prompt} is replaced")
    g.add_argument("--seed", type=int, default=1234)


def setup(args, n_seq: int | None = None) -> tuple[Engine, np.ndarray, Template]:
    """Engine, initial soft prompt [K, n_embd] and template from the common arguments."""
    eng = Engine(args.model, args.n_gpu_layers, args.ctx, args.batch, n_seq or args.par, args.threads, args.max_outputs)
    if args.init_file:
        theta, tpl, _ = load_soft_prompt(args.init_file)
        print(f"loaded {theta.shape[0]} soft tokens from {args.init_file}", file=sys.stderr)
    else:
        tpl = Template(args.before, args.after, args.example)
        ids = eng.tokenize(args.init_text)
        theta = eng.token_embeddings(ids) if len(ids) else np.zeros((0, eng.n_embd), np.float32)
        print(f"init from {len(ids)} tokens of {args.init_text!r}", file=sys.stderr)
    if args.n_soft > 0 and len(theta) and args.n_soft != len(theta):
        theta = theta[np.arange(args.n_soft) % len(theta)].copy()
    eng.set_before(eng.tokenize(tpl.before))
    return eng, theta, tpl
