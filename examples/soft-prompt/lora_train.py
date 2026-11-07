#!/usr/bin/env python3
"""LoRA fine-tuning (also NoRA, MiSS) of a GGUF model, quantized or not, with exact gradients from the state-gradient
mode (RWKV7, EdgeRock): the base weights stay frozen as they are (e.g. Q8_0 or Q4_K: QLoRA), the adapter is F32.

Start from an adapter made by lora_init.py (it records which factors are trained) or a saved one. Data: JSONL, one
example per line, {"prompt", "response"} (only the response is trained), {"text"} (all of it) or {"segments": [[text,
trained], ...]} (several trained spans, e.g. the assistant turns of an agent loop; the first segment must not be trained;
the segments are tokenized one by one, so split them where the tokenizer does not merge), formatted as the model expects. Every example is one sequence from the initial state (zero, or the frozen state of --state, e.g. a chat
state); --max-len cuts longer ones. Long examples: --bptt-chunk N backpropagates through time in chunks of N tokens
(exact, the memory of one chunk: --max-len is then limited by time only; --truncate: truncated BPTT instead).

The loss is the mean NLL per response token of a batch (--batch-examples); AdamW with warmup and cosine decay, global
gradient-norm clipping. One log line per step (stderr). Saves OUT (best validation loss, or the last without --val),
OUT.last and OUT.opt.npz (optimizer state, for --resume).

    lora_train.py -m MODEL.gguf --adapter INIT.gguf --train train.jsonl [--val val.jsonl] -o OUT.gguf --lr 1e-4 --steps 500
"""
from __future__ import annotations

import argparse, json, math, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import gguf
from softprompt import Engine, load_state, read_jsonl


def log(s: str):
    print(s, file=sys.stderr, flush=True)


def adapter_meta(path: str) -> dict:
    r = gguf.GGUFReader(path)

    def val(k, default=None):
        f = r.fields.get(k)
        if f is None:
            return default
        x = f.parts[f.data[0]]
        return bytes(x).decode() if f.types[0] == gguf.GGUFValueType.STRING else x.tolist()[0]
    return {"arch": val("general.architecture"), "alpha": float(val("adapter.lora.alpha", 0.0)),
            "method": val("adapter.training.method", "lora"), "train_a": bool(val("adapter.training.train_a", True)),
            "train_b": bool(val("adapter.training.train_b", True)), "rank": val("adapter.training.rank", 0),
            "meta": val("adapter.training.meta", "{}")}


def save_adapter(path: str, eng: Engine, meta: dict, info: dict):
    w = gguf.GGUFWriter(path, meta["arch"])
    w.add_type(gguf.GGUFType.ADAPTER)
    w.add_string(gguf.Keys.Adapter.TYPE, "lora")
    w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, meta["alpha"])
    w.add_string("adapter.training.method", meta["method"])
    w.add_bool("adapter.training.train_a", meta["train_a"])
    w.add_bool("adapter.training.train_b", meta["train_b"])
    if meta["rank"]:
        w.add_uint32("adapter.training.rank", int(meta["rank"]))
    w.add_string("adapter.training.meta", json.dumps(info))
    for i, n in enumerate(eng.lora_names):
        w.add_tensor(n, eng.lora_get(i))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def load_examples(eng: Engine, path: str, max_len: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(prompt tokens, response tokens, response mask: 1 = trained)"""
    out, cut = [], 0
    for r in read_jsonl(path):
        if "segments" in r:
            segs = r["segments"]
            if not segs or segs[0][1]:
                raise ValueError(f"{path}: the first segment must not be trained")
            post = eng.tokenize(segs[0][0])
            parts = [(eng.tokenize(t), bool(tr)) for t, tr in segs[1:] if t]
            resp = np.concatenate([p for p, _ in parts]) if parts else np.zeros(0, np.int32)
            mask = np.concatenate([np.full(len(p), float(tr), np.float32) for p, tr in parts]) if parts else np.zeros(0, np.float32)
        elif "text" in r:
            ids = eng.tokenize(r["text"])
            post, resp = ids[:1], ids[1:]
            mask = np.ones(len(resp), np.float32)
        else:
            post, resp = eng.tokenize(r["prompt"]), eng.tokenize(r["response"])
            mask = np.ones(len(resp), np.float32)
        if len(post) + len(resp) - 1 > max_len:
            cut += 1
            keep = max(0, max_len + 1 - len(post))
            resp, mask = resp[:keep], mask[:keep]
        if len(post) and len(resp) and mask.sum() > 0:
            out.append((post, resp, mask))
    log(f"{path}: {len(out)} examples ({cut} cut at {max_len} tokens), {int(sum(m.sum() for _, _, m in out))} trained tokens, "
        f"{sum(len(p) + len(r) for p, r, _ in out)} in all")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model", required=True)
    ap.add_argument("--adapter", required=True, help="initial adapter (lora_init.py or a saved one)")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--state", help="frozen initial state (a soft prompt file with a state, e.g. a chat state)")
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps (default: --epochs over the training set)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-examples", type=int, default=8, help="examples per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-5, help="NoRA 2e-5 (rwkv_lightning); MiSS and plain LoRA usually higher")
    ap.add_argument("--lr-final", type=float, default=0.0, help="end of the cosine decay (default: lr / 10)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--wd", type=float, default=0.0, help="AdamW weight decay")
    ap.add_argument("--clip", type=float, default=1.0, help="global gradient norm clip (0: off)")
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--max-len", type=int, default=2048, help="tokens per example (prompt + response)")
    ap.add_argument("--grad-seqs", type=int, default=4, help="examples per decode (similar lengths, padded)")
    ap.add_argument("--ckpt", type=int, default=4, help="gradient checkpointing: layers per segment (0: off)")
    ap.add_argument("--bptt-chunk", type=int, default=0,
                    help="tokens per chunk of backpropagation through time (0: every example in one pass); GPU memory "
                         "follows grad-seqs x chunk instead of grad-seqs x max-len")
    ap.add_argument("--truncate", action="store_true", help="with --bptt-chunk: truncated BPTT (no gradient across chunks)")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--n-val", type=int, default=0, help="validation examples used (0: all)")
    ap.add_argument("--save-every", type=int, default=0, help="also save OUT.stepN every N steps")
    ap.add_argument("--resume", action="store_true", help="continue from OUT.opt.npz (the adapter: --adapter OUT.last)")
    ap.add_argument("--train-a", type=int, choices=[0, 1], help="override the adapter's choice")
    ap.add_argument("--train-b", type=int, choices=[0, 1])
    ap.add_argument("-ngl", "--n-gpu-layers", type=int, default=999)
    ap.add_argument("-t", "--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    meta = adapter_meta(args.adapter)
    if args.train_a is not None:
        meta["train_a"] = bool(args.train_a)
    if args.train_b is not None:
        meta["train_b"] = bool(args.train_b)
    g = max(1, args.grad_seqs)
    n_batch = g * args.bptt_chunk if args.bptt_chunk > 0 else args.max_len + 16
    eng = Engine(args.model, args.n_gpu_layers, g * (args.max_len + 16), n_batch, g, args.threads)
    eng.lib.sp_set_grad_ckpt(eng.ctx, args.ckpt)
    eng.lib.sp_set_grad_seqs(eng.ctx, g)
    eng.set_grad_chunk(args.bptt_chunk, not args.truncate)
    names = eng.lora_load(args.adapter, 1.0, meta["train_a"], meta["train_b"])
    trained = [i for i, n in enumerate(names) if (meta["train_a"] if n.endswith(".lora_a") else meta["train_b"])]
    n_par = sum(int(np.prod(eng.lora_shapes[i])) for i in trained)
    log(f"adapter {args.adapter}: {meta['method']}, alpha {meta['alpha']:g}, {len(names) // 2} weights, "
        f"trained {'A' if meta['train_a'] else ''}{'B' if meta['train_b'] else ''}: {n_par / 1e6:.2f} M parameters")

    if args.bptt_chunk > 0:
        log(f"BPTT in chunks of {args.bptt_chunk} tokens ({'truncated' if args.truncate else 'exact'})")
    st = load_state(args.state)
    st = np.zeros((1, eng.n_state), np.float32) if st is None else st.reshape(1, -1)
    train = load_examples(eng, args.train, args.max_len)
    val = load_examples(eng, args.val, args.max_len) if args.val else []
    if args.n_val:
        val = val[:args.n_val]
    bs = args.batch_examples
    steps = args.steps or max(1, math.ceil(args.epochs * len(train) / bs))
    lr_final = args.lr_final if args.lr_final > 0 else args.lr / 10

    def lr_at(s: int) -> float:
        if s <= args.warmup:
            return args.lr * s / max(1, args.warmup)
        p = (s - args.warmup) / max(1, steps - args.warmup)
        return lr_final + 0.5 * (args.lr - lr_final) * (1 + math.cos(math.pi * min(1.0, p)))

    th = np.zeros((1, 0, eng.n_embd), np.float32)

    def val_loss() -> float:
        if not val:
            return float("nan")
        lp = eng.score(th, [0] * len(val), [p for p, _, _ in val], [r for _, r, _ in val], states=st)
        return float(-sum((x * m).sum() for x, (_, _, m) in zip(lp, val)) / sum(m.sum() for _, _, m in val))

    params = {i: eng.lora_get(i) for i in trained}
    m = {i: np.zeros_like(p) for i, p in params.items()}
    v = {i: np.zeros_like(p) for i, p in params.items()}
    rng = np.random.default_rng(args.seed)
    order, cursor, step0 = rng.permutation(len(train)), 0, 0
    if args.resume:
        z = np.load(args.out + ".opt.npz")
        for i in trained:
            m[i], v[i] = z[f"m{i}"], z[f"v{i}"]
        order, cursor, step0 = z["order"], int(z["cursor"]), int(z["step"])
        rng = np.random.default_rng(args.seed + step0)
        log(f"resumed at step {step0}")

    def save(path: str, step: int, vl: float):
        save_adapter(path, eng, meta, {"step": step, "val_loss": vl, "train": args.train, "lr": args.lr, "base": args.model})

    def save_opt(step: int):
        np.savez(args.out + ".opt.npz", step=step, order=order, cursor=cursor, **{f"m{i}": m[i] for i in trained},
                 **{f"v{i}": v[i] for i in trained})

    best = vl = val_loss()
    log(f"step {step0:5d}: val {vl:.5f} (init), {steps} steps of {bs} examples, lr {args.lr:g} -> {lr_final:g}")
    tok_tot, t_start = 0, time.time()
    for step in range(step0 + 1, steps + 1):
        t0 = time.time()
        batch = []
        for _ in range(bs):
            if cursor >= len(order):
                order, cursor = rng.permutation(len(train)), 0
            batch.append(train[order[cursor]])
            cursor += 1
        n_tok = int(sum(m.sum() for _, _, m in batch))
        n_all = sum(len(p) + len(r) for p, r, _ in batch)
        eng.lora_grad_clear()
        lp, _ = eng.state_grad(st, [0] * len(batch), [p for p, _, _ in batch], [r for _, r, _ in batch],
                               [m / n_tok for _, _, m in batch])
        loss = float(-sum((x * m).sum() for x, (_, _, m) in zip(lp, batch)) / n_tok)
        # the gradient of the mean log-prob; the loss is its negative
        grads = {i: -eng.lora_grad(i) for i in trained}
        gn = math.sqrt(sum(float((x.astype(np.float64) ** 2).sum()) for x in grads.values()))
        sc = min(1.0, args.clip / (gn + 1e-12)) if args.clip > 0 else 1.0
        lr = lr_at(step)
        b1, b2 = 0.9, args.beta2
        for i in trained:
            gi = grads[i] * sc
            m[i] = b1 * m[i] + (1 - b1) * gi
            v[i] = b2 * v[i] + (1 - b2) * gi * gi
            mh, vh = m[i] / (1 - b1 ** step), v[i] / (1 - b2 ** step)
            if args.wd:
                params[i] *= 1 - lr * args.wd
            params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
            eng.lora_set(i, params[i])
        tok_tot += n_all
        dt = time.time() - t0
        log(f"step {step:5d}: train {loss:.5f}, |g| {gn:.3e}, lr {lr:.2e}, {n_tok} trained of {n_all} tokens, {dt:.1f} s "
            f"({n_all / dt:.0f} tok/s)")
        if step % args.eval_every == 0 or step == steps:
            vl = val_loss()
            improved = not val or vl < best
            if improved:
                best = vl
                save(args.out, step, vl)
            save(args.out + ".last", step, vl)
            save_opt(step)
            log(f"step {step:5d}: val {vl:.5f} (best {best:.5f}){' saved' if improved else ''}, "
                f"{tok_tot / (time.time() - t_start):.0f} tok/s overall")
        if args.save_every and step % args.save_every == 0:
            save(f"{args.out}.step{step}", step, vl)
    log(f"DONE {steps} steps, best val {best:.5f}")


if __name__ == "__main__":
    main()
