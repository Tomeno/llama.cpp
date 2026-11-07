"""A new F32 GGUF LoRA adapter for a model (llama.cpp format: <weight>.lora_a [rank, in], <weight>.lora_b [out, rank],
delta W = alpha / rank * B @ A), initialized for one of these methods:

  lora  A Kaiming-uniform, B zero (PEFT's default), both trained
  nora  NoRA-init (Normalized LoRA, as rwkv_lightning_cuda): A uniform in +-1/sqrt(in), then every column (one input
        feature, over the rank) scaled to unit norm, once; B zero; alpha = rank; both trained
  miss  MiSS (as rwkv_lightning_cuda, the efficient form): input feature k goes to bucket k % rank, Y += alpha / rank *
        D @ S; here A is that fixed 0/1 bucket matrix and B = D (zero), so only B is trained (lora_train.py reads
        adapter.training.train_a / train_b from the file)

The targets are the base weights named by --targets (default: the six RWKV7 projections of every layer).

    lora_init.py MODEL.gguf OUT.gguf --method nora --rank 16 [--alpha A] [--targets ...] [--layers 0-31] [--seed 1234]
"""
import argparse, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "gguf-py"))
import numpy as np
import gguf

RWKV7 = ["time_mix_receptance", "time_mix_key", "time_mix_value", "time_mix_output", "channel_mix_key", "channel_mix_value"]

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out")
ap.add_argument("--method", choices=["lora", "nora", "miss"], default="nora")
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--alpha", type=float, default=0, help="default: rank (scale 1)")
ap.add_argument("--targets", default=",".join(RWKV7), help="comma-separated weight names without blk.N. (output: the head)")
ap.add_argument("--layers", default="", help="e.g. 0-31 or 8,9,10 (default: all)")
ap.add_argument("--seed", type=int, default=1234)
args = ap.parse_args()
alpha = args.alpha or float(args.rank)
if args.method == "nora" and alpha != args.rank:
    sys.exit("NoRA needs alpha = rank")

rd = gguf.GGUFReader(args.model)
arch = bytes(rd.fields["general.architecture"].parts[-1]).decode()
targets = args.targets.split(",")
layers = None
if args.layers:
    layers = set()
    for part in args.layers.split(","):
        a, _, b = part.partition("-")
        layers.update(range(int(a), int(b or a) + 1))

w = gguf.GGUFWriter(args.out, arch)
w.add_type(gguf.GGUFType.ADAPTER)
w.add_string(gguf.Keys.Adapter.TYPE, "lora")
w.add_float32(gguf.Keys.Adapter.LORA_ALPHA, alpha)
w.add_string("adapter.training.method", args.method)
w.add_bool("adapter.training.train_a", args.method != "miss")
w.add_bool("adapter.training.train_b", True)
w.add_uint32("adapter.training.rank", args.rank)

rng = np.random.default_rng(args.seed)
n_par = n_t = 0
for t in sorted(rd.tensors, key=lambda t: t.name):
    m = re.fullmatch(r"blk\.(\d+)\.(\w+)\.weight", t.name)
    if m:
        if m[2] not in targets or (layers is not None and int(m[1]) not in layers):
            continue
    elif not (t.name == "output.weight" and "output" in targets):
        continue
    n_in, n_out = int(t.shape[0]), int(t.shape[1])  # ggml ne0 = input width, ne1 = output rows
    r = args.rank
    if args.method == "miss":
        a = np.zeros((r, n_in), np.float32)
        a[np.arange(n_in) % r, np.arange(n_in)] = 1.0
    else:
        bound = 1 / np.sqrt(n_in)
        a = rng.uniform(-bound, bound, (r, n_in)).astype(np.float32)
        if args.method == "nora":
            a /= np.maximum(np.linalg.norm(a, axis=0, keepdims=True), 1e-12)
    b = np.zeros((n_out, r), np.float32)
    base = t.name.removesuffix(".weight") + ".weight"
    w.add_tensor(base + ".lora_a", a)
    w.add_tensor(base + ".lora_b", b)
    n_par += a.size * (args.method != "miss") + b.size
    n_t += 1

if not n_t:
    sys.exit("no target weights found")
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"{args.out}: {args.method}, rank {args.rank}, alpha {alpha:g}, {n_t} weights, {n_par / 1e6:.1f} M trained parameters")
