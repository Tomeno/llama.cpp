"""A soft prompt file that holds a text prefix (e.g. a persona system prompt) and the recurrent state a tuned initial
state (a train_state.py --no-soft-prompt file) reaches after reading it: served with --soft-prompt, every request
behaves as that state + that system prompt, with no system message in the request.

The server processes the prefix from the zero state and then replaces the recurrent state by the saved one, so the
window KV of the prefix is the one of the zero state (close, not identical to reading it after the tuned state).

    python prefix_state.py -m M.gguf --state state.gguf --system "You are ..." -o persona.gguf
"""
import argparse, sys
import numpy as np
from softprompt import Engine, Template, load_state, save_state

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True)
ap.add_argument("--state", required=True, help="state-only soft prompt file the prefix starts from")
ap.add_argument("--system", help="system prompt text (wrapped in the ChatML system turn)")
ap.add_argument("--before", help="raw prefix text instead of --system")
ap.add_argument("-ngl", "--n-gpu-layers", type=int, default=0)
ap.add_argument("-o", "--out", required=True)
args = ap.parse_args()
if (args.system is None) == (args.before is None):
    sys.exit("give one of --system / --before")
before = args.before if args.before is not None else f"<|im_start|>system\n{args.system}<|im_end|>\n"

eng = Engine(args.model, args.n_gpu_layers, 4096, 2048, 1)
eng.set_before(eng.tokenize(before))
eng.set_init_state(load_state(args.state))
empty = np.zeros((0, eng.n_embd), np.float32)
save_state(args.out, empty, eng.get_state(empty), Template(before, "", Template.example), {"state": args.state})
print(f"{args.out}: prefix of {len(eng.tokenize(before))} tokens + state", file=sys.stderr)
