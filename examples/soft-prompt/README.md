# llama.cpp/examples/soft-prompt

Tunes a *soft prompt* (K free input embeddings in front of every request) with forward passes only: antithetic
evolution strategies (ES) with Adam. No backward pass is needed, so it works with every model llama.cpp can run
from embedding inputs, including recurrent and hybrid models such as EdgeRock where llama.cpp has no backward ops.

```
[before text] [K soft embeddings] [after text + formatted prompt] [response]
```

The default texts make the soft prompt a ChatML system turn and the prompt a user turn followed by
`<|im_start|>assistant\n<think>\n`, the EdgeRock format. Initializing from `--init-text` uses the token embeddings
of that text, so step 0 is exactly the text prompt.

## Layout

The performance-critical parts are a C++ shared library; everything else is Python (numpy, ctypes, gguf-py).

| file | |
| - | - |
| `soft-prompt.cpp/.h` | `libllama-soft-prompt`: batched sampling and teacher-forced log-prob scoring, every sequence of one context with its own soft prompt; tokenizer and token embeddings |
| `softprompt.py` | ctypes bindings (`Engine`), template, soft prompt files (GGUF), shared arguments |
| `es.py` | antithetic ES + Adam, optional low-rank (EGGROLL-style) noise, rank shaping |
| `rewards.py` | meta-talk regexes, sentence repetition, token loops, answer matching, Kimi length reward, DAPO overlong penalty |
| `rollout.py` | sampling groups of responses and rewarding them, external reward command |
| `train_nll.py` | minimize the NLL of given responses (`--eval-only`: NLL with and without the soft prompt) |
| `probe.py` | sample and report the reward terms, and how many groups have a reward spread |
| `grpo.py` | Dr. GRPO advantages, DAPO dynamic sampling / clip-higher / token-level or constant normalization, k3 KL to the initial soft prompt; the clipped surrogate is evaluated with teacher forcing for every ES member |
| `generate.py` | sample one response |
| `slice_model.py` | copy of a GGUF model with a subset of its blocks, for quick tests |

Build the library with the `llama-soft-prompt` target; the scripts find it in `build/bin` (or set `SOFTPROMPT_LIB`).

## Usage

```bash
cmake --build build --target llama-soft-prompt
cd examples/soft-prompt

# base rate of the meta-talk with a text prompt (prompts: JSONL {"prompt", optional "answer"})
python probe.py -m EdgeRock-4B-Q4_K_M.gguf --prompts simple.jsonl --group 8 --gen-max 256 --init-text "..."

# GRPO on the meta-talk / loop reward with short rollouts
python grpo.py -m EdgeRock-4B-Q4_K_M.gguf --train simple.jsonl --val simple_val.jsonl --init-text "..." \
    --prompts 16 --group 8 --gen-min 128 --gen-max 384 --pairs 8 --es-steps 2 -o sp.gguf --samples-out samples.jsonl

# teacher-forced NLL on given responses (JSONL {"prompt", "response"})
python train_nll.py -m EdgeRock-4B-Q4_K_M.gguf --train train.jsonl --val val.jsonl --init-text "..." -o sp.gguf

python generate.py -m EdgeRock-4B-Q4_K_M.gguf --init-file sp.gguf -p "What is 17*23?"
```

`--par` sets the parallel sequences of the context (each holds a full sequence state, about 165 MiB for EdgeRock-4B);
the population needs `2 x --pairs` of them.

## Verified

- the soft prompt initialized from a text gives the same loss as that text as tokens, for any batch size and any
  number of parallel sequences (exactly; on quantized models only when the batch splits match)
- ES lowers the held-out loss on a toy EdgeRock (random weights, synthetic targets)
- probe and grpo run end to end on the toy model; not yet run on the real model
- tested on CPU only

## Plan

### Option 2: soft prompt (these scripts)

1. Data. Teacher-forced NLL needs target responses. The intended source is the model itself:
   sample N responses per training problem with the best text prompt (temperature 0.6, top-p 0.95, top-k 20),
   keep the ones whose final answer matches the reference (the EdgeRock repo ships reference answers for its
   evaluation sets; use problems outside the evaluation split for training), prefer short ones and ones that do
   not recite a meta-instruction. This is rejection-sampling self-distillation.
2. Train the soft prompt on that data with this tool; select by validation NLL.
3. Evaluate with generation on held-out problems: accuracy and thinking length vs the text prompt.
4. Iterate: regenerate the data with the trained soft prompt, filter, retrain.
5. Serving: a soft prompt is a fixed prefix, so its state can be computed once and restored for every request.
   Needs a small server addition (`--soft-prompt FILE`): process the prefix into a slot state at startup and
   restore it for new sequences, since a soft prompt has no token ids to match in the prompt cache.

Open items for this tool: noise generation on the device instead of the host for large populations, and a reward
objective (see the RL note below).

### Option 3: initial state tuning

The parameters are the per-sequence state itself (EdgeRock: delta-net states, RMA records, window KV; about
30-35M values) instead of K input embeddings (32 x 2560 = 82k values).

1. Start from the state the best soft/text prompt produces, so the search starts from a state the model can reach.
2. Perturb the state per population member: needs an API to add noise to a sequence's state on the device
   (generated from a seed, so that nothing is copied per member), and to read back / write the updated state.
3. Objective and data as in option 2, starting each example from the perturbed state.
4. Constrain the search to stay near reachable states: small steps, separate noise scales per state part
   (recurrent, RMA keys/values, RMA scores and flags are not continuous parameters and stay fixed).
5. Store the result as a state file restored per request (no server change needed: it is a slot state).

400x more dimensions than option 2 makes ES much noisier; it is only worth it if option 2 saturates.

### Why not Dust's virtual population here

Dust perturbs activations independently at every token and credits each perturbation with that token's own
(and nearby) loss change, so one forward pass evaluates thousands of population members. That relies on
locality: a perturbation at token t mostly affects the loss at t. A soft prompt is read by every scored token, so
there is no per-token credit, and its population must stay a set of separate sequences. The part that transfers
is evaluating many cheap noisy draws instead of few averaged ones (`--assign split`). Dust becomes the right tool
for training weights (or adapters), where the per-token activation noise is the whole point; that needs graph
hooks for per-layer, per-token noise inputs and the clean-pass caching that Dust uses.
