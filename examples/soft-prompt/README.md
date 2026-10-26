# llama.cpp/examples/soft-prompt

Tunes a *soft prompt* (K free input embeddings in front of every request) with forward passes only: antithetic
evolution strategies (ES) with Adam. No backward pass is needed, so it works with every model llama.cpp can run
from embedding inputs, including recurrent and hybrid models such as EdgeRock where llama.cpp has no backward ops.

```
[before text] [K soft embeddings] [after text] [example prompt] [example response]
                                                                 ^ scored (teacher-forced NLL)
```

The default texts make the soft prompt a ChatML system turn and the example a user turn followed by
`<|im_start|>assistant\n<think>\n`, the EdgeRock format. Initializing from `--init-text` uses the token embeddings
of that text, so step 0 is exactly the text prompt (the tool's `eval` mode reproduces the text prompt's loss).

## Usage

Data is JSONL, one `{"prompt": ..., "response": ...}` per line. The response is scored as is, so for a thinking
model it starts inside the `<think>` block and contains the closing `</think>` and the answer.

```bash
# train (16 GB GPU: a 4B Q4 model with 16-32 population members fits)
llama-soft-prompt -m EdgeRock-4B-Q4_K_M.gguf --train train.jsonl --val val.jsonl \
    --init-text "Solve the problem independently with a concise, logically rigorous derivation." \
    --pairs 8 --examples 4 --steps 500 --sigma 0.05 --lr 0.01 -o sp.gguf

# validation loss of a soft prompt vs no soft prompt
llama-soft-prompt -m EdgeRock-4B-Q4_K_M.gguf --mode eval --val val.jsonl --init-file sp.gguf

# generate with a soft prompt
llama-soft-prompt -m EdgeRock-4B-Q4_K_M.gguf --mode generate --init-file sp.gguf -p "What is 17*23?" -n 2048
```

`sp.gguf` holds the tensor `soft_prompt.embd` (F32, `[n_embd, K]`) and the before/after/example texts it was
trained with; `sp.gguf.last` is the latest step, `sp.gguf` the best validation loss.

Options that matter:

- `--pairs N`: population 2N, run as 2N parallel sequences of one context; memory is 2N sequence states.
- `--examples N`: examples per pair and step. A step costs 2 x pairs x examples forward passes.
- `--assign shared|split`: every pair on the same examples (default), or every pair on its own examples, which
  sees more data per step at the same cost. Both members of a pair always share their examples, so the antithetic
  difference cancels the difficulty of the examples.
- `--shaping none|rank`: weights of the pair differences; `rank` is robust to outlier examples.
- `--rank R`: low-rank noise `A B^T` over the `[K, n_embd]` soft prompt (EGGROLL-style); 0 = full rank.
- `--sigma`, `--lr`: relative to the RMS of the init embeddings.
- `--max-outputs`: logits rows per decode call; host memory is rows x n_vocab x 4 bytes.

## Verified

- the soft prompt initialized from a text gives the same loss as that text as tokens, for any batch size and any
  number of parallel sequences
- ES lowers the held-out loss on a toy EdgeRock (random weights, synthetic targets): 20.27 -> 17.68 in 60 steps
- tested on CPU only

## Plan

### Option 2: soft prompt (this tool)

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
