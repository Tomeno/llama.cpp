# llama.cpp/examples/soft-prompt

Forward-only tuning of a *soft prompt*: K free input embeddings placed in front of every request, optimized with
antithetic evolution strategies (ES) and Adam. Only forward passes are used, so it works with every model llama.cpp
can run from embedding inputs, including recurrent/hybrid models such as EdgeRock, for which llama.cpp has no
backward ops. Two objectives are implemented:

- `train_nll.py`: teacher-forced NLL of given responses (supervised / self-distillation)
- `grpo.py`: reward of sampled responses, GRPO-style (Dr. GRPO advantages, DAPO tricks), evaluated with teacher forcing

See `docs/edgerock.md` for the model, the llama.cpp implementation, the findings that motivated this, and the
status of the project.

```
[before text] [K soft embeddings] [after text + formatted prompt] [response]
 --before       --init-text/-file   --after + --example ({prompt})
```

With the defaults the soft prompt is a ChatML system turn, the prompt a user turn, and the assistant turn opens with
`<think>\n`, which is the official EdgeRock format (`runtime/inference.py` of the model repo; no system prompt there).
For plain continuation use `--before "" --after "" --example "{prompt}"`.

`--init-text` initializes the soft prompt from the token embeddings of a text (dequantized rows of
`token_embd.weight`), so **step 0 is exactly that text prompt**; K is its number of tokens. `--init-file` continues
from a saved soft prompt (it also restores the template it was trained with).

## Contents

| file | |
| - | - |
| `soft-prompt.cpp`, `soft-prompt.h` | `libllama-soft-prompt` (C API): batched sampling and teacher-forced log-prob scoring, every sequence of one context with its own soft prompt; tokenizer; token embeddings |
| `softprompt.py` | ctypes bindings (`Engine`), `Template`, soft prompt files (GGUF), common arguments, `setup()` |
| `es.py` | antithetic ES + Adam; optional low-rank (EGGROLL-style) noise; rank fitness shaping |
| `rewards.py` | meta-talk regexes, sentence repetition, token-level loops, answer matching (final / contains / prefix), Kimi-style length reward, DAPO overlong penalty |
| `rollout.py` | sample groups of responses with a soft prompt, reward them, optional external reward command, summaries |
| `train_nll.py` | NLL training; `--eval-only` prints the NLL with and without the soft prompt |
| `probe.py` | sample + reward report, and how many groups have a reward spread (what GRPO can learn from) |
| `grpo.py` | the GRPO loop |
| `generate.py` | sample one response with a soft prompt |
| `make_data.py` | synthetic data: arithmetic continuation (`arith`), simple chat questions with answers (`simple`) |
| `slice_model.py` | copy of a GGUF model with a subset of its blocks (fast experiments) |

## Setup

```bash
# CUDA build (the library and the llama.cpp tools)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-soft-prompt llama-completion llama-server
# python: numpy; gguf-py is imported from the repo (gguf-py/)
pip install numpy
cd examples/soft-prompt
```

The scripts find `build/bin/libllama-soft-prompt.so` themselves (or set `SOFTPROMPT_LIB`). Models:
<https://huggingface.co/hkust-aaron/EdgeRock-4B-GGUF> (Q4_K_M, Q8_0, BF16, ...). The GGUFs there load as they are
(architecture `edgerock`).

Memory: every parallel sequence (`--par`, and at least `2 x --pairs` for the ES population) holds a full EdgeRock
sequence state, about 130 MiB of recurrent + RMA state plus up to ~48 MiB of window KV. On a 16 GB GPU with the Q4_K_M
model (2.9 GB) `--par 32` (~5-6 GB) leaves room for compute buffers; start there. Scoring computes the log-probs of
the targets on the device (`llama_set_logprobs`), so no logits are copied to the host.

## Data formats (JSONL)

- `train_nll.py`: `{"prompt": ..., "response": ...}`. The response is scored as is; for a thinking model it starts
  inside `<think>` and contains `</think>` and the answer. `--max-resp` truncates.
- `probe.py`, `grpo.py`: `{"prompt": ..., "answer": optional}`. Correctness uses `--answer-mode`:
  `final` (the extracted final answer after `</think>` equals the reference: last `\boxed{}`, else "answer is X" /
  "Answer: X", else the last line; single letters match multiple-choice answers), `contains` (the reference appears as
  a whole word after `</think>`), `prefix` (continuation: the response starts with the reference).

## Scripts

Common options (all scripts): `-m`, `-ngl` (default all layers), `-c` context size, `-b` batch, `--par` parallel
sequences, `-t` threads, `--init-text` / `--init-file` / `--n-soft`, `--before` / `--after` /
`--example`, `--seed`. ES options (`train_nll.py`, `grpo.py`): `--pairs` (population 2N), `--sigma` and `--lr`
(relative to the RMS of the init embeddings), `--rank` (low-rank noise, 0 = full), `--shaping none|rank`, `--steps`,
`--eval-every`, `-o` (best by validation; `<out>.last` is the latest).

### train_nll.py

```bash
python train_nll.py -m M.gguf --train train.jsonl --val val.jsonl --init-text "..." \
    --pairs 8 --examples 8 --steps 200 --sigma 0.05 --lr 0.03 -o sp.gguf
python train_nll.py -m M.gguf --val val.jsonl --init-file sp.gguf --pairs 1 --eval-only
```

`--examples` per pair and step; `--assign shared` (all pairs on the same examples, default) or `split` (every pair
its own examples: more data per step at the same cost). A step scores `2 x pairs x examples` sequences.

### probe.py

```bash
python probe.py -m M.gguf --prompts simple_val.jsonl --group 8 --gen-max 256 --answer-mode contains --init-text "..." \
    --samples-out probe.jsonl
```

Prints the mean reward and its terms (meta-talk share and rate, repetition, loops, thinking / total tokens, finished
share, accuracy) and how many prompt groups have a reward spread. **Run this before GRPO**: if (nearly) every sample
recites the meta-instruction, all groups get the same reward and GRPO has no signal; use the graded terms or start
from a better text prompt. `--temp 0` samples greedily (use `--group 1`).

### grpo.py

```bash
python grpo.py -m M.gguf --train simple_train.jsonl --val simple_val.jsonl --init-text "..." --answer-mode contains \
    --prompts 16 --group 8 --gen-min 128 --gen-max 384 --pairs 8 --es-steps 2 --kl 0.01 \
    --steps 100 --eval-every 10 -o sp.gguf --samples-out samples.jsonl
```

Per step:

1. Sample `--group` responses for `--prompts` prompts with the current soft prompt (`--temp 0.6 --top-p 0.95 --top-k 20`
   by default, the model card's settings). The token budget is drawn from `[--gen-min, --gen-max]` every step (so the
   prompt cannot learn to postpone meta-talk past a fixed window).
2. Reward (`rewards.py`, weights `--w-meta --w-any --w-rep --w-loop --w-correct --w-len`, `--overlong`, optional
   `--reward-cmd`).
3. Dr. GRPO advantages `A = r - mean(group)` (no std division); groups without a reward spread are dropped (DAPO
   dynamic sampling; `--no-dynamic` keeps them).
4. Log-probs of the samples under the sampling soft prompt (`old`) and the initial soft prompt (`ref`), at `--temp`.
5. `--es-steps` ES updates on the clipped surrogate, every population member scoring all kept samples:

   ```
   J(theta) = 1/Z * sum_i sum_t [ min(rho_t A_i, clip(rho_t, 1 - clip_low, 1 + clip_high) A_i) - kl * k3_t ]
   rho_t = exp(logp_theta - logp_old),  k3_t = exp(logp_ref - logp_theta) - (logp_ref - logp_theta) - 1
   Z = samples x budget (--norm const, Dr. GRPO) or the token count (--norm token, DAPO)
   ```

   with DAPO clip-higher defaults `--clip-low 0.2 --clip-high 0.28`.
6. Every `--eval-every` steps: sample the validation prompts with a fixed seed; keep the best mean reward in `-o`.

Log columns: reward and its terms, `groups kept/total`, `max_new`, `objective` (population mean of J), `|dL|` (loss
difference of the first antithetic pair, i.e. the signal strength), `KL` (k3 between the sampling and the initial
soft prompt on the samples), `drift` (RMS distance from the init, in units of the init embedding RMS).

The reward for the meta-talk stage (see `docs/edgerock.md`): `-w_meta * share of thinking sentences matching a
pattern - w_any * [any match] - w_rep * share of sentences repeated 3+ times - w_loop * share of repeated 12-token
n-grams + w_correct * correct + w_len * length term + overlong term`. The patterns (`rewards.DEFAULT_PATTERNS`, or
`--patterns FILE`, one regex per line) cover the observed recitation ("meta instruction", "treat every explicit",
"logically rigorous", "logical leaps", "we need to ensure we don't", unrequested `\boxed`, ...). Expect the prompt to
find paraphrases; read `--samples-out` and extend the patterns.

`--reward-cmd CMD` runs `CMD IN.jsonl OUT.txt` per batch: one JSON sample per line in (prompt, answer, response,
reward terms), one float per line out, added to the reward (e.g. the math/code verifiers in the EdgeRock repo, or an
LLM judge).

### generate.py

```bash
python generate.py -m M.gguf --init-file sp.gguf -p "What is 17*23?" -n 2048
```

### llama-server

A trained file (soft prompt, or soft prompt + state) can be served: `--soft-prompt` evaluates the prefix once at load
time and every request starts from it. It replaces the system turn, so requests should have only user and assistant
messages, and the chat template has to give the prompt format used in training (for EdgeRock the official one, which
opens `<think>`):

```bash
llama-server -m EdgeRock-4B-Q4_K_M.gguf --soft-prompt sp.gguf --jinja --chat-template-file ../edgerock/ref/chat_template.jinja
```

In the prompt cache the soft rows stand as placeholder tokens (the pad token), so `prompt_tokens` in the usage counts
them; a slot that has to start over restores the saved prefix instead of processing them. Not with media input or a
draft model. The thinking is not split off into `reasoning_content` with this template.

### make_data.py, slice_model.py

```bash
python make_data.py arith  --n 256 -o arith_train.jsonl
python make_data.py arith  --n 32  -o arith_val.jsonl --seed 1 --exclude arith_train.jsonl
python make_data.py simple --n 256 -o simple_train.jsonl
python slice_model.py EdgeRock-4B-Q4_K_M.gguf EdgeRock-slice8.gguf --layers 0-3,28-31
```

## Sanity check: arithmetic continuation

A test with a clear signal and short sequences (a few tokens per sample):

```bash
python make_data.py arith --n 256 -o arith_train.jsonl && python make_data.py arith --n 32 -o arith_val.jsonl --seed 1 --exclude arith_train.jsonl
python probe.py -m M.gguf --prompts arith_val.jsonl --group 1 --temp 0 --gen-max 6 --answer-mode prefix \
    --before "" --after "" --example "{prompt}" --init-text "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
python train_nll.py -m M.gguf --train arith_train.jsonl --val arith_val.jsonl --before "" --after "" --example "{prompt}" \
    --init-text "Lorem ipsum dolor sit amet, consectetur adipiscing elit." --pairs 8 --examples 8 --steps 40 --eval-every 10 \
    --sigma 0.05 --lr 0.03 -o arith_sp.gguf
```

Results on EdgeRock-4B Q4_K_M (CPU, 2-digit multiplication only, 32 validation problems):

| prompt | val NLL | greedy accuracy |
| - | - | - |
| "Lorem ipsum ..." (deliberately unhelpful start) | 0.402 | 71.9% |
| none | 0.269 | |
| "Multiplication table practice with answers:\n12*12=144\n" | 0.012 | 100% |
| soft prompt from "Lorem ipsum ..." after ES steps 10 / 20 / 30 / 40 | 0.218 / 0.147 / 0.106 / 0.077 | 93.8% (after 40) |

(40 steps x 16 members x 8 examples took ~2 h on the 4-core CPU.) ES learns on the real model; from a good text prompt there is little left to learn on this task, so use a
harder task (or a weaker start) to see improvements over a normal prompt.

## Verified

- the soft prompt initialized from a text gives exactly the loss of that text as tokens (F32 tiny model: identical for
  every batch size and number of parallel sequences; quantized models: identical when the batch splits match, the
  quantized kernels themselves depend on the batch shape)
- ES lowers held-out NLL on a toy model and on EdgeRock-4B (arithmetic, above: 0.402 -> 0.077, accuracy 71.9% -> 93.8%)
- probe and grpo run end to end (toy model); GRPO has not been run on the real model yet
- CUDA (RTX A4000): the text-init equivalence, the arithmetic ES run and `probe.py` work; ~4 s per arithmetic ES step (see `docs/edgerock.md`, "Next steps")

## Performance notes and possible improvements

- `SP_PROFILE=1` prints the wall time of the scoring phases (prefix, state, decode, log-probs) after every
  `sp_score` call; `SP_PROFILE=2` also times every graph node by op (it synchronizes after each node, so the totals
  are inflated, but the ranking holds). `runs/meta-talk/prof/bench_score.py` is a fixed workload (one ES step) that
  checks the losses against a saved run.
- EdgeRock 4B Q4_K_M, A4000, one state-tuning step (24 members x 8 examples, `--par 24`): 19.6 s before the RMA
  commit / read counts were made exact (every ubatch ran a full commit and two reads over 24 x 8 MB of records per
  layer, also with nothing to commit), 11.0 s after (bit-identical results), 9.8 s with the shared prefix (rounding
  level changes: the prefix KV cells are shared), 7.9 s with the log-probs on the device (full 85-token chunks per
  sequence instead of 21, so that the logits fit on the host; same chunks: within 1.3e-7 of the host log-softmax),
  5.3 s with one state upload per distinct state and the jobs sorted by length (similar lengths per round: recurrent
  ubatches need equal tokens per sequence, so ragged rounds split into many small ubatches). Soft prompt ES: 5.9 s.
- Every distinct soft prompt of a round (compared by content) is evaluated once, every distinct state set once, and
  copied to the other sequences (`seq_cp` plus `llama_memory_seq_unshare`); state tuning evaluates a single prefix
  per round. `Engine.score` runs the longest jobs first; `es.score_population` does too, keeping the mirrored slots.
  Prompts are still re-run per job; processing the prompt once per sample group would be the next step.
- Device log-probs are a softmax plus a gather, clamped at about -87.3 (probabilities below FLT_MIN).
- Noise is generated on the host (numpy) and the members are copied in every call; fine for K x n_embd = 82k values.

## Plan and design notes

### Option 2: soft prompt (these scripts)

1. Measure the base rate of the meta-talk with `probe.py` on simple prompts with short budgets, for: no system turn,
   an empty system turn, and candidate text prompts (e.g. the recited instruction itself, or "The user's message is
   complete; there are no additional instructions."). Pick the best as the init.
2. Stage A (degeneration): `grpo.py` with short randomized budgets (128-384 tokens), meta-talk / repetition / loop
   penalties, `contains` correctness on simple prompts, KL to the init. Check `--samples-out` for paraphrases.
3. Stage B (length and accuracy): problems with reference answers (GSM8K / MATH train splits, NuminaMath,
   OpenR1-Math, ...; keep EdgeRock's evaluation benchmarks out), full budgets (2-4k), `--w-len` (shorter correct
   answers win; wrong answers never gain), `--overlong` (DAPO soft penalty before the budget), `--norm const`.
4. Cheaper alternative / warm start for B: rejection-sampling self-distillation: sample N answers per problem, keep
   correct + short + no meta-talk, train with `train_nll.py` (expert iteration when repeated).
5. Serving: a soft prompt has no token ids, so the server prompt cache cannot match it. Needs `--soft-prompt FILE` in
   the server (process the prefix once per slot and restore the state for new requests) - not implemented.

### Option 3: initial state tuning

The parameters are the per-sequence state itself (EdgeRock: delta-net states, RMA records, window KV; ~30-35M values)
instead of K input embeddings (e.g. 32 x 2560 = 82k values).

1. Start from the state the best soft/text prompt produces (a reachable state).
2. Perturb the state per population member: needs a C API to add seeded noise to a sequence's state on the device
   and to read/write it (`llama_state_seq_get/set_data` work but copy ~165 MiB per member).
3. Objective and data as in option 2, every example starting from the perturbed state.
4. Stay near reachable states: small steps, separate noise scales per state part; RMA scores and valid flags are not
   continuous parameters and stay fixed.
5. The result is a state file restored per request; no server change needed beyond restoring a slot.

~400x more dimensions than option 2 makes ES much noisier; only worth it if option 2 saturates.

### EGGROLL and Dust

- EGGROLL's main gain is computational (low-rank weight perturbations applied inside batched matmuls, so members are
  never materialized). A soft prompt member is 82k floats, so that does not apply; its low-rank noise is `--rank`.
- Dust perturbs activations independently at every token and credits each perturbation with that token's (and
  nearby) loss change, so one forward pass evaluates thousands of members. That needs locality; a soft prompt is read
  by every scored token, so there is no per-token credit and its population stays a set of sequences. What transfers
  is many cheap noisy draws (`--assign split`). Dust is the right tool for training weights or adapters; that needs
  graph hooks for per-layer, per-token noise inputs and clean-pass caching in llama.cpp.
