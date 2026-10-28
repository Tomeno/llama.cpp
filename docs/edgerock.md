# EdgeRock in llama.cpp (handoff notes)

Start here. This documents the EdgeRock support on this branch, how it was verified, how to control its state and
caches, what was learned about the model, and the soft prompt tuning work built on top. Everything was developed and
tested on a 4-core CPU without a GPU; GPU results are in "Next steps".

- Model: <https://huggingface.co/rockmatrix-ai/EdgeRock> (reference Python code, `modeling_edgerock.py`)
- GGUFs: <https://huggingface.co/hkust-aaron/EdgeRock-4B-GGUF> (their `backend/` folder holds the official llama.cpp
  patch, reviewed below; the GGUFs themselves load with this branch as they are)
- Branch: `claude/relaxed-ptolemy-ndq1zj` of the Tomeno/llama.cpp fork

| what | where |
| - | - |
| model implementation | `src/models/edgerock.cpp`, registration in `src/llama-arch.*`, `src/llama-model.*`, `src/llama-hparams.*` |
| per-layer recurrent state sizes | `src/llama-hparams.cpp` (`n_embd_r(il)`, `n_embd_s(il)`), `src/llama-memory-recurrent.cpp` |
| server checkpoint fix for hybrid + SWA | `tools/server/server-context.cpp` (`n_swa_thold`) |
| HF -> GGUF conversion | `conversion/edgerock.py`, `gguf-py/gguf/constants.py`, `gguf-py/gguf/tensor_mapping.py` |
| verification tools | `examples/edgerock/` |
| soft prompt tuning (forward-only, ES / GRPO) | `examples/soft-prompt/` (own README) |

## Quick start on a GPU machine

```bash
git checkout claude/relaxed-ptolemy-ndq1zj
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
huggingface-cli download hkust-aaron/EdgeRock-4B-GGUF EdgeRock-4B-Q4_K_M.gguf --local-dir models

# smoke test
build/bin/llama-completion -m models/EdgeRock-4B-Q4_K_M.gguf -p "The capital of France is" -n 32 -no-cnv --temp 0
# chat: the GGUF has no chat template; the official one is ChatML that opens <think> (fetch_reference.py downloads it)
build/bin/llama-completion -m models/EdgeRock-4B-Q4_K_M.gguf --jinja --chat-template-file ref/chat_template.jinja \
    -cnv -st -p "What is 17*23?" --temp 0.6 --top-p 0.95 --top-k 20 -n 4096
```

Expected on CPU (Q4_K_M, 4 threads): "Paris. Paris is known for ...", ~4 tokens/s.

## The model

EdgeRock-4B is Qwen3.5-4B converted to bounded memory ("constant persistent history"):

- 32 layers: 24 gated delta-net (recurrent) layers and 8 attention layers (every 4th: 3, 7, ..., 31), hidden 2560,
  vocab 248320, tied embeddings, interleaved M-RoPE (rotary dims 64, theta 1e7).
- The attention layers use a **1024-token sliding window** (reference: key `>= query - 1024 + 1`).
- Each attention layer has an **RMA** module ("Rock Memory Anchors", `IndependentGQAMemory` in the reference): a
  fixed-capacity bank of up to 1024 key/value records per KV head (4 KV heads, head dim 256). Every token writes a
  *pending* record (keys/values from `u + writer_mlp(u)`, `u = RMSNorm(residual)`, RoPE'd keys, a learned admission
  score per KV head). Every 256 tokens (`min(max_commit_chunk, window)`) the pending records are merged into the
  bank, keeping the 1024 highest scores (ties: earlier position first). Tokens read only the committed bank, with an
  extra zero "sink" logit; the read is gated per head (`2 tanh(read_scale) sigmoid(q . w / sqrt(d) + b)`, q before
  RoPE), projected and added to the attention output. Memory reads use the residual stream *before* the layer's
  input norm.
- Constant state per sequence: delta-net states (~52 MiB), RMA banks and pending records (~84 MiB), window KV
  (<= ~48 MiB at f16).

Prompt format (official runtime `runtime/inference.py`): no system prompt,
`<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n`. Sampling used for the model card's
evaluations: temperature 0.6, top-p 0.95, top-k 20.

## Implementation (llama.cpp)

`LLM_ARCH_EDGEROCK` ("edgerock") is its own architecture; `llama_model_edgerock` derives from `llama_model_qwen35`
for hparams and tensors and has its own graph (`src/models/edgerock.cpp`; the Qwen3.5 helpers are copied, as
qwen35moe does).

- **Hparams** (`edgerock.memory.capacity / commit_chunk / admission_rank / writer_intermediate`,
  `edgerock.attention.window` or `.sliding_window`) -> `hparams.n_mem_*`, `n_swa`. All attention layers are SWA
  layers (`LLAMA_SWA_TYPE_STANDARD`), so the hybrid iSWA memory gives them a ring cache of `n_swa + n_ubatch` cells.
- **RMA state lives in `llama_memory_recurrent`**, in the rows of the attention layers, like RWKV keeps its state:
  `r` holds the pending records (K, V `[d, chunk, kv_heads]` + scores), `s` the committed bank (K, V
  `[d, capacity, kv_heads]` + scores + valid flags). This needed per-layer state sizes: `llama_hparams::n_embd_r(il)` /
  `n_embd_s(il)` (the recurrent memory now allocates, saves and loads per layer). Consequences: `seq_cp`, `seq_rm`,
  state save/load, server checkpoints and slot files all handle RMA with no special code.
- **RMA in the graph** uses only standard ggml ops (`top_k`, batched `get_rows`, `set_rows`, `mul_mat`, `out_prod`,
  `soft_max` with sinks, `scale_bias`, `step`, `concat`), so it can run on any backend. Commit points are derived from positions (they
  start at 0 and are consecutive per sequence; the pending count is `pos % chunk`): a ubatch is split into segments
  at chunk boundaries; for every possible commit the graph gathers the chunk's records (from the stored pending rows
  and the ubatch's new records), takes the top `capacity` scores over bank + chunk, and blends the result in with a
  per-sequence 0/1 input (so the graph shape only depends on the ubatch shape). Each segment reads its own bank
  state. Indices and masks come from `llm_graph_input_mem_rma::set_input`. Invalid records get a large negative
  logit; the sink is a zero logit (soft_max sink) with a zero value.
- **Decode reads the states in place**: when a ubatch moves no recurrent state (`is_in_place`), `build_rs` returns
  views of the cache rows instead of gathering copies; the RMA read multiplies the committed K / V views directly
  (`out_prod` sums over the records, so V needs no transpose), and without a commit the new pending records are
  written to their slots with `set_rows`. In a single-token ubatch only the sequences that reach a commit point are
  gathered, merged and written back (`set_rows`), instead of merging and blending the bank of every sequence. Copies of the whole state per token (~130 MiB per sequence) made batched
  generation scale linearly with the number of sequences; this cut the decode time of 24 parallel sequences by 2.3x.
- Cost/limit: the graph has `ceil(n_seq_tokens / chunk) + 1` reads and commits per memory layer, so very small
  commit chunks with huge ubatches blow up the graph (tiny test models: keep `-ub <= 64`). For EdgeRock-4B (chunk 256)
  a 2048-token ubatch has 9 reads per layer.
- Not supported: MTP (the GGUFs have no MTP tensors), recurrent rollback (`n_rs_seq`), partial `seq_rm` (as for every
  recurrent model; checkpoints are used instead), context shift (positions must start at 0).
- Exact ties of admission scores may be ordered differently than the reference (`top_k` is unordered); with float
  scores this does not happen in practice.

### The official patchset (`backend/` of the GGUF repo): why not used

- `set_swa_pattern(1)` marks *no* layer as SWA: the attention layers get full attention over a growing KV cache
  (8 GiB at 262k context) instead of the trained 1024-token window, so it is neither constant-memory nor the model.
- It aliases `edgerock` to `qwen35`, adds an arbitrary tensor name override to `LLM_TN`, puts EdgeRock fields into
  the generic hparams, hard-codes the architecture name in `common.cpp`, and patches `llama_memory_recurrent` for all
  models.
- The RMA state is a host `std::map` mutated by a single-threaded CPU custom op (`ggml_map_custom3`) inside graph
  compute; it throws inside compute on a position mismatch, is invisible to memory accounting, and has its own save
  format.
- `ollama-runtime.patch` is unrelated (moves Ollama's config/key directory).

With only its window bug fixed it agrees with this implementation (perplexity below).

## Verification

Tools in `examples/edgerock/` (build target `llama-edgerock-*`; the reference scripts need `torch`,
`transformers==5.7.0` for `ref_full.py`, `safetensors`, `numpy`).

| check | how | result |
| - | - | - |
| RMA module vs the reference `IndependentGQAMemory` (verbatim code) on the exact layer inputs, BF16 weights | `llama-edgerock-dump` + `ref_memory.py --bf16-act`, 1800 tokens: 1736 prefill in 300-token ubatches + 64 single-token decodes; also ubatch 37 and 1 | layers 3 and 7: mean rel. error ~3e-4, max < 2e-3 (the reference's own stateless vs stateful: ~1e-3); tokens before the first commit read exactly 0 on both sides |
| whole model vs the reference `EdgeRockForCausalLM` | `make_tiny.py` (8 layers, window 16, capacity 8, chunk 4, random weights) -> convert -> `llama-edgerock-logits` -> `ref_full.py`, 120 tokens (~30 commits) | max abs logit error 1.2e-3 (one batch) / 1.8e-3 (ubatch 7 + decode, token-by-token) on logits of scale 34; argmax 120/120 |
| perplexity vs the official patch, 2 x 2048 tokens, Q4_K_M | `llama-perplexity -c 2048 -b 2048 -ub 512` | this: 7.7259; official: 7.7380 (window bug); official with the window fixed: 7.7229 |
| several sequences per ubatch | perplexity with `-np 2` | identical to one sequence |
| state round trips | `llama-edgerock-state-test` | save/restore, restore into another sequence, `seq_cp`, partial (checkpoint) restore: bit-exact; partial `seq_rm` refused |
| converter | tiny checkpoint through `convert_hf_to_gguf.py` | metadata, `1 + w` on the memory norms (as the reference RMSNorm), loads and matches the reference |

Reproduce (RMA module, needs the BF16 GGUF):

```bash
cd examples/edgerock
python fetch_reference.py ref/ --memory 3 7
../../build/bin/llama-edgerock-dump models/EdgeRock-4B-BF16.gguf some_text.txt dump/ 1736 300 1800 l_out-2 mem_out-3 l_out-6 mem_out-7
python ref_memory.py --ref ref/ --dump dump/ --layer 3 --chunk 300 --bf16-act
python make_tiny.py ref/ tiny/ && python ../../convert_hf_to_gguf.py tiny/ --outtype f32 --outfile tiny/tiny-f32.gguf
../../build/bin/llama-edgerock-logits tiny/tiny-f32.gguf some_text.txt 120 90 7 tiny/logits_90_7.bin
python ref_full.py tiny/ tiny/logits_90_7.bin
../../build/bin/llama-edgerock-state-test models/EdgeRock-4B-Q4_K_M.gguf some_text.txt 700
```

(`dump/` must be empty: the dump tool appends.) On quantized models, results depend slightly on the batch shape
(the quantized CPU kernels do); e.g. perplexity moves ~0.2% between `-ub 1300` and `-ub 37`. Compare like with like.

## State, memory and caches

Per sequence: ~130 MiB recurrent + RMA state, plus the window KV: 32 KiB per cell (8 layers, f16 K and V); one
sequence needs 1024 + n_ubatch cells (48 MiB with `-ub 512`), N parallel sequences about N x 1024 + n_ubatch. None
of it grows with the context: at `-c 262144` the KV cache of one sequence is 48 MiB. A
server context checkpoint is the full state (~163 MiB): with no full-attention layer, "partial" is everything.

From the C API:

- `llama_state_seq_get_data` / `llama_state_seq_set_data` (or `_save_file` / `_load_file`): snapshot / restore one
  sequence, also into another sequence id.
- `llama_memory_seq_cp(mem, src, dst, -1, -1)`: fork a sequence (e.g. branch a conversation).
- `llama_state_seq_*_ext(..., LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)`: what server checkpoints use (= the full state here).
- Partial `seq_rm` is refused (recurrent state cannot be truncated): to roll back, restore an earlier snapshot.
- Positions must start at 0 and be consecutive per sequence (`set_input` asserts it); no context shift.

From `llama-server`:

- `--slot-save-path DIR` + `POST /slots/{id}?action=save|restore|erase`: a slot file is the state plus the slot's
  checkpoints (1708-token prompt with 3 checkpoints: 650 MiB, restored in 0.5 s; a resumed prompt then processed 4
  tokens instead of 1699). There is no option to save without checkpoints.
- `--ctx-checkpoints N` (default 32 x ~163 MiB, up to ~5 GiB per slot): 2-8 is plenty. `--checkpoint-min-step`
  (default 8192) only evicts close checkpoints when the list is full.
- `--cache-ram MiB` / `--cache-idle-slots`: swap idle slots to host RAM.
- Do not use `--swa-full` (full-size window cache: not constant any more).
- Fix on this branch: for hybrid memories the server's `pos_min` of a state is the recurrent part's last position, so
  the SWA relaxation of the checkpoint threshold (`pos_eval - n_swa + 2`) made every checkpoint unusable (full
  reprocessing on every request). `n_swa_thold` drops the relaxation for recurrent/hybrid models.

## Findings about the model

- In its thinking it often recites an instruction that was never given ("The user also gave a meta instruction:
  Solve the problem independently with a concise, logically rigorous derivation ... place it in \boxed{}", and
  "Treat every explicit content, count, format, language, ordering, and placement constraint as absolute") and then
  checks compliance with it, sometimes in a loop ("We need to ensure we don't ..." repeated) - with greedy decoding
  it looped until stopped; with temperature 0.6 / top-p 0.95 / top-k 20 it answered "What is the capital of France
  and why?" correctly after ~1800 thinking tokens.
- That text appears nowhere in the release: not in the README, docs, runtime/evaluation code, whitepaper sources,
  nor in any of the 114,511 evaluation prompts (no system prompts there either; prompts end with short task lines
  such as "Give the answer letter on the final line." or "Put your final answer in \boxed{}."). The whitepaper
  describes four stages: RMA adaptation, joint adaptation, supervised post-training, RLVR; that data is not released.
  Most likely the instruction was attached to post-training prompts and was learned into the weights.
- There is no pretrained initial state: the checkpoint holds only weights, and every sequence starts empty (zero
  delta-net/conv states, empty RMA banks, empty window), in the reference and here.
- `--reasoning-budget` of the llama.cpp tools does not apply with the custom chat template (not detected as a
  thinking format).

## Soft prompt tuning

`examples/soft-prompt/` (see its README for usage, options, data formats and the plan). Summary:

- C++ library (`libllama-soft-prompt`, C API) for batched sampling and teacher-forced scoring with one soft prompt per
  sequence; Python (ctypes, numpy) for rewards, ES + Adam and the training loops.
- `train_nll.py` (NLL), `grpo.py` (Dr. GRPO + DAPO dynamic sampling / clip-higher / normalization + k3 KL, the
  surrogate evaluated with teacher forcing for every ES member), `probe.py` (reward base rates), `generate.py`.
- Verified: a soft prompt initialized from text reproduces that text prompt exactly; ES lowers held-out NLL on
  EdgeRock-4B (arithmetic continuation from an unhelpful init: val NLL 0.402 -> 0.077 and greedy accuracy 71.9% ->
  93.8% in 40 ES steps; a good text prompt starts at 0.012 / 100% on that task, see the README).
- Plan: probe the meta-talk base rate -> GRPO stage A on short rollouts (meta-talk / repetition / loop penalties) ->
  stage B on full rollouts with length rewards; option 3 (initial state tuning) only if soft prompts saturate.

## Next steps

GPU results (RTX A4000 16 GB, CUDA 13.4, `-DCMAKE_CUDA_ARCHITECTURES=86`):

| check | result |
| - | - |
| smoke test, Q4_K_M | "Paris. Paris is known for ...", 57 tokens/s generation |
| CPU fallback | none: 2 graph splits (embedding lookup + rest), every RMA op runs on CUDA |
| perplexity GPU vs CPU (`-dev none`), 2 x 2048 tokens | BF16: 8.1659 vs 8.1686 (0.03%); Q4_K_M: 8.3953 vs 8.3788 (quantized kernels differ) |
| `llama-edgerock-state-test`, 700 tokens | bit-exact on GPU and CPU |
| RMA module vs reference (BF16) | GPU: mean rel. error ~1e-3, max 4e-2; CPU on the same machine: 3e-4 / 2e-3 |
| tiny model vs reference | CPU: <= 1.5e-3, argmax 120/120 for every batch shape; GPU: 1.2e-3 token by token, 1.6e-2 (ub 512) and 5.4e-2 (ub 7, argmax 119/120) batched |
| soft prompt from text vs the same text as `--before` | identical val NLL |
| arithmetic ES, 40 steps x 16 members x 8 examples | val NLL 0.705 -> 0.315, greedy accuracy 56.2% -> 71.9%, ~4 s per step |
| `probe.py`, 32 simple prompts x 4 samples, 256 tokens | meta-talk in 65.6% of samples, 30 of 32 groups with a reward spread |

Fixed on the way:

- CUDA `top_k` returns the kept records in a different order every run (its radix path places them with atomics), so the RMA bank order and the read summation order changed between runs (~0.4 logit drift on Q4_K_M after the first commit). The commit now uses `ggml_argsort_top_k`.
- The state test used a unified KV cache; there the other sequence's cells change the attention reduction order (not bit-exact on CUDA, rounding level: <= 0.08 logits on BF16). It now uses per-sequence KV.
- `make_tiny.py` used linear head dim 8; the CUDA gated delta net supports 16, 32, 64, 128 only. It now uses 16.

Known:

- llama.cpp runs F32 matmuls in TF32 on Ampere+ (cuBLAS math mode in `common.cuh`, and the `mmf` kernel's TF32 mma for small batches). This is the larger GPU error of the RMA read (F32 x F32) and of the F32 tiny model in batches. `NVIDIA_TF32_OVERRIDE=0` turns off the cuBLAS part only.
- Past the window (e.g. 1300 tokens), state restores are not bit-exact on GPU or CPU (BF16: 0.0027 logits, tokens match); `seq_cp` is. Likely the different window cell order after a restore; not proven.
- The arithmetic numbers in the soft-prompt README were measured on multiplication-only data; `make_data.py arith` mixes `*`, `+`, `-`.

Open work:

- soft prompts in `llama-server` (`--soft-prompt FILE`; the prompt cache cannot match a prefix without token ids)
- scoring speed: share the prefix across jobs (`seq_cp`), GPU-side target log-probs
- option 3 (state tuning) needs a seeded on-device state perturbation API
- weight / adapter training with Dust-style per-token activation perturbation would need graph hooks
- the RMA graph could cap the reads per ubatch (split ubatches at commit boundaries) for very small commit chunks

## Gotchas met on the way

- Embedding batches of M-RoPE models need `n_pos_per_embd` (4) positions per row, section-major
  (`pos[j * n_tokens + i]`), `[p, p, p, 0]` for text; with one position per row llama.cpp reads past the array.
- `llama_get_logits_ith` may reorder the outputs on first access: never call it from several threads at once.
- `n_batch` must be at least the number of sequences that get a token in the same call.
- A `-march=native` build from one host can crash on another CPU (SIGILL); rebuild the CPU backend from scratch.
