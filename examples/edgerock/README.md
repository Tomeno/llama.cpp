# llama.cpp/examples/edgerock

Verification tools for the EdgeRock architecture; see `docs/edgerock.md` ("Verification") for the procedure and the
expected results.

| file | |
| - | - |
| `edgerock-dump.cpp` | `llama-edgerock-dump`: decode a text, append named graph tensors (e.g. `l_out-2`, `mem_out-3`) of every ubatch to files |
| `edgerock-logits.cpp` | `llama-edgerock-logits`: logits of every token (prefill in ubatches, then token by token) |
| `edgerock-state-test.cpp` | `llama-edgerock-state-test`: sequence state save/restore, `seq_cp`, partial restore must be bit-exact |
| `fetch_reference.py` | download the reference code/config/tokenizer and range-read the RMA weights of some layers |
| `ref_memory.py` | run the reference `IndependentGQAMemory` on dumped layer inputs and compare |
| `make_tiny.py` | tiny random EdgeRock checkpoint (HF format) for end-to-end tests |
| `ref_full.py` | compare llama.cpp logits with the reference `EdgeRockForCausalLM` (transformers==5.7.0) |
