// C API of libllama-soft-prompt, see soft-prompt.cpp and softprompt.py
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sp_ctx sp_ctx;

const char * sp_last_error(void);

sp_ctx * sp_init(const char * model_path, int32_t n_gpu_layers, int32_t n_ctx, int32_t n_batch,
                 int32_t n_seq, int32_t n_threads);
void     sp_free(sp_ctx * c);

int32_t sp_n_embd (const sp_ctx * c);
int32_t sp_n_vocab(const sp_ctx * c);
int32_t sp_n_seq  (const sp_ctx * c);
int64_t sp_n_state(const sp_ctx * c);
int32_t sp_is_eog (const sp_ctx * c, int32_t tok);

int32_t sp_tokenize  (const sp_ctx * c, const char * text, int32_t len, int32_t * out, int32_t cap);
int32_t sp_detokenize(const sp_ctx * c, const int32_t * toks, int32_t n, char * out, int32_t cap);

void    sp_set_before      (sp_ctx * c, const int32_t * toks, int32_t n);
void    sp_set_init_state  (sp_ctx * c, const float * state);
int32_t sp_token_embeddings(const sp_ctx * c, const int32_t * ids, int32_t n, float * out);
int32_t sp_get_state       (sp_ctx * c, const float * theta, int32_t n_soft, float * out);

int32_t sp_score(sp_ctx * c, const float * thetas, const float * states, int32_t n_soft, const int32_t * theta_idx, int32_t n_jobs,
                 const int32_t * post, const int64_t * post_off, const int32_t * post_len,
                 const int32_t * resp, const int64_t * resp_off, const int32_t * resp_len,
                 float temp, float * out);

// gradient of sum_t w[t] * log p(resp[t]) with respect to the initial recurrent state (before the first prompt token, no
// prefix): job j starts from states[state_idx[j]] and adds its gradient to grad[state_idx[j]] ([n_states, n_state], not
// cleared); logp gets the log-probs (laid out as resp, w too). Every job must fit in one batch, unless chunked.
// sp_state_grad with gradient checkpointing in segments of n_layers layers (0: off)
void    sp_set_grad_ckpt(sp_ctx * c, int32_t n_layers);
// sp_state_grad decodes up to n_seqs jobs of similar length at once (default 1), padded to the longest; all of them must fit
// in one batch together
void    sp_set_grad_seqs(sp_ctx * c, int32_t n_seqs);
// sp_state_grad: jobs longer than n_tokens (0: off, the default) are backpropagated through time in chunks of n_tokens
// tokens (RWKV7): any length with the memory of one chunk, one more forward pass. carry != 0: exact (the gradient
// crosses the chunk boundaries); 0: truncated BPTT (every chunk from its forward state, no gradient from later chunks)
void    sp_set_grad_chunk(sp_ctx * c, int32_t n_tokens, int32_t carry);

int32_t sp_state_grad(sp_ctx * c, const float * states, const int32_t * state_idx, int32_t n_jobs,
                      const int32_t * post, const int64_t * post_off, const int32_t * post_len,
                      const int32_t * resp, const int64_t * resp_off, const int32_t * resp_len,
                      const float * w, float temp, float * logp, float * grad);

// LoRA: loads an F32 GGUF adapter and activates it at scale; train_a / train_b: its A / B tensors are parameters of
// sp_state_grad, which adds their gradients to an accumulator (sp_lora_grad, cleared by sp_lora_grad_clear). Returns the
// number of tensors (A and B of every weight, sorted by name), -1 on error. One adapter per context.
int32_t      sp_lora_load      (sp_ctx * c, const char * path, float scale, int32_t train_a, int32_t train_b);
void         sp_lora_scale     (sp_ctx * c, float scale);
const char * sp_lora_name      (const sp_ctx * c, int32_t i);
int64_t      sp_lora_shape     (const sp_ctx * c, int32_t i, int64_t * ne);
int64_t      sp_lora_get       (const sp_ctx * c, int32_t i, float * dst);
int64_t      sp_lora_set       (sp_ctx * c, int32_t i, const float * src);
int64_t      sp_lora_grad      (sp_ctx * c, int32_t i, float * dst);
void         sp_lora_grad_clear(sp_ctx * c);

// with SP_CAPTURE=1 in the environment: the outputs of all layers but the last for the last n_last tokens of the last
// decode, [n_layer - 1, n_last, n_embd]; the decode must hold one sequence
int32_t sp_capture(const sp_ctx * c, int32_t n_last, float * out);

int32_t sp_generate(sp_ctx * c, const float * theta, const float * state, int32_t n_soft, int32_t n_prompts,
                    const int32_t * post, const int64_t * post_off, const int32_t * post_len,
                    int32_t max_new, float temp, float top_p, int32_t top_k, uint64_t seed,
                    int32_t * out_tok, int32_t * out_len, int32_t * out_finished);

#ifdef __cplusplus
}
#endif
