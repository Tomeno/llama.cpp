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
// cleared); logp gets the log-probs (laid out as resp, w too). Every job must fit in one batch.
int32_t sp_state_grad(sp_ctx * c, const float * states, const int32_t * state_idx, int32_t n_jobs,
                      const int32_t * post, const int64_t * post_off, const int32_t * post_len,
                      const int32_t * resp, const int64_t * resp_off, const int32_t * resp_len,
                      const float * w, float temp, float * logp, float * grad);

int32_t sp_generate(sp_ctx * c, const float * theta, const float * state, int32_t n_soft, int32_t n_prompts,
                    const int32_t * post, const int64_t * post_off, const int32_t * post_len,
                    int32_t max_new, float temp, float top_p, int32_t top_k, uint64_t seed,
                    int32_t * out_tok, int32_t * out_len, int32_t * out_finished);

#ifdef __cplusplus
}
#endif
