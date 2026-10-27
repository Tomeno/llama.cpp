// soft-prompt: the performance-critical parts of forward-only soft prompt tuning, as a C API for Python (ctypes)
//
// A soft prompt is K free input embeddings placed in front of every request:
//
//     [before tokens] [K soft embeddings] [post tokens (rest of the template + prompt)] [response]
//
// The library evaluates many soft prompts at once, one per sequence of a single context:
//   - sp_score:    log-probs of given responses (teacher forcing), every job with its own soft prompt
//   - sp_generate: sampled responses, every prompt with the same soft prompt
// Optionally the recurrent state after the prefix is replaced too (initial state tuning, see sp_get_state).
// Everything else (rewards, ES, Adam, data, training loops) lives in the Python scripts next to this file.

#include "soft-prompt.h"

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#    define SP_API extern "C" __declspec(dllexport)
#else
#    define SP_API extern "C" __attribute__((visibility("default")))
#endif

struct sp_ctx {
    std::string         model_path;
    llama_model       * model = nullptr;
    llama_context     * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    int32_t n_vocab     = 0;
    int32_t n_embd      = 0;
    int32_t n_pos       = 1;  // positions per embedding row (4 for M-RoPE)
    int32_t n_seq       = 1;  // sequences of the context
    int32_t n_batch     = 0;
    int32_t n_threads   = 1;
    int32_t max_outputs = 512;

    std::vector<llama_token> before;

    std::vector<std::pair<int32_t, int64_t>> state_layers; // (layer, floats) of the recurrent states
    int64_t n_state = 0;
};

static std::string g_error;

// phase timing (SP_PROFILE=1): synchronizes after every phase and prints a summary after every sp_score call
struct sp_prof {
    enum { PREFIX, STATE, DECODE, LOGITS, SOFTMAX, N };
    bool    on = getenv("SP_PROFILE") != nullptr;
    int64_t t[N] = {};
    int64_t n_dec = 0, n_tok = 0, n_out = 0;
    int64_t t0 = 0;
    void start(llama_context * ctx) { if (on) { llama_synchronize(ctx); t0 = ggml_time_us(); } }
    void stop(llama_context * ctx, int phase) { if (on) { llama_synchronize(ctx); const int64_t t1 = ggml_time_us(); t[phase] += t1 - t0; t0 = t1; } }
};
static sp_prof g_prof;

// per-op timing (SP_PROFILE=2): the scheduler synchronizes after every observed node (fusion is off, so totals are inflated)
struct sp_op_prof {
    std::map<std::string, std::pair<int64_t, int64_t>> t; // "op name" -> (us, count)
    int64_t t0 = 0;
};
static sp_op_prof g_op_prof;

static bool op_prof_cb(ggml_tensor * t, bool ask, void *) {
    const int64_t now = ggml_time_us();
    if (ask) {
        g_op_prof.t0 = now;
        return true;
    }
    std::string name = t->name;
    name.erase(std::find_if(name.begin(), name.end(), [](char ch) { return ch == '-' || ch == ' ' || ch == '('; }), name.end());
    while (!name.empty() && isdigit((unsigned char) name.back())) {
        name.pop_back();
    }
    for (const std::string & k : { std::string(ggml_op_desc(t)), std::string(ggml_op_desc(t)) + " " + name }) {
        auto & e = g_op_prof.t[k];
        e.first += now - g_op_prof.t0;
        e.second++;
    }
    return true;
}

static void op_prof_print() {
    std::vector<std::pair<std::string, std::pair<int64_t, int64_t>>> v(g_op_prof.t.begin(), g_op_prof.t.end());
    std::sort(v.begin(), v.end(), [](const auto & a, const auto & b) { return a.second.first > b.second.first; });
    int64_t tot = 0;
    for (auto & x : v) {
        tot += x.first.find(' ') == std::string::npos ? x.second.first : 0;
    }
    fprintf(stderr, "ops: %.2f s total\n", tot*1e-6);
    for (size_t i = 0; i < std::min<size_t>(v.size(), 60); ++i) {
        fprintf(stderr, "  %6.2f s %5.1f%% %7lld x  %s\n", v[i].second.first*1e-6, 100.0*v[i].second.first/tot,
                (long long) v[i].second.second, v[i].first.c_str());
    }
    g_op_prof = sp_op_prof();
}

struct sp_item {
    llama_seq_id seq;
    llama_token  tok;
    llama_pos    pos;
    bool         output;
};

static void decode_items(sp_ctx * c, const std::vector<sp_item> & items) {
    llama_batch b = llama_batch_init((int32_t) items.size(), 0, 1);
    b.n_tokens = (int32_t) items.size();
    for (size_t k = 0; k < items.size(); ++k) {
        b.token[k]     = items[k].tok;
        b.pos[k]       = items[k].pos;
        b.n_seq_id[k]  = 1;
        b.seq_id[k][0] = items[k].seq;
        b.logits[k]    = items[k].output;
    }
    const int ret = llama_decode(c->ctx, b);
    llama_batch_free(b);
    if (ret != 0) {
        throw std::runtime_error("llama_decode failed: " + std::to_string(ret));
    }
}

// decode a different soft prompt on every sequence
static void decode_soft(sp_ctx * c, const std::vector<const float *> & embd, int n_soft, llama_pos pos0) {
    const int ns = (int) embd.size();
    const int n  = n_soft * ns;
    llama_batch b = llama_batch_init(n, c->n_embd, 1);

    // embedding rows of M-RoPE models carry n_pos positions each, section-major;
    // [p, p, p, 0] is what a text token at position p gets
    std::vector<llama_pos> pos((size_t) n * c->n_pos);
    llama_pos * pos_own = b.pos;
    b.pos = pos.data();

    b.n_tokens = 0;
    for (int s = 0; s < ns; ++s) {
        for (int i = 0; i < n_soft; ++i) {
            const int k = b.n_tokens++;
            memcpy(b.embd + (size_t) k*c->n_embd, embd[s] + (size_t) i*c->n_embd, c->n_embd*sizeof(float));
            for (int j = 0; j < c->n_pos; ++j) {
                pos[(size_t) j*n + k] = c->n_pos == 1 || j < 3 ? pos0 + i : 0;
            }
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = s;
            b.logits[k]    = false;
        }
    }
    const int ret = llama_decode(c->ctx, b);
    b.pos = pos_own;
    llama_batch_free(b);
    if (ret != 0) {
        throw std::runtime_error("llama_decode (embd) failed: " + std::to_string(ret));
    }
}

// replaces the recurrent state of sequence s ([n_state] floats, layer after layer)
static void set_state(sp_ctx * c, llama_seq_id s, const float * state) {
    for (auto [il, n] : c->state_layers) {
        if (llama_memory_seq_state_set(c->ctx, s, il, state) != n) {
            throw std::runtime_error("failed to set the state of layer " + std::to_string(il));
        }
        state += n;
    }
}

// tokens per sequence that fit in one batch
static int chunk(const sp_ctx * c, int ns) {
    return std::max(1, c->n_batch / ns);
}

// clears the memory and processes the prefix (before + soft prompt) on sequences 0 .. embd.size()-1,
// returns the next position
static llama_pos prefix(sp_ctx * c, const std::vector<const float *> & embd, int n_soft) {
    const int ns = (int) embd.size();
    llama_memory_clear(llama_get_memory(c->ctx), true);
    llama_pos pos = 0;
    for (int i = 0; i < (int) c->before.size(); i += chunk(c, ns)) {
        const int n = std::min(chunk(c, ns), (int) c->before.size() - i);
        std::vector<sp_item> items;
        for (int s = 0; s < ns; ++s) {
            for (int k = 0; k < n; ++k) {
                items.push_back({ s, c->before[i + k], pos + k, false });
            }
        }
        decode_items(c, items);
        pos += n;
    }
    for (int i = 0; i < n_soft; i += chunk(c, ns)) {
        const int n = std::min(chunk(c, ns), n_soft - i);
        std::vector<const float *> part(embd.size());
        for (size_t s = 0; s < embd.size(); ++s) {
            part[s] = embd[s] + (size_t) i*c->n_embd;
        }
        decode_soft(c, part, n, pos);
        pos += n;
    }
    return pos;
}

struct sp_seq_job {
    const float       * theta;
    const float       * state; // nullptr: the state the prefix leaves
    const llama_token * post;
    int                 n_post;
    const llama_token * resp;
    int                 n_resp;
    float             * logp; // n_resp outputs
};

// one round: sequence s scores job s
static void score_round(sp_ctx * c, const std::vector<sp_seq_job> & jobs, int n_soft, float temp) {
    const int ns = (int) jobs.size();
    std::vector<const float *> embd(ns);
    for (int s = 0; s < ns; ++s) {
        embd[s] = jobs[s].theta;
    }
    g_prof.start(c->ctx);
    const llama_pos pos = prefix(c, embd, n_soft);
    g_prof.stop(c->ctx, sp_prof::PREFIX);
    g_prof.start(c->ctx);
    for (int s = 0; s < ns; ++s) {
        if (jobs[s].state) {
            set_state(c, s, jobs[s].state);
        }
    }
    g_prof.stop(c->ctx, sp_prof::STATE);

    // logits at s0 .. n_dec-1 predict the response tokens; the last token is only a target
    std::vector<std::vector<llama_token>> toks(ns);
    std::vector<int> n_dec(ns), s0(ns), n_done(ns, 0);
    int n_max = 0;
    for (int s = 0; s < ns; ++s) {
        toks[s].assign(jobs[s].post, jobs[s].post + jobs[s].n_post);
        toks[s].insert(toks[s].end(), jobs[s].resp, jobs[s].resp + jobs[s].n_resp);
        n_dec[s] = (int) toks[s].size() - 1;
        s0[s]    = jobs[s].n_post - 1;
        n_max = std::max(n_max, n_dec[s]);
    }
    const int   n_big   = chunk(c, ns);
    const int   n_small = std::max(1, std::min(n_big, c->max_outputs / ns));
    const float inv_t   = 1.0f / temp;

    for (int i0 = 0; i0 < n_max; ) {
        // small chunks once any sequence produces outputs in the window, so that the logits fit
        int n = n_big;
        for (int s = 0; s < ns; ++s) {
            if (i0 < n_dec[s] && i0 + n_big > s0[s]) {
                n = n_small;
            }
        }
        struct row { int seq; int idx; llama_token target; float * dst; }; // idx: position in the batch
        std::vector<sp_item> items;
        std::vector<row>     rows;
        for (int s = 0; s < ns; ++s) {
            for (int i = i0; i < std::min(i0 + n, n_dec[s]); ++i) {
                const bool out = i >= s0[s];
                if (out) {
                    rows.push_back({ s, (int) items.size(), toks[s][i + 1], jobs[s].logp + n_done[s]++ });
                }
                items.push_back({ s, toks[s][i], pos + i, out });
            }
        }
        g_prof.start(c->ctx);
        decode_items(c, items);
        g_prof.stop(c->ctx, sp_prof::DECODE);
        g_prof.n_dec++;
        g_prof.n_tok += (int64_t) items.size();
        g_prof.n_out += (int64_t) rows.size();

        // llama_get_logits_ith can reorder the outputs on first access, so it must not run concurrently
        std::vector<const float *> logits(rows.size());
        for (size_t r = 0; r < rows.size(); ++r) {
            logits[r] = llama_get_logits_ith(c->ctx, rows[r].idx);
        }
        g_prof.stop(c->ctx, sp_prof::LOGITS);
        g_prof.start(c->ctx);

        std::atomic<size_t> next(0);
        auto worker = [&]() {
            for (size_t r; (r = next++) < rows.size(); ) {
                const float * l = logits[r];
                float mx = l[0];
                for (int v = 1; v < c->n_vocab; ++v) {
                    mx = std::max(mx, l[v]);
                }
                double sum = 0.0;
                for (int v = 0; v < c->n_vocab; ++v) {
                    sum += std::exp((double) ((l[v] - mx) * inv_t));
                }
                *rows[r].dst = (float) ((double) ((l[rows[r].target] - mx) * inv_t) - std::log(sum));
            }
        };
        std::vector<std::thread> th;
        for (int k = 0; k < std::min<int>(c->n_threads, (int) rows.size()); ++k) {
            th.emplace_back(worker);
        }
        for (auto & x : th) {
            x.join();
        }
        g_prof.stop(c->ctx, sp_prof::SOFTMAX);
        i0 += n;
    }
}

//
// C API
//

SP_API const char * sp_last_error(void) {
    return g_error.c_str();
}

SP_API sp_ctx * sp_init(const char * model_path, int32_t n_gpu_layers, int32_t n_ctx, int32_t n_batch,
                        int32_t n_seq, int32_t n_threads, int32_t max_outputs) {
    try {
        llama_backend_init();
        auto * c = new sp_ctx();
        c->model_path  = model_path;
        c->n_seq       = std::max(1, n_seq);
        c->n_batch     = std::max(n_batch, c->n_seq); // every decode call carries at least one token per sequence
        c->n_threads   = n_threads > 0 ? n_threads : (int32_t) std::max(1u, std::thread::hardware_concurrency());
        c->max_outputs = max_outputs;

        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;
        c->model = llama_model_load_from_file(model_path, mparams);
        if (!c->model) {
            throw std::runtime_error(std::string("failed to load ") + model_path);
        }
        c->vocab   = llama_model_get_vocab(c->model);
        c->n_vocab = llama_vocab_n_tokens(c->vocab);
        c->n_embd  = llama_model_n_embd_inp(c->model);

        const auto rope = llama_model_rope_type(c->model);
        c->n_pos = rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;

        auto cparams = llama_context_default_params();
        cparams.n_ctx           = n_ctx;
        cparams.n_batch         = c->n_batch;
        cparams.n_ubatch        = c->n_batch;
        cparams.n_seq_max       = c->n_seq;
        cparams.kv_unified      = true;
        cparams.n_threads       = c->n_threads;
        cparams.n_threads_batch = c->n_threads;
        if (const char * e = getenv("SP_PROFILE"); e && atoi(e) >= 2) {
            cparams.cb_eval = op_prof_cb;
        }
        c->ctx = llama_init_from_model(c->model, cparams);
        if (!c->ctx) {
            llama_model_free(c->model);
            throw std::runtime_error("failed to create the context");
        }
        for (int32_t il = 0; il < llama_model_n_layer(c->model); ++il) {
            const int64_t n = llama_model_layer_is_recr(c->model, il) ? llama_memory_seq_state_get(c->ctx, 0, il, nullptr) : 0;
            if (n > 0) {
                c->state_layers.emplace_back(il, n);
                c->n_state += n;
            }
        }
        return c;
    } catch (const std::exception & e) {
        g_error = e.what();
        return nullptr;
    }
}

SP_API void sp_free(sp_ctx * c) {
    if (c) {
        llama_free(c->ctx);
        llama_model_free(c->model);
        delete c;
    }
}

SP_API int32_t sp_n_embd (const sp_ctx * c) { return c->n_embd;  }
SP_API int32_t sp_n_vocab(const sp_ctx * c) { return c->n_vocab; }
SP_API int32_t sp_n_seq  (const sp_ctx * c) { return c->n_seq;   }
SP_API int64_t sp_n_state(const sp_ctx * c) { return c->n_state; }

// the recurrent state after the prefix (before + soft prompt theta): out [n_state]
SP_API int32_t sp_get_state(sp_ctx * c, const float * theta, int32_t n_soft, float * out) {
    try {
        prefix(c, { theta }, n_soft);
        for (auto [il, n] : c->state_layers) {
            if (llama_memory_seq_state_get(c->ctx, 0, il, out) != n) {
                throw std::runtime_error("failed to get the state of layer " + std::to_string(il));
            }
            out += n;
        }
        return 0;
    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}

SP_API int32_t sp_is_eog(const sp_ctx * c, int32_t tok) {
    return llama_vocab_is_eog(c->vocab, tok);
}

// returns the number of tokens, or -(needed capacity)
SP_API int32_t sp_tokenize(const sp_ctx * c, const char * text, int32_t len, int32_t * out, int32_t cap) {
    return llama_tokenize(c->vocab, text, len, out, cap, false, true);
}

// text of the tokens, special tokens included; returns the length, or -(needed capacity)
SP_API int32_t sp_detokenize(const sp_ctx * c, const int32_t * toks, int32_t n, char * out, int32_t cap) {
    std::string s;
    char buf[256];
    for (int i = 0; i < n; ++i) {
        const int k = llama_token_to_piece(c->vocab, toks[i], buf, sizeof(buf), 0, true);
        s.append(buf, std::max(k, 0));
    }
    if ((int) s.size() > cap) {
        return -(int32_t) s.size();
    }
    memcpy(out, s.data(), s.size());
    return (int32_t) s.size();
}

// the fixed tokens before the soft prompt
SP_API void sp_set_before(sp_ctx * c, const int32_t * toks, int32_t n) {
    c->before.assign(toks, toks + n);
}

// rows of token_embd.weight for the ids, dequantized, read from the model file: out [n][n_embd]
SP_API int32_t sp_token_embeddings(const sp_ctx * c, const int32_t * ids, int32_t n, float * out) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * g = gguf_init_from_file(c->model_path.c_str(), ip);
    if (!g) {
        g_error = "failed to read " + c->model_path;
        return -1;
    }
    const int64_t tid = gguf_find_tensor(g, "token_embd.weight");
    const ggml_tensor * t = tid >= 0 ? ggml_get_tensor(meta, "token_embd.weight") : nullptr;
    int32_t ret = 0;
    if (!t || t->ne[0] != c->n_embd) {
        g_error = "token_embd.weight not found (split models are not supported)";
        ret = -1;
    } else {
        const size_t row_size = ggml_row_size(t->type, c->n_embd);
        const size_t base     = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, tid);
        const auto * traits   = ggml_get_type_traits(t->type);
        std::vector<uint8_t> row(row_size);
        FILE * f = fopen(c->model_path.c_str(), "rb");
        for (int i = 0; i < n && ret == 0; ++i) {
            if (ids[i] < 0 || ids[i] >= t->ne[1] ||
                fseeko(f, (off_t) (base + (size_t) ids[i] * row_size), SEEK_SET) != 0 || fread(row.data(), 1, row_size, f) != row_size) {
                g_error = "failed to read token embeddings";
                ret = -1;
            } else if (t->type == GGML_TYPE_F32) {
                memcpy(out + (size_t) i*c->n_embd, row.data(), row_size);
            } else {
                traits->to_float(row.data(), out + (size_t) i*c->n_embd, c->n_embd);
            }
        }
        fclose(f);
    }
    gguf_free(g);
    ggml_free(meta);
    return ret;
}

// teacher-forced log-probs at temperature temp; job j uses soft prompt thetas[theta_idx[j]] ([n_soft][n_embd] each),
// prompt tokens post[post_off[j] .. + post_len[j]] and response tokens resp[resp_off[j] .. + resp_len[j]];
// writes resp_len[j] log-probs to out[resp_off[j] ..]. Jobs run n_seq at a time.
// states: nullptr, or one recurrent state per soft prompt ([n_state] each) that replaces the one after the prefix.
SP_API int32_t sp_score(sp_ctx * c, const float * thetas, const float * states, int32_t n_soft, const int32_t * theta_idx, int32_t n_jobs,
                        const int32_t * post, const int64_t * post_off, const int32_t * post_len,
                        const int32_t * resp, const int64_t * resp_off, const int32_t * resp_len,
                        float temp, float * out) {
    try {
        const size_t n_theta = (size_t) n_soft * c->n_embd;
        for (int j0 = 0; j0 < n_jobs; j0 += c->n_seq) {
            std::vector<sp_seq_job> jobs;
            for (int j = j0; j < std::min(n_jobs, j0 + c->n_seq); ++j) {
                if (post_len[j] < 1 || resp_len[j] < 1) {
                    throw std::runtime_error("every job needs at least one prompt and one response token");
                }
                jobs.push_back({ thetas + (size_t) theta_idx[j]*n_theta, states ? states + (size_t) theta_idx[j]*c->n_state : nullptr,
                                 post + post_off[j], post_len[j],
                                 resp + resp_off[j], resp_len[j], out + resp_off[j] });
            }
            score_round(c, jobs, n_soft, temp);
        }
        if (g_prof.on) {
            const auto & t = g_prof.t;
            fprintf(stderr, "sp_score: prefix %.2f s, state %.2f s, decode %.2f s, logits %.2f s, softmax %.2f s;"
                    " %lld decodes, %lld tokens, %lld outputs\n", t[0]*1e-6, t[1]*1e-6, t[2]*1e-6, t[3]*1e-6, t[4]*1e-6,
                    (long long) g_prof.n_dec, (long long) g_prof.n_tok, (long long) g_prof.n_out);
            g_prof = sp_prof();
            if (!g_op_prof.t.empty()) {
                op_prof_print();
            }
        }
        return 0;
    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}

// samples a response for every prompt with the soft prompt theta (and the recurrent state, if not nullptr), n_seq sequences at a time;
// out_tok [n_prompts][max_new], out_len: tokens generated (the end-of-generation token included), out_finished: 1 if it ended
SP_API int32_t sp_generate(sp_ctx * c, const float * theta, const float * state, int32_t n_soft, int32_t n_prompts,
                           const int32_t * post, const int64_t * post_off, const int32_t * post_len,
                           int32_t max_new, float temp, float top_p, int32_t top_k, uint64_t seed,
                           int32_t * out_tok, int32_t * out_len, int32_t * out_finished) {
    try {
        for (int w0 = 0; w0 < n_prompts; w0 += c->n_seq) {
            const int ns = std::min(c->n_seq, n_prompts - w0);
            const llama_pos pos0 = prefix(c, std::vector<const float *>(ns, theta), n_soft);
            for (int s = 0; state && s < ns; ++s) {
                set_state(c, s, state);
            }

            std::vector<llama_sampler *> smpl(ns);
            for (int s = 0; s < ns; ++s) {
                smpl[s] = llama_sampler_chain_init(llama_sampler_chain_default_params());
                if (temp <= 0.0f) {
                    llama_sampler_chain_add(smpl[s], llama_sampler_init_greedy());
                } else {
                    llama_sampler_chain_add(smpl[s], llama_sampler_init_top_k(top_k));
                    llama_sampler_chain_add(smpl[s], llama_sampler_init_top_p(top_p, 1));
                    llama_sampler_chain_add(smpl[s], llama_sampler_init_temp(temp));
                    llama_sampler_chain_add(smpl[s], llama_sampler_init_dist((uint32_t) (seed + w0 + s)));
                }
                out_len[w0 + s]      = 0;
                out_finished[w0 + s] = 0;
            }

            std::vector<llama_pos>   pos(ns);
            std::vector<llama_token> last(ns, -1);
            std::vector<bool>        active(ns, true);

            auto take = [&](int s, llama_token t) {
                const int k = w0 + s;
                out_tok[(size_t) k*max_new + out_len[k]++] = t;
                last[s] = t;
                if (llama_vocab_is_eog(c->vocab, t)) {
                    out_finished[k] = 1;
                    active[s] = false;
                } else if (out_len[k] >= max_new) {
                    active[s] = false;
                }
            };

            // the prompts; the first token of a sequence is sampled right after its last prompt token
            int n_max = 0;
            for (int s = 0; s < ns; ++s) {
                n_max = std::max(n_max, post_len[w0 + s]);
            }
            for (int i0 = 0; i0 < n_max; i0 += chunk(c, ns)) {
                std::vector<sp_item> items;
                std::vector<std::pair<int, int>> outs; // (sequence, batch index)
                for (int s = 0; s < ns; ++s) {
                    const int32_t * pt = post + post_off[w0 + s];
                    const int       np = post_len[w0 + s];
                    for (int i = i0; i < std::min(i0 + chunk(c, ns), np); ++i) {
                        const bool o = i + 1 == np;
                        if (o) {
                            outs.emplace_back(s, (int) items.size());
                        }
                        items.push_back({ s, pt[i], pos0 + i, o });
                    }
                }
                decode_items(c, items);
                for (auto [s, idx] : outs) {
                    pos[s] = pos0 + post_len[w0 + s];
                    take(s, llama_sampler_sample(smpl[s], c->ctx, idx));
                }
            }

            // one token per active sequence and step
            while (true) {
                std::vector<sp_item> items;
                std::vector<int>     seqs;
                for (int s = 0; s < ns; ++s) {
                    if (active[s]) {
                        seqs.push_back(s);
                        items.push_back({ s, last[s], pos[s]++, true });
                    }
                }
                if (items.empty()) {
                    break;
                }
                decode_items(c, items);
                for (size_t k = 0; k < seqs.size(); ++k) {
                    take(seqs[k], llama_sampler_sample(smpl[seqs[k]], c->ctx, (int32_t) k));
                }
            }
            for (auto * s : smpl) {
                llama_sampler_free(s);
            }
        }
        return 0;
    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}
