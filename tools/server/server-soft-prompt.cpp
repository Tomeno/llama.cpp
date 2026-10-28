#include "server-soft-prompt.h"

#include "common.h"
#include "gguf.h"
#include "src/llama-ext.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>

static void decode_or_throw(llama_context * ctx, llama_batch batch, const char * what) {
    if (llama_decode(ctx, batch) != 0) {
        throw std::runtime_error(std::string("soft prompt: failed to decode the ") + what);
    }
}

// sets the recurrent state of seq_id from src, layer after layer; throws if src does not match the model's state
static void set_state(llama_context * ctx, llama_seq_id seq_id, const float * src, int64_t n_left) {
    const llama_model * model = llama_get_model(ctx);
    for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
        if (!llama_model_layer_is_recr(model, il)) {
            continue;
        }
        const int64_t n = llama_memory_seq_state_get(ctx, seq_id, il, nullptr);
        if (n <= 0) {
            continue;
        }
        if (n > n_left || llama_memory_seq_state_set(ctx, seq_id, il, src) != n) {
            throw std::runtime_error("soft prompt: the state does not match the recurrent state of layer " + std::to_string(il));
        }
        src    += n;
        n_left -= n;
    }
    if (n_left != 0) {
        throw std::runtime_error("soft prompt: the state has " + std::to_string(n_left) + " values more than the model's recurrent state");
    }
}

static void decode_tokens(llama_context * ctx, std::vector<llama_token> toks) {
    const int n_batch = (int) llama_n_batch(ctx);
    for (int i = 0; i < (int) toks.size(); i += n_batch) {
        decode_or_throw(ctx, llama_batch_get_one(toks.data() + i, std::min(n_batch, (int) toks.size() - i)), "prefix tokens");
    }
}

void server_soft_prompt::init(llama_context * ctx, const std::string & path) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    ggml_context * gctx = nullptr;
    gguf_init_params gparams = { /*.no_alloc =*/ false, /*.ctx =*/ &gctx };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gguf(gguf_init_from_file(path.c_str(), gparams), gguf_free);
    std::unique_ptr<ggml_context, decltype(&ggml_free)> gctx_ptr(gctx, ggml_free);
    if (!gguf) {
        throw std::runtime_error("soft prompt: failed to read " + path);
    }
    // the writer leaves out empty strings, so a missing template part is an empty one
    auto str = [&](const char * key) {
        const int64_t k = gguf_find_key(gguf.get(), key);
        return k < 0 ? std::string() : std::string(gguf_get_val_str(gguf.get(), k));
    };
    const ggml_tensor * embd = ggml_get_tensor(gctx, "soft_prompt.embd");
    const ggml_tensor * st   = ggml_get_tensor(gctx, "soft_prompt.state");
    const int n_embd = llama_model_n_embd_inp(model);
    if (!embd || embd->type != GGML_TYPE_F32 || embd->ne[0] != n_embd) {
        throw std::runtime_error("soft prompt: " + path + " has no F32 soft_prompt.embd with " + std::to_string(n_embd) + " columns");
    }
    const int n_soft = (int) embd->ne[1];

    const std::vector<llama_token> before = common_tokenize(vocab, str("soft_prompt.before"), false, true);
    const std::vector<llama_token> after  = common_tokenize(vocab, str("soft_prompt.after"),  false, true);

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    if (before.empty() && after.empty() && n_soft == 0) {
        if (!st || st->type != GGML_TYPE_F32) {
            throw std::runtime_error("soft prompt: " + path + " has neither a prefix nor an F32 state");
        }
        init_state.assign((const float *) st->data, (const float *) st->data + ggml_nelements(st));
        // checks the size on a seeded sequence
        if (!llama_memory_seq_seed(ctx, 0)) {
            throw std::runtime_error("soft prompt: failed to seed a sequence");
        }
        set_state(ctx, 0, init_state.data(), (int64_t) init_state.size());
        llama_memory_seq_rm(mem, 0, -1, -1);
        return;
    }

    decode_tokens(ctx, before);

    // embedding rows of M-RoPE models carry 4 positions each, section-major; [p, p, p, 0] is what a text token gets
    const auto rope  = llama_model_rope_type(model);
    const int  n_pos = rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;
    const int  n_batch = (int) llama_n_batch(ctx);
    llama_pos  pos0 = (llama_pos) before.size();
    for (int i0 = 0; i0 < n_soft; i0 += n_batch) {
        const int n = std::min(n_batch, n_soft - i0);
        llama_batch b = llama_batch_init(n, n_embd, 1);
        std::vector<llama_pos> pos((size_t) n*n_pos);
        llama_pos * pos_own = b.pos;
        b.pos = pos.data();
        b.n_tokens = n;
        memcpy(b.embd, (const float *) embd->data + (size_t) i0*n_embd, (size_t) n*n_embd*sizeof(float));
        for (int k = 0; k < n; ++k) {
            for (int j = 0; j < n_pos; ++j) {
                pos[(size_t) j*n + k] = n_pos == 1 || j < 3 ? pos0 + i0 + k : 0;
            }
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = 0;
            b.logits[k]    = false;
        }
        const int ret = llama_decode(ctx, b);
        b.pos = pos_own;
        llama_batch_free(b);
        if (ret != 0) {
            throw std::runtime_error("soft prompt: failed to decode the soft embeddings");
        }
    }

    if (st) {
        if (st->type != GGML_TYPE_F32) {
            throw std::runtime_error("soft prompt: soft_prompt.state is not F32");
        }
        set_state(ctx, 0, (const float *) st->data, ggml_nelements(st));
    }

    decode_tokens(ctx, after);

    state.resize(llama_state_seq_get_size(ctx, 0));
    if (llama_state_seq_get_data(ctx, state.data(), state.size(), 0) != state.size()) {
        throw std::runtime_error("soft prompt: failed to save the prefix state");
    }
    llama_memory_seq_rm(mem, 0, -1, -1);

    // the soft rows stand as a placeholder token in the prompt cache
    const llama_token ph = llama_vocab_pad(vocab) != LLAMA_TOKEN_NULL ? llama_vocab_pad(vocab) : 0;
    tokens = before;
    tokens.insert(tokens.end(), n_soft, ph);
    tokens.insert(tokens.end(), after.begin(), after.end());
}

bool server_soft_prompt::restore(llama_context * ctx, llama_seq_id seq_id) const {
    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);
    if (tokens.empty()) {
        if (!llama_memory_seq_seed(ctx, seq_id)) {
            return false;
        }
        try {
            set_state(ctx, seq_id, init_state.data(), (int64_t) init_state.size());
        } catch (const std::exception &) {
            return false;
        }
        return true;
    }
    return llama_state_seq_set_data(ctx, state.data(), state.size(), seq_id) == state.size();
}

bool server_soft_prompt::is_prefix_of(const std::vector<llama_token> & toks) const {
    return toks.size() >= tokens.size() && std::equal(tokens.begin(), tokens.end(), toks.begin());
}
