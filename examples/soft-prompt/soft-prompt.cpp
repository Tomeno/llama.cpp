// soft-prompt: tune a soft prompt (K free input embeddings) without backpropagation
//
// The soft prompt is placed in a fixed position in front of every example:
//
//     [before tokens] [K soft embeddings] [after tokens] [example prompt (formatted)] [example response]
//
// and the objective is the mean negative log-likelihood of the response tokens (teacher forcing).
// Optimization is antithetic evolution strategies (ES) with Adam: each step evaluates the population
// theta +- sigma*eps_i as parallel sequences of one context, and estimates the gradient from the loss differences.
// Only forward passes are used, so every model llama.cpp can run with embedding inputs can be tuned.
//
// Initializing from a text gives the token embeddings of that text, so step 0 is exactly the text prompt.

#include "llama.h"
#include "ggml.h"
#include "gguf.h"
#include "json.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#define SP_LOG(...)  do { fprintf(stderr, __VA_ARGS__); fflush(stderr); } while (0)

struct sp_params {
    std::string model;
    std::string train_file;
    std::string val_file;
    std::string init_text  = "You are a helpful assistant.";
    std::string init_file;            // soft prompt GGUF to start from
    std::string out_file   = "soft-prompt.gguf";
    std::string mode       = "train"; // train | eval | generate

    // the soft prompt sits between these two texts, the example is formatted with {prompt} and followed by the response
    std::string before     = "<|im_start|>system\n";
    std::string after      = "<|im_end|>\n";
    std::string example    = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n";

    int32_t n_soft         = 0;       // 0: as many as the tokens of the init text
    int32_t n_gpu_layers   = 999;
    int32_t n_threads      = (int32_t) std::max(1u, std::thread::hardware_concurrency());
    int32_t n_ctx          = 32768;
    int32_t n_batch        = 2048;
    int32_t max_outputs    = 512;     // logits rows per decode call (host memory: rows * n_vocab * 4 bytes)
    int32_t max_resp       = 1024;    // response tokens scored per example

    int32_t n_pairs        = 8;       // population = 2 * n_pairs
    int32_t rank           = 0;       // 0: full-rank noise, r > 0: low-rank noise A B^T
    int32_t n_examples     = 4;       // examples per step
    int32_t n_steps        = 200;
    int32_t eval_every     = 25;
    int32_t n_val          = 0;       // 0: whole validation set
    float   sigma          = 0.05f;   // noise scale, relative to the RMS of the init embeddings
    float   lr             = 0.01f;   // Adam step size, relative to the RMS of the init embeddings
    float   beta1          = 0.9f;
    float   beta2          = 0.999f;
    std::string assign     = "shared"; // shared: all pairs on the same examples, split: every pair on its own examples
    std::string shaping    = "none";   // none | rank (centered ranks of the pair differences)
    uint64_t seed          = 1234;

    // generate mode
    std::string prompt;
    int32_t n_predict      = 512;
    float   temp           = 0.6f;
    float   top_p          = 0.95f;
    int32_t top_k          = 20;
};

static void print_usage(const char * argv0) {
    const sp_params d;
    printf("usage: %s -m MODEL [options]\n\n", argv0);
    printf("modes:\n");
    printf("  --mode train|eval|generate   (default: %s)\n", d.mode.c_str());
    printf("data (JSONL, one {\"prompt\": ..., \"response\": ...} per line):\n");
    printf("  --train FILE                 training examples\n");
    printf("  --val FILE                   validation examples (eval mode uses this one)\n");
    printf("  --max-resp N                 response tokens scored per example (default: %d)\n", d.max_resp);
    printf("soft prompt:\n");
    printf("  --init-text TEXT             start from the token embeddings of TEXT (default: \"%s\")\n", d.init_text.c_str());
    printf("  --init-file FILE             start from a saved soft prompt (overrides --init-text)\n");
    printf("  --n-soft N                   number of soft tokens, padded/truncated from the init (default: init length)\n");
    printf("  --before TEXT / --after TEXT text around the soft prompt (default: a ChatML system turn)\n");
    printf("  --example TEXT               example format, {prompt} is replaced (default: ChatML user turn + <think>)\n");
    printf("  -o FILE                      output soft prompt GGUF (default: %s)\n", d.out_file.c_str());
    printf("optimization (ES + Adam):\n");
    printf("  --pairs N                    antithetic pairs, population = 2N (default: %d)\n", d.n_pairs);
    printf("  --rank R                     low-rank noise rank, 0 = full rank (default: %d)\n", d.rank);
    printf("  --examples N                 examples per pair and step (default: %d)\n", d.n_examples);
    printf("  --steps N                    (default: %d)\n", d.n_steps);
    printf("  --sigma F                    noise scale relative to the init embedding RMS (default: %.3f)\n", d.sigma);
    printf("  --lr F                       Adam step relative to the init embedding RMS (default: %.3f)\n", d.lr);
    printf("  --assign shared|split        pairs score the same examples, or each pair its own examples (default: %s)\n", d.assign.c_str());
    printf("  --shaping none|rank          fitness shaping of the pair differences (default: %s)\n", d.shaping.c_str());
    printf("  --eval-every N               validation interval in steps (default: %d)\n", d.eval_every);
    printf("  --n-val N                    validation examples used, 0 = all (default: %d)\n", d.n_val);
    printf("  --seed N\n");
    printf("generate mode:\n");
    printf("  -p TEXT                      user prompt\n");
    printf("  -n N, --temp F, --top-p F, --top-k N\n");
    printf("runtime:\n");
    printf("  -ngl N, -t N, -c N, -b N     (defaults: %d, %d, %d, %d)\n", d.n_gpu_layers, d.n_threads, d.n_ctx, d.n_batch);
    printf("  --max-outputs N              logits rows per decode call (default: %d)\n", d.max_outputs);
}

static bool parse_args(int argc, char ** argv, sp_params & p) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + a);
            }
            return argv[++i];
        };
        if      (a == "-m" || a == "--model")   p.model        = next();
        else if (a == "--mode")                 p.mode         = next();
        else if (a == "--train")                p.train_file   = next();
        else if (a == "--val")                  p.val_file     = next();
        else if (a == "--init-text")            p.init_text    = next();
        else if (a == "--init-file")            p.init_file    = next();
        else if (a == "-o" || a == "--out")     p.out_file     = next();
        else if (a == "--before")               p.before       = next();
        else if (a == "--after")                p.after        = next();
        else if (a == "--example")              p.example      = next();
        else if (a == "--n-soft")               p.n_soft       = std::stoi(next());
        else if (a == "-ngl")                   p.n_gpu_layers = std::stoi(next());
        else if (a == "-t")                     p.n_threads    = std::stoi(next());
        else if (a == "-c")                     p.n_ctx        = std::stoi(next());
        else if (a == "-b")                     p.n_batch      = std::stoi(next());
        else if (a == "--max-outputs")          p.max_outputs  = std::stoi(next());
        else if (a == "--max-resp")             p.max_resp     = std::stoi(next());
        else if (a == "--pairs")                p.n_pairs      = std::stoi(next());
        else if (a == "--rank")                 p.rank         = std::stoi(next());
        else if (a == "--examples")             p.n_examples   = std::stoi(next());
        else if (a == "--steps")                p.n_steps      = std::stoi(next());
        else if (a == "--eval-every")           p.eval_every   = std::stoi(next());
        else if (a == "--n-val")                p.n_val        = std::stoi(next());
        else if (a == "--sigma")                p.sigma        = std::stof(next());
        else if (a == "--lr")                   p.lr           = std::stof(next());
        else if (a == "--assign")               p.assign       = next();
        else if (a == "--shaping")              p.shaping      = next();
        else if (a == "--seed")                 p.seed         = std::stoull(next());
        else if (a == "-p")                     p.prompt       = next();
        else if (a == "-n")                     p.n_predict    = std::stoi(next());
        else if (a == "--temp")                 p.temp         = std::stof(next());
        else if (a == "--top-p")                p.top_p        = std::stof(next());
        else if (a == "--top-k")                p.top_k        = std::stoi(next());
        else if (a == "-h" || a == "--help")  { print_usage(argv[0]); exit(0); }
        else {
            fprintf(stderr, "unknown argument: %s\n", a.c_str());
            print_usage(argv[0]);
            return false;
        }
    }
    if (p.model.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

//
// soft prompt storage
//

struct soft_prompt {
    int64_t n_embd = 0;
    int64_t n_soft = 0;
    std::vector<float> data; // [n_soft][n_embd]
};

// rows of token_embd.weight, dequantized, read straight from the model file
static std::vector<float> read_token_embeddings(const std::string & fname, const std::vector<llama_token> & ids, int64_t n_embd) {
    ggml_context * meta = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ true, /*.ctx =*/ &meta };
    gguf_context * gctx = gguf_init_from_file(fname.c_str(), ip);
    if (!gctx) {
        throw std::runtime_error("failed to read " + fname);
    }
    const int64_t tid = gguf_find_tensor(gctx, "token_embd.weight");
    if (tid < 0) {
        throw std::runtime_error("token_embd.weight not found in " + fname + " (split models are not supported here)");
    }
    const ggml_tensor * t = ggml_get_tensor(meta, "token_embd.weight");
    GGML_ASSERT(t->ne[0] == n_embd);

    const size_t row_size = ggml_row_size(t->type, n_embd);
    const size_t base     = gguf_get_data_offset(gctx) + gguf_get_tensor_offset(gctx, tid);
    const auto * traits   = ggml_get_type_traits(t->type);

    std::vector<float> out(ids.size() * n_embd);
    std::vector<uint8_t> row(row_size);
    FILE * f = fopen(fname.c_str(), "rb");
    for (size_t i = 0; i < ids.size(); ++i) {
        GGML_ASSERT(ids[i] >= 0 && ids[i] < t->ne[1]);
        if (fseeko(f, (off_t) (base + (size_t) ids[i] * row_size), SEEK_SET) != 0 || fread(row.data(), 1, row_size, f) != row_size) {
            throw std::runtime_error("failed to read token embeddings");
        }
        if (t->type == GGML_TYPE_F32) {
            memcpy(out.data() + i*n_embd, row.data(), row_size);
        } else {
            traits->to_float(row.data(), out.data() + i*n_embd, n_embd);
        }
    }
    fclose(f);
    gguf_free(gctx);
    ggml_free(meta);
    return out;
}

static void save_soft_prompt(const std::string & fname, const soft_prompt & sp, const sp_params & p, int step, float val_loss) {
    ggml_init_params ip = { ggml_tensor_overhead() + sp.data.size()*sizeof(float) + 1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, sp.n_embd, sp.n_soft);
    ggml_set_name(t, "soft_prompt.embd");
    memcpy(t->data, sp.data.data(), ggml_nbytes(t));

    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.type", "soft_prompt");
    gguf_set_val_str(g, "soft_prompt.before",  p.before.c_str());
    gguf_set_val_str(g, "soft_prompt.after",   p.after.c_str());
    gguf_set_val_str(g, "soft_prompt.example", p.example.c_str());
    gguf_set_val_str(g, "soft_prompt.init_text", p.init_text.c_str());
    gguf_set_val_i32(g, "soft_prompt.step", step);
    gguf_set_val_f32(g, "soft_prompt.val_loss", val_loss);
    gguf_add_tensor(g, t);
    gguf_write_to_file(g, fname.c_str(), false);
    gguf_free(g);
    ggml_free(ctx);
}

static soft_prompt load_soft_prompt(const std::string & fname, sp_params & p) {
    ggml_context * ctx = nullptr;
    gguf_init_params ip = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx };
    gguf_context * g = gguf_init_from_file(fname.c_str(), ip);
    if (!g) {
        throw std::runtime_error("failed to read " + fname);
    }
    auto get_str = [&](const char * key, std::string & dst) {
        const int64_t k = gguf_find_key(g, key);
        if (k >= 0) {
            dst = gguf_get_val_str(g, k);
        }
    };
    // the soft prompt only makes sense with the texts it was trained with
    get_str("soft_prompt.before",  p.before);
    get_str("soft_prompt.after",   p.after);
    get_str("soft_prompt.example", p.example);

    const ggml_tensor * t = ggml_get_tensor(ctx, "soft_prompt.embd");
    if (!t || t->type != GGML_TYPE_F32) {
        throw std::runtime_error("soft_prompt.embd (F32) not found in " + fname);
    }
    soft_prompt sp;
    sp.n_embd = t->ne[0];
    sp.n_soft = t->ne[1];
    sp.data.assign((const float *) t->data, (const float *) t->data + ggml_nelements(t));
    gguf_free(g);
    ggml_free(ctx);
    return sp;
}

//
// data
//

struct sp_example {
    std::vector<llama_token> post; // after text + formatted prompt
    std::vector<llama_token> resp; // scored tokens
};

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> out(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), false, true);
    if (n < 0) {
        out.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), out.data(), (int32_t) out.size(), false, true);
    }
    out.resize(std::max(n, 0));
    return out;
}

static std::string format_example(const std::string & fmt, const std::string & prompt) {
    std::string s = fmt;
    const size_t pos = s.find("{prompt}");
    if (pos == std::string::npos) {
        return s + prompt;
    }
    return s.replace(pos, 8, prompt);
}

static std::vector<sp_example> load_examples(const std::string & fname, const llama_vocab * vocab, const sp_params & p) {
    std::ifstream in(fname);
    if (!in) {
        throw std::runtime_error("failed to open " + fname);
    }
    std::vector<sp_example> out;
    std::string line;
    size_t n_resp = 0;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
            continue;
        }
        const auto j = common_json::parse(line);
        sp_example ex;
        ex.post = tokenize(vocab, p.after + format_example(p.example, j.at("prompt").get<std::string>()));
        ex.resp = tokenize(vocab, j.at("response").get<std::string>());
        if ((int) ex.resp.size() > p.max_resp) {
            ex.resp.resize(p.max_resp);
        }
        if (ex.resp.empty() || ex.post.empty()) {
            continue;
        }
        n_resp += ex.resp.size();
        out.push_back(std::move(ex));
    }
    SP_LOG("%s: %zu examples, %zu scored tokens\n", fname.c_str(), out.size(), n_resp);
    return out;
}

//
// scoring
//

struct sp_scorer {
    llama_context     * ctx;
    const llama_vocab * vocab;
    int32_t n_vocab;
    int32_t n_embd;
    int32_t n_pos;   // positions per embedding row
    const sp_params & p;
    std::vector<llama_token> before;

    struct item {
        llama_seq_id seq;
        llama_token  tok;
        llama_pos    pos;
        bool         output;
    };

    void decode_items(const std::vector<item> & items) {
        llama_batch b = llama_batch_init((int32_t) items.size(), 0, 1);
        b.n_tokens = (int32_t) items.size();
        for (size_t k = 0; k < items.size(); ++k) {
            b.token[k]     = items[k].tok;
            b.pos[k]       = items[k].pos;
            b.n_seq_id[k]  = 1;
            b.seq_id[k][0] = items[k].seq;
            b.logits[k]    = items[k].output;
        }
        const int ret = llama_decode(ctx, b);
        llama_batch_free(b);
        if (ret != 0) {
            throw std::runtime_error("llama_decode failed: " + std::to_string(ret));
        }
    }

    // decode a different soft prompt on every sequence
    void decode_soft(const std::vector<const float *> & embd, int n_soft, llama_pos pos0) {
        const int n_seq = (int) embd.size();
        const int n     = n_soft * n_seq;
        llama_batch b = llama_batch_init(n, n_embd, 1);

        // embedding rows of M-RoPE models carry n_pos positions each, section-major;
        // [p, p, p, 0] is what a text token at position p gets
        std::vector<llama_pos> pos((size_t) n * n_pos);
        llama_pos * pos_own = b.pos;
        b.pos = pos.data();

        b.n_tokens = 0;
        for (int s = 0; s < n_seq; ++s) {
            for (int i = 0; i < n_soft; ++i) {
                const int k = b.n_tokens++;
                memcpy(b.embd + (size_t) k*n_embd, embd[s] + (size_t) i*n_embd, n_embd*sizeof(float));
                for (int j = 0; j < n_pos; ++j) {
                    pos[(size_t) j*n + k] = n_pos == 1 || j < 3 ? pos0 + i : 0;
                }
                b.n_seq_id[k]  = 1;
                b.seq_id[k][0] = s;
                b.logits[k]    = false;
            }
        }
        const int ret = llama_decode(ctx, b);
        b.pos = pos_own;
        llama_batch_free(b);
        if (ret != 0) {
            throw std::runtime_error("llama_decode (embd) failed: " + std::to_string(ret));
        }
    }

    // tokens per sequence that fit in one batch
    int chunk(int n_seq) const {
        return std::max(1, p.n_batch / n_seq);
    }

    // processes the prefix (before + soft prompt) on sequences 0 .. embd.size()-1, returns the next position
    llama_pos prefix(const std::vector<const float *> & embd, int n_soft) {
        const int n_seq = (int) embd.size();
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_pos pos = 0;
        for (int i = 0; i < (int) before.size(); i += chunk(n_seq)) {
            const int n = std::min(chunk(n_seq), (int) before.size() - i);
            std::vector<item> items;
            for (int s = 0; s < n_seq; ++s) {
                for (int k = 0; k < n; ++k) {
                    items.push_back({ s, before[i + k], pos + k, false });
                }
            }
            decode_items(items);
            pos += n;
        }
        for (int i = 0; i < n_soft; i += chunk(n_seq)) {
            const int n = std::min(chunk(n_seq), n_soft - i);
            std::vector<const float *> part(embd.size());
            for (size_t s = 0; s < embd.size(); ++s) {
                part[s] = embd[s] + (size_t) i*n_embd;
            }
            decode_soft(part, n, pos);
            pos += n;
        }
        return pos;
    }

    // sequence s runs soft prompt embd[s] on example exs[s]: adds the response NLL to nll[s] and the token count to n_tok[s]
    void score(const std::vector<const float *> & embd, int n_soft, const std::vector<const sp_example *> & exs,
               std::vector<double> & nll, std::vector<int64_t> & n_tok) {
        const int n_seq = (int) embd.size();
        const llama_pos pos = prefix(embd, n_soft);

        // logits at s0 .. n_dec-1 predict the response tokens; the last token is only a target
        std::vector<std::vector<llama_token>> toks(n_seq);
        std::vector<int> n_dec(n_seq), s0(n_seq);
        int n_max = 0;
        for (int s = 0; s < n_seq; ++s) {
            toks[s] = exs[s]->post;
            toks[s].insert(toks[s].end(), exs[s]->resp.begin(), exs[s]->resp.end());
            n_dec[s] = (int) toks[s].size() - 1;
            s0[s]    = (int) exs[s]->post.size() - 1;
            n_tok[s] += n_dec[s] - s0[s];
            n_max = std::max(n_max, n_dec[s]);
        }
        const int n_big   = chunk(n_seq);
        const int n_small = std::max(1, std::min(n_big, p.max_outputs / n_seq));

        for (int i0 = 0; i0 < n_max; ) {
            // small chunks once any sequence produces outputs in the window, so that the logits fit
            int n = n_big;
            for (int s = 0; s < n_seq; ++s) {
                if (i0 < n_dec[s] && i0 + n_big > s0[s]) {
                    n = n_small;
                }
            }
            std::vector<item> items;
            struct row { int seq; int idx; llama_token target; }; // idx: position in the batch
            std::vector<row> rows;
            for (int s = 0; s < n_seq; ++s) {
                for (int i = i0; i < std::min(i0 + n, n_dec[s]); ++i) {
                    const bool out = i >= s0[s];
                    if (out) {
                        rows.push_back({ s, (int) items.size(), toks[s][i + 1] });
                    }
                    items.push_back({ s, toks[s][i], pos + i, out });
                }
            }
            decode_items(items);

            // llama_get_logits_ith can reorder the outputs on first access, so it must not run concurrently
            std::vector<const float *> logits(rows.size());
            for (size_t r = 0; r < rows.size(); ++r) {
                logits[r] = llama_get_logits_ith(ctx, rows[r].idx);
            }

            std::vector<double> row_nll(rows.size());
            std::atomic<size_t> next(0);
            auto worker = [&]() {
                for (size_t r; (r = next++) < rows.size(); ) {
                    const float * l = logits[r];
                    float mx = l[0];
                    for (int v = 1; v < n_vocab; ++v) {
                        mx = std::max(mx, l[v]);
                    }
                    double sum = 0.0;
                    for (int v = 0; v < n_vocab; ++v) {
                        sum += std::exp((double) (l[v] - mx));
                    }
                    row_nll[r] = (double) mx + std::log(sum) - (double) l[rows[r].target];
                }
            };
            std::vector<std::thread> th;
            for (int k = 0; k < std::min<int>(p.n_threads, (int) rows.size()); ++k) {
                th.emplace_back(worker);
            }
            for (auto & x : th) {
                x.join();
            }
            for (size_t r = 0; r < rows.size(); ++r) {
                nll[rows[r].seq] += row_nll[r];
            }
            i0 += n;
        }
    }

    // mean NLL per scored token; sequence s runs embd[s] on the examples exs[s] (all lists equally long)
    std::vector<double> loss(const std::vector<const float *> & embd, int n_soft, const std::vector<std::vector<const sp_example *>> & exs) {
        const size_t n_seq = embd.size();
        std::vector<double>  nll(n_seq, 0.0);
        std::vector<int64_t> n_tok(n_seq, 0);
        for (size_t r = 0; r < exs[0].size(); ++r) {
            std::vector<const sp_example *> cur(n_seq);
            for (size_t s = 0; s < n_seq; ++s) {
                cur[s] = exs[s][r];
            }
            score(embd, n_soft, cur, nll, n_tok);
        }
        for (size_t s = 0; s < n_seq; ++s) {
            nll[s] /= std::max<int64_t>(1, n_tok[s]);
        }
        return nll;
    }

    // all sequences on the same examples
    std::vector<double> loss(const std::vector<const float *> & embd, int n_soft, const std::vector<const sp_example *> & exs) {
        return loss(embd, n_soft, std::vector<std::vector<const sp_example *>>(embd.size(), exs));
    }
};

//
// main
//

static double rms(const std::vector<float> & v) {
    double s = 0.0;
    for (float x : v) {
        s += (double) x*x;
    }
    return std::sqrt(s / std::max<size_t>(1, v.size()));
}

int main(int argc, char ** argv) {
    sp_params p;
    try {
        if (!parse_args(argc, argv, p)) {
            return 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = p.n_gpu_layers;
    llama_model * model = llama_model_load_from_file(p.model.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", p.model.c_str());
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_embd  = llama_model_n_embd_inp(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    try {
        // soft prompt init
        soft_prompt sp;
        if (!p.init_file.empty()) {
            sp = load_soft_prompt(p.init_file, p);
            SP_LOG("loaded %" PRId64 " soft tokens from %s\n", sp.n_soft, p.init_file.c_str());
        } else {
            const auto ids = tokenize(vocab, p.init_text);
            sp.n_embd = n_embd;
            sp.n_soft = (int64_t) ids.size();
            sp.data   = read_token_embeddings(p.model, ids, n_embd);
            SP_LOG("init from %zu tokens of \"%s\"\n", ids.size(), p.init_text.c_str());
        }
        GGML_ASSERT(sp.n_embd == n_embd);
        if (p.n_soft > 0 && p.n_soft != sp.n_soft) {
            // extra slots repeat the init rows cyclically
            std::vector<float> d(p.n_soft * n_embd);
            for (int i = 0; i < p.n_soft; ++i) {
                memcpy(d.data() + (size_t) i*n_embd, sp.data.data() + (size_t) (i % sp.n_soft)*n_embd, n_embd*sizeof(float));
            }
            sp.data   = std::move(d);
            sp.n_soft = p.n_soft;
        }
        const double emb_rms = rms(sp.data);
        SP_LOG("soft prompt: %" PRId64 " x %" PRId64 " (%zu parameters), embedding RMS %.4f\n", sp.n_soft, sp.n_embd, sp.data.size(), emb_rms);

        // training: one sequence per population member; eval: validation examples in parallel
        const int n_pop = p.mode == "train" ? 2*p.n_pairs : p.mode == "eval" ? std::max(1, p.n_pairs) : 1;

        // every decode call carries at least one token per sequence
        if (p.n_batch < n_pop) {
            SP_LOG("raising the batch size from %d to the population size %d\n", p.n_batch, n_pop);
            p.n_batch = n_pop;
        }

        auto cparams = llama_context_default_params();
        cparams.n_ctx           = p.n_ctx;
        cparams.n_batch         = p.n_batch;
        cparams.n_ubatch        = p.n_batch;
        cparams.n_seq_max       = n_pop;
        cparams.kv_unified      = true;
        cparams.n_threads       = p.n_threads;
        cparams.n_threads_batch = p.n_threads;
        llama_context * ctx = llama_init_from_model(model, cparams);
        if (!ctx) {
            throw std::runtime_error("failed to create the context");
        }

        const auto rope   = llama_model_rope_type(model);
        const int  n_pos  = rope == LLAMA_ROPE_TYPE_MROPE || rope == LLAMA_ROPE_TYPE_IMROPE ? 4 : 1;

        sp_scorer sc = { ctx, vocab, n_vocab, n_embd, n_pos, p, tokenize(vocab, p.before) };

        if (p.mode == "generate") {
            const std::vector<const float *> embd = { sp.data.data() };
            llama_pos pos = sc.prefix(embd, (int) sp.n_soft);
            auto toks = tokenize(vocab, p.after + format_example(p.example, p.prompt));
            for (int i = 0; i + 1 < (int) toks.size(); i += sc.chunk(1)) {
                std::vector<sp_scorer::item> items;
                for (int k = i; k < std::min(i + sc.chunk(1), (int) toks.size() - 1); ++k) {
                    items.push_back({ 0, toks[k], pos++, false });
                }
                sc.decode_items(items);
            }
            llama_token cur = toks.back();

            llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(p.top_k));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(p.top_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(p.temp));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist((uint32_t) p.seed));
            for (int i = 0; i < p.n_predict; ++i) {
                sc.decode_items({ { 0, cur, pos++, true } });
                cur = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, cur)) {
                    break;
                }
                char buf[256];
                const int n = llama_token_to_piece(vocab, cur, buf, sizeof(buf), 0, true);
                fwrite(buf, 1, std::max(n, 0), stdout);
                fflush(stdout);
            }
            printf("\n");
            llama_sampler_free(smpl);
            llama_free(ctx);
            llama_model_free(model);
            return 0;
        }

        const auto val = p.val_file.empty() ? std::vector<sp_example>() : load_examples(p.val_file, vocab, p);
        std::vector<const sp_example *> val_set;
        for (size_t i = 0; i < val.size() && (p.n_val <= 0 || (int) i < p.n_val); ++i) {
            val_set.push_back(&val[i]);
        }

        // mean NLL of one soft prompt over a set of examples, n_pop examples at a time
        auto eval_set = [&](const float * theta, int n_soft, const std::vector<const sp_example *> & exs) {
            double  nll   = 0.0;
            int64_t n_tok = 0;
            for (size_t i = 0; i < exs.size(); i += n_pop) {
                const size_t n = std::min<size_t>(n_pop, exs.size() - i);
                std::vector<double>  l(n, 0.0);
                std::vector<int64_t> c(n, 0);
                sc.score(std::vector<const float *>(n, theta), n_soft, { exs.begin() + i, exs.begin() + i + n }, l, c);
                for (size_t s = 0; s < n; ++s) {
                    nll   += l[s];
                    n_tok += c[s];
                }
            }
            return nll / std::max<int64_t>(1, n_tok);
        };
        auto eval_center = [&](const std::vector<float> & theta) {
            return val_set.empty() ? std::nan("") : eval_set(theta.data(), (int) sp.n_soft, val_set);
        };

        if (p.mode == "eval") {
            if (val_set.empty()) {
                throw std::runtime_error("eval mode needs --val");
            }
            SP_LOG("val loss: soft prompt %.5f, no soft prompt %.5f (over %zu examples)\n",
                    eval_set(sp.data.data(), (int) sp.n_soft, val_set), eval_set(sp.data.data(), 0, val_set), val_set.size());
            llama_free(ctx);
            llama_model_free(model);
            return 0;
        }

        if (p.mode != "train") {
            throw std::runtime_error("unknown mode " + p.mode);
        }
        const auto train = load_examples(p.train_file, vocab, p);
        if (train.empty()) {
            throw std::runtime_error("no training examples");
        }

        const size_t  n_par   = sp.data.size();
        const int64_t K       = sp.n_soft;
        const int     R       = p.rank;
        const float   sigma   = (float) (p.sigma * emb_rms);
        const float   lr      = (float) (p.lr * emb_rms);

        std::vector<float> theta = sp.data;
        std::vector<float> m(n_par, 0.0f), v(n_par, 0.0f), g(n_par);
        std::vector<std::vector<float>> eps(p.n_pairs, std::vector<float>(n_par));
        std::vector<std::vector<float>> members(2*p.n_pairs, std::vector<float>(n_par));
        std::vector<float> a((size_t) K * std::max(R, 1)), bb((size_t) n_embd * std::max(R, 1));

        std::mt19937_64 rng(p.seed);
        std::normal_distribution<float> normal(0.0f, 1.0f);

        double best_val = eval_center(theta);
        SP_LOG("step %5d: val %.5f (init)\n", 0, best_val);
        save_soft_prompt(p.out_file, sp, p, 0, (float) best_val);

        std::vector<size_t> order(train.size());
        for (size_t i = 0; i < order.size(); ++i) {
            order[i] = i;
        }
        size_t cursor = order.size();

        for (int step = 1; step <= p.n_steps; ++step) {
            const auto t0 = std::chrono::steady_clock::now();

            // examples, without replacement within an epoch:
            // shared: every pair sees the same n_examples, split: every pair sees its own n_examples
            auto draw = [&]() {
                if (cursor >= order.size()) {
                    std::shuffle(order.begin(), order.end(), rng);
                    cursor = 0;
                }
                return &train[order[cursor++]];
            };
            std::vector<std::vector<const sp_example *>> exs(2*p.n_pairs);
            for (int i = 0; i < p.n_pairs; ++i) {
                if (i == 0 || p.assign == "split") {
                    for (int k = 0; k < p.n_examples; ++k) {
                        exs[2*i].push_back(draw());
                    }
                } else {
                    exs[2*i] = exs[0];
                }
                // both members of a pair see the same examples, so their difference cancels the example difficulty
                exs[2*i + 1] = exs[2*i];
            }

            // antithetic population theta +- sigma*eps_i
            for (int i = 0; i < p.n_pairs; ++i) {
                auto & e = eps[i];
                if (R <= 0) {
                    for (auto & x : e) {
                        x = normal(rng);
                    }
                } else {
                    for (auto & x : a)  x = normal(rng);
                    for (auto & x : bb) x = normal(rng);
                    const float s = 1.0f / std::sqrt((float) R);
                    for (int64_t k = 0; k < K; ++k) {
                        for (int d = 0; d < n_embd; ++d) {
                            float acc = 0.0f;
                            for (int r = 0; r < R; ++r) {
                                acc += a[k*R + r] * bb[(size_t) d*R + r];
                            }
                            e[k*n_embd + d] = acc * s;
                        }
                    }
                }
                for (size_t j = 0; j < n_par; ++j) {
                    members[2*i + 0][j] = theta[j] + sigma*e[j];
                    members[2*i + 1][j] = theta[j] - sigma*e[j];
                }
            }
            std::vector<const float *> embd;
            for (auto & x : members) {
                embd.push_back(x.data());
            }

            const auto L = sc.loss(embd, (int) K, exs);

            // gradient estimate from the antithetic differences
            std::vector<double> diff(p.n_pairs);
            double mean_pop = 0.0;
            for (int i = 0; i < p.n_pairs; ++i) {
                diff[i] = L[2*i] - L[2*i + 1];
                mean_pop += L[2*i] + L[2*i + 1];
            }
            mean_pop /= 2*p.n_pairs;
            // fitness shaping; Adam makes the overall scale irrelevant, only the weights of the pairs matter
            std::vector<double> w = diff;
            if (p.shaping == "rank" && p.n_pairs > 1) {
                // centered ranks of the differences, robust to outlier examples
                std::vector<int> idx(p.n_pairs);
                for (int i = 0; i < p.n_pairs; ++i) {
                    idx[i] = i;
                }
                std::sort(idx.begin(), idx.end(), [&](int x, int y) { return diff[x] < diff[y]; });
                for (int r = 0; r < p.n_pairs; ++r) {
                    w[idx[r]] = (double) r / (p.n_pairs - 1) - 0.5;
                }
            }
            std::fill(g.begin(), g.end(), 0.0f);
            for (int i = 0; i < p.n_pairs; ++i) {
                const float c = (float) (w[i] / (2.0 * sigma * p.n_pairs));
                for (size_t j = 0; j < n_par; ++j) {
                    g[j] += c * eps[i][j];
                }
            }

            // Adam
            const float bc1 = 1.0f - std::pow(p.beta1, (float) step);
            const float bc2 = 1.0f - std::pow(p.beta2, (float) step);
            for (size_t j = 0; j < n_par; ++j) {
                m[j] = p.beta1*m[j] + (1.0f - p.beta1)*g[j];
                v[j] = p.beta2*v[j] + (1.0f - p.beta2)*g[j]*g[j];
                theta[j] -= lr * (m[j]/bc1) / (std::sqrt(v[j]/bc2) + 1e-8f);
            }

            double drift = 0.0;
            for (size_t j = 0; j < n_par; ++j) {
                drift += (double) (theta[j] - sp.data[j]) * (theta[j] - sp.data[j]);
            }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            SP_LOG("step %5d: train %.5f (population mean), |dL| %.2e, drift %.3f RMS, %.1f s\n",
                    step, mean_pop, std::fabs(diff[0]), std::sqrt(drift / n_par) / emb_rms, dt);

            if (step % p.eval_every == 0 || step == p.n_steps) {
                const double vl = eval_center(theta);
                soft_prompt cur = sp;
                cur.data = theta;
                if (val_set.empty() || vl < best_val) {
                    best_val = vl;
                    save_soft_prompt(p.out_file, cur, p, step, (float) vl);
                }
                save_soft_prompt(p.out_file + ".last", cur, p, step, (float) vl);
                SP_LOG("step %5d: val %.5f (best %.5f)\n", step, vl, best_val);
            }
        }

        llama_free(ctx);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        llama_model_free(model);
        return 1;
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
