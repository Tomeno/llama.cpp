// beam-search: length-normalized beam search over a JSONL of prompts, several prompts at a time
//
//   llama-beam-search -m MODEL -i PROMPTS.jsonl -o OUT.jsonl [--beams 4] [-n 2048] [--alpha 1.0] [--state SP.gguf]
//                     [--par 4] [-c 65536] [--template T] [--no-repeat-ngram N] [--verify]
//
// Every input row {"prompt": ...} (other fields are copied to the output) is put into --template ("{}" = the prompt)
// and searched with W beams: every step decodes all live beams of all running prompts in one batch, and the beams
// that survive are forked from their parents with llama_memory_seq_cp (each prompt owns 2 W sequences, used in turns).
// A hypothesis ends at an end-of-generation token; the search of a prompt stops once W have ended and no live beam can
// beat them (HF early_stopping=False), or at -n tokens. The best by logp / length^alpha is written with its thinking
// and answer split at </think>, as runs/chat-eval/chat_eval.py rows.
// --state: a soft-prompt file (examples/soft-prompt) that holds only an initial recurrent state, set before the prompt.
// --no-repeat-ngram N: a beam never generates the same N-gram twice (HF no_repeat_ngram_size).
// --verify: re-scores every result teacher-forced on a fresh sequence; the sum of log-probs must match the search's.

#include "llama.h"
#include "llama-ext.h"
#include "common.h"
#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

struct beam {
    llama_seq_id             seq;
    std::vector<llama_token> toks;
    double                   logp;
    int                      i_batch = -1; // row of its logits in the last decode
};

struct hyp {
    std::vector<llama_token> toks;
    double                   logp;
    double                   score;
    bool                     ended;
};

struct job {
    json                     row;
    std::vector<llama_token> prompt;
    int                      seq0;     // owns seq0 .. seq0 + 2 W - 1
    int                      side = 0; // which half holds the live beams
    std::vector<beam>        beams;
    std::vector<hyp>         done;
    bool                     finished = false;
    int                      n_steps = 0;
};

static llama_context * ctx;
static const llama_vocab * vocab;
static std::vector<float> init_state;

static void fail(const std::string & msg) {
    fprintf(stderr, "beam-search: %s\n", msg.c_str());
    exit(1);
}

static void load_state(const std::string & path) {
    ggml_context * gctx = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &gctx };
    gguf_context * g = gguf_init_from_file(path.c_str(), gp);
    if (!g) {
        fail("failed to read " + path);
    }
    const ggml_tensor * st   = ggml_get_tensor(gctx, "soft_prompt.state");
    const ggml_tensor * embd = ggml_get_tensor(gctx, "soft_prompt.embd");
    if (!st || st->type != GGML_TYPE_F32 || (embd && embd->ne[1] > 0)) {
        fail(path + " is not a state-only soft prompt (F32 soft_prompt.state, no soft rows)");
    }
    init_state.assign((const float *) st->data, (const float *) st->data + ggml_nelements(st));
    gguf_free(g);
    ggml_free(gctx);
}

// an empty sequence with the initial state (or nothing without --state)
static void seed(llama_seq_id seq) {
    llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
    if (init_state.empty()) {
        return;
    }
    if (!llama_memory_seq_seed(ctx, seq)) {
        fail("failed to seed a sequence");
    }
    const llama_model * model = llama_get_model(ctx);
    const float * src = init_state.data();
    int64_t n_left = (int64_t) init_state.size();
    for (int32_t il = 0; il < llama_model_n_layer(model); ++il) {
        if (!llama_model_layer_is_recr(model, il)) {
            continue;
        }
        const int64_t n = llama_memory_seq_state_get(ctx, seq, il, nullptr);
        if (n <= 0) {
            continue;
        }
        if (n > n_left || llama_memory_seq_state_set(ctx, seq, il, src) != n) {
            fail("the state does not match the recurrent state of layer " + std::to_string(il));
        }
        src += n;
        n_left -= n;
    }
    if (n_left != 0) {
        fail("the state is larger than the model's recurrent state");
    }
}

// log-softmax of a logits row: (log Z, top-k token ids by logit)
static double log_z(const float * l, int n_vocab) {
    float mx = l[0];
    for (int j = 1; j < n_vocab; ++j) {
        mx = std::max(mx, l[j]);
    }
    double s = 0.0;
    for (int j = 0; j < n_vocab; ++j) {
        s += std::exp((double) (l[j] - mx));
    }
    return mx + std::log(s);
}

static std::vector<llama_token> top_k(const float * l, int n_vocab, int k) {
    std::vector<llama_token> ids(n_vocab);
    for (int j = 0; j < n_vocab; ++j) {
        ids[j] = j;
    }
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](llama_token a, llama_token b) { return l[a] > l[b]; });
    ids.resize(k);
    return ids;
}

static std::string text_of(const std::vector<llama_token> & toks) {
    std::string s;
    for (llama_token t : toks) {
        if (!llama_vocab_is_eog(vocab, t)) {
            s += common_token_to_piece(vocab, t, true);
        }
    }
    return s;
}

// decodes toks on seq from position pos0 in n_batch pieces, logits of the last token only
static void decode_seq(llama_seq_id seq, const std::vector<llama_token> & toks, int pos0) {
    const int n_batch = (int) llama_n_batch(ctx);
    const int n = (int) toks.size();
    for (int i0 = 0; i0 < n; i0 += n_batch) {
        const int m = std::min(n_batch, n - i0);
        llama_batch b = llama_batch_init(m, 0, 1);
        for (int k = 0; k < m; ++k) {
            b.token[k] = toks[i0 + k];
            b.pos[k] = pos0 + i0 + k;
            b.n_seq_id[k] = 1;
            b.seq_id[k][0] = seq;
            b.logits[k] = i0 + k == n - 1;
        }
        b.n_tokens = m;
        if (llama_decode(ctx, b)) {
            fail("decode failed");
        }
        llama_batch_free(b);
    }
}

// sum of log-probs of toks after prompt, teacher-forced on seq (left empty) the way the search decodes them:
// the prompt in one piece, then token by token
static double rescore(llama_seq_id seq, const std::vector<llama_token> & prompt, const std::vector<llama_token> & toks) {
    seed(seq);
    decode_seq(seq, prompt, 0);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    double lp = 0.0;
    for (size_t i = 0; i < toks.size(); ++i) {
        const float * l = llama_get_logits_ith(ctx, -1);
        lp += l[toks[i]] - log_z(l, n_vocab);
        if (i + 1 < toks.size()) {
            decode_seq(seq, { toks[i] }, (int) (prompt.size() + i));
        }
    }
    llama_memory_seq_rm(llama_get_memory(ctx), seq, -1, -1);
    return lp;
}

int main(int argc, char ** argv) {
    std::string model_path, in_path, out_path, state_path;
    std::string tmpl = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n";
    int W = 4, n_max = 2048, n_par = 4, n_ctx = 65536, ngl = 999, no_repeat = 0;
    double alpha = 1.0;
    bool verify = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fail("missing value of " + a);
            }
            return argv[++i];
        };
        if      (a == "-m")         model_path = next();
        else if (a == "-i")         in_path    = next();
        else if (a == "-o")         out_path   = next();
        else if (a == "--state")    state_path = next();
        else if (a == "--template") tmpl       = next();
        else if (a == "--beams")    W          = std::stoi(next());
        else if (a == "-n")         n_max      = std::stoi(next());
        else if (a == "--par")      n_par      = std::stoi(next());
        else if (a == "-c")         n_ctx      = std::stoi(next());
        else if (a == "-ngl")       ngl        = std::stoi(next());
        else if (a == "--alpha")    alpha      = std::stod(next());
        else if (a == "--no-repeat-ngram") no_repeat = std::stoi(next());
        else if (a == "--verify")   verify     = true;
        else fail("unknown argument " + a);
    }
    if (model_path.empty() || in_path.empty() || out_path.empty() || W < 1 || n_par < 1) {
        fail("usage: -m MODEL -i PROMPTS.jsonl -o OUT.jsonl [--beams W] [-n N] [--alpha A] [--state SP] [--par P] "
            "[-c CTX] [--template T] [--no-repeat-ngram N] [--verify]");
    }
    // unescape \n in a template given on the command line
    for (size_t p; (p = tmpl.find("\\n")) != std::string::npos;) {
        tmpl.replace(p, 2, "\n");
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fail("failed to load " + model_path);
    }
    vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = n_ctx;
    cp.n_batch    = 2048;
    cp.n_ubatch   = 512;
    cp.n_seq_max  = n_par * 2 * W + (verify ? 1 : 0);
    cp.kv_unified = true; // seq_cp shares the cells of the attention layers
    ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fail("failed to create the context");
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (!state_path.empty()) {
        load_state(state_path);
    }
    const llama_seq_id seq_verify = n_par * 2 * W;

    std::vector<json> rows;
    {
        std::ifstream in(in_path);
        for (std::string line; std::getline(in, line);) {
            if (!line.empty()) {
                rows.push_back(json::parse(line));
            }
        }
    }
    std::ofstream out(out_path);

    std::vector<std::unique_ptr<job>> slots(n_par);
    size_t next_row = 0, n_done = 0;
    double max_dlp = 0.0;

    // starts row r in slot s: prompt prefill on its first sequence, one live beam
    auto start = [&](int s, size_t r) {
        auto j = std::make_unique<job>();
        j->row = rows[r];
        std::string p = tmpl;
        p.replace(p.find("{}"), 2, j->row["prompt"].get<std::string>());
        j->prompt = common_tokenize(vocab, p, false, true);
        j->seq0 = s * 2 * W;
        for (int k = 0; k < 2 * W; ++k) {
            llama_memory_seq_rm(mem, j->seq0 + k, -1, -1);
        }
        seed(j->seq0);
        decode_seq(j->seq0, j->prompt, 0);
        // the root's logits are in the last row of that decode; it is expanded right away, before other slots decode
        beam root{ j->seq0, {}, 0.0, -1 };
        j->beams.push_back(root);
        return j;
    };

    // expands the live beams of j from the logits rows (root: the last prompt row) into the next W; true if j ended
    auto expand = [&](job & j, bool from_prompt) {
        struct cand { int parent; llama_token tok; double logp; };
        std::vector<cand> cands;
        for (int b = 0; b < (int) j.beams.size(); ++b) {
            const float * l = llama_get_logits_ith(ctx, from_prompt ? -1 : j.beams[b].i_batch);
            const double lz = log_z(l, n_vocab);
            // --no-repeat-ngram: the tokens that would complete an n-gram this beam already generated
            std::vector<llama_token> banned;
            const std::vector<llama_token> & t = j.beams[b].toks;
            const int m = no_repeat - 1, n = (int) t.size();
            if (no_repeat > 0 && n >= m) {
                for (int i = 0; i + m < n; ++i) {
                    if (std::equal(t.begin() + i, t.begin() + i + m, t.end() - m)) {
                        banned.push_back(t[i + m]);
                    }
                }
            }
            int n_cand = 0;
            for (llama_token tok : top_k(l, n_vocab, 2 * W + (int) banned.size())) {
                if (n_cand < 2 * W && std::find(banned.begin(), banned.end(), tok) == banned.end()) {
                    cands.push_back({ b, tok, j.beams[b].logp + l[tok] - lz });
                    n_cand++;
                }
            }
        }
        std::sort(cands.begin(), cands.end(), [](const cand & a, const cand & b) { return a.logp > b.logp; });
        const int len = (int) j.beams[0].toks.size() + 1;
        std::vector<cand> keep;
        for (int rank = 0; rank < (int) cands.size() && (int) keep.size() < W; ++rank) {
            const cand & c = cands[rank];
            if (llama_vocab_is_eog(vocab, c.tok)) {
                if (rank < W) { // HF: an ended hypothesis counts only from the top W
                    std::vector<llama_token> t = j.beams[c.parent].toks;
                    t.push_back(c.tok);
                    j.done.push_back({ t, c.logp, c.logp / std::pow((double) len, alpha), true });
                }
                continue;
            }
            keep.push_back(c);
        }
        std::sort(j.done.begin(), j.done.end(), [](const hyp & a, const hyp & b) { return a.score > b.score; });
        if (j.done.size() > (size_t) W) {
            j.done.resize(W);
        }
        // no live beam can beat the W ended ones (HF early_stopping=False: the best live score at the current length)
        const bool full = (int) j.done.size() >= W && !keep.empty() &&
                          keep[0].logp / std::pow((double) len, alpha) <= j.done.back().score;
        if (len >= n_max || keep.empty() || full) {
            if (len >= n_max) {
                for (const cand & c : keep) {
                    std::vector<llama_token> t = j.beams[c.parent].toks;
                    t.push_back(c.tok);
                    j.done.push_back({ t, c.logp, c.logp / std::pow((double) len, alpha), false });
                }
            }
            j.finished = true;
            return;
        }
        // fork the survivors into the other half of the job's sequences, then free this half
        const int other = 1 - j.side;
        std::vector<beam> nb;
        for (int k = 0; k < (int) keep.size(); ++k) {
            const beam & p = j.beams[keep[k].parent];
            const llama_seq_id dst = j.seq0 + other * W + k;
            llama_memory_seq_rm(mem, dst, -1, -1);
            llama_memory_seq_cp(mem, p.seq, dst, -1, -1);
            beam b{ dst, p.toks, keep[k].logp, -1 };
            b.toks.push_back(keep[k].tok);
            nb.push_back(std::move(b));
        }
        for (const beam & b : j.beams) {
            llama_memory_seq_rm(mem, b.seq, -1, -1);
        }
        j.beams = std::move(nb);
        j.side = other;
        j.n_steps++;
    };

    auto finish = [&](int s) {
        job & j = *slots[s];
        for (int k = 0; k < 2 * W; ++k) {
            llama_memory_seq_rm(mem, j.seq0 + k, -1, -1);
        }
        std::sort(j.done.begin(), j.done.end(), [](const hyp & a, const hyp & b) { return a.score > b.score; });
        const hyp & h = j.done[0];
        const std::string text = text_of(h.toks);
        const size_t k = text.find("</think>");
        json r = j.row;
        r["think"]  = k == std::string::npos ? text : text.substr(0, k);
        r["answer"] = k == std::string::npos ? "" : text.substr(k + strlen("</think>"));
        r["finish"] = h.ended ? "stop" : "length";
        r["tokens"] = (int) h.toks.size();
        r["logp"]   = h.logp;
        r["score"]  = h.score;
        r["n_ended"] = (int) std::count_if(j.done.begin(), j.done.end(), [](const hyp & x) { return x.ended; });
        if (verify) {
            const double lp = rescore(seq_verify, j.prompt, h.toks);
            r["logp_rescored"] = lp;
            max_dlp = std::max(max_dlp, std::fabs(lp - h.logp));
        }
        out << r.dump() << "\n";
        out.flush();
        n_done++;
        fprintf(stderr, "[%zu/%zu] %d tokens, %s, logp %.2f, score %.4f%s\n", n_done, rows.size(), (int) h.toks.size(),
                h.ended ? "stop" : "length", h.logp, h.score,
                verify ? (", rescored " + std::to_string(r["logp_rescored"].get<double>())).c_str() : "");
        slots[s].reset();
    };

    const int64_t t0 = ggml_time_us();
    int64_t n_decoded = 0;
    while (true) {
        // fill free slots (the prompt is expanded straight from its prefill logits)
        for (int s = 0; s < n_par; ++s) {
            while (!slots[s] && next_row < rows.size()) {
                slots[s] = start(s, next_row++);
                expand(*slots[s], true);
                if (slots[s]->finished) {
                    finish(s);
                }
            }
        }
        // one decode: the last token of every live beam
        int n_tok = 0;
        for (auto & j : slots) {
            if (j) {
                n_tok += (int) j->beams.size();
            }
        }
        if (n_tok == 0) {
            break;
        }
        llama_batch b = llama_batch_init(n_tok, 0, 1);
        int i = 0;
        for (auto & j : slots) {
            if (!j) {
                continue;
            }
            for (beam & bm : j->beams) {
                b.token[i] = bm.toks.back();
                b.pos[i] = (llama_pos) (j->prompt.size() + bm.toks.size() - 1);
                b.n_seq_id[i] = 1;
                b.seq_id[i][0] = bm.seq;
                b.logits[i] = true;
                bm.i_batch = i++;
            }
        }
        b.n_tokens = n_tok;
        if (llama_decode(ctx, b)) {
            fail("decode failed (beams)");
        }
        llama_batch_free(b);
        n_decoded += n_tok;
        // every slot reads its rows of this decode before finish() decodes anything (--verify)
        for (auto & j : slots) {
            if (j) {
                expand(*j, false);
            }
        }
        for (int s = 0; s < n_par; ++s) {
            if (slots[s] && slots[s]->finished) {
                finish(s);
            }
        }
    }
    const double dt = (ggml_time_us() - t0) / 1e6;
    fprintf(stderr, "beam-search: %zu prompts, %lld beam tokens decoded in %.1f s (%.1f tok/s)%s\n", rows.size(),
            (long long) n_decoded, dt, n_decoded / dt,
            verify ? (", max |logp - rescored| " + std::to_string(max_dlp)).c_str() : "");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
