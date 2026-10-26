// edgerock-state-test: sequence state round trips must reproduce generation bit-exactly
//
//   llama-edgerock-state-test MODEL TEXT_FILE N_PROMPT
//
// Decodes N_PROMPT tokens of the text, saves the sequence state (full and partial = what server checkpoints use),
// generates greedily, then checks: restore into seq 0, restore into seq 1, seq_cp 0 -> 1, partial restore.
// Also prints whether a partial seq_rm is accepted (it must not be for recurrent/hybrid memory).

#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static llama_context * ctx;
static int n_vocab;

static void decode(const std::vector<llama_token> & toks, int pos0, int seq, bool last_logits) {
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); ++i) {
        b.token[i] = toks[i];
        b.pos[i] = pos0 + (int) i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = seq;
        b.logits[i] = last_logits && i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    if (llama_decode(ctx, b)) {
        fprintf(stderr, "decode failed\n");
        exit(1);
    }
    llama_batch_free(b);
}

static std::vector<llama_token> gen(llama_token first, int pos0, int seq, int n, std::vector<float> & logits) {
    std::vector<llama_token> out;
    logits.clear();
    llama_token t = first;
    for (int i = 0; i < n; ++i) {
        decode({ t }, pos0 + i, seq, true);
        const float * l = llama_get_logits_ith(ctx, -1);
        logits.insert(logits.end(), l, l + n_vocab);
        int best = 0;
        for (int j = 1; j < n_vocab; ++j) {
            if (l[j] > l[best]) {
                best = j;
            }
        }
        t = best;
        out.push_back(t);
    }
    return out;
}

static bool check(const char * name, const std::vector<llama_token> & a, const std::vector<llama_token> & b,
                  const std::vector<float> & la, const std::vector<float> & lb) {
    float md = 0;
    for (size_t i = 0; i < la.size(); ++i) {
        md = fmaxf(md, fabsf(la[i] - lb[i]));
    }
    printf("%-42s tokens %s, max |dlogit| = %g\n", name, a == b ? "match" : "DIFFER", md);
    return a == b && md == 0.0f;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL TEXT_FILE N_PROMPT\n", argv[0]);
        return 1;
    }
    std::ifstream in(argv[2]);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    const int n_prompt = atoi(argv[3]), n_gen = 24;

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(text.size() + 16);
    toks.resize(llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true));
    if ((int) toks.size() < n_prompt + 1) {
        fprintf(stderr, "the text has only %zu tokens\n", toks.size());
        return 1;
    }
    toks.resize(n_prompt + 1);
    const llama_token next = toks.back();
    toks.pop_back();

    auto cp = llama_context_default_params();
    cp.n_ctx = 8192;
    cp.n_batch = 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = 2;
    cp.kv_unified = true;
    ctx = llama_init_from_model(model, cp);
    llama_memory_t mem = llama_get_memory(ctx);

    decode(toks, 0, 0, false);
    std::vector<uint8_t> full(llama_state_seq_get_size(ctx, 0));
    full.resize(llama_state_seq_get_data(ctx, full.data(), full.size(), 0));
    std::vector<uint8_t> part(llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    part.resize(llama_state_seq_get_data_ext(ctx, part.data(), part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY));
    printf("prompt %d tokens: seq state %.1f MiB, partial (checkpoint) state %.1f MiB\n", n_prompt, full.size()/1048576.0, part.size()/1048576.0);

    bool ok = true;
    std::vector<float> l0, l1;
    const auto ref = gen(next, n_prompt, 0, n_gen, l0);

    llama_memory_clear(mem, true);
    ok &= llama_state_seq_set_data(ctx, full.data(), full.size(), 0) > 0;
    ok &= check("restore into seq 0", ref, gen(next, n_prompt, 0, n_gen, l1), l0, l1);

    llama_memory_clear(mem, true);
    ok &= llama_state_seq_set_data(ctx, full.data(), full.size(), 1) > 0;
    ok &= check("restore into seq 1", ref, gen(next, n_prompt, 1, n_gen, l1), l0, l1);

    llama_memory_clear(mem, true);
    decode(toks, 0, 0, false);
    llama_memory_seq_cp(mem, 0, 1, -1, -1);
    ok &= check("seq_cp 0 -> 1, generate on 1", ref, gen(next, n_prompt, 1, n_gen, l1), l0, l1);
    ok &= check("then generate on 0 (untouched by seq 1)", ref, gen(next, n_prompt, 0, n_gen, l1), l0, l1);

    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_memory_seq_rm(mem, 1, -1, -1);
    ok &= llama_state_seq_set_data_ext(ctx, part.data(), part.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) > 0;
    ok &= check("partial (checkpoint) restore into seq 0", ref, gen(next, n_prompt, 0, n_gen, l1), l0, l1);

    const bool rm = llama_memory_seq_rm(mem, 0, n_prompt + n_gen - 1, -1);
    printf("partial seq_rm (drop the last token) accepted: %s\n", rm ? "yes (unexpected)" : "no (expected)");

    printf("%s\n", ok && !rm ? "PASS" : "FAIL");
    llama_free(ctx);
    llama_model_free(model);
    return ok && !rm ? 0 : 1;
}
