// edgerock-logits: logits of every token of a text, for comparisons with a reference implementation
//
//   llama-edgerock-logits MODEL TEXT_FILE N_TOTAL N_PREFILL N_UBATCH OUT_FILE
//
// The first N_PREFILL tokens are decoded in one batch (ubatches of N_UBATCH), the rest one token at a time, so the
// prefill, ubatch splitting and single-token decode paths all contribute. OUT_FILE: int32 tokens [N_TOTAL], then
// f32 logits [N_TOTAL][n_vocab].

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s MODEL TEXT_FILE N_TOTAL N_PREFILL N_UBATCH OUT_FILE\n", argv[0]);
        return 1;
    }
    std::ifstream in(argv[2]);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    const int n_total   = atoi(argv[3]);
    const int n_prefill = atoi(argv[4]);
    const int n_ubatch  = atoi(argv[5]);

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    std::vector<llama_token> toks(text.size() + 16);
    toks.resize(llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true));
    if ((int) toks.size() < n_total) {
        fprintf(stderr, "the text has only %zu tokens\n", toks.size());
        return 1;
    }
    toks.resize(n_total);

    auto cp = llama_context_default_params();
    cp.n_ctx    = std::max(4096, n_total + 16);
    cp.n_batch  = std::max(n_prefill, 1);
    cp.n_ubatch = n_ubatch;
    llama_context * ctx = llama_init_from_model(model, cp);

    FILE * f = fopen(argv[6], "wb");
    fwrite(toks.data(), sizeof(llama_token), n_total, f);
    auto run = [&](int i0, int n) {
        llama_batch b = llama_batch_init(n, 0, 1);
        for (int i = 0; i < n; ++i) {
            b.token[i] = toks[i0 + i];
            b.pos[i] = i0 + i;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = 0;
            b.logits[i] = 1;
        }
        b.n_tokens = n;
        if (llama_decode(ctx, b)) {
            fprintf(stderr, "decode failed\n");
            exit(1);
        }
        for (int i = 0; i < n; ++i) {
            fwrite(llama_get_logits_ith(ctx, i), sizeof(float), n_vocab, f);
        }
        llama_batch_free(b);
    };
    run(0, n_prefill);
    for (int i = n_prefill; i < n_total; ++i) {
        run(i, 1);
    }
    fclose(f);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
