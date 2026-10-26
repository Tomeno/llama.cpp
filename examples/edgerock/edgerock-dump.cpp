// edgerock-dump: decodes a text and appends named graph tensors of every ubatch to <outdir>/<name>.bin (f32, row-major)
//
//   llama-edgerock-dump MODEL TEXT_FILE OUTDIR N_PREFILL N_UBATCH N_TOTAL NAME...
//
// The first N_PREFILL tokens are decoded in one batch (split into ubatches of N_UBATCH), the rest one token at a time.
// Names are graph tensor names, e.g. l_out-2 (output of layer 2 = input of layer 3) and mem_out-3 (RMA output of layer 3).
// The tokens go to OUTDIR/tokens.bin (int32).

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

static std::set<std::string> g_names;
static std::string           g_outdir;

static bool cb_eval(ggml_tensor * t, bool ask, void *) {
    if (!g_names.count(t->name)) {
        return false;
    }
    if (ask) {
        return true;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t));
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    FILE * f = fopen((g_outdir + "/" + t->name + ".bin").c_str(), "ab");
    fwrite(buf.data(), sizeof(float), buf.size(), f);
    fclose(f);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 8) {
        fprintf(stderr, "usage: %s MODEL TEXT_FILE OUTDIR N_PREFILL N_UBATCH N_TOTAL NAME...\n", argv[0]);
        return 1;
    }
    std::ifstream in(argv[2]);
    std::stringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    g_outdir = argv[3];
    const int n_prefill = atoi(argv[4]);
    const int n_ubatch  = atoi(argv[5]);
    const int n_total   = atoi(argv[6]);
    for (int i = 7; i < argc; ++i) {
        g_names.insert(argv[i]);
    }

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(text.size() + 16);
    toks.resize(llama_tokenize(vocab, text.c_str(), (int32_t) text.size(), toks.data(), (int32_t) toks.size(), false, true));
    if ((int) toks.size() < n_total) {
        fprintf(stderr, "the text has only %zu tokens\n", toks.size());
        return 1;
    }
    toks.resize(n_total);
    FILE * ft = fopen((g_outdir + "/tokens.bin").c_str(), "wb");
    fwrite(toks.data(), sizeof(llama_token), toks.size(), ft);
    fclose(ft);

    auto cp = llama_context_default_params();
    cp.n_ctx    = std::max(8192, n_total + 16);
    cp.n_batch  = std::max(n_prefill, 1);
    cp.n_ubatch = n_ubatch;
    cp.cb_eval  = cb_eval;
    llama_context * ctx = llama_init_from_model(model, cp);

    if (llama_decode(ctx, llama_batch_get_one(toks.data(), n_prefill))) {
        fprintf(stderr, "decode failed\n");
        return 1;
    }
    for (int i = n_prefill; i < n_total; ++i) {
        if (llama_decode(ctx, llama_batch_get_one(toks.data() + i, 1))) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
    }
    fprintf(stderr, "dumped %d tokens\n", n_total);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
