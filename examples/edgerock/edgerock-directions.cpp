// edgerock-directions: mean layer outputs over the last tokens of a set of prompts, for finding a linear direction
//
//   llama-edgerock-directions MODEL PROMPTS OUT N_LAST
//
// PROMPTS: complete formatted prompts separated by NUL bytes. OUT: raw f32 [n_prompts, n_layer, N_LAST, n_embd]
// (l_out-il at each of the N_LAST last tokens of the prompt), one row per prompt so that a caller can hold some out.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static int g_n_last  = 0;
static int g_n_embd  = 0;
static int g_n_layer = 0;
static std::vector<float> g_row;

static bool cb_eval(ggml_tensor * t, bool ask, void *) {
    int il = -1;
    if (sscanf(t->name, "l_out-%d", &il) != 1) {
        return false;
    }
    if (ask) {
        return true;
    }
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && t->ne[0] == g_n_embd);
    std::vector<float> buf(ggml_nelements(t));
    ggml_backend_tensor_get(t, buf.data(), 0, ggml_nbytes(t));
    // the last layer may hold only the output tokens; its missing rows stay zero
    for (int k = 0; k < g_n_last && k < t->ne[1]; ++k) {
        const int64_t src = t->ne[1] - 1 - k;
        std::copy_n(buf.data() + src*g_n_embd, g_n_embd,
                    g_row.data() + ((int64_t) il*g_n_last + g_n_last - 1 - k)*g_n_embd);
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL PROMPTS OUT N_LAST\n", argv[0]);
        return 1;
    }
    g_n_last = atoi(argv[4]);

    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss;
    ss << in.rdbuf();
    std::vector<std::string> prompts;
    std::string p;
    std::istringstream is(ss.str());
    while (std::getline(is, p, '\0')) {
        prompts.push_back(p);
    }

    llama_backend_init();
    llama_model * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) {
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    g_n_embd  = llama_model_n_embd(model);
    g_n_layer = llama_model_n_layer(model);
    g_row.resize((int64_t) g_n_layer*g_n_last*g_n_embd);

    auto cp = llama_context_default_params();
    cp.n_ctx    = 4096;
    cp.n_batch  = 4096;
    cp.n_ubatch = 4096;
    cp.n_seq_max = 1;
    cp.cb_eval  = cb_eval;
    llama_context * ctx = llama_init_from_model(model, cp);

    FILE * f = fopen(argv[3], "wb");
    for (size_t i = 0; i < prompts.size(); ++i) {
        std::vector<llama_token> toks(prompts[i].size() + 16);
        toks.resize(llama_tokenize(vocab, prompts[i].c_str(), (int32_t) prompts[i].size(), toks.data(), (int32_t) toks.size(), false, true));
        llama_memory_clear(llama_get_memory(ctx), true);
        std::fill(g_row.begin(), g_row.end(), 0.0f);
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), (int32_t) toks.size()))) {
            fprintf(stderr, "decode failed\n");
            return 1;
        }
        llama_synchronize(ctx);
        fwrite(g_row.data(), sizeof(float), g_row.size(), f);
    }
    fclose(f);
    fprintf(stderr, "%zu prompts, %d layers, last %d tokens\n", prompts.size(), g_n_layer, g_n_last);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
