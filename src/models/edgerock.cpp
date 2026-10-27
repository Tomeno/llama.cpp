#include "models.h"
#include "llama-memory-recurrent.h"

#include <cmath>

// EdgeRock: Qwen3.5 with sliding window attention layers, each paired with a Retained Memory Attention (RMA) module.
// RMA keeps up to n_mem_capacity K/V records per KV head. Every token writes a pending record with a learned
// admission score; each n_mem_chunk tokens the pending records are merged into the committed ones, keeping the
// top n_mem_capacity scores. Tokens read only the committed records, with an extra zero "sink" logit.
//
// The records live in the recurrent memory rows of the attention layers:
//   r: pending   K, V [n_embd_head, n_mem_chunk,    n_head_kv] + scores [n_mem_chunk, n_head_kv]
//   s: committed K, V [n_embd_head, n_mem_capacity, n_head_kv] + scores, valid flags [n_mem_capacity, n_head_kv]
// The pending count and the commit points follow from the positions, which start at 0 in every sequence.

// large negative logit/score for the records that are not valid, finite to keep 0*x and x-x well defined
static constexpr float RMA_INVALID = 1e30f;

// gather indices and masks of the commits that fall inside the ubatch, computed from the positions
class llm_graph_input_mem_rma : public llm_graph_input_i {
public:
    llm_graph_input_mem_rma(int64_t n_chunk, int64_t n_head_kv, int64_t n_seq_tokens, int64_t n_seqs) :
        n_chunk(n_chunk), n_head_kv(n_head_kv), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs),
        n_commit((n_chunk - 1 + n_seq_tokens) / n_chunk),
        n_read  ((n_chunk - 2 + n_seq_tokens) / n_chunk + 1) {}

    void set_input(const llama_ubatch * ubatch) override {
        GGML_ASSERT(ubatch->n_seq_tokens == n_seq_tokens && ubatch->n_seqs == n_seqs);

        const int64_t P = n_chunk;
        const int64_t T = n_seq_tokens;

        std::vector<int32_t> commit(P*n_head_kv*n_seqs*n_commit, 0);
        std::vector<float>   on    (n_seqs*n_commit, 0.0f);
        std::vector<int32_t> pend  (P*n_head_kv*n_seqs, 0);
        std::vector<float>   sel   (T*n_seqs*n_read, 0.0f);

        // the pending record of position p is kept in slot p % P, the ubatch tokens follow the P pending slots
        for (int64_t s = 0; s < n_seqs; ++s) {
            const llama_pos p0   = ubatch->pos[s*T];
            const llama_pos base = p0 - p0 % P;

            for (int64_t t = 0; t < T; ++t) {
                GGML_ASSERT(ubatch->pos[s*T + t] == p0 + t && "EdgeRock RMA needs consecutive positions");
                sel[(((p0 + t)/P - p0/P)*n_seqs + s)*T + t] = 1.0f;
            }

            auto src = [&](llama_pos q, int64_t j) -> int32_t {
                return q < p0 ? j : (q < p0 + T ? P + q - p0 : j);
            };

            for (int64_t k = 0; k < n_commit; ++k) {
                const llama_pos b = base + (k + 1)*P;
                if (b > p0 + T) {
                    continue;
                }
                on[k*n_seqs + s] = 1.0f;
                for (int64_t g = 0; g < n_head_kv; ++g) {
                    for (int64_t j = 0; j < P; ++j) {
                        commit[((k*n_seqs + s)*n_head_kv + g)*P + j] = src(b - P + j, j);
                    }
                }
            }

            const llama_pos cb = ((p0 + T)/P)*P;
            for (int64_t g = 0; g < n_head_kv; ++g) {
                for (int64_t j = 0; j < P; ++j) {
                    pend[(s*n_head_kv + g)*P + j] = src(cb + j, j);
                }
            }
        }

        ggml_backend_tensor_set(commit_idx, commit.data(), 0, ggml_nbytes(commit_idx));
        ggml_backend_tensor_set(commit_on,  on.data(),     0, ggml_nbytes(commit_on));
        ggml_backend_tensor_set(pend_idx,   pend.data(),   0, ggml_nbytes(pend_idx));
        if (read_sel) {
            ggml_backend_tensor_set(read_sel, sel.data(), 0, ggml_nbytes(read_sel));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return params.ubatch.n_seq_tokens == n_seq_tokens && params.ubatch.n_seqs == n_seqs;
    }

    const int64_t n_chunk;
    const int64_t n_head_kv;
    const int64_t n_seq_tokens;
    const int64_t n_seqs;
    const int64_t n_commit; // max commits in a ubatch
    const int64_t n_read;   // max committed states read by the tokens of a ubatch

    ggml_tensor * commit_idx = nullptr; // I32 [n_chunk, n_head_kv, n_seqs, n_commit]
    ggml_tensor * commit_on  = nullptr; // F32 [n_seqs, n_commit], 1 if the commit happens
    ggml_tensor * pend_idx   = nullptr; // I32 [n_chunk, n_head_kv, n_seqs]
    ggml_tensor * read_sel   = nullptr; // F32 [n_seq_tokens, n_seqs, n_read], 1 if the token reads that state (n_read > 1)
};

void llama_model_edgerock::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_hparams(ml);

    ml.get_key(LLM_KV_MEMORY_CAPACITY,            hparams.n_mem_capacity);
    ml.get_key(LLM_KV_MEMORY_COMMIT_CHUNK,        hparams.n_mem_chunk);
    ml.get_key(LLM_KV_MEMORY_ADMISSION_RANK,      hparams.n_mem_rank);
    ml.get_key(LLM_KV_MEMORY_WRITER_INTERMEDIATE, hparams.n_mem_writer);

    if (!ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false)) {
        ml.get_key(LLM_KV_ATTENTION_WINDOW, hparams.n_swa);
    }

    // all attention layers are sliding window layers
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = il < hparams.n_layer() && !hparams.is_recr(il);
    }

    // the reference commits every min(commit_chunk, window) tokens
    hparams.n_mem_chunk = std::min(hparams.n_mem_chunk, hparams.n_swa);

    GGML_ASSERT(hparams.n_mem_capacity > 0 && hparams.n_mem_chunk > 0);
}

void llama_model_edgerock::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int flags = nextn_flags(ml).trunk;

    const int64_t n_mem_rank   = hparams.n_mem_rank;
    const int64_t n_mem_writer = hparams.n_mem_writer;

    for (int il = 0; il < n_layer; ++il) {
        if (!hparams.has_mem(il)) {
            continue;
        }

        auto & mem = layers[il].mem;

        mem.input_norm     = create_tensor(tn(LLM_TENSOR_MEM_INPUT_NORM,     "weight", il), { n_embd }, flags);
        mem.q_norm         = create_tensor(tn(LLM_TENSOR_MEM_Q_NORM,         "weight", il), { n_embd_head_k }, flags);
        mem.k_norm         = create_tensor(tn(LLM_TENSOR_MEM_K_NORM,         "weight", il), { n_embd_head_k }, flags);
        mem.wq             = create_tensor(tn(LLM_TENSOR_MEM_Q,              "weight", il), { n_embd, n_embd_head_k * n_head }, flags);
        mem.wk             = create_tensor(tn(LLM_TENSOR_MEM_K,              "weight", il), { n_embd, n_embd_k_gqa }, flags);
        mem.wv             = create_tensor(tn(LLM_TENSOR_MEM_V,              "weight", il), { n_embd, n_embd_v_gqa }, flags);
        mem.wo             = create_tensor(tn(LLM_TENSOR_MEM_OUT,            "weight", il), { n_embd_head_v * n_head, n_embd }, flags);
        mem.write_gate     = create_tensor(tn(LLM_TENSOR_MEM_WRITE_GATE,     "weight", il), { n_embd, n_mem_writer }, flags);
        mem.write_up       = create_tensor(tn(LLM_TENSOR_MEM_WRITE_UP,       "weight", il), { n_embd, n_mem_writer }, flags);
        mem.write_down     = create_tensor(tn(LLM_TENSOR_MEM_WRITE_DOWN,     "weight", il), { n_mem_writer, n_embd }, flags);
        mem.admission_down = create_tensor(tn(LLM_TENSOR_MEM_ADMISSION_DOWN, "weight", il), { n_embd, n_mem_rank }, flags);
        mem.admission_up   = create_tensor(tn(LLM_TENSOR_MEM_ADMISSION_UP,   "weight", il), { n_mem_rank, n_head_kv }, flags);
        mem.read_scale     = create_tensor(tn(LLM_TENSOR_MEM_READ_SCALE,               il), { n_head }, flags);
        mem.read_gate      = create_tensor(tn(LLM_TENSOR_MEM_READ_GATE,                il), { n_embd_head_k, n_head }, flags);
        mem.read_gate_b    = create_tensor(tn(LLM_TENSOR_MEM_READ_GATE_BIAS,           il), { n_head }, flags);
    }
}

std::unique_ptr<llm_graph_context> llama_model_edgerock::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("EdgeRock does not support MTP");
    }
    return std::make_unique<graph>(*this, params);
}

llama_model_edgerock::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    int sections[4];
    std::copy(std::begin(hparams.rope_sections), std::begin(hparams.rope_sections) + 4, sections);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    cb(inpL, "model.input_embed", -1);

    auto * inp = build_inp_mem_hybrid_iswa();

    ggml_tensor * inp_pos     = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    llm_graph_input_mem_rma * inp_mem;
    {
        auto inp_rma = std::make_unique<llm_graph_input_mem_rma>(hparams.n_mem_chunk, n_head_kv, ubatch.n_seq_tokens, ubatch.n_seqs);

        const int64_t P = inp_rma->n_chunk;
        const int64_t S = inp_rma->n_seqs;

        inp_rma->commit_idx = ggml_new_tensor_4d(ctx0, GGML_TYPE_I32, P, n_head_kv, S, inp_rma->n_commit);
        inp_rma->commit_on  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, S, inp_rma->n_commit);
        inp_rma->pend_idx   = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, P, n_head_kv, S);
        ggml_set_input(inp_rma->commit_idx);
        ggml_set_input(inp_rma->commit_on);
        ggml_set_input(inp_rma->pend_idx);

        if (inp_rma->n_read > 1) {
            inp_rma->read_sel = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ubatch.n_seq_tokens, S, inp_rma->n_read);
            ggml_set_input(inp_rma->read_sel);
        }

        inp_mem = (llm_graph_input_mem_rma *) res->add_input(std::move(inp_rma));
    }

    for (int il = 0; il < n_layer; ++il) {
        res->t_layer_inp[il] = inpL;

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            // Linear attention layer (gated delta net)
            cur = build_layer_attn_linear(inp->get_recr(), cur, il);
        } else {
            // Sliding window attention layer + RMA on the un-normed layer input
            cur = build_layer_attn(inp->get_attn(), cur, inp_pos, sections, il);

            ggml_tensor * mem_out = build_layer_mem(inp->get_recr(), inp_mem, inpSA, inp_pos, sections, il);
            cur = ggml_add(ctx0, cur, mem_out);
            cb(cur, "attn_mem_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // Residual connection
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        ggml_tensor * attn_post_norm = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(attn_post_norm, "attn_post_norm", il);

        cur = build_layer_ffn(attn_post_norm, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "post_ffn", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // LM head
    cur = build_lora_mm(model.output, cur, model.output_s);

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

ggml_tensor * llama_model_edgerock::graph::build_layer_mem(
        llm_graph_input_rs      * inp_rs,
        llm_graph_input_mem_rma * inp_mem,
        ggml_tensor             * cur,
        ggml_tensor             * inp_pos,
        int                     * sections,
        int                       il) {
    const auto & mem = model.layers[il].mem;

    const int64_t D   = hparams.n_embd_head_k(il);
    const int64_t H   = hparams.n_head(il);
    const int64_t G   = hparams.n_head_kv(il);
    const int64_t Hg  = H / G;
    const int64_t C   = hparams.n_mem_capacity;
    const int64_t P   = hparams.n_mem_chunk;
    const int64_t T   = ubatch.n_seq_tokens;
    const int64_t S   = ubatch.n_seqs;
    const size_t  esz = sizeof(float);

    GGML_ASSERT(H % G == 0);

    ggml_tensor * u = build_norm(cur, mem.input_norm, nullptr, LLM_NORM_RMS, il);
    cb(u, "mem_input_norm", il);

    // query and read gate (the gate uses the query before RoPE)
    ggml_tensor * q = build_lora_mm(mem.wq, u);
    q = ggml_reshape_3d(ctx0, q, D, H, n_tokens);
    q = build_norm(q, mem.q_norm, nullptr, LLM_NORM_RMS, il);
    cb(q, "mem_q_normed", il);

    ggml_tensor * gate = ggml_sum_rows(ctx0, ggml_mul(ctx0, q, mem.read_gate)); // [1, H, n_tokens]
    gate = ggml_reshape_2d(ctx0, gate, H, n_tokens);
    gate = ggml_scale(ctx0, gate, 1.0f/sqrtf(float(D)));
    gate = ggml_sigmoid(ctx0, ggml_add(ctx0, gate, mem.read_gate_b));
    gate = ggml_mul(ctx0, gate, ggml_scale(ctx0, ggml_tanh(ctx0, mem.read_scale), 2.0f));
    cb(gate, "mem_read_gate", il);

    q = ggml_rope_multi(ctx0, q, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    // records written by this ubatch: the writer MLP refines the input before the K/V projections
    ggml_tensor * z = build_ffn(u,
            mem.write_up,   nullptr, nullptr,
            mem.write_gate, nullptr, nullptr,
            mem.write_down, nullptr, nullptr,
            nullptr,
            LLM_FFN_SILU, LLM_FFN_PAR, il);
    z = ggml_add(ctx0, z, u);
    cb(z, "mem_writer", il);

    ggml_tensor * k = build_lora_mm(mem.wk, z);
    k = ggml_reshape_3d(ctx0, k, D, G, n_tokens);
    k = build_norm(k, mem.k_norm, nullptr, LLM_NORM_RMS, il);
    k = ggml_rope_multi(ctx0, k, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow);

    ggml_tensor * v = build_lora_mm(mem.wv, z);

    ggml_tensor * score = build_lora_mm(mem.admission_up, ggml_silu(ctx0, build_lora_mm(mem.admission_down, u))); // [G, n_tokens]
    cb(score, "mem_admission", il);

    // [D, G, T, S] -> [D, T, G, S]
    k     = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, k, D, G, T, S), 0, 2, 1, 3));
    v     = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, v, D, G, T, S), 0, 2, 1, 3));
    score = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, score, G, T, S), 1, 0, 2, 3));

    // load the records of the sequences in the ubatch
    const auto * mctx_cur = inp_rs->mctx;
    const auto   kv_head  = mctx_cur->get_head();

    ggml_tensor * r_all = mctx_cur->get_r_l(il);
    ggml_tensor * s_all = mctx_cur->get_s_l(il);

    ggml_tensor * pend = build_rs(inp_rs, r_all, hparams.n_embd_r(il), S); // [n_embd_r, S]
    ggml_tensor * comm = build_rs(inp_rs, s_all, hparams.n_embd_s(il), S); // [n_embd_s, S]

    ggml_tensor * pend_k = ggml_view_4d(ctx0, pend, D, P, G, S, D*esz, D*P*esz, pend->nb[1], 0);
    ggml_tensor * pend_v = ggml_view_4d(ctx0, pend, D, P, G, S, D*esz, D*P*esz, pend->nb[1], D*P*G*esz);
    ggml_tensor * pend_s = ggml_view_3d(ctx0, pend,    P, G, S,    P*esz,       pend->nb[1], 2*D*P*G*esz);

    ggml_tensor * comm_k = ggml_cont(ctx0, ggml_view_4d(ctx0, comm, D, C, G, S, D*esz, D*C*esz, comm->nb[1], 0));
    ggml_tensor * comm_v = ggml_cont(ctx0, ggml_view_4d(ctx0, comm, D, C, G, S, D*esz, D*C*esz, comm->nb[1], D*C*G*esz));
    ggml_tensor * comm_s = ggml_cont(ctx0, ggml_view_3d(ctx0, comm,    C, G, S,    C*esz,       comm->nb[1], 2*D*C*G*esz));
    ggml_tensor * comm_f = ggml_cont(ctx0, ggml_view_3d(ctx0, comm,    C, G, S,    C*esz,       comm->nb[1], (2*D*C*G + C*G)*esz));

    // pending slots followed by the records of the ubatch, the commit and pending indices point into these
    ggml_tensor * ext_k = ggml_concat(ctx0, pend_k, k, 1);                                 // [D, P + T, G, S]
    ggml_tensor * ext_v = ggml_concat(ctx0, pend_v, v, 1);
    ggml_tensor * ext_s = ggml_concat(ctx0, pend_s, score, 0);                             // [P + T, G, S]
    ext_s = ggml_reshape_4d(ctx0, ext_s, 1, P + T, G, S);

    auto masked_score = [&](ggml_tensor * s, ggml_tensor * f) {
        // valid: s, invalid: s - RMA_INVALID
        return ggml_add(ctx0, s, ggml_scale_bias(ctx0, f, RMA_INVALID, -RMA_INVALID));
    };

    struct rma_state {
        ggml_tensor * k; // [D, C, G, S]
        ggml_tensor * v; // [D, C, G, S]
        ggml_tensor * s; // [C, G, S]
        ggml_tensor * f; // [C, G, S]
    };

    std::vector<rma_state> states = { { comm_k, comm_v, comm_s, comm_f } };

    for (int64_t ic = 0; ic < inp_mem->n_commit; ++ic) {
        const rma_state & st = states.back();

        ggml_tensor * idx = ggml_view_3d(ctx0, inp_mem->commit_idx, P, G, S,
                inp_mem->commit_idx->nb[1], inp_mem->commit_idx->nb[2], ic*inp_mem->commit_idx->nb[3]);
        ggml_tensor * on  = ggml_view_3d(ctx0, inp_mem->commit_on, 1, 1, S,
                esz, esz, ic*inp_mem->commit_on->nb[1]);
        ggml_tensor * off = ggml_scale_bias(ctx0, on, -1.0f, 1.0f);

        // candidates: the committed records and the P records of the chunk, keep the C best scores
        ggml_tensor * cand_k = ggml_concat(ctx0, st.k, ggml_get_rows(ctx0, ext_k, idx), 1);   // [D, C + P, G, S]
        ggml_tensor * cand_v = ggml_concat(ctx0, st.v, ggml_get_rows(ctx0, ext_v, idx), 1);
        ggml_tensor * cand_s = ggml_concat(ctx0, masked_score(st.s, st.f),
                ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, ext_s, idx), P, G, S), 0);           // [C + P, G, S]

        // argsort instead of top_k: the order of the kept records must not depend on the backend run (CUDA top_k is unordered)
        ggml_tensor * top = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, cand_s, C));              // [C, G, S]

        ggml_tensor * new_k = ggml_get_rows(ctx0, cand_k, top);
        ggml_tensor * new_v = ggml_get_rows(ctx0, cand_v, top);
        ggml_tensor * new_s = ggml_get_rows(ctx0, ggml_reshape_4d(ctx0, cand_s, 1, C + P, G, S), top);
        new_s = ggml_reshape_3d(ctx0, new_s, C, G, S);
        ggml_tensor * new_f = ggml_step(ctx0, ggml_scale_bias(ctx0, new_s, 1.0f, 0.5f*RMA_INVALID));

        // only the sequences that reach a commit point take the merged records
        auto blend = [&](ggml_tensor * a, ggml_tensor * b, ggml_tensor * m_on, ggml_tensor * m_off) {
            return ggml_add(ctx0, ggml_mul(ctx0, a, m_off), ggml_mul(ctx0, b, m_on));
        };

        ggml_tensor * on4  = ggml_reshape_4d(ctx0, on,  1, 1, 1, S);
        ggml_tensor * off4 = ggml_reshape_4d(ctx0, off, 1, 1, 1, S);

        states.push_back({
            blend(st.k, new_k, on4, off4),
            blend(st.v, new_v, on4, off4),
            blend(st.s, new_s, on,  off),
            blend(st.f, new_f, on,  off),
        });
    }

    // read: attention of every token over the committed records it can see, plus a zero sink
    // the query rows of a KV head are grouped as [D, Hg*T, G, S] so that the masks broadcast over them
    ggml_tensor * qg = ggml_reshape_4d(ctx0, q, D*Hg, G, T, S);
    qg = ggml_cont(ctx0, ggml_permute(ctx0, qg, 0, 2, 1, 3));
    qg = ggml_reshape_4d(ctx0, qg, D, Hg*T, G, S);

    ggml_tensor * read = nullptr;

    for (int64_t ir = 0; ir < inp_mem->n_read; ++ir) {
        const rma_state & st = states[ir];

        ggml_tensor * ks = ggml_pad(ctx0, st.k, 0, 1, 0, 0);                                   // [D, C + 1, G, S]
        ggml_tensor * vs = ggml_pad(ctx0, st.v, 0, 1, 0, 0);
        vs = ggml_cont(ctx0, ggml_transpose(ctx0, vs));                                        // [C + 1, D, G, S]

        ggml_tensor * bias = ggml_scale_bias(ctx0, st.f, RMA_INVALID, -RMA_INVALID);
        bias = ggml_pad(ctx0, ggml_reshape_4d(ctx0, bias, C, 1, G, S), 1, 0, 0, 0);           // [C + 1, 1, G, S]

        ggml_tensor * kq = ggml_mul_mat(ctx0, ks, qg);                                         // [C + 1, Hg*T, G, S]
        kq = ggml_scale(ctx0, kq, 1.0f/sqrtf(float(D)));
        kq = ggml_soft_max(ctx0, ggml_add(ctx0, kq, bias));

        ggml_tensor * out = ggml_mul_mat(ctx0, vs, kq);                                        // [D, Hg*T, G, S]
        out = ggml_reshape_4d(ctx0, out, D*Hg, T, G, S);
        out = ggml_cont(ctx0, ggml_permute(ctx0, out, 0, 2, 1, 3));                            // [D*Hg, G, T, S]
        out = ggml_reshape_3d(ctx0, out, D*H, T, S);

        if (inp_mem->n_read > 1) {
            ggml_tensor * sel = ggml_view_3d(ctx0, inp_mem->read_sel, 1, T, S,
                    esz, inp_mem->read_sel->nb[1], ir*inp_mem->read_sel->nb[2]);
            out = ggml_mul(ctx0, out, sel);
        }

        read = read ? ggml_add(ctx0, read, out) : out;
    }
    cb(read, "mem_read", il);

    // store the records
    {
        const rma_state & st = states.back();

        ggml_tensor * new_pk = ggml_get_rows(ctx0, ext_k, inp_mem->pend_idx);                  // [D, P, G, S]
        ggml_tensor * new_pv = ggml_get_rows(ctx0, ext_v, inp_mem->pend_idx);
        ggml_tensor * new_ps = ggml_get_rows(ctx0, ext_s, inp_mem->pend_idx);                  // [1, P, G, S]

        auto store = [&](ggml_tensor * src, ggml_tensor * dst_all, int64_t n, int64_t offset) {
            ggml_tensor * dst = ggml_view_2d(ctx0, dst_all, n, S, dst_all->nb[1], kv_head*dst_all->nb[1] + offset*esz);
            ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_reshape_2d(ctx0, src, n, S), dst));
        };

        store(new_pk, r_all, D*P*G, 0);
        store(new_pv, r_all, D*P*G, D*P*G);
        store(new_ps, r_all,   P*G, 2*D*P*G);

        store(st.k, s_all, D*C*G, 0);
        store(st.v, s_all, D*C*G, D*C*G);
        store(st.s, s_all,   C*G, 2*D*C*G);
        store(st.f, s_all,   C*G, 2*D*C*G + C*G);
    }

    // gated output projection
    read = ggml_reshape_3d(ctx0, read, D, H, n_tokens);
    read = ggml_mul(ctx0, read, ggml_reshape_3d(ctx0, gate, 1, H, n_tokens));
    read = ggml_reshape_2d(ctx0, read, D*H, n_tokens);

    ggml_tensor * out = build_lora_mm(mem.wo, read);
    cb(out, "mem_out", il);

    return out;
}

std::pair<ggml_tensor *, ggml_tensor *> llama_model_edgerock::graph::build_qkvz(
                ggml_tensor * input,
                        int   il) {
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    ggml_tensor * qkv_mixed = build_lora_mm(model.layers[il].wqkv, input, model.layers[il].wqkv_s);
    qkv_mixed = ggml_reshape_3d(ctx0, qkv_mixed, qkv_mixed->ne[0], n_seq_tokens, n_seqs);
    cb(qkv_mixed, "linear_attn_qkv_mixed", il);

    ggml_tensor * z = build_lora_mm(model.layers[il].wqkv_gate, input, model.layers[il].wqkv_gate_s);
    cb(z, "z", il);

    return { qkv_mixed, z };
}

ggml_tensor * llama_model_edgerock::graph::build_norm_gated(
        ggml_tensor * input,
        ggml_tensor * weights,
        ggml_tensor * gate,
        int           layer) {
    ggml_tensor * normalized = build_norm(input, weights, nullptr, LLM_NORM_RMS, layer);
    ggml_tensor * gated_silu = ggml_silu(ctx0, gate);

    return ggml_mul(ctx0, normalized, gated_silu);
}

ggml_tensor * llama_model_edgerock::graph::build_layer_attn(
        llm_graph_input_attn_kv_iswa * inp,
        ggml_tensor *             cur,
        ggml_tensor *             inp_pos,
        int *                     sections,
        int                       il) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    // Order: joint QG projection, QG split, Q norm, KV projection, K norm, RoPE, attention

    // Qwen3Next uses a single Q projection that outputs query + gate
    auto [Qcur_full, Kcur, Vcur] = build_qkv(model.layers[il], cur,
            n_embd_head * 2, n_head,
            n_embd_head,     n_head_kv,
            n_embd_head,     n_head_kv,
            il, false);
    cb(Qcur_full, "Qcur_full", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    ggml_tensor * Qcur = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head, 0);
    cb(Qcur, "Qcur_reshaped", il);

    // Apply Q normalization
    Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
    cb(Qcur, "Qcur_normed", il);

    // Apply K normalization
    Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
    Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
    cb(Kcur, "Kcur_normed", il);

    ggml_tensor * gate = ggml_view_3d(ctx0, Qcur_full, n_embd_head, n_head, n_tokens,
        ggml_element_size(Qcur_full) * n_embd_head * 2,
        ggml_element_size(Qcur_full) * n_embd_head * 2 * n_head,
        ggml_element_size(Qcur_full) * n_embd_head);
    gate = ggml_cont_2d(ctx0, gate, n_embd_head * n_head, n_tokens);
    cb(gate, "gate_reshaped", il);

    Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

    // Apply MRoPE
    Qcur = ggml_rope_multi(
            ctx0, Qcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    Kcur = ggml_rope_multi(
            ctx0, Kcur, inp_pos, nullptr,
            n_rot, sections, rope_type, n_ctx_orig, freq_base, freq_scale,
            ext_factor, attn_factor, beta_fast, beta_slow
            );

    cb(Qcur, "Qcur", il);
    cb(Kcur, "Kcur", il);
    cb(Vcur, "Vcur", il);

    // Attention computation
    const float kq_scale = hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    cur = build_attn(inp,
                nullptr, nullptr, nullptr,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
    cb(cur, "attn_pregate", il);

    ggml_tensor * gate_sigmoid = ggml_sigmoid(ctx0, gate);
    cb(gate_sigmoid, "gate_sigmoid", il);

    cur = ggml_mul(ctx0, cur, gate_sigmoid);
    cb(cur, "attn_gated", il);

    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
    cb(cur, "attn_output", il);

    return cur;
}

ggml_tensor * llama_model_edgerock::graph::build_layer_attn_linear(
        llm_graph_input_rs * inp,
        ggml_tensor *        cur,
        int                  il) {
    const auto * mctx_cur = inp->mctx;

    const int64_t d_inner      = hparams.ssm_d_inner;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t head_k_dim   = hparams.ssm_d_state;
    const int64_t num_k_heads  = hparams.ssm_n_group;
    const int64_t num_v_heads  = hparams.ssm_dt_rank;
    const int64_t head_v_dim   = d_inner / num_v_heads;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // Input projections
    auto qkvz = build_qkvz(cur, il);
    ggml_tensor * qkv_mixed = qkvz.first;
    ggml_tensor * z         = qkvz.second;

    ggml_tensor * beta = build_lora_mm(model.layers[il].ssm_beta, cur, model.layers[il].ssm_beta_s);
    beta = ggml_reshape_4d(ctx0, beta, 1, num_v_heads, n_seq_tokens, n_seqs);
    cb(beta, "beta", il);

    beta = ggml_sigmoid(ctx0, beta);
    cb(beta, "beta_sigmoid", il);

    ggml_tensor * alpha = build_lora_mm(model.layers[il].ssm_alpha, cur, model.layers[il].ssm_alpha_s);
    alpha = ggml_reshape_3d(ctx0, alpha, num_v_heads, n_seq_tokens, n_seqs);
    cb(alpha, "alpha", il);

    ggml_tensor * alpha_biased   = ggml_add(ctx0, alpha, model.layers[il].ssm_dt);
    ggml_tensor * alpha_softplus = ggml_softplus(ctx0, alpha_biased);
    cb(alpha_softplus, "a_softplus", il);

    ggml_tensor * gate = ggml_mul(ctx0, alpha_softplus, model.layers[il].ssm_a);  // -A_log.exp() * softplus
    cb(gate, "gate", il);

    gate = ggml_reshape_4d(ctx0, gate, 1, num_v_heads, n_seq_tokens, n_seqs);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * ssm_states_all  = mctx_cur->get_s_l(il);

    ggml_tensor * conv_kernel      = model.layers[il].ssm_conv1d;
    const int64_t conv_kernel_size = conv_kernel->ne[0];
    const int64_t conv_channels    = d_inner + 2 * hparams.ssm_n_group * hparams.ssm_d_state;

    ggml_tensor * conv_input = build_conv_state(inp, conv_states_all, qkv_mixed, conv_kernel_size, conv_channels, il);

    ggml_tensor * state = build_rs(inp, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_v_dim, head_v_dim, num_v_heads, n_seqs);
    cb(state, "state_predelta", il);

    ggml_tensor * conv_output_proper = ggml_ssm_conv(ctx0, conv_input, conv_kernel);
    cb(conv_output_proper, "conv_output_raw", il);

    ggml_tensor * conv_output_silu = ggml_silu(ctx0, conv_output_proper);
    cb(conv_output_silu, "conv_output_silu", il);

    ggml_tensor * conv_qkv_mix = conv_output_silu;

    // Calculate the total conv dimension
    int64_t qkv_dim = head_k_dim * num_k_heads * 2 + head_v_dim * num_v_heads;
    int64_t nb1_qkv = ggml_row_size(conv_qkv_mix->type, qkv_dim);

    // Extract the convolved Q, K, V from conv_output
    ggml_tensor * q_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            0);

    ggml_tensor * k_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_k_dim, num_k_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_k_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            head_k_dim * num_k_heads * ggml_element_size(conv_qkv_mix));

    ggml_tensor * v_conv = ggml_view_4d(ctx0, conv_qkv_mix, head_v_dim, num_v_heads, n_seq_tokens, n_seqs,
            ggml_row_size(conv_qkv_mix->type, head_v_dim),
            nb1_qkv,
            nb1_qkv * n_seq_tokens,
            ggml_row_size(conv_qkv_mix->type, 2 * head_k_dim * num_k_heads));

    cb(q_conv, "q_conv", il);
    cb(k_conv, "k_conv", il);
    cb(v_conv, "v_conv", il);


    const float eps_norm = hparams.f_norm_rms_eps;

    q_conv = build_gdn_l2_norm(ctx0, q_conv, eps_norm);
    k_conv = build_gdn_l2_norm(ctx0, k_conv, eps_norm);

    //q_conv = ggml_cont_4d(ctx0, q_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //k_conv = ggml_cont_4d(ctx0, k_conv, head_k_dim, num_k_heads, n_seq_tokens, n_seqs);
    //v_conv = ggml_cont_4d(ctx0, v_conv, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // if head keys and value keys are different, repeat to force tensors into matching shapes
    // note: need explicit repeat only if we are not using the fused GDN.
    if (num_k_heads != num_v_heads && (!cparams.fused_gdn_ar || !cparams.fused_gdn_ch)) {
        GGML_ASSERT(num_v_heads % num_k_heads == 0);
        q_conv = ggml_repeat_4d(ctx0, q_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
        k_conv = ggml_repeat_4d(ctx0, k_conv, head_k_dim, num_v_heads, n_seq_tokens, n_seqs);
    }

    cb(q_conv, "q_conv_predelta", il);
    cb(k_conv, "k_conv_predelta", il);
    cb(v_conv, "v_conv_predelta", il);

    ggml_tensor * output = build_recurrent_attn(inp, ssm_states_all, q_conv, k_conv, v_conv, gate, beta, state, il);

    // z: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * n_tokens * n_seqs, head_dim]
    ggml_tensor * z_2d = ggml_reshape_4d(ctx0, z, head_v_dim, num_v_heads, n_seq_tokens, n_seqs);

    // Apply gated normalization: self.norm(core_attn_out, z)
    ggml_tensor * attn_out_norm = build_norm_gated(output, model.layers[il].ssm_norm, z_2d, il);

    // Final reshape: [head_dim, n_heads, n_tokens, n_seqs] -> [n_heads * head_dim , n_tokens * n_seqs ]
    ggml_tensor * final_output = ggml_reshape_2d(ctx0, attn_out_norm, head_v_dim * num_v_heads, n_seq_tokens * n_seqs);
    cb(final_output, "final_output", il);

    // Output projection
    cur = build_lora_mm(model.layers[il].ssm_out, final_output, model.layers[il].ssm_out_s);
    cb(cur, "linear_attn_out", il);

    return cur;
}

ggml_tensor * llama_model_edgerock::graph::build_layer_ffn(ggml_tensor * cur, const int il) {
    cur = build_ffn(cur,
        model.layers[il].ffn_up, NULL, model.layers[il].ffn_up_s,
        model.layers[il].ffn_gate, NULL, model.layers[il].ffn_gate_s,
        model.layers[il].ffn_down, NULL, model.layers[il].ffn_down_s,
        NULL,
        LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(cur, "ffn_out", il);

    return cur;
}
