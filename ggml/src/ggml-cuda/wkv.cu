#include "common.cuh"
#include "wkv.cuh"

template <int block_size>
static __global__ void rwkv_wkv_f32(const int B, const int T, const int C, const int H, const float * k, const float * v, const float * r, const float * tf, const float * td, const float * s, float * dst) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _k[head_size], _r[head_size], _tf[head_size], _td[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + i * head_size + tid];
    }

    __syncthreads();
    _tf[tid] = tf[head_i * head_size + tid];
    __syncthreads();

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        __syncthreads();

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& k = (float4&)(_k[j]);
            const float4& r = (float4&)(_r[j]);
            const float4& tf = (float4&)(_tf[j]);
            const float4& td = (float4&)(_td[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            y += r.x * (tf.x * kv.x + s.x);
            y += r.y * (tf.y * kv.y + s.y);
            y += r.z * (tf.z * kv.z + s.z);
            y += r.w * (tf.w * kv.w + s.w);

            s.x = s.x * td.x + kv.x;
            s.y = s.y * td.y + kv.y;
            s.z = s.z * td.z + kv.z;
            s.w = s.w * td.w + kv.w;
        }
        dst[t] = y;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        dst[T * C + batch_i * state_size + head_i * head_size * head_size + i * head_size + tid] = state[i];
    }
}

// a thread per row of the state; the vectors of the next token are loaded into registers while the current one is
// computed, and go through a double buffer in shared memory (one barrier per token)
template <int block_size>
static __global__ void rwkv_wkv7_f32(const int B, const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst, float * s_out) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    const int head_size = block_size;
    const int batch_i = bid / H;
    const int head_i = bid % H;
    const int state_size = C * head_size;
    const int n_seq_tokens = T / B;

    float state[head_size];
    __shared__ float _r[2][head_size], _w[2][head_size], _k[2][head_size], _a[2][head_size], _b[2][head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    const int t0 = batch_i * n_seq_tokens * C + head_i * head_size + tid;
    const int t1 = t0 + n_seq_tokens * C;
    float nr = 0, nw = 0, nk = 0, na = 0, nb = 0, nv = 0;
    if (t0 < t1) {
        nr = r[t0]; nw = w[t0]; nk = k[t0]; na = a[t0]; nb = b[t0]; nv = v[t0];
    }
    for (int t = t0, bf = 0; t < t1; t += C, bf ^= 1) {
        _r[bf][tid] = nr;
        _w[bf][tid] = nw;
        _k[bf][tid] = nk;
        _a[bf][tid] = na;
        _b[bf][tid] = nb;
        const float _v = nv;
        __syncthreads();
        if (t + C < t1) {
            nr = r[t + C]; nw = w[t + C]; nk = k[t + C]; na = a[t + C]; nb = b[t + C]; nv = v[t + C];
        }

        float sa4[4] = { 0, 0, 0, 0 };
        #pragma unroll
        for (int j = 0; j < head_size; j += 4) {
            const float4 & a = (float4 &) (_a[bf][j]);
            const float4 & s = (float4 &) (state[j]);
            sa4[0] += a.x * s.x;
            sa4[1] += a.y * s.y;
            sa4[2] += a.z * s.z;
            sa4[3] += a.w * s.w;
        }
        const float sa = (sa4[0] + sa4[1]) + (sa4[2] + sa4[3]);

        float y4[4] = { 0, 0, 0, 0 };
        #pragma unroll
        for (int j = 0; j < head_size; j += 4) {
            const float4 & r = (float4 &) (_r[bf][j]);
            const float4 & w = (float4 &) (_w[bf][j]);
            const float4 & k = (float4 &) (_k[bf][j]);
            const float4 & b = (float4 &) (_b[bf][j]);
            float4 & s = (float4 &) (state[j]);

            s.x = s.x * w.x + k.x * _v + sa * b.x;
            s.y = s.y * w.y + k.y * _v + sa * b.y;
            s.z = s.z * w.z + k.z * _v + sa * b.z;
            s.w = s.w * w.w + k.w * _v + sa * b.w;

            y4[0] += s.x * r.x;
            y4[1] += s.y * r.y;
            y4[2] += s.z * r.z;
            y4[3] += s.w * r.w;
        }
        dst[t] = (y4[0] + y4[1]) + (y4[2] + y4[3]);
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        s_out[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

// one token per sequence: a warp per R rows of a head (W warps per block), every lane two columns (r, w, k, a, b of them in registers);
// the R rows of state are loaded before any of them is used
template <int R, int W>
static __global__ void __launch_bounds__(WARP_SIZE * W)
rwkv_wkv7_f32_t1(const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst, float * s_out) {
    constexpr int N = CUDA_WKV_BLOCK_SIZE;
    const int lane = threadIdx.x;
    const int row0 = (blockIdx.y * W + threadIdx.y) * R;
    const int bi   = blockIdx.x / H;
    const int h    = blockIdx.x % H;
    const int64_t vo = (int64_t) bi * C + h * N;
    const int64_t so = (int64_t) bi * C * N + (int64_t) h * N * N + (int64_t) row0 * N + 2 * lane;

    float2 S[R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        S[j] = *(const float2 *) (s + so + j * N);
    }
    const float2 r2 = *(const float2 *) (r + vo + 2 * lane);
    const float2 w2 = *(const float2 *) (w + vo + 2 * lane);
    const float2 k2 = *(const float2 *) (k + vo + 2 * lane);
    const float2 a2 = *(const float2 *) (a + vo + 2 * lane);
    const float2 b2 = *(const float2 *) (b + vo + 2 * lane);
    const float  vl = lane < R ? v[vo + row0 + lane] : 0.0f;

    float sa[R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        sa[j] = S[j].x * a2.x + S[j].y * a2.y;
    }
#pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
            sa[j] += __shfl_xor_sync(0xffffffff, sa[j], o);
        }
    }
    float y[R];
#pragma unroll
    for (int j = 0; j < R; ++j) {
        const float vt = __shfl_sync(0xffffffff, vl, j);
        S[j].x = S[j].x * w2.x + k2.x * vt + sa[j] * b2.x;
        S[j].y = S[j].y * w2.y + k2.y * vt + sa[j] * b2.y;
        *(float2 *) (s_out + so + j * N) = S[j];
        y[j] = S[j].x * r2.x + S[j].y * r2.y;
    }
#pragma unroll
    for (int o = WARP_SIZE / 2; o > 0; o >>= 1) {
#pragma unroll
        for (int j = 0; j < R; ++j) {
            y[j] += __shfl_xor_sync(0xffffffff, y[j], o);
        }
    }
    if (lane < R) {
        float yl = y[0];
#pragma unroll
        for (int j = 1; j < R; ++j) {
            yl = lane == j ? y[j] : yl;
        }
        dst[vo + row0 + lane] = yl;
    }
}

void ggml_cuda_op_rwkv_wkv6(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * k_d  = (const float *)dst->src[0]->data;
    const float * v_d  = (const float *)dst->src[1]->data;
    const float * r_d  = (const float *)dst->src[2]->data;
    const float * tf_d = (const float *)dst->src[3]->data;
    const float * td_d = (const float *)dst->src[4]->data;
    const float * s_d  = (const float *)dst->src[5]->data;

    const int64_t B = dst->src[5]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d);
    } else {
        rwkv_wkv_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, k_d, v_d, r_d, tf_d, td_d, s_d, dst_d);
    }
}

void ggml_cuda_op_rwkv_wkv7(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const float * r_d = (const float *)dst->src[0]->data;
    const float * w_d = (const float *)dst->src[1]->data;
    const float * k_d = (const float *)dst->src[2]->data;
    const float * v_d = (const float *)dst->src[3]->data;
    const float * a_d = (const float *)dst->src[4]->data;
    const float * b_d = (const float *)dst->src[5]->data;
    const float * s_d = (const float *)dst->src[6]->data;

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t C = dst->ne[0];
    const int64_t H = dst->src[0]->ne[1];

    float * dst_d = (float *)dst->data;
    // in place (op_params[0]): the final state over the input state
    float * s_out = ggml_get_op_params_i32(dst, 0) ? (float *) dst->src[6]->data : dst_d + T * C;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32);
    GGML_ASSERT(C % H == 0);
    GGML_ASSERT(C / H == CUDA_WKV_BLOCK_SIZE || C / H == CUDA_WKV_BLOCK_SIZE * 2);

    if (T / B == 1 && C / H == CUDA_WKV_BLOCK_SIZE) {
        constexpr int R = 8, warps = 8;
        rwkv_wkv7_f32_t1<R, warps><<<dim3(B * H, CUDA_WKV_BLOCK_SIZE / R / warps), dim3(WARP_SIZE, warps), 0, stream>>>(C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, s_out);
    } else if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, s_out);
    } else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, s_out);
    }
}

// backward pass of rwkv_wkv7: one block per (sequence, head), Q threads per row of the state (E = N/Q columns each,
// interleaved: thread q of a row holds the columns q, q + Q, ...)
// - forward: the state every CK tokens (ck) and S a of every token and row (sa) to the scratch
// - backward, the last token first: the state before token t from the one after it,
//   S_{t-1} = (S_t - v k^T - sa b^T) / w (w >= exp(-0.606531)), exact again at every checkpoint, so the rounding grows
//   by at most w^-(CK-1); the sums over the rows (the gradients of r, w, k, a, b) go through shared memory.
// The per-token vectors of the next token are loaded into registers while the current one is computed (double buffer).
// The shared memory pipe is the limit: the column vectors are stored by column group (padded, no bank conflicts) and
// read as float4, and the rows of shS/shG are padded so that both the row and the column accesses are conflict-free.
template <int N, int Q, int CK>
static __global__ void __launch_bounds__(N * Q, 2) rwkv_wkv7_back_f32(const int T, const int Ts, const int C, const int H,
        const float * r, const float * w, const float * k, const float * v, const float * a, const float * b,
        const float * s0, const float * gy, float * d, float * ck, float * sa_buf) {
    constexpr int E  = N / Q;
    constexpr int NV = 7;     // r, w, k, a, b, v, dy
    constexpr int CP = E + 4; // a column group in shared memory
    constexpr int SP = N + 4; // a row of shS, shG
    static_assert(E % 4 == 0 && Q == 4, "rwkv_wkv7_back_f32: the layouts assume Q = 4");
    const int tid = threadIdx.x;
    const int i   = tid / Q; // row
    const int q   = tid % Q; // columns q + e*Q
    const int bid = blockIdx.x;
    const int bi  = bid / H;
    const int h   = bid % H;

    const float * gs = gy + (int64_t) C * T;
    float * dr = d;
    float * dw = d + (int64_t) 1 * C * T;
    float * dk = d + (int64_t) 2 * C * T;
    float * dv = d + (int64_t) 3 * C * T;
    float * da = d + (int64_t) 4 * C * T;
    float * db = d + (int64_t) 5 * C * T;
    float * ds = d + (int64_t) 6 * C * T;

    const int64_t so = (int64_t) bi * C * N + (int64_t) h * N * N + (int64_t) i * N + q;
    float * ckb = ck + (int64_t) bid * ((Ts + CK - 1) / CK) * N * N + tid; // + e*N*Q: this kernel's own layout
    float * sab = sa_buf + (int64_t) bid * Ts * N;
    const float * src[NV] = { r, w, k, a, b, v, gy };
    auto off = [&](int t) { return ((int64_t) bi * Ts + t) * C + h * N; };

    enum { R, W, K, A, B, IW };
    __shared__ __align__(16) float col[2][6][Q * CP]; // r, w, k, a, b, 1/w: column j at (j % Q)*CP + j / Q
    __shared__ __align__(16) float row[2][2][N];      // v, dy
    __shared__ __align__(16) float _sa[N], _dsa[N];
    __shared__ float shS[N][SP], shG[N][SP];
    __shared__ float red[Q][5][N];

    float S[E], G[E], nxt[NV];

    // reduction over the Q threads of a row (consecutive lanes)
    auto row_sum = [](float x) {
#pragma unroll
        for (int o = Q / 2; o > 0; o >>= 1) {
            x += __shfl_xor_sync(0xffffffff, x, o, Q);
        }
        return x;
    };
    // the E values of this thread's columns of a column vector
    auto cols = [&](int bf, int x, float (&o)[E]) {
        const float4 * p = (const float4 *) &col[bf][x][q * CP];
#pragma unroll
        for (int m = 0; m < E / 4; ++m) {
            const float4 f = p[m];
            o[4*m + 0] = f.x; o[4*m + 1] = f.y; o[4*m + 2] = f.z; o[4*m + 3] = f.w;
        }
    };
    // the vectors of token t: threads tid < N load element tid of each into registers, then store them to buffer bf
    auto fetch = [&](int t) {
        if (tid < N && t >= 0 && t < Ts) {
            const int64_t o = off(t);
#pragma unroll
            for (int x = 0; x < NV; ++x) {
                nxt[x] = src[x][o + tid];
            }
        }
    };
    auto store = [&](int bf) {
        if (tid < N) {
            const int pos = (tid % Q) * CP + tid / Q;
#pragma unroll
            for (int x = 0; x < 5; ++x) {
                col[bf][x][pos] = nxt[x];
            }
            col[bf][IW][pos] = 1.0f / nxt[1];
            row[bf][0][tid]  = nxt[5];
            row[bf][1][tid]  = nxt[6];
        }
    };

#pragma unroll
    for (int e = 0; e < E; ++e) {
        S[e] = s0[so + e * Q];
    }
    fetch(0);
    store(0);
    __syncthreads();
    for (int t = 0; t < Ts; ++t) {
        const int bf = t & 1;
        fetch(t + 1);
        if (t % CK == 0) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                ckb[(int64_t) (t / CK) * N * N + e * N * Q] = S[e];
            }
        }
        float x0[E], x1[E], x2[E];
        cols(bf, A, x0);
        float sa = 0.0f;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            sa += S[e] * x0[e];
        }
        sa = row_sum(sa);
        if (q == 0) {
            sab[(int64_t) t * N + i] = sa;
        }
        const float vi = row[bf][0][i];
        cols(bf, W, x0);
        cols(bf, K, x1);
        cols(bf, B, x2);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            S[e] = S[e] * x0[e] + vi * x1[e] + sa * x2[e];
        }
        store(bf ^ 1);
        __syncthreads();
    }

    // S: the final state
#pragma unroll
    for (int e = 0; e < E; ++e) {
        G[e] = gs[so + e * Q];
    }
    fetch(Ts - 1);
    store(0);
    __syncthreads();
    float sa_nxt = sab[(int64_t) (Ts - 1) * N + i];
    for (int t = Ts - 1, n = 0; t >= 0; --t, ++n) {
        const int bf = n & 1;
        fetch(t - 1);
        const float sa = sa_nxt;
        if (t > 0) {
            sa_nxt = sab[(int64_t) (t - 1) * N + i];
        }
        const int64_t o = off(t);
        const float vi  = row[bf][0][i];
        const float dyi = row[bf][1][i];

        float x0[E], x1[E], x2[E];
        cols(bf, K, x1);
        cols(bf, B, x2);
        // the state before token t
        if (t % CK == 0) {
#pragma unroll
            for (int e = 0; e < E; ++e) {
                S[e] = ckb[(int64_t) (t / CK) * N * N + e * N * Q];
            }
        } else {
            cols(bf, IW, x0);
#pragma unroll
            for (int e = 0; e < E; ++e) {
                S[e] = (S[e] - vi * x1[e] - sa * x2[e]) * x0[e];
            }
        }

        cols(bf, R, x0);
        float dvi = 0.0f, dsai = 0.0f;
#pragma unroll
        for (int e = 0; e < E; ++e) {
            G[e] += dyi * x0[e];
            dvi  += G[e] * x1[e];
            dsai += G[e] * x2[e];
            shS[i][e * Q + q] = S[e];
            shG[i][e * Q + q] = G[e];
        }
        dvi  = row_sum(dvi);
        dsai = row_sum(dsai);
        if (q == 0) {
            dv[o + i] = dvi;
            _sa[i]  = sa;
            _dsa[i] = dsai;
        }
        // G for token t - 1 (shG has the one of token t); nothing of buffer bf is read after the barrier below
        cols(bf, W, x0);
        cols(bf, A, x1);
#pragma unroll
        for (int e = 0; e < E; ++e) {
            G[e] = G[e] * x0[e] + dsai * x1[e];
        }
        const int j = tid % N;
        const int p = tid / N;
        const int pos = (j % Q) * CP + j / Q;
        const float wj = col[bf][W][pos], kj = col[bf][K][pos], bj = col[bf][B][pos];
        store(bf ^ 1);
        __syncthreads();

        // column j, rows p*N/Q .. (p+1)*N/Q - 1
        {
            float drj = 0.0f, dkj = 0.0f, dbj = 0.0f, dwj = 0.0f, daj = 0.0f;
#pragma unroll
            for (int x = 0; x < N / Q; x += 4) {
                const int i0 = p * (N / Q) + x;
                const float4 v4  = *(const float4 *) &row[bf][0][i0];
                const float4 dy4 = *(const float4 *) &row[bf][1][i0];
                const float4 sa4 = *(const float4 *) &_sa[i0];
                const float4 ds4 = *(const float4 *) &_dsa[i0];
                const float vv[4] = { v4.x, v4.y, v4.z, v4.w }, dyv[4] = { dy4.x, dy4.y, dy4.z, dy4.w };
                const float sv[4] = { sa4.x, sa4.y, sa4.z, sa4.w }, dsv[4] = { ds4.x, ds4.y, ds4.z, ds4.w };
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const float sp = shS[i0 + u][j];
                    const float g  = shG[i0 + u][j];
                    drj += (sp * wj + vv[u] * kj + sv[u] * bj) * dyv[u];
                    dkj += g * vv[u];
                    dbj += g * sv[u];
                    dwj += g * sp;
                    daj += dsv[u] * sp;
                }
            }
            red[p][0][j] = drj;
            red[p][1][j] = dwj;
            red[p][2][j] = dkj;
            red[p][3][j] = daj;
            red[p][4][j] = dbj;
        }
        __syncthreads();
        for (int idx = tid; idx < 5 * N; idx += N * Q) {
            const int c = idx / N;
            const int j = idx % N;
            float x = 0.0f;
#pragma unroll
            for (int p = 0; p < Q; ++p) {
                x += red[p][c][j];
            }
            float * dst = c == 0 ? dr : c == 1 ? dw : c == 2 ? dk : c == 3 ? da : db;
            dst[o + j] = x;
        }
    }
#pragma unroll
    for (int e = 0; e < E; ++e) {
        ds[so + e * Q] = G[e];
    }
}

void ggml_cuda_op_rwkv_wkv7_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int Q  = 4;
    constexpr int CK = 8;

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t N = dst->src[0]->ne[0];
    const int64_t H = dst->src[0]->ne[1];
    const int64_t C = N * H;
    const int64_t Ts = T / B;
    const int64_t n_ck = (Ts + CK - 1) / CK;

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32 && dst->src[7]->type == GGML_TYPE_F32);
    GGML_ASSERT(N == CUDA_WKV_BLOCK_SIZE);

    ggml_cuda_pool_alloc<float> ck(ctx.pool(), B * H * n_ck * N * N);
    ggml_cuda_pool_alloc<float> sa(ctx.pool(), B * H * Ts * N);

    rwkv_wkv7_back_f32<CUDA_WKV_BLOCK_SIZE, Q, CK><<<B * H, N * Q, 0, ctx.stream()>>>(T, Ts, C, H,
        (const float *) dst->src[0]->data, (const float *) dst->src[1]->data, (const float *) dst->src[2]->data,
        (const float *) dst->src[3]->data, (const float *) dst->src[4]->data, (const float *) dst->src[5]->data,
        (const float *) dst->src[6]->data, (const float *) dst->src[7]->data, (float *) dst->data, ck.get(), sa.get());
}
