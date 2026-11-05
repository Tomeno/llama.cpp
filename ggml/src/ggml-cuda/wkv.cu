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
    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        state[i] = s[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i];
    }

    for (int t = batch_i * n_seq_tokens * C + head_i * head_size + tid; t < (batch_i + 1) * n_seq_tokens * C + head_i * head_size + tid; t += C) {
        __syncthreads();
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < head_size; j += 4)
        {
            const float4& a = (float4&)(_a[j]);
            const float4& s = (float4&)(state[j]);
            sa += a.x * s.x;
            sa += a.y * s.y;
            sa += a.z * s.z;
            sa += a.w * s.w;
        }

        const float _v = v[t];
        float y = 0;
        for (int j = 0; j < head_size; j += 4) {
            const float4& r = (float4&)(_r[j]);
            const float4& w = (float4&)(_w[j]);
            const float4& k = (float4&)(_k[j]);
            const float4& b = (float4&)(_b[j]);
            float4& s = (float4&)(state[j]);
            float4 kv;

            kv.x = k.x * _v;
            kv.y = k.y * _v;
            kv.z = k.z * _v;
            kv.w = k.w * _v;

            s.x = s.x * w.x + kv.x + sa * b.x;
            s.y = s.y * w.y + kv.y + sa * b.y;
            s.z = s.z * w.z + kv.z + sa * b.z;
            s.w = s.w * w.w + kv.w + sa * b.w;

            y += s.x * r.x;
            y += s.y * r.y;
            y += s.z * r.z;
            y += s.w * r.w;
        }
        dst[t] = y;
    }

    #pragma unroll
    for (int i = 0; i < head_size; i++) {
        s_out[batch_i * state_size + head_i * head_size * head_size + tid * head_size + i] = state[i];
    }
}

template <int rows_per_block>
static __global__ void __launch_bounds__(WARP_SIZE * rows_per_block, 2)
rwkv_wkv7_f32_t1_warp_row(const int T, const int C, const int H, const float * r, const float * w, const float * k, const float * v, const float * a, const float * b, const float * s, float * dst, float * s_out) {
    constexpr int head_size = CUDA_WKV_BLOCK_SIZE;
    constexpr int half_head = head_size / 2;

    const int lane = threadIdx.x;
    const int row  = blockIdx.y * rows_per_block + threadIdx.y;
    const int bid  = blockIdx.x;

    const int batch_i = bid / H;
    const int head_i  = bid % H;
    const int state_size = C * head_size;
    const int head_off = head_i * head_size;
    const int t = batch_i * C + head_off + row;

    __shared__ float _r[head_size], _w[head_size], _k[head_size], _a[head_size], _b[head_size];

    if (threadIdx.y == 0) {
        _r[lane] = r[batch_i * C + head_off + lane];
        _w[lane] = w[batch_i * C + head_off + lane];
        _k[lane] = k[batch_i * C + head_off + lane];
        _a[lane] = a[batch_i * C + head_off + lane];
        _b[lane] = b[batch_i * C + head_off + lane];

        _r[lane + half_head] = r[batch_i * C + head_off + lane + half_head];
        _w[lane + half_head] = w[batch_i * C + head_off + lane + half_head];
        _k[lane + half_head] = k[batch_i * C + head_off + lane + half_head];
        _a[lane + half_head] = a[batch_i * C + head_off + lane + half_head];
        _b[lane + half_head] = b[batch_i * C + head_off + lane + half_head];
    }
    __syncthreads();

    const int64_t state_base = batch_i * state_size + head_i * head_size * head_size + row * head_size;
    const float s0 = s[state_base + lane];
    const float s1 = s[state_base + lane + half_head];
    const float sa = warp_reduce_sum(_a[lane] * s0 + _a[lane + half_head] * s1);

    const float vt  = v[t];
    const float st0 = s0 * _w[lane]             + _k[lane]             * vt + sa * _b[lane];
    const float st1 = s1 * _w[lane + half_head] + _k[lane + half_head] * vt + sa * _b[lane + half_head];
    const float y   = warp_reduce_sum(st0 * _r[lane] + st1 * _r[lane + half_head]);

    s_out[state_base + lane]             = st0;
    s_out[state_base + lane + half_head] = st1;

    if (lane == 0) {
        dst[t] = y;
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
        constexpr int rows_per_block = 4;
        rwkv_wkv7_f32_t1_warp_row<rows_per_block><<<dim3(B * H, CUDA_WKV_BLOCK_SIZE / rows_per_block), dim3(WARP_SIZE, rows_per_block), 0, stream>>>(T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, s_out);
    } else if (C / H == CUDA_WKV_BLOCK_SIZE) {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, s_out);
    } else {
        rwkv_wkv7_f32<CUDA_WKV_BLOCK_SIZE * 2><<<B * H, C / H, 0, stream>>>(B, T, C, H, r_d, w_d, k_d, v_d, a_d, b_d, s_d, dst_d, s_out);
    }
}

// backward pass of rwkv_wkv7: one block per (sequence, head), thread i holds row i of the state (and of its gradient);
// a forward pass keeps the state every CH tokens (ck), then every chunk is recomputed from its checkpoint and
// backpropagated, the last one first; the sums over the rows go through shared memory
template <int N, int CH>
static __global__ void rwkv_wkv7_back_f32(const int T, const int Ts, const int C, const int H,
        const float * r, const float * w, const float * k, const float * v, const float * a, const float * b,
        const float * s0, const float * gy, float * d, float * ck) {
    const int i   = threadIdx.x;
    const int bid = blockIdx.x;
    const int bi  = bid / H;
    const int h   = bid % H;
    const int n_ch = (Ts + CH - 1) / CH;

    const float * gs = gy + (int64_t) C * T;
    float * dr = d;
    float * dw = d + (int64_t) 1 * C * T;
    float * dk = d + (int64_t) 2 * C * T;
    float * dv = d + (int64_t) 3 * C * T;
    float * da = d + (int64_t) 4 * C * T;
    float * db = d + (int64_t) 5 * C * T;
    float * ds = d + (int64_t) 6 * C * T;

    const int64_t so = (int64_t) bi * C * N + (int64_t) h * N * N + (int64_t) i * N;
    float * ckb = ck + (int64_t) bid * n_ch * N * N + (int64_t) i * N;

    __shared__ float _r[N], _w[N], _k[N], _a[N], _b[N], _v[N], _dy[N], _sa[N], _dsa[N];
    __shared__ float shS[N][N + 1], shG[N][N + 1];

    float S[N], G[N];
    float Sb[CH][N];
    float sab[CH];

    auto load = [&](int64_t o) {
        __syncthreads();
        _r[i] = r[o + i];
        _w[i] = w[o + i];
        _k[i] = k[o + i];
        _a[i] = a[o + i];
        _b[i] = b[o + i];
        _v[i] = v[o + i];
        _dy[i] = gy[o + i];
        __syncthreads();
    };
    // S <- S w + v k + sa b, returns sa
    auto step = [&]() {
        float sa = 0.0f;
#pragma unroll
        for (int j = 0; j < N; ++j) {
            sa += S[j] * _a[j];
        }
        const float vi = _v[i];
#pragma unroll
        for (int j = 0; j < N; ++j) {
            S[j] = S[j] * _w[j] + vi * _k[j] + sa * _b[j];
        }
        return sa;
    };

#pragma unroll
    for (int j = 0; j < N; ++j) {
        S[j] = s0[so + j];
    }
    for (int t = 0; t < Ts; ++t) {
        if (t % CH == 0) {
#pragma unroll
            for (int j = 0; j < N; ++j) {
                ckb[(int64_t) (t / CH) * N * N + j] = S[j];
            }
        }
        load(((int64_t) bi * Ts + t) * C + h * N);
        step();
    }

#pragma unroll
    for (int j = 0; j < N; ++j) {
        G[j] = gs[so + j];
    }
    for (int c = n_ch - 1; c >= 0; --c) {
        const int tb = c * CH;
        const int te = min(Ts, tb + CH);
#pragma unroll
        for (int j = 0; j < N; ++j) {
            S[j] = ckb[(int64_t) c * N * N + j];
        }
        for (int t = tb; t < te; ++t) {
            load(((int64_t) bi * Ts + t) * C + h * N);
#pragma unroll
            for (int j = 0; j < N; ++j) {
                Sb[t - tb][j] = S[j];
            }
            sab[t - tb] = step();
        }
        for (int t = te - 1; t >= tb; --t) {
            const int64_t o = ((int64_t) bi * Ts + t) * C + h * N;
            load(o);
            const float dyi = _dy[i];
            float dvi = 0.0f, dsai = 0.0f;
#pragma unroll
            for (int j = 0; j < N; ++j) {
                G[j] += dyi * _r[j];
                dvi  += G[j] * _k[j];
                dsai += G[j] * _b[j];
                shS[i][j] = Sb[t - tb][j];
                shG[i][j] = G[j];
            }
            _sa[i]  = sab[t - tb];
            _dsa[i] = dsai;
            __syncthreads();

            // column i: sums over the rows
            const float wj = _w[i], kj = _k[i], bj = _b[i];
            float drj = 0.0f, dkj = 0.0f, dbj = 0.0f, dwj = 0.0f, daj = 0.0f;
#pragma unroll 8
            for (int q = 0; q < N; ++q) {
                const float sp = shS[q][i];
                const float g  = shG[q][i];
                drj += (sp * wj + _v[q] * kj + _sa[q] * bj) * _dy[q];
                dkj += g * _v[q];
                dbj += g * _sa[q];
                dwj += g * sp;
                daj += _dsa[q] * sp;
            }
            dr[o + i] = drj;
            dw[o + i] = dwj;
            dk[o + i] = dkj;
            dv[o + i] = dvi;
            da[o + i] = daj;
            db[o + i] = dbj;

#pragma unroll
            for (int j = 0; j < N; ++j) {
                G[j] = G[j] * _w[j] + dsai * _a[j];
            }
        }
    }
#pragma unroll
    for (int j = 0; j < N; ++j) {
        ds[so + j] = G[j];
    }
}

void ggml_cuda_op_rwkv_wkv7_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    constexpr int CH = 16;

    const int64_t B = dst->src[6]->ne[1];
    const int64_t T = dst->src[0]->ne[2];
    const int64_t N = dst->src[0]->ne[0];
    const int64_t H = dst->src[0]->ne[1];
    const int64_t C = N * H;
    const int64_t Ts = T / B;
    const int64_t n_ch = (Ts + CH - 1) / CH;

    GGML_ASSERT(dst->src[6]->type == GGML_TYPE_F32 && dst->src[7]->type == GGML_TYPE_F32);
    GGML_ASSERT(N == CUDA_WKV_BLOCK_SIZE);

    ggml_cuda_pool_alloc<float> ck(ctx.pool(), B * H * n_ch * N * N);

    rwkv_wkv7_back_f32<CUDA_WKV_BLOCK_SIZE, CH><<<B * H, N, 0, ctx.stream()>>>(T, Ts, C, H,
        (const float *) dst->src[0]->data, (const float *) dst->src[1]->data, (const float *) dst->src[2]->data,
        (const float *) dst->src[3]->data, (const float *) dst->src[4]->data, (const float *) dst->src[5]->data,
        (const float *) dst->src[6]->data, (const float *) dst->src[7]->data, (float *) dst->data, ck.get());
}
