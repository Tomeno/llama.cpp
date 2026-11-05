#include "rwkv7.cuh"

// fused element-wise parts of an RWKV7 layer (GGML_OP_RWKV7) and their backward passes (GGML_OP_RWKV7_BACK);
// the CPU kernels in ggml-cpu/ops.cpp are the reference

static __device__ __forceinline__ float rwkv7_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// LERP: one thread per (token, channel), all K lerps
template <bool back>
static __global__ void rwkv7_lerp_f32(const float * x, const char * sh, const int64_t sh_nb2, const float * mu,
        const float * gr, float * d, const int64_t C, const int64_t T, const int64_t Ts, const int K) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= C*T) {
        return;
    }
    const int64_t t = i / C;
    const int64_t c = i - t*C;
    if (!back) {
        const float xc = x[i];
        const float pv = t % Ts ? x[i - C] : *(const float *) (sh + (t / Ts)*sh_nb2 + c*sizeof(float));
        for (int k = 0; k < K; ++k) {
            d[k*C*T + i] = xc + (pv - xc) * mu[k*C + c];
        }
        return;
    }
    const bool last = (t + 1) % Ts == 0;
    float dx = 0.0f, ds = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float m = mu[k*C + c];
        const float g = gr[k*C*T + i];
        dx += g * (1.0f - m);
        ds += g * m;
        if (!last) {
            dx += gr[k*C*T + i + C] * m;
        }
    }
    d[i] = dx;
    if (t % Ts == 0) {
        d[C*T + (t / Ts)*C + c] = ds;
    }
}

// VRES and DECAY: one thread per element
template <bool back>
static __global__ void rwkv7_vres_f32(const float * v, const float * vf, const float * v12, const float * v0,
        const float * gr, float * d, const int64_t C, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float gate = rwkv7_sigmoid(v12[i] + v0[i % C]);
    if (!back) {
        d[i] = v[i] + (vf[i] - v[i]) * gate;
    } else {
        const float g = gr[i];
        d[i]       = g * (1.0f - gate);
        d[n + i]   = g * gate;
        d[2*n + i] = g * (vf[i] - v[i]) * gate * (1.0f - gate);
    }
}

template <bool back>
static __global__ void rwkv7_decay_f32(const float * w, const float * w0, const float * gr, float * d, const int64_t C, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float sg = rwkv7_sigmoid(w[i] + w0[i % C]);
    const float e  = expf(-0.606531f * sg);
    d[i] = back ? gr[i] * e * -0.606531f * sg * (1.0f - sg) : e;
}

// KKA and OUT: one warp per (token, head), E = head_size/32 elements per lane
template <int E, bool back>
static __global__ void rwkv7_kka_f32(const float * k, const float * a, const float * a0, const float * k_k, const float * k_a,
        const float * gr, float * d, const int64_t C, const int64_t T) {
    const int N = 32*E;
    const int64_t row = (int64_t) blockIdx.x*blockDim.y + threadIdx.y; // t*H + h
    if (row >= T*(C/N)) {
        return;
    }
    const int lane = threadIdx.x;
    const int64_t o  = row*N;       // == t*C + h*N
    const int64_t cb = o % C;       // h*N
    const int64_t n  = C*T;

    float kv[E], kkv[E], av[E], q = 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int j = lane + 32*e;
        kv[e] = k[o + j];
        const float u = kv[e] * k_k[cb + j];
        q += u*u;
        av[e] = rwkv7_sigmoid(a[o + j] + a0[cb + j]);
    }
    q = warp_reduce_sum(q);
    const float inv = 1.0f / fmaxf(sqrtf(q), 1e-12f);
#pragma unroll
    for (int e = 0; e < E; ++e) {
        kkv[e] = kv[e] * k_k[cb + lane + 32*e] * inv;
    }
    if (!back) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int j = lane + 32*e;
            d[o + j]       = kv[e] * (1.0f + k_a[cb + j] * (av[e] - 1.0f));
            d[n + o + j]   = -kkv[e];
            d[2*n + o + j] = kkv[e] * av[e];
        }
        return;
    }
    float dkk[E], dot = 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int j = lane + 32*e;
        dkk[e] = -gr[n + o + j] + gr[2*n + o + j] * av[e];
        dot += dkk[e] * kkv[e];
    }
    dot = warp_reduce_sum(dot);
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int j = lane + 32*e;
        const float ka = k_a[cb + j];
        const float gk = gr[o + j], gb = gr[2*n + o + j];
        const float du = (dkk[e] - kkv[e] * dot) * inv;
        d[o + j]     = gk * (1.0f + ka * (av[e] - 1.0f)) + du * k_k[cb + j];
        d[n + o + j] = (gk * kv[e] * ka + gb * kkv[e]) * av[e] * (1.0f - av[e]);
    }
}

template <int E, bool back>
static __global__ void rwkv7_out_f32(const float * y, const float * r, const float * k, const float * v, const float * g,
        const float * lw, const float * lb, const float * rk, const float * gr, float * d, const int64_t C, const int64_t T,
        const float eps) {
    const int N = 32*E;
    const int64_t row = (int64_t) blockIdx.x*blockDim.y + threadIdx.y;
    if (row >= T*(C/N)) {
        return;
    }
    const int lane = threadIdx.x;
    const int64_t o  = row*N;
    const int64_t cb = o % C;
    const int64_t n  = C*T;

    float yv[E], mean = 0.0f, bonus = 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int j = lane + 32*e;
        yv[e] = y[o + j];
        mean  += yv[e];
        bonus += r[o + j] * k[o + j] * rk[cb + j];
    }
    mean  = warp_reduce_sum(mean) / N;
    bonus = warp_reduce_sum(bonus);
    float var = 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        yv[e] -= mean;
        var += yv[e]*yv[e];
    }
    var = warp_reduce_sum(var) / N;
    const float rstd = rsqrtf(var + eps);
    if (!back) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            const int j = lane + 32*e;
            const float z = yv[e] * rstd * lw[cb + j] + lb[cb + j] + bonus * v[o + j];
            d[o + j] = g ? z * g[o + j] : z;
        }
        return;
    }
    float dz[E], s_dn = 0.0f, s_dnn = 0.0f, s_dzv = 0.0f;
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int j = lane + 32*e;
        dz[e] = g ? gr[o + j] * g[o + j] : gr[o + j];
        const float dn = dz[e] * lw[cb + j];
        s_dn  += dn;
        s_dnn += dn * yv[e] * rstd;
        s_dzv += dz[e] * v[o + j];
    }
    s_dn  = warp_reduce_sum(s_dn);
    s_dnn = warp_reduce_sum(s_dnn);
    s_dzv = warp_reduce_sum(s_dzv);
#pragma unroll
    for (int e = 0; e < E; ++e) {
        const int j = lane + 32*e;
        const float yn = yv[e] * rstd;
        d[o + j]       = (dz[e] * lw[cb + j] - s_dn / N - yn * s_dnn / N) * rstd;
        d[n + o + j]   = s_dzv * k[o + j] * rk[cb + j];
        d[2*n + o + j] = s_dzv * r[o + j] * rk[cb + j];
        d[3*n + o + j] = dz[e] * bonus;
        d[4*n + o + j] = g ? gr[o + j] * (yn * lw[cb + j] + lb[cb + j] + bonus * v[o + j]) : 0.0f;
    }
}

template <int E, bool back>
static void rwkv7_heads(const ggml_tensor * dst, cudaStream_t stream, const float * const * S, const float * gr, float * d,
        int64_t C, int64_t T, float eps) {
    const int64_t rows = T*(C/(32*E));
    const dim3 block(32, 4);
    const dim3 grid((rows + 3) / 4);
    if (ggml_get_rwkv7_op(dst) == GGML_RWKV7_OP_KKA) {
        rwkv7_kka_f32<E, back><<<grid, block, 0, stream>>>(S[0], S[1], S[2], S[3], S[4], gr, d, C, T);
    } else {
        rwkv7_out_f32<E, back><<<grid, block, 0, stream>>>(S[0], S[1], S[2], S[3], S[4], S[5], S[6], S[7], gr, d, C, T, eps);
    }
}

template <bool back>
static void rwkv7_op(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    cudaStream_t stream = ctx.stream();
    const float * S[GGML_MAX_SRC] = {};
    for (int i = 0; i < GGML_MAX_SRC - 1; ++i) {
        S[i] = dst->src[i] ? (const float *) dst->src[i]->data : nullptr;
    }
    const float * gr = back ? (const float *) dst->src[GGML_MAX_SRC - 1]->data : nullptr;
    float * d = (float *) dst->data;
    const int64_t C   = dst->src[0]->ne[0];
    const int64_t T   = ggml_nelements(dst->src[0]) / C;
    const int     N   = ggml_get_op_params_i32(dst, 1);
    const float   eps = ggml_get_op_params_f32(dst, 2);
    const int     bs  = 256;
    const int64_t nb  = (C*T + bs - 1) / bs;

    switch (ggml_get_rwkv7_op(dst)) {
        case GGML_RWKV7_OP_LERP: {
            const int K = (int) (ggml_nelements(dst->src[2]) / C);
            rwkv7_lerp_f32<back><<<nb, bs, 0, stream>>>(S[0], (const char *) dst->src[1]->data, dst->src[1]->nb[2], S[2],
                gr, d, C, T, dst->src[0]->ne[1], K);
        } break;
        case GGML_RWKV7_OP_VRES:
            rwkv7_vres_f32<back><<<nb, bs, 0, stream>>>(S[0], S[1], S[2], S[3], gr, d, C, C*T);
            break;
        case GGML_RWKV7_OP_DECAY:
            rwkv7_decay_f32<back><<<nb, bs, 0, stream>>>(S[0], S[1], gr, d, C, C*T);
            break;
        case GGML_RWKV7_OP_KKA:
        case GGML_RWKV7_OP_OUT:
            switch (N) {
                case 32:  rwkv7_heads<1, back>(dst, stream, S, gr, d, C, T, eps); break;
                case 64:  rwkv7_heads<2, back>(dst, stream, S, gr, d, C, T, eps); break;
                case 128: rwkv7_heads<4, back>(dst, stream, S, gr, d, C, T, eps); break;
                default: GGML_ABORT("unsupported RWKV7 head size %d", N);
            }
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

void ggml_cuda_op_rwkv7(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    rwkv7_op<false>(ctx, dst);
}

void ggml_cuda_op_rwkv7_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    rwkv7_op<true>(ctx, dst);
}

bool ggml_cuda_rwkv7_supported(const ggml_tensor * op) {
    const int N = ggml_get_op_params_i32(op, 1);
    switch (ggml_get_rwkv7_op(op)) {
        case GGML_RWKV7_OP_KKA:
        case GGML_RWKV7_OP_OUT:
            return N == 32 || N == 64 || N == 128;
        default:
            return true;
    }
}
