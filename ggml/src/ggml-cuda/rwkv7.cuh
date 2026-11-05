#include "common.cuh"

void ggml_cuda_op_rwkv7(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rwkv7_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_rwkv7_supported(const ggml_tensor * op);
