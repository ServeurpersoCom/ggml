#include "snake.cuh"

// Fused Snake activation: y = x + sin^2(a * x) * inv_b
// x: [T, C] (T contiguous), a: [C], inv_b: [C]
// 1 thread per element, coalesced T-reads for adjacent threads

static __global__ void kernel_snake(
        const float * __restrict__ x,
        const float * __restrict__ a,
        const float * __restrict__ inv_b,
        float * __restrict__ dst,
        const int T,
        const int C) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= T * C) return;

    const int c = idx / T;

    const float xi = x[idx];
    const float s  = sinf(a[c] * xi);
    dst[idx] = xi + s * s * inv_b[c];
}

void ggml_cuda_op_snake(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];

    const float * x_d     = (const float *)src0->data;
    const float * a_d     = (const float *)src1->data;
    const float * inv_b_d = (const float *)src2->data;
    float       * dst_d   = (float       *)dst->data;

    const int T = (int)src0->ne[0];
    const int C = (int)src0->ne[1];
    const int total = T * C;

    const int block_size = 256;
    const int grid_size  = (total + block_size - 1) / block_size;

    cudaStream_t stream = ctx.stream();
    kernel_snake<<<grid_size, block_size, 0, stream>>>(x_d, a_d, inv_b_d, dst_d, T, C);
}
