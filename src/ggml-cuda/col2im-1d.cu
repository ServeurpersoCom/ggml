#include "col2im-1d.cuh"

// col2im_1d: scatter-add GEMM columns to 1D signal (gather approach)
// columns: [K*OC, T_in]  ->  output: [T_out, OC]
// Each output element gathers ceil(K/s) values - typically 2 for our VAE.
static __global__ void col2im_1d_kernel(
        const float * __restrict__ col,
        float * __restrict__ dst,
        const int T_in, const int T_out,
        const int OC, const int K, const int K_OC,
        const int s0, const int total) {

    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= total) return;

    // dst layout: [T_out, OC] - ne[0]=T_out fastest
    const int t_out = idx % T_out;
    const int oc    = idx / T_out;

    // Gather: find all (t_in, k) where t_in*s + k == t_out, 0 <= k < K
    int t_in_min = (t_out - K + s0) / s0;  // ceil((t_out - K + 1) / s)
    if (t_in_min < 0) t_in_min = 0;
    int t_in_max = t_out / s0;
    if (t_in_max >= T_in) t_in_max = T_in - 1;

    float sum = 0.0f;
    for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
        const int k = t_out - t_in * s0;
        // col layout: [K*OC, T_in] - ne[0]=K*OC, order oc*K+k (K fastest)
        sum += col[(oc * K + k) + t_in * K_OC];
    }

    dst[idx] = sum;
}

void ggml_cuda_op_col2im_1d(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t OC = ((const int32_t *)(dst->op_params))[1];

    const int K_OC = (int) src0->ne[0];
    const int T_in = (int) src0->ne[1];
    const int K    = K_OC / OC;
    const int T_out = (int) dst->ne[0];

    const int total = T_out * OC;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    col2im_1d_kernel<<<num_blocks, block_size, 0, stream>>>(
        src0_d, dst_d, T_in, T_out, OC, K, K_OC, s0, total);
}
