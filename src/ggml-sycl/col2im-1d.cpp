#include "col2im-1d.hpp"

// col2im_1d: scatter-add GEMM columns to 1D signal (gather approach)
// columns: [K*OC, T_in]  ->  output: [T_out, OC]
// Supports F32, F16, BF16 data with F32 accumulator.

template <typename T>
static void col2im_1d_kernel(
        const T * __restrict__ col, T * __restrict__ dst,
        const int T_in, const sycl::uint3 T_out_fd,
        const int K, const int K_OC,
        const int s0, const int p0, const int total,
        const sycl::nd_item<1> & item) {

    const int idx = (int)item.get_global_id(0);
    if (idx >= total) return;

    // dst layout: [T_out, OC], ne[0]=T_out fastest
    const sycl::uint2 qr = fast_div_modulo((uint32_t)idx, T_out_fd);  // qr.x() = idx / T_out, qr.y() = idx % T_out
    const int oc    = (int)qr.x();
    const int t_out = (int)qr.y();
    const int t_abs = t_out + p0;  // absolute position in uncropped signal

    // Gather: find all (t_in, k) where t_in*s + k == t_abs, 0 <= k < K
    int t_in_min = (t_abs - K + s0) / s0;  // ceil((t_abs - K + 1) / s)
    if (t_in_min < 0) t_in_min = 0;
    int t_in_max = t_abs / s0;
    if (t_in_max >= T_in) t_in_max = T_in - 1;

    float sum = 0.0f;
    for (int t_in = t_in_min; t_in <= t_in_max; t_in++) {
        const int k = t_abs - t_in * s0;
        // col layout: [K*OC, T_in], column index = oc * K + k
        sum += static_cast<float>(col[(oc * K + k) + t_in * K_OC]);
    }

    dst[idx] = static_cast<T>(sum);
}

template <typename T>
static void col2im_1d_sycl(
        const T * col, T * dst,
        const int T_in, const int T_out,
        const int OC, const int K, const int K_OC,
        const int s0, const int p0,
        dpct::queue_ptr stream) {

    const sycl::uint3 T_out_fd = init_fastdiv_values((uint32_t)T_out);

    const int total = T_out * OC;
    const int num_blocks = (total + SYCL_COL2IM_1D_BLOCK_SIZE - 1) / SYCL_COL2IM_1D_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>((size_t)num_blocks * SYCL_COL2IM_1D_BLOCK_SIZE, SYCL_COL2IM_1D_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            col2im_1d_kernel(col, dst, T_in, T_out_fd, K, K_OC, s0, p0, total, item);
        });
}

void ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);

    const ggml_tensor * src0 = dst->src[0];
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(dst->type == src0->type);

    const int32_t s0 = ((const int32_t *)(dst->op_params))[0];
    const int32_t OC = ((const int32_t *)(dst->op_params))[1];
    const int32_t p0 = ((const int32_t *)(dst->op_params))[2];

    const int K_OC  = (int) src0->ne[0];
    const int T_in  = (int) src0->ne[1];
    const int K     = K_OC / OC;
    const int T_out = (int) dst->ne[0];

    switch (src0->type) {
        case GGML_TYPE_F32:
            col2im_1d_sycl((const float *)src0->data, (float *)dst->data,
                           T_in, T_out, OC, K, K_OC, s0, p0, stream);
            break;
        case GGML_TYPE_F16:
            col2im_1d_sycl((const sycl::half *)src0->data, (sycl::half *)dst->data,
                           T_in, T_out, OC, K, K_OC, s0, p0, stream);
            break;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            col2im_1d_sycl((const sycl::ext::oneapi::bfloat16 *)src0->data, (sycl::ext::oneapi::bfloat16 *)dst->data,
                           T_in, T_out, OC, K, K_OC, s0, p0, stream);
            break;
#endif
        default:
            GGML_ABORT("col2im_1d: unsupported type");
    }
}
