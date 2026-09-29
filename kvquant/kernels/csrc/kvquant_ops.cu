#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// ---------------- block 归约辅助 ----------------
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float smem[BLOCK];
    smem[threadIdx.x] = v;
    __syncthreads();
    #pragma unroll
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) smem[threadIdx.x] += smem[threadIdx.x + off];
        __syncthreads();
    }
    return smem[0];
}

// ============================================================
// RoPE（element-wise, 论文式 (3)）
//   x, out : [B, H, S, D] fp16
//   cos/sin: [S_total, D/2] fp32
// ============================================================
__global__ void rope_kernel(
    const half* __restrict__ x,
    half* __restrict__ out,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int B, int H, int S, int D, int pos_offset
) {
    const int half_d = D >> 1;
    const int64_t total = (int64_t)B * H * S * half_d;
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int pair = (int)(idx % half_d);
    const int64_t t = idx / half_d;
    const int s  = (int)(t % S);
    const int64_t bh = t / S;
    const int h = (int)(bh % H);
    const int b = (int)(bh / H);

    const int pos = s + pos_offset;
    const float c  = __ldg(&cos_cache[(int64_t)pos * half_d + pair]);
    const float sn = __ldg(&sin_cache[(int64_t)pos * half_d + pair]);

    const int64_t base = (((int64_t)b * H + h) * S + s) * D;
    const float x1 = __half2float(x[base + pair]);
    const float x2 = __half2float(x[base + pair + half_d]);
    out[base + pair]          = __float2half(x1 * c - x2 * sn);
    out[base + pair + half_d] = __float2half(x1 * sn + x2 * c);
}

// ============================================================
// LUT 稠密 matvec
//   out[i] = Σ_j ( lut[qmat[i,j]] * scale[s] + zero[s] ) * vec[j]
//   per_channel=1 -> s=j（Key 用）  per_channel=0 -> s=i（Value 用）
// ============================================================
template <int BLOCK, int LUT_MAX>
__global__ void lut_matvec_kernel(
    const uint8_t* __restrict__ qmat,   // [N, D]
    const half*    __restrict__ lut,    // [lut_size]
    const half*    __restrict__ scale,  // [D] 或 [N]
    const half*    __restrict__ zero,   // [D] 或 [N]
    const half*    __restrict__ vec,    // [D]
    half*          __restrict__ out,    // [N]
    int N, int D, int lut_size, int per_channel
) {
    __shared__ half s_lut[LUT_MAX];
    for (int i = threadIdx.x; i < lut_size; i += BLOCK) s_lut[i] = lut[i];
    __syncthreads();

    const int row = blockIdx.x;
    if (row >= N) return;

    const uint8_t* rp = qmat + (int64_t)row * D;
    float acc = 0.f;

    if (per_channel) {
        for (int j = threadIdx.x; j < D; j += BLOCK) {
            const float v = __half2float(s_lut[rp[j]])
                          * __half2float(scale[j])
                          + __half2float(zero[j]);
            acc = fmaf(v, __half2float(vec[j]), acc);
        }
    } else {
        const float sc = __half2float(scale[row]);
        const float zc = __half2float(zero[row]);
        for (int j = threadIdx.x; j < D; j += BLOCK) {
            const float v = __half2float(s_lut[rp[j]]) * sc + zc;
            acc = fmaf(v, __half2float(vec[j]), acc);
        }
    }

    const float total = block_reduce_sum<BLOCK>(acc);
    if (threadIdx.x == 0) out[row] = __float2half(total);
}

// ============================================================
// 稀疏 matvec（CSR）
//   out[r] = Σ values[k] * vec[col_idx[k]]   (k ∈ [row_ptr[r], row_ptr[r+1]))
// ============================================================
template <int BLOCK>
__global__ void sparse_matvec_kernel(
    const int*  __restrict__ row_ptr,
    const int*  __restrict__ col_idx,
    const half* __restrict__ values,
    const half* __restrict__ vec,
    half*       __restrict__ out,
    int R
) {
    const int row = blockIdx.x;
    if (row >= R) return;
    const int start = row_ptr[row];
    const int end   = row_ptr[row + 1];

    float acc = 0.f;
    for (int k = start + threadIdx.x; k < end; k += BLOCK) {
        acc = fmaf(__half2float(values[k]),
                   __half2float(vec[col_idx[k]]), acc);
    }
    const float total = block_reduce_sum<BLOCK>(acc);
    if (threadIdx.x == 0) out[row] = __float2half(total);
}

// ============================================================
// LUT 最近邻查找（替代 Python 端 (flat - cb).abs().argmin）
// ============================================================
__global__ void lut_lookup_kernel(
    const float* __restrict__ x,
    const float* __restrict__ cb,
    int64_t*     __restrict__ idx_out,
    float*       __restrict__ dequant_out,
    int64_t N, int lut_size
) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float v = x[i];
    float best_d = fabsf(v - cb[0]);
    int   best   = 0;
    #pragma unroll 4
    for (int k = 1; k < lut_size; ++k) {
        const float d = fabsf(v - cb[k]);
        if (d < best_d) { best_d = d; best = k; }
    }
    idx_out[i]     = best;
    dequant_out[i] = cb[best];
}

// ============================================================
// 主机侧包装函数
// ============================================================
torch::Tensor rope_apply(torch::Tensor x,
                         torch::Tensor cos_cache,
                         torch::Tensor sin_cache,
                         int64_t pos_offset) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(x.dim() == 4, "x must be [B,H,S,D]");
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");

    auto xc  = x.contiguous();
    auto out = torch::empty_like(xc);

    const int B = xc.size(0), H = xc.size(1), S = xc.size(2), D = xc.size(3);
    const int half_d = D / 2;
    const int64_t total = (int64_t)B * H * S * half_d;

    const int block = 256;
    const int grid  = (int)((total + block - 1) / block);
    auto stream = at::cuda::getCurrentCUDAStream();

    rope_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const half*>(xc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        cos_cache.contiguous().data_ptr<float>(),
        sin_cache.contiguous().data_ptr<float>(),
        B, H, S, D, (int)pos_offset);
    return out;
}

torch::Tensor lut_matvec(torch::Tensor qmat, torch::Tensor lut,
                         torch::Tensor scale, torch::Tensor zero,
                         torch::Tensor vec, bool per_channel) {
    TORCH_CHECK(qmat.is_cuda() && lut.is_cuda() && scale.is_cuda()
                && zero.is_cuda() && vec.is_cuda(), "all must be CUDA");

    auto qc = qmat.to(torch::kUInt8).contiguous();
    auto lc = lut.contiguous();
    auto sc = scale.contiguous();
    auto zc = zero.contiguous();
    auto vc = vec.contiguous();

    const int N = qc.size(0);
    const int D = qc.size(1);
    const int lut_size = lc.numel();
    TORCH_CHECK(lut_size <= 256, "lut_size too large");

    auto out = torch::empty({N}, vc.options());

    constexpr int BLOCK = 128;
    constexpr int LUT_MAX = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    lut_matvec_kernel<BLOCK, LUT_MAX><<<N, BLOCK, 0, stream>>>(
        qc.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(lc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zc.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(vc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        N, D, lut_size, (int)per_channel);
    return out;
}

torch::Tensor sparse_matvec_csr(torch::Tensor row_ptr, torch::Tensor col_idx,
                                torch::Tensor values, torch::Tensor vec,
                                int64_t R) {
    TORCH_CHECK(row_ptr.is_cuda() && col_idx.is_cuda()
                && values.is_cuda() && vec.is_cuda(), "all must be CUDA");

    auto rp = row_ptr.to(torch::kInt32).contiguous();
    auto ci = col_idx.to(torch::kInt32).contiguous();
    auto vs = values.contiguous();
    auto vc = vec.contiguous();

    auto out = torch::empty({R}, vc.options());

    constexpr int BLOCK = 128;
    auto stream = at::cuda::getCurrentCUDAStream();

    sparse_matvec_kernel<BLOCK><<<(int)R, BLOCK, 0, stream>>>(
        rp.data_ptr<int>(), ci.data_ptr<int>(),
        reinterpret_cast<const half*>(vs.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(vc.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        (int)R);
    return out;
}

std::vector<torch::Tensor> lut_lookup(torch::Tensor x, torch::Tensor codebook) {
    TORCH_CHECK(x.is_cuda() && codebook.is_cuda(), "all must be CUDA");
    auto xc = x.reshape(-1).to(torch::kFloat32).contiguous();
    auto cb = codebook.reshape(-1).to(torch::kFloat32).contiguous();
    const int64_t N = xc.numel();
    const int lut_size = cb.numel();

    auto idx = torch::empty({N}, torch::dtype(torch::kInt64).device(xc.device()));
    auto dq  = torch::empty({N}, torch::dtype(torch::kFloat32).device(xc.device()));

    const int block = 256;
    const int64_t grid = (N + block - 1) / block;
    auto stream = at::cuda::getCurrentCUDAStream();

    lut_lookup_kernel<<<(int)grid, block, 0, stream>>>(
        xc.data_ptr<float>(), cb.data_ptr<float>(),
        idx.data_ptr<int64_t>(), dq.data_ptr<float>(),
        N, lut_size);

    return {idx, dq};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_apply",        &rope_apply,        "RoPE apply (CUDA)");
    m.def("lut_matvec",        &lut_matvec,        "LUT dense matvec (CUDA)");
    m.def("sparse_matvec_csr", &sparse_matvec_csr, "Sparse matvec CSR (CUDA)");
    m.def("lut_lookup",        &lut_lookup,        "LUT lookup (CUDA)");
}