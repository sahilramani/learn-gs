// Differentiable tiled rasterizer: notebook 11's forward, extended to
// store per-pixel final transmittance and last-contributor count, plus a
// backward kernel that walks each tile back to front with notebook 08's
// recurrence and accumulates per-splat gradients with atomics.
// A minimal diff-gaussian-rasterization; wrapped by notebook 12 in a
// torch.autograd.Function.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE 16
#define BATCH 256

// ---------------------------------------------------------------- forward
__global__ void forward_kernel(
    int H, int W,
    const int* __restrict__ ranges,
    const int* __restrict__ point_list,
    const float* __restrict__ means2d,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    float alpha_min, float alpha_max, float t_stop,
    float* __restrict__ img,
    float* __restrict__ T_final,
    int* __restrict__ n_contrib)          // (H, W) last contributor count
{
    int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
    int px = blockIdx.x * TILE + threadIdx.x;
    int py = blockIdx.y * TILE + threadIdx.y;
    bool inside = (px < W) && (py < H);
    bool done = !inside;

    int start = ranges[2 * tile_id];
    int end = ranges[2 * tile_id + 1];
    int tid = threadIdx.y * TILE + threadIdx.x;

    __shared__ float2 s_xy[BATCH];
    __shared__ float4 s_co[BATCH];
    __shared__ float3 s_rgb[BATCH];

    float T = 1.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    int last = 0;

    for (int base = start; base < end; base += BATCH) {
        if (__syncthreads_count(done) == TILE * TILE)
            break;
        int j = base + tid;
        if (j < end) {
            int s = point_list[j];
            s_xy[tid] = make_float2(means2d[2 * s], means2d[2 * s + 1]);
            s_co[tid] = make_float4(conics[3 * s], conics[3 * s + 1],
                                    conics[3 * s + 2], opacities[s]);
            s_rgb[tid] = make_float3(colors[3 * s], colors[3 * s + 1],
                                     colors[3 * s + 2]);
        }
        __syncthreads();

        int nbatch = min(BATCH, end - base);
        if (!done) {
            for (int k = 0; k < nbatch; k++) {
                float dx = (float)px - s_xy[k].x;
                float dy = (float)py - s_xy[k].y;
                float power = -0.5f * (s_co[k].x * dx * dx
                                       + s_co[k].z * dy * dy)
                              - s_co[k].y * dx * dy;
                if (power > 0.0f) continue;
                float alpha = fminf(s_co[k].w * expf(power), alpha_max);
                if (alpha < alpha_min) continue;
                float w = alpha * T;
                r += w * s_rgb[k].x;
                g += w * s_rgb[k].y;
                b += w * s_rgb[k].z;
                T *= (1.0f - alpha);
                last = base - start + k + 1;      // count within tile list
                if (T < t_stop) { done = true; break; }
            }
        }
    }
    if (inside) {
        int p = py * W + px;
        img[3 * p] = r;
        img[3 * p + 1] = g;
        img[3 * p + 2] = b;
        T_final[p] = T;
        n_contrib[p] = last;
    }
}

// --------------------------------------------------------------- backward
__global__ void backward_kernel(
    int H, int W,
    const int* __restrict__ ranges,
    const int* __restrict__ point_list,
    const float* __restrict__ means2d,
    const float* __restrict__ conics,
    const float* __restrict__ colors,
    const float* __restrict__ opacities,
    const float* __restrict__ T_final,
    const int* __restrict__ n_contrib,
    const float* __restrict__ dL_dimg,    // (H, W, 3)
    float alpha_min, float alpha_max,
    float* __restrict__ d_means2d,        // (N, 2)
    float* __restrict__ d_conics,         // (N, 3)
    float* __restrict__ d_colors,         // (N, 3)
    float* __restrict__ d_opac)           // (N,)
{
    int tile_id = blockIdx.y * gridDim.x + blockIdx.x;
    int px = blockIdx.x * TILE + threadIdx.x;
    int py = blockIdx.y * TILE + threadIdx.y;
    bool inside = (px < W) && (py < H);
    int p = py * W + px;

    int start = ranges[2 * tile_id];
    int end = ranges[2 * tile_id + 1];
    int tid = threadIdx.y * TILE + threadIdx.x;

    __shared__ float2 s_xy[BATCH];
    __shared__ float4 s_co[BATCH];
    __shared__ float3 s_rgb[BATCH];
    __shared__ int s_id[BATCH];

    // per-pixel end state, the input to notebook 08's reconstruction
    int last = inside ? n_contrib[p] : 0;
    float T_run = inside ? T_final[p] : 1.0f;
    float dLr = inside ? dL_dimg[3 * p] : 0.0f;
    float dLg = inside ? dL_dimg[3 * p + 1] : 0.0f;
    float dLb = inside ? dL_dimg[3 * p + 2] : 0.0f;
    float Sr = 0.0f, Sg = 0.0f, Sb = 0.0f;   // color blended behind i

    for (int hi = end; hi > start; hi -= BATCH) {
        int lo = max(start, hi - BATCH);
        __syncthreads();                      // previous batch fully used
        int j = lo + tid;
        if (j < hi) {
            int s = point_list[j];
            s_xy[tid] = make_float2(means2d[2 * s], means2d[2 * s + 1]);
            s_co[tid] = make_float4(conics[3 * s], conics[3 * s + 1],
                                    conics[3 * s + 2], opacities[s]);
            s_rgb[tid] = make_float3(colors[3 * s], colors[3 * s + 1],
                                     colors[3 * s + 2]);
            s_id[tid] = s;
        }
        __syncthreads();

        for (int k = (hi - lo) - 1; k >= 0; k--) {
            int rel = (lo - start) + k;
            if (rel >= last) continue;        // never composited forward
            float dx = (float)px - s_xy[k].x;
            float dy = (float)py - s_xy[k].y;
            float A = s_co[k].x, B = s_co[k].y, C = s_co[k].z;
            float power = -0.5f * (A * dx * dx + C * dy * dy) - B * dx * dy;
            if (power > 0.0f) continue;
            float G = expf(power);
            float alpha_raw = s_co[k].w * G;
            float alpha = fminf(alpha_raw, alpha_max);
            if (alpha < alpha_min) continue;

            // notebook 08, line for line: recover T before this splat,
            // form dL/dalpha from the suffix, then push the suffix
            T_run = T_run / (1.0f - alpha);
            float w = alpha * T_run;
            int s = s_id[k];

            atomicAdd(&d_colors[3 * s], dLr * w);
            atomicAdd(&d_colors[3 * s + 1], dLg * w);
            atomicAdd(&d_colors[3 * s + 2], dLb * w);

            float inv1a = 1.0f / (1.0f - alpha);
            float dLda = dLr * (s_rgb[k].x * T_run - Sr * inv1a)
                       + dLg * (s_rgb[k].y * T_run - Sg * inv1a)
                       + dLb * (s_rgb[k].z * T_run - Sb * inv1a);

            Sr += s_rgb[k].x * w;
            Sg += s_rgb[k].y * w;
            Sb += s_rgb[k].z * w;

            if (alpha_raw > alpha_max)
                continue;                     // clamped: d alpha / d * = 0

            atomicAdd(&d_opac[s], dLda * G);
            float dLdpow = dLda * alpha;
            atomicAdd(&d_means2d[2 * s], dLdpow * (A * dx + B * dy));
            atomicAdd(&d_means2d[2 * s + 1], dLdpow * (C * dy + B * dx));
            atomicAdd(&d_conics[3 * s], dLdpow * (-0.5f * dx * dx));
            atomicAdd(&d_conics[3 * s + 1], dLdpow * (-dx * dy));
            atomicAdd(&d_conics[3 * s + 2], dLdpow * (-0.5f * dy * dy));
        }
    }
}

// ------------------------------------------------------------------ host
static void check_f32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> forward(
    torch::Tensor ranges, torch::Tensor point_list,
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, int64_t H, int64_t W,
    double alpha_min, double alpha_max, double t_stop)
{
    check_f32(means2d, "means2d"); check_f32(conics, "conics");
    check_f32(colors, "colors"); check_f32(opacities, "opacities");
    int tx = (W + TILE - 1) / TILE;
    int ty = (H + TILE - 1) / TILE;
    TORCH_CHECK(ranges.size(0) == (int64_t)tx * ty, "ranges/tile mismatch");

    auto img = torch::zeros({H, W, 3}, means2d.options());
    auto T_final = torch::ones({H, W}, means2d.options());
    auto n_contrib = torch::zeros({H, W},
                                  means2d.options().dtype(torch::kInt32));
    dim3 block(TILE, TILE);
    dim3 grid(tx, ty);
    forward_kernel<<<grid, block>>>(
        (int)H, (int)W, ranges.data_ptr<int>(), point_list.data_ptr<int>(),
        means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(),
        (float)alpha_min, (float)alpha_max, (float)t_stop,
        img.data_ptr<float>(), T_final.data_ptr<float>(),
        n_contrib.data_ptr<int>());
    return {img, T_final, n_contrib};
}

std::vector<torch::Tensor> backward(
    torch::Tensor ranges, torch::Tensor point_list,
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, torch::Tensor T_final, torch::Tensor n_contrib,
    torch::Tensor dL_dimg, int64_t H, int64_t W,
    double alpha_min, double alpha_max)
{
    check_f32(dL_dimg, "dL_dimg");
    int64_t N = means2d.size(0);
    auto d_means2d = torch::zeros({N, 2}, means2d.options());
    auto d_conics = torch::zeros({N, 3}, means2d.options());
    auto d_colors = torch::zeros({N, 3}, means2d.options());
    auto d_opac = torch::zeros({N}, means2d.options());
    int tx = (W + TILE - 1) / TILE;
    int ty = (H + TILE - 1) / TILE;
    dim3 block(TILE, TILE);
    dim3 grid(tx, ty);
    backward_kernel<<<grid, block>>>(
        (int)H, (int)W, ranges.data_ptr<int>(), point_list.data_ptr<int>(),
        means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(),
        T_final.data_ptr<float>(), n_contrib.data_ptr<int>(),
        dL_dimg.contiguous().data_ptr<float>(),
        (float)alpha_min, (float)alpha_max,
        d_means2d.data_ptr<float>(), d_conics.data_ptr<float>(),
        d_colors.data_ptr<float>(), d_opac.data_ptr<float>());
    return {d_means2d, d_conics, d_colors, d_opac};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward,
          "tiled forward storing T_final and n_contrib");
    m.def("backward", &backward,
          "tiled backward: grads for means2d, conics, colors, opacities");
}
