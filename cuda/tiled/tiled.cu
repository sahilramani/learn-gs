// Tiled rasterizer: one block per 16x16 tile, per-tile splat lists,
// shared-memory batches of 256, block-wide early exit. The actual 3DGS
// forward design at notebook scale. Host (torch) builds the tile lists.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE 16
#define BATCH 256

__global__ void rasterize_tiled_kernel(
    int H, int W,
    const int* __restrict__ ranges,       // (n_tiles, 2) start, end
    const int* __restrict__ point_list,   // (n_dup,) splat ids, sorted
    const float* __restrict__ means2d,    // (N, 2)
    const float* __restrict__ conics,     // (N, 3)
    const float* __restrict__ colors,     // (N, 3)
    const float* __restrict__ opacities,  // (N,)
    float alpha_min, float alpha_max, float t_stop,
    float* __restrict__ img,              // (H, W, 3)
    float* __restrict__ T_final)          // (H, W)
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
    __shared__ float4 s_co[BATCH];        // conic A, B, C + opacity
    __shared__ float3 s_rgb[BATCH];

    float T = 1.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;

    for (int base = start; base < end; base += BATCH) {
        // barrier doubles as the vote: if every pixel in the tile is done,
        // the whole block stops fetching
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
    }
}

static void check_f32(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

std::vector<torch::Tensor> rasterize(
    torch::Tensor ranges, torch::Tensor point_list,
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, int64_t H, int64_t W,
    double alpha_min, double alpha_max, double t_stop)
{
    TORCH_CHECK(ranges.dtype() == torch::kInt32 && ranges.is_contiguous());
    TORCH_CHECK(point_list.dtype() == torch::kInt32
                && point_list.is_contiguous());
    check_f32(means2d, "means2d");
    check_f32(conics, "conics");
    check_f32(colors, "colors");
    check_f32(opacities, "opacities");

    int tx = (W + TILE - 1) / TILE;
    int ty = (H + TILE - 1) / TILE;
    TORCH_CHECK(ranges.size(0) == (int64_t)tx * ty, "ranges/tile mismatch");

    auto img = torch::zeros({H, W, 3}, means2d.options());
    auto T_final = torch::ones({H, W}, means2d.options());
    dim3 block(TILE, TILE);
    dim3 grid(tx, ty);
    rasterize_tiled_kernel<<<grid, block>>>(
        (int)H, (int)W,
        ranges.data_ptr<int>(), point_list.data_ptr<int>(),
        means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(),
        (float)alpha_min, (float)alpha_max, (float)t_stop,
        img.data_ptr<float>(), T_final.data_ptr<float>());
    return {img, T_final};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize", &rasterize,
          "tiled shared-memory rasterizer (img, T_final)");
}
