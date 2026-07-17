// Naive rasterizer: one thread per pixel, loop over every sorted splat.
// Correct and slow on purpose; notebook 10 measures exactly how slow.
// Compiled by torch.utils.cpp_extension.load from the notebook.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>

__global__ void rasterize_naive_kernel(
    int H, int W, int N,
    const float* __restrict__ means2d,    // (N, 2) u, v
    const float* __restrict__ conics,     // (N, 3) A, B, C (inverse cov)
    const float* __restrict__ colors,     // (N, 3)
    const float* __restrict__ opacities,  // (N,)
    float alpha_min, float alpha_max, float t_stop,
    float* __restrict__ img,              // (H, W, 3)
    float* __restrict__ T_final)          // (H, W)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;

    float T = 1.0f;
    float r = 0.0f, g = 0.0f, b = 0.0f;
    for (int i = 0; i < N; i++) {
        float dx = (float)px - means2d[2 * i];
        float dy = (float)py - means2d[2 * i + 1];
        float A = conics[3 * i], B = conics[3 * i + 1], C = conics[3 * i + 2];
        float power = -0.5f * (A * dx * dx + C * dy * dy) - B * dx * dy;
        if (power > 0.0f) continue;               // numerical guard
        float alpha = fminf(opacities[i] * expf(power), alpha_max);
        if (alpha < alpha_min) continue;
        float w = alpha * T;
        r += w * colors[3 * i];
        g += w * colors[3 * i + 1];
        b += w * colors[3 * i + 2];
        T *= (1.0f - alpha);
        if (T < t_stop) break;                    // pixel saturated
    }
    int p = py * W + px;
    img[3 * p] = r;
    img[3 * p + 1] = g;
    img[3 * p + 2] = b;
    T_final[p] = T;
}

static void check_input(const torch::Tensor& t, int64_t dim0, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.size(0) == dim0, name, " has wrong length");
}

std::vector<torch::Tensor> rasterize(
    torch::Tensor means2d, torch::Tensor conics, torch::Tensor colors,
    torch::Tensor opacities, int64_t H, int64_t W,
    double alpha_min, double alpha_max, double t_stop)
{
    int64_t N = means2d.size(0);
    check_input(means2d, N, "means2d");
    check_input(conics, N, "conics");
    check_input(colors, N, "colors");
    check_input(opacities, N, "opacities");

    auto img = torch::zeros({H, W, 3}, means2d.options());
    auto T_final = torch::ones({H, W}, means2d.options());
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    rasterize_naive_kernel<<<grid, block>>>(
        (int)H, (int)W, (int)N,
        means2d.data_ptr<float>(), conics.data_ptr<float>(),
        colors.data_ptr<float>(), opacities.data_ptr<float>(),
        (float)alpha_min, (float)alpha_max, (float)t_stop,
        img.data_ptr<float>(), T_final.data_ptr<float>());
    return {img, T_final};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize", &rasterize,
          "naive per-pixel rasterizer (img, T_final)");
}
