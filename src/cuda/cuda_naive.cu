// Step 5a: CUDA, naive. One thread per cell on the byte grid of step 1; each thread reads
// its 8 neighbors and itself straight from global memory (9 loads per cell, served mostly
// by L1/L2 because neighboring threads read overlapping cells) and wraps with modulo on
// both axes. The modulo by a runtime N is an integer division per neighbor index, which
// is part of what cuda_shared removes. See docs/cuda.md.
#include <cstddef>
#include <cstdint>

#include "src/cuda/byte_device_backend.cuh"

namespace gol::cuda {

namespace {

constexpr int kBlockX = 32;  // one warp spans 32 consecutive cells of a row
constexpr int kBlockY = 8;

__global__ void step_naive(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int n) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= n || c >= n) return;

    int count = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        const std::size_t row = static_cast<std::size_t>((r + dr + n) % n) * n;
        for (int dc = -1; dc <= 1; ++dc)
            if (dr != 0 || dc != 0) count += in[row + (c + dc + n) % n];
    }
    const int alive = in[static_cast<std::size_t>(r) * n + c];
    out[static_cast<std::size_t>(r) * n + c] = static_cast<uint8_t>((count == 3) | (alive & (count == 2)));
}

struct NaiveStep {
    static constexpr const char* kName = "cuda_naive";
    static void prepare() {}
    static void launch(const uint8_t* in, uint8_t* out, int n) {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid((n + kBlockX - 1) / kBlockX, (n + kBlockY - 1) / kBlockY);
        step_naive<<<grid, block>>>(in, out, n);
    }
};

}  // namespace

}  // namespace gol::cuda

GOL_REGISTER_BACKEND_IF("cuda_naive", gol::cuda::ByteDeviceBackend<gol::cuda::NaiveStep>,
                        gol::cuda::device_available);
