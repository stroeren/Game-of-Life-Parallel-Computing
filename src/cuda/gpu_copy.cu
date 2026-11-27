// build/gpu_copy: achievable GPU memory bandwidth, the GPU counterpart of STREAM.
//
//   build/gpu_copy [size_MiB ...]        default sizes: 16 64 256 1024
//
// For each buffer size it times (cudaEvent, best of kReps after a warm-up) a
// device-to-device copy dst[i] = src[i] three ways:
//   vec16    grid-stride kernel, one 16-byte uint4 per thread per iteration (the ceiling)
//   byte     grid-stride kernel, one uint8_t per thread per iteration (the access size the
//            byte-grid backends use)
//   memcpy   cudaMemcpy DeviceToDevice, for reference
// Bandwidth counts read + write, 2 * bytes / seconds, in GiB/s (base 1024), the same
// traffic model as gib_per_s in the CSV. Sizes that fit in L2 measure L2, not DRAM; the
// last line reports vec16 at the largest size, which bench/hwinfo.sh copies into
// docs/hardware.md as "GPU measured, device copy kernel".
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "src/cuda/cuda_util.cuh"

namespace {

using gol::cuda::DeviceBuffer;
using gol::cuda::EventTimer;

constexpr int kReps = 20;
constexpr int kThreadsPerBlock = 256;

__global__ void copy_vec16(const uint4* __restrict__ src, uint4* __restrict__ dst, std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
        dst[i] = src[i];
}

__global__ void copy_byte(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst, std::size_t count) {
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += stride)
        dst[i] = src[i];
}

enum class Method { vec16, u8, d2d };

const char* method_name(Method m) {
    switch (m) {
        case Method::vec16: return "vec16";
        case Method::u8: return "byte";
        case Method::d2d: return "memcpy";
    }
    return "?";
}

void copy_once(Method m, const DeviceBuffer& src, DeviceBuffer& dst, int blocks) {
    const std::size_t bytes = src.bytes();
    switch (m) {
        case Method::vec16:
            copy_vec16<<<blocks, kThreadsPerBlock>>>(src.get<uint4>(), dst.get<uint4>(), bytes / sizeof(uint4));
            break;
        case Method::u8:
            copy_byte<<<blocks, kThreadsPerBlock>>>(src.get<uint8_t>(), dst.get<uint8_t>(), bytes);
            break;
        case Method::d2d:
            GOL_CUDA_CHECK(cudaMemcpy(dst.get(), src.get(), bytes, cudaMemcpyDeviceToDevice));
            break;
    }
}

// Best GiB/s over kReps timed copies, after one untimed warm-up copy.
double measure(Method m, const DeviceBuffer& src, DeviceBuffer& dst, int blocks) {
    copy_once(m, src, dst, blocks);
    GOL_CUDA_CHECK(cudaGetLastError());
    GOL_CUDA_CHECK(cudaDeviceSynchronize());
    double best = 0.0;
    EventTimer t;
    for (int rep = 0; rep < kReps; ++rep) {
        t.start();
        copy_once(m, src, dst, blocks);
        const double s = t.stop_seconds();
        GOL_CUDA_CHECK(cudaGetLastError());
        if (s > 0.0) best = std::max(best, gol::cuda::gib(2.0 * static_cast<double>(src.bytes())) / s);
    }
    return best;
}

int run(int argc, char** argv) {
    std::vector<std::size_t> sizes_mib;
    for (int i = 1; i < argc; ++i) {
        const long v = std::strtol(argv[i], nullptr, 10);
        if (v <= 0) {
            std::fprintf(stderr, "usage: %s [size_MiB ...]  (positive integers)\n", argv[0]);
            return 2;
        }
        sizes_mib.push_back(static_cast<std::size_t>(v));
    }
    if (sizes_mib.empty()) sizes_mib = {16, 64, 256, 1024};
    std::sort(sizes_mib.begin(), sizes_mib.end());

    int device = 0;
    GOL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    GOL_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    // Enough resident blocks to cover every SM several times; the grid-stride loop does
    // the rest.
    const int blocks = prop.multiProcessorCount * 16;
    std::printf("gpu_copy: %s, %d SMs, L2 %.1f MiB, %.1f GiB global memory, %d blocks x %d threads, best of %d\n",
                prop.name, prop.multiProcessorCount, prop.l2CacheSize / (1024.0 * 1024.0),
                gol::cuda::gib(static_cast<double>(prop.totalGlobalMem)), blocks, kThreadsPerBlock, kReps);
    std::printf("size_mib,method,gib_per_s\n");

    double dram_vec16 = 0.0;
    for (std::size_t mib : sizes_mib) {
        const std::size_t bytes = mib * 1024 * 1024;
        DeviceBuffer src(bytes), dst(bytes);
        GOL_CUDA_CHECK(cudaMemset(src.get(), 0x5a, bytes));
        GOL_CUDA_CHECK(cudaMemset(dst.get(), 0, bytes));
        for (Method m : {Method::vec16, Method::u8, Method::d2d}) {
            const double g = measure(m, src, dst, blocks);
            std::printf("%zu,%s,%.1f\n", mib, method_name(m), g);
            if (m == Method::vec16) dram_vec16 = g;  // sizes are sorted: ends at the largest
        }
    }
    const bool in_l2 = sizes_mib.back() * 1024 * 1024 <= static_cast<std::size_t>(prop.l2CacheSize) * 4;
    std::printf("GPU copy kernel: %.1f GiB/s (vec16, %zu MiB per buffer%s)\n", dram_vec16, sizes_mib.back(),
                in_l2 ? "; WARNING: buffer is not much larger than L2, pass a bigger size" : "");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gpu_copy: %s\n", e.what());
        return 1;
    }
}
