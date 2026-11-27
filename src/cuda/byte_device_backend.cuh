// The part of a step 5 backend that is not the kernel: a byte grid (one uint8_t per cell,
// as in step 1) that stays on the device for all generations.
//
//   load   unpack the canonical grid on the host, copy it to the device (H2D), zero the
//          second buffer. Not timed; the H2D time goes to stderr.
//   run    launch one kernel per generation and swap the two device pointers. Timed with
//          cudaEvents around the launches only (CLAUDE.md: no host-device copies in
//          `seconds`).
//   store  copy the current buffer back (D2H, time to stderr) and pack it into the
//          canonical layout for the checksum.
//
// `Step` supplies the kernel: `static constexpr const char* kName`,
// `static void prepare()`, called once in load (outside the timed loop) for any one-time
// setup such as reading a configuration, and
// `static void launch(const uint8_t* in, uint8_t* out, int n)`, which enqueues one
// generation on the default stream. Kernels on the default stream run in order, so the
// pointer swap after each launch is safe without synchronizing.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "common/grid.hpp"
#include "common/registry.hpp"
#include "src/cuda/cuda_util.cuh"

namespace gol::cuda {

inline void report_copy(const char* impl, const char* direction, std::size_t bytes, double seconds) {
    std::fprintf(stderr, "%s: %s copy %.1f MiB in %.3f ms (%.2f GiB/s), not included in seconds\n",
                 impl, direction, static_cast<double>(bytes) / (1024.0 * 1024.0), seconds * 1e3,
                 seconds > 0.0 ? gib(static_cast<double>(bytes)) / seconds : 0.0);
}

template <class Step>
class ByteDeviceBackend : public Backend {
public:
    void load(const Grid& initial, const RunConfig&) override {
        Step::prepare();
        n_ = initial.n;
        const std::size_t bytes = static_cast<std::size_t>(n_) * static_cast<std::size_t>(n_);
        const std::vector<uint8_t> host = unpack(initial);
        cur_ = DeviceBuffer(bytes);
        next_ = DeviceBuffer(bytes);

        EventTimer t;
        t.start();
        GOL_CUDA_CHECK(cudaMemcpy(cur_.get(), host.data(), bytes, cudaMemcpyHostToDevice));
        report_copy(Step::kName, "H2D", bytes, t.stop_seconds());
        GOL_CUDA_CHECK(cudaMemset(next_.get(), 0, bytes));
        GOL_CUDA_CHECK(cudaDeviceSynchronize());
    }

    double run(int gens) override {
        uint8_t* in = cur_.get();
        uint8_t* out = next_.get();
        EventTimer t;
        t.start();
        for (int g = 0; g < gens; ++g) {
            Step::launch(in, out, n_);
            std::swap(in, out);  // swap device pointers, never data
        }
        const double seconds = t.stop_seconds();  // waits for the last kernel
        GOL_CUDA_CHECK(cudaGetLastError());        // launch errors (bad configuration etc.)
        if (gens % 2 != 0) std::swap(cur_, next_);
        return seconds;
    }

    Grid store() override {
        std::vector<uint8_t> host(cur_.bytes());
        EventTimer t;
        t.start();
        GOL_CUDA_CHECK(cudaMemcpy(host.data(), cur_.get(), host.size(), cudaMemcpyDeviceToHost));
        report_copy(Step::kName, "D2H", host.size(), t.stop_seconds());
        return pack(host.data(), n_);
    }

    std::size_t grid_bytes() const override { return cur_.bytes(); }

private:
    int n_ = 0;
    DeviceBuffer cur_, next_;
};

}  // namespace gol::cuda
