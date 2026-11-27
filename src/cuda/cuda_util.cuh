// Host-side CUDA helpers shared by the step 5 backends and build/gpu_copy: error checks,
// an owning device buffer, and cudaEvent timing. No kernels and no gol:: dependencies.
#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace gol::cuda {

inline void check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

#define GOL_CUDA_CHECK(call) ::gol::cuda::check((call), #call)

// True if at least one CUDA device can be used. The backends register with this as their
// runtime predicate, so a CUDA build on a machine without a GPU just hides them.
inline bool device_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();  // clear the (non-sticky) error
        return false;
    }
    return count > 0;
}

// Owning cudaMalloc buffer. Movable, so two of them can be std::swap-ed.
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) { GOL_CUDA_CHECK(cudaMalloc(&ptr_, bytes)); }
    ~DeviceBuffer() { release(); }

    DeviceBuffer(DeviceBuffer&& o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            release();
            ptr_ = std::exchange(o.ptr_, nullptr);
            bytes_ = std::exchange(o.bytes_, 0);
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    template <class T = uint8_t>
    T* get() const { return static_cast<T*>(ptr_); }
    std::size_t bytes() const { return bytes_; }

private:
    void release() {
        if (ptr_ != nullptr) cudaFree(ptr_);
        ptr_ = nullptr;
        bytes_ = 0;
    }
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

// Elapsed GPU time between two events on the default stream.
class EventTimer {
public:
    EventTimer() {
        GOL_CUDA_CHECK(cudaEventCreate(&start_));
        GOL_CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~EventTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

    void start() { GOL_CUDA_CHECK(cudaEventRecord(start_)); }
    // Records the stop event, waits for it and returns the seconds since start().
    double stop_seconds() {
        GOL_CUDA_CHECK(cudaEventRecord(stop_));
        GOL_CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.0f;
        GOL_CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms) / 1e3;
    }

private:
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

inline double gib(double bytes) { return bytes / (1024.0 * 1024.0 * 1024.0); }

}  // namespace gol::cuda
