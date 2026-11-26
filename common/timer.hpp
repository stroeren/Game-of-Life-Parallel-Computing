// Monotonic wall-clock timer for the generation loop.
#pragma once

#include <chrono>

namespace gol {

class Timer {
public:
    Timer() : start_(Clock::now()) {}
    void reset() { start_ = Clock::now(); }
    double seconds() const { return std::chrono::duration<double>(Clock::now() - start_).count(); }

private:
    using Clock = std::chrono::steady_clock;  // monotonic: never jumps with NTP/wall-clock changes
    static_assert(Clock::is_steady, "steady_clock must be monotonic");
    Clock::time_point start_;
};

}  // namespace gol
