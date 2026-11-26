// Step 1: the byte baseline. One byte per cell, two buffers swapped each generation,
// torus wrap. Deliberately simple: every other implementation is compared to this one.
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "common/registry.hpp"
#include "common/timer.hpp"
#include "src/byte/byte_kernel.hpp"

namespace gol::byte {

namespace {

// Conway's rule B3/S23: born with exactly 3 neighbors, survives with 2 or 3.
inline uint8_t rule(uint8_t alive, int neighbors) {
    return static_cast<uint8_t>((neighbors == 3) | (alive & (neighbors == 2)));
}

// Cell (mid row, column c) given the column indices of its left and right neighbors.
inline uint8_t next_cell(const uint8_t* up, const uint8_t* mid, const uint8_t* dn, int l, int c,
                         int r) {
    int neighbors = up[l] + up[c] + up[r] + mid[l] + mid[r] + dn[l] + dn[c] + dn[r];
    return rule(mid[c], neighbors);
}

}  // namespace

void step_region(const uint8_t* in, uint8_t* out, int rows, int cols, int r0, int r1, int c0,
                 int c1) {
    const std::size_t pitch = static_cast<std::size_t>(cols);
    for (int row = r0; row < r1; ++row) {
        const int above = row == 0 ? rows - 1 : row - 1;
        const int below = row == rows - 1 ? 0 : row + 1;
        const uint8_t* up = in + above * pitch;
        const uint8_t* mid = in + row * pitch;
        const uint8_t* dn = in + below * pitch;
        uint8_t* dst = out + row * pitch;

        // Interior columns need no wrap, so this loop is branch-free and vectorizes.
        const int lo = std::max(c0, 1);
        const int hi = std::min(c1, cols - 1);
        for (int c = lo; c < hi; ++c) dst[c] = next_cell(up, mid, dn, c - 1, c, c + 1);

        // The two edge columns wrap around the torus.
        if (c0 == 0) dst[0] = next_cell(up, mid, dn, cols - 1, 0, cols > 1 ? 1 : 0);
        if (c1 == cols && cols > 1) dst[cols - 1] = next_cell(up, mid, dn, cols - 2, cols - 1, 0);
    }
}

namespace {

class ByteBackend : public Backend {
public:
    void load(const Grid& initial, const RunConfig&) override {
        n_ = initial.n;
        cur_ = unpack(initial);
        next_.assign(cur_.size(), 0);
    }

    double run(int gens) override {
        Timer t;
        for (int g = 0; g < gens; ++g) {
            step_rows(cur_.data(), next_.data(), n_, n_, 0, n_);
            std::swap(cur_, next_);
        }
        return t.seconds();
    }

    Grid store() override { return pack(cur_.data(), n_); }

    std::size_t grid_bytes() const override { return cur_.size(); }

private:
    int n_ = 0;
    std::vector<uint8_t> cur_, next_;
};

}  // namespace

}  // namespace gol::byte

GOL_REGISTER_BACKEND("byte", gol::byte::ByteBackend);
