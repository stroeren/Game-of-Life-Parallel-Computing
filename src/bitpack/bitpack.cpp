// Step 2: bit-packed, single-threaded. The grid stays in the canonical layout (N/64 words
// per row), two buffers swapped each generation. See bitpack_kernel.hpp for the adder tree.
#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

#include "common/registry.hpp"
#include "common/timer.hpp"
#include "src/bitpack/bitpack_kernel.hpp"

namespace gol::bitpack {

// Auto-vectorization is disabled for this function so that step 2 measures the scalar
// bit-sliced kernel and step 3 (src/simd) measures what explicit SIMD adds. Without this,
// GCC -O3 -march=native vectorizes the interior loop on its own (see docs/bitpack.md).
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("no-tree-vectorize")))
#endif
void step_region(const uint64_t* in, uint64_t* out, int rows, int words_per_row, int r0, int r1,
                 int w0, int w1) {
    const std::size_t pitch = static_cast<std::size_t>(words_per_row);
    for (int row = r0; row < r1; ++row) {
        const int above = row == 0 ? rows - 1 : row - 1;
        const int below = row == rows - 1 ? 0 : row + 1;
        const uint64_t* up = in + above * pitch;
        const uint64_t* mid = in + row * pitch;
        const uint64_t* dn = in + below * pitch;
        uint64_t* dst = out + row * pitch;

        // Interior words: both neighbor words exist in this row, no wrap.
        const int lo = std::max(w0, 1);
        const int hi = std::min(w1, words_per_row - 1);
        for (int w = lo; w < hi; ++w)
            dst[w] = next_word(up[w - 1], up[w], up[w + 1], mid[w - 1], mid[w], mid[w + 1],
                               dn[w - 1], dn[w], dn[w + 1]);

        // First and last word of the row take their carry bit from the other end.
        if (w0 == 0) dst[0] = next_word_at(up, mid, dn, words_per_row, 0);
        if (w1 == words_per_row && words_per_row > 1)
            dst[words_per_row - 1] = next_word_at(up, mid, dn, words_per_row, words_per_row - 1);
    }
}

namespace {

class BitpackBackend : public Backend {
public:
    void load(const Grid& initial, const RunConfig&) override {
        n_ = initial.n;
        cur_ = initial.words;
        next_.assign(cur_.size(), 0);
    }

    double run(int gens) override {
        const int wpr = n_ / 64;
        Timer t;
        for (int g = 0; g < gens; ++g) {
            step_rows(cur_.data(), next_.data(), n_, wpr, 0, n_);
            std::swap(cur_, next_);
        }
        return t.seconds();
    }

    Grid store() override {
        Grid g(n_);
        g.words = cur_;
        return g;
    }

    std::size_t grid_bytes() const override { return cur_.size() * sizeof(uint64_t); }

private:
    int n_ = 0;
    std::vector<uint64_t> cur_, next_;
};

}  // namespace

}  // namespace gol::bitpack

GOL_REGISTER_BACKEND("bitpack", gol::bitpack::BitpackBackend);
