// Step 3: AVX2, single-threaded. The bit-sliced adder tree of step 2 (see
// src/bitpack/bitpack_kernel.hpp) applied to 4 consecutive words of a row per instruction:
// each __m256i lane is one 64-cell word, and the shifts are per-lane 64-bit shifts.
//
// Carries across word boundaries: unaligned loads at offset -1 and +1 word.
//   For words w..w+3 we load three vectors per row:
//     left  = loadu(row + w - 1)   lanes hold words w-1 .. w+2
//     mid   = loadu(row + w)       lanes hold words w   .. w+3
//     right = loadu(row + w + 1)   lanes hold words w+1 .. w+4
//   so lane i of `left` is exactly the word to the left of lane i of `mid`, and
//   west = (mid << 1) | (left >> 63), east = (mid >> 1) | (right << 63), lane by lane.
// Why loads rather than permutes: the alternative keeps the previous/next vector in a
// register and builds the shifted-by-one-word vector with _mm256_permute4x64_epi64 plus a
// blend (AVX2 has no single cross-lane 64-bit shift). Those are 2 extra shuffle-port uops
// with 3-cycle latency per neighbor vector, i.e. 12 per output vector for 3 rows x 2
// directions, all competing for the one shuffle port. The offset loads instead hit lines
// that are already in L1 (the same 3 rows are being streamed), the core issues 2-3 loads
// per cycle, and the cost of the loads that straddle a 64-byte line is small on current
// x86 cores. So the loads are cheaper and the code is simpler.
//
// Row ends: word 0 needs the last word of the row and the last word needs word 0 (torus),
// and the vector loop needs words w-1 .. w+4 to exist. Words not covered by whole vectors
// (word 0, the last word, and up to 3 tail words) are computed with the scalar step 2
// function gol::bitpack::next_word_at, which handles the wrap. For N = 64 (1 word per
// row) everything goes through that scalar path.
//
// Operation count: the same 45 bitwise ops as step 2, now per 4 words (45 vector uops per
// 256 cells, or 11.25 per 64-cell word), plus 9 loads and 1 store per vector.
#include "src/simd/simd_kernel.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "common/registry.hpp"
#include "common/timer.hpp"
#include "src/bitpack/bitpack_kernel.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define GOL_HAVE_X86 1
// Compile the AVX2 code with AVX2 enabled even if the rest of the build were not, so the
// runtime check (not the build flags) decides whether it is used.
#define GOL_AVX2 __attribute__((target("avx2")))
#else
#define GOL_HAVE_X86 0
#define GOL_AVX2
#endif

namespace gol::simd {

bool avx2_supported() {
#if GOL_HAVE_X86
    return __builtin_cpu_supports("avx2");
#else
    return false;
#endif
}

#if GOL_HAVE_X86

namespace {

using V = __m256i;

GOL_AVX2 inline V load(const uint64_t* p) { return _mm256_loadu_si256(reinterpret_cast<const V*>(p)); }
GOL_AVX2 inline void store(uint64_t* p, V v) { _mm256_storeu_si256(reinterpret_cast<V*>(p), v); }
GOL_AVX2 inline V vxor(V a, V b) { return _mm256_xor_si256(a, b); }
GOL_AVX2 inline V vand(V a, V b) { return _mm256_and_si256(a, b); }
GOL_AVX2 inline V vor(V a, V b) { return _mm256_or_si256(a, b); }

// Same definitions as gol::bitpack::west/east, on 4 lanes.
GOL_AVX2 inline V west(V x, V left) { return vor(_mm256_slli_epi64(x, 1), _mm256_srli_epi64(left, 63)); }
GOL_AVX2 inline V east(V x, V right) { return vor(_mm256_srli_epi64(x, 1), _mm256_slli_epi64(right, 63)); }

GOL_AVX2 inline void full_add(V a, V b, V c, V& sum, V& carry) {
    const V ab = vxor(a, b);
    sum = vxor(ab, c);
    carry = vor(vand(a, b), vand(ab, c));
}

// Words w..w+3 of the row `mid`, from rows up/mid/dn. Requires words w-1 .. w+4 to exist.
GOL_AVX2 inline V next4(const uint64_t* up, const uint64_t* mid, const uint64_t* dn, int w) {
    const V u = load(up + w), m = load(mid + w), d = load(dn + w);
    V u0, u1, d0, d1, s0, c0, t0, t1;
    full_add(west(u, load(up + w - 1)), u, east(u, load(up + w + 1)), u0, u1);
    full_add(west(d, load(dn + w - 1)), d, east(d, load(dn + w + 1)), d0, d1);
    const V mw = west(m, load(mid + w - 1)), me = east(m, load(mid + w + 1));
    const V m0 = vxor(mw, me), m1 = vand(mw, me);  // half adder
    full_add(u0, d0, m0, s0, c0);
    full_add(u1, d1, m1, t0, t1);
    const V two_or_three = _mm256_andnot_si256(t1, vxor(t0, c0));  // ~t1 & (t0 ^ c0)
    return vand(two_or_three, vor(s0, m));
}

}  // namespace

GOL_AVX2 void step_region_avx2(const uint64_t* in, uint64_t* out, int rows, int words_per_row,
                               int r0, int r1, int w0, int w1) {
    const std::size_t pitch = static_cast<std::size_t>(words_per_row);
    for (int row = r0; row < r1; ++row) {
        const int above = row == 0 ? rows - 1 : row - 1;
        const int below = row == rows - 1 ? 0 : row + 1;
        const uint64_t* up = in + above * pitch;
        const uint64_t* mid = in + row * pitch;
        const uint64_t* dn = in + below * pitch;
        uint64_t* dst = out + row * pitch;

        int w = w0;
        if (w == 0 && w < w1) {  // word 0 wraps to the end of the row
            dst[0] = bitpack::next_word_at(up, mid, dn, words_per_row, 0);
            w = 1;
        }
        // Whole vectors: words w..w+3 inside the region, and w+4 (read by the +1 load)
        // still inside the row.
        for (; w + 4 <= w1 && w + 4 < words_per_row; w += 4) store(dst + w, next4(up, mid, dn, w));
        // Tail, including the last word of the row (wraps to word 0).
        for (; w < w1; ++w) dst[w] = bitpack::next_word_at(up, mid, dn, words_per_row, w);
    }
}

#else

void step_region_avx2(const uint64_t*, uint64_t*, int, int, int, int, int, int) {}

#endif

namespace {

class SimdBackend : public Backend {
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
            step_rows_avx2(cur_.data(), next_.data(), n_, wpr, 0, n_);
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

}  // namespace gol::simd

GOL_REGISTER_BACKEND_IF("simd", gol::simd::SimdBackend, gol::simd::avx2_supported);
