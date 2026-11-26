// Step 3 kernel: the step 2 bit-sliced algorithm on 4 words (256 cells) per AVX2
// instruction. Same region/ghost conventions as gol::bitpack::step_region.
#pragma once

#include <cstdint>

namespace gol::simd {

// True if the CPU running this process supports AVX2 (checked at runtime).
bool avx2_supported();

// Compute words [w0, w1) of rows [r0, r1); see gol::bitpack::step_region. Requires
// avx2_supported().
void step_region_avx2(const uint64_t* in, uint64_t* out, int rows, int words_per_row, int r0,
                      int r1, int w0, int w1);

inline void step_rows_avx2(const uint64_t* in, uint64_t* out, int rows, int words_per_row, int r0,
                           int r1) {
    step_region_avx2(in, out, rows, words_per_row, r0, r1, 0, words_per_row);
}

}  // namespace gol::simd
