// Step 2 kernel: 64 cells per uint64_t in the canonical layout, bit-sliced neighbor count.
//
// Other backends call these instead of copying them: src/simd uses next_word_at for the
// words at the row ends, OpenMP calls step_rows per block of rows, and MPI calls
// step_region on a local buffer with ghost rows (and ghost words for block decomposition).
//
// ======================================================================================
// The bit-sliced adder tree
// ======================================================================================
// A word holds 64 cells of one row: bit j is column 64*w + j. Instead of counting the
// neighbors of one cell at a time, we build 8 "neighbor words" N0..N7 whose bit j is the
// state of cell j's 1st..8th neighbor, then add those eight 1-bit numbers in all 64 bit
// positions ("lanes") at once using only XOR/AND/OR, like a hardware adder circuit:
//
//   row above:  N0 = west(up)  N1 = up       N2 = east(up)
//   same row:   N3 = west(mid)               N4 = east(mid)
//   row below:  N5 = west(dn)  N6 = dn       N7 = east(dn)
//
//   west(x) = (x << 1) | (word_to_the_left >> 63)   bit j <- column c-1
//   east(x) = (x >> 1) | (word_to_the_right << 63)  bit j <- column c+1
//   (the carried-in bit comes from the adjacent word; at the row ends that word wraps
//    around to the other end of the row, which makes the grid a torus horizontally)
//
//   full adder FA(a,b,c) -> (sum, carry):  sum = a^b^c,  carry = (a&b) | (c&(a^b))
//   half adder HA(a,b)   -> (sum, carry):  sum = a^b,    carry = a&b
//
//   Level 1, per row:   (u0,u1) = FA(N0,N1,N2)   up count   = u0 + 2*u1   (0..3)
//                       (d0,d1) = FA(N5,N6,N7)   down count = d0 + 2*d1   (0..3)
//                       (m0,m1) = HA(N3,N4)      mid count  = m0 + 2*m1   (0..2)
//   Level 2, ones:      (s0,c0) = FA(u0,d0,m0)   s0 is bit 0 of the total, c0 has weight 2
//   Level 2, twos:      (t0,t1) = FA(u1,d1,m1)   t0 has weight 2, t1 has weight 4
//
//   count = s0 + 2*(c0 + t0) + 4*t1
//
// The rule only needs to know whether count is 2 or 3, i.e. whether the weight-2 part is
// exactly 1 and nothing of weight >= 4 is set:
//
//   two_or_three = ~t1 & (t0 ^ c0)            (t0 = c0 = 1 would mean count >= 4)
//   next         = two_or_three & (s0 | alive)
//
// which is next = (count == 3) | (alive & (count == 2)): with count in {2,3}, s0 = 1 means
// 3 (born or survives) and s0 = 0 means 2 (survives only if alive).
//
// Operation count per 64-cell word (for the arithmetic-intensity estimate)
//   6 shifted neighbor words x 3 ops (shl, shr, or)             = 18
//   4 full adders x 5 ops (2 xor, 2 and, 1 or)                  = 20
//   1 half adder (xor, and)                                     =  2
//   rule (xor, not, and, or, and)                               =  5
//   total                                                       = 45 bitwise ops
// (compilers fuse not+and into one andn with BMI1, giving 44). Memory traffic in the
// CLAUDE.md model is 16 bytes per word (read 8, write 8), so the arithmetic intensity is
// 45 / 16 = 2.8 integer ops per byte. These are integer ops, not FLOPs. The 9 loads per
// word (3 words from each of 3 rows) hit L1 except for the first touch of each line.
// ======================================================================================
#pragma once

#include <cstdint>

namespace gol::bitpack {

// Bitwise operations per word in next_word (see above).
inline constexpr int kOpsPerWord = 45;

inline uint64_t west(uint64_t x, uint64_t left) { return (x << 1) | (left >> 63); }
inline uint64_t east(uint64_t x, uint64_t right) { return (x >> 1) | (right << 63); }

inline void full_add(uint64_t a, uint64_t b, uint64_t c, uint64_t& sum, uint64_t& carry) {
    const uint64_t ab = a ^ b;
    sum = ab ^ c;
    carry = (a & b) | (ab & c);
}

inline void half_add(uint64_t a, uint64_t b, uint64_t& sum, uint64_t& carry) {
    sum = a ^ b;
    carry = a & b;
}

// Next state of word `m` given its 8 surrounding words (l = word to the left, i.e. lower
// columns; r = word to the right).
inline uint64_t next_word(uint64_t ul, uint64_t u, uint64_t ur, uint64_t ml, uint64_t m,
                          uint64_t mr, uint64_t dl, uint64_t d, uint64_t dr) {
    uint64_t u0, u1, d0, d1, m0, m1, s0, c0, t0, t1;
    full_add(west(u, ul), u, east(u, ur), u0, u1);
    full_add(west(d, dl), d, east(d, dr), d0, d1);
    half_add(west(m, ml), east(m, mr), m0, m1);
    full_add(u0, d0, m0, s0, c0);
    full_add(u1, d1, m1, t0, t1);
    const uint64_t two_or_three = ~t1 & (t0 ^ c0);
    return two_or_three & (s0 | m);
}

// Word w of a row, with the left/right words wrapping around the row (torus). Works for
// words_per_row == 1 too (the word is its own left and right neighbor).
inline uint64_t next_word_at(const uint64_t* up, const uint64_t* mid, const uint64_t* dn,
                             int words_per_row, int w) {
    const int l = w == 0 ? words_per_row - 1 : w - 1;
    const int r = w == words_per_row - 1 ? 0 : w + 1;
    return next_word(up[l], up[w], up[r], mid[l], mid[w], mid[r], dn[l], dn[w], dn[r]);
}

// Compute words [w0, w1) of rows [r0, r1) of a `rows` x `words_per_row` grid (row pitch =
// words_per_row) from `in` into `out`. Rows and words wrap (torus), with the same ghost
// conventions as gol::byte::step_region: a buffer with ghost rows/words passes a region
// that excludes them, so the wrap is never taken there.
void step_region(const uint64_t* in, uint64_t* out, int rows, int words_per_row, int r0, int r1,
                 int w0, int w1);

// Full-width rows [r0, r1).
inline void step_rows(const uint64_t* in, uint64_t* out, int rows, int words_per_row, int r0,
                      int r1) {
    step_region(in, out, rows, words_per_row, r0, r1, 0, words_per_row);
}

}  // namespace gol::bitpack
