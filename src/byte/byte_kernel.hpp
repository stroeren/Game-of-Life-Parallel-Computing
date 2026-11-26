// Step 1 kernel: one byte per cell (0 = dead, 1 = alive), row-major.
//
// Other backends call these instead of copying them (OpenMP over row blocks, MPI over a
// local strip or block with ghost cells).
#pragma once

#include <cstdint>

namespace gol::byte {

// Compute the next generation of cells [r0, r1) x [c0, c1) of a `rows` x `cols` byte grid
// (row pitch = cols) from `in` into `out`. Neighbors wrap modulo rows and cols (torus).
//
// - Whole torus:            step_region(in, out, n, n, 0, n, 0, n)
// - A block of rows (OpenMP): step_region(in, out, n, n, r0, r1, 0, n)
// - Local strip with one ghost row above and below (MPI): rows = local + 2, update
//   [1, local + 1). The region never touches row 0 or rows - 1 as a center, so the
//   vertical wrap is never taken and the ghost rows supply the neighbors.
// - Local block with ghost rows and columns: rows = h + 2, cols = w + 2, update
//   [1, h + 1) x [1, w + 1).
void step_region(const uint8_t* in, uint8_t* out, int rows, int cols, int r0, int r1, int c0,
                 int c1);

// Full-width rows [r0, r1).
inline void step_rows(const uint8_t* in, uint8_t* out, int rows, int cols, int r0, int r1) {
    step_region(in, out, rows, cols, r0, r1, 0, cols);
}

}  // namespace gol::byte
