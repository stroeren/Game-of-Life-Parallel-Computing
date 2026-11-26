# Step 1: byte baseline

`src/byte/byte.cpp`, impl `byte`.

- One `uint8_t` per cell (0 or 1), N x N row-major, so `grid_bytes = N^2`.
- Two buffers, `cur` and `next`; each generation writes `next` from `cur`, then swaps the
  vectors (no copy).
- Torus wrap: the row above/below is chosen once per row (`row == 0 ? N-1 : row-1`). Within
  a row, the interior columns `1..N-2` use `c-1`/`c+1` directly, so that loop has no
  branches and GCC auto-vectorizes it at `-O3 -march=native`. The two edge columns wrap
  explicitly.
- Rule: `next = (count == 3) | (alive & (count == 2))`, the same form the bit-sliced
  kernels use.
- The kernel is exposed as `gol::byte::step_region(in, out, rows, cols, r0, r1, c0, c1)`
  in `src/byte/byte_kernel.hpp` (and `step_rows` for full-width rows), so OpenMP calls it
  per block of rows and MPI calls it on a local buffer with ghost rows/columns.
- Traffic model: each generation reads and writes the grid once, so
  `gib_per_s = 2 * N^2 * gens / 1024^3 / seconds`. The three input rows a cell needs stay
  in L1/L2, so the real DRAM traffic is close to this model once `N^2` exceeds the LLC
  (N = 4096 is 16 MiB, N = 16384 is 256 MiB per buffer).
