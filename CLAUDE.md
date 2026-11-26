# Game of Life: CSI 4650 course project

## Goal
An optimization path for Conway's Game of Life on a torus. Every step must be correct
(same checksum as the byte baseline) and measured (cells/s, GiB/s) against hardware
bandwidth and Golly.

## Layout
common/        grid init, checksum, timer, CLI, implementation registry (owner: Person A;
               change only through a PR that A reviews)
src/byte/      step 1: one byte per cell, two swapped buffers
src/bitpack/   step 2: 64 cells per uint64_t, bit-sliced neighbor count
src/simd/      step 3: AVX2 / AVX-512 on bit-packed words
src/omp/       step 4: OpenMP over blocks of rows
src/cuda/      step 5: CUDA, tile + halo in shared memory
src/mpi/       step 6: MPI halo exchange (binary gol_mpi)
tests/         blinker, glider, cross-implementation checksum (CTest)
bench/         benchmark scripts, results/*.csv, plot scripts
docs/          hardware.md and per-step notes
report/        report sources and figures

## Contracts (do not break)
- The grid is N x N, N a multiple of 64, wrapping on both axes (torus).
- Initial state: splitmix64 seeded with --seed fills the canonical bit-packed grid one
  uint64_t at a time in row-major order. Bit j of word w in row r is cell (r, 64*w + j).
  Every implementation starts from this grid.
- Checksum: FNV-1a 64 over the canonical bit-packed words as little-endian bytes. Every
  implementation converts its state to the canonical layout before hashing.
- Driver: gol --impl <name> --n <N> --gens <K> --seed <S> [--threads T] [--check]
  prints one CSV row:
  impl,n,gens,threads,ranks,seconds,cells_per_sec,grid_bytes,gib_per_s,checksum
- MPI uses gol_mpi with the same flags plus --decomp strip|block,
  --mode blocking|nonblocking and --kernel byte|bitpack.
- seconds covers the generation loop only: no init, no host-device copies (report copies
  separately on stderr).
- gib_per_s = 2 * grid_bytes * gens / 1024^3 / seconds, because each generation reads and
  writes the grid. All sizes are base-1024 (GiB). Multiply vendor GB/s by 0.931.

## Rules
- An implementation is done only when ctest passes: blinker (period 2), glider (equals
  itself shifted one cell diagonally after 4 generations, including across the wrap), and
  checksum equal to byte after 100 generations at N=256 and N=1024 with seed 42.
- Never report a speedup from a run whose checksum does not match.
- Benchmarks: median of 5 runs, gens chosen so each run takes at least 1 s,
  N = 1024, 4096, 16384.
- Build: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
  && ctest --test-dir build --output-on-failure
- Flags live in CMakeLists.txt: -O3 -march=native. CUDA and MPI targets build only when
  found; nothing else may depend on them.
- Add a backend with one new source file plus one registry line. Do not edit another
  person's kernel; call it.
- Ask before running anything with sudo.

## How the harness works (added by prompt 0)
- A backend is a class derived from `gol::Backend` (common/registry.hpp) with
  `load` (canonical grid in, not timed), `run(gens)` (returns seconds of the generation
  loop), `store` (canonical grid out, not timed) and `grid_bytes`. Register it at the
  bottom of its own source file with `GOL_REGISTER_BACKEND("name", Class)` (or
  `GOL_REGISTER_BACKEND_IF("name", Class, predicate)` when it needs a runtime CPU/GPU check).
- The one registry line goes in src/CMakeLists.txt: `gol_backend(dir/file.cpp)` or
  `gol_cuda_backend(dir/file.cu)`. Tests and bench/run.sh pick up every registered backend.
- Reusable kernels expose a row/region function in a header next to the source, e.g.
  `gol::byte::step_rows` in src/byte/byte_kernel.hpp. Ghost rows/columns are handled by
  passing a region that excludes them.
- `gol --list` prints the available backends; `gol --csv-header` prints the CSV header.
- bench/run.sh [impl...] appends to bench/results/<hostname>.csv; bench/summarize.py prints
  the median table with speedup over byte. bench/hwinfo.sh regenerates docs/hardware.md and
  bench/stream.sh measures STREAM Triad.
