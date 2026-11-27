# Step 4: OpenMP

`src/omp/omp.cpp`, impls `omp_byte`, `omp_bitpack`, `omp_simd` (`omp_simd` only with AVX2).

> **Status: not built or measured yet.** This was written on a machine with no compiler
> toolchain, so the code below has not been compiled, ctest has not been run, and no numbers
> exist. The analysis in "Why static and dynamic perform as they do" is a prediction to
> check against the sweep, not a result.

## Design

- **The existing kernels, not copies.** Each generation is one `#pragma omp for` over the
  rows, and each iteration calls the kernel's row-range function for its row:
  `gol::byte::step_rows`, `gol::bitpack::step_rows` or `gol::simd::step_rows_avx2`. A
  kernel reads only `in` and writes only its own rows of `out`, so the rows of one
  generation are independent.
- **One parallel region for all generations.** Threads are created once per `run()`. Each
  generation costs one barrier, the implicit one at the end of the `omp for`, and no
  fork/join. Each thread swaps its private copies of the `in`/`out` pointers after the
  barrier.
- **Schedule from the environment.** The loop is `schedule(runtime)`, so
  `OMP_SCHEDULE=static | dynamic,<c> | guided` picks the schedule without a rebuild. When
  `OMP_SCHEDULE` is unset the backend sets `static` itself, because libgomp's default for
  `schedule(runtime)` is `dynamic,1`, the worst choice here. The schedule is not in the CSV
  row (the contract fixes its columns), so `bench/omp_sweep.sh` records it in its own
  columns.
- **Buffers.** Two buffers, 64-byte aligned and uninitialized at allocation. Rows are whole
  cache lines (byte: N bytes; bit-packed: N/8 bytes for N >= 512), so threads writing
  neighboring rows never share a line. Without the alignment, the 16-byte-aligned base that
  glibc returns for large allocations would put a shared line at every block or chunk
  boundary.
- **First touch.** `load` fills both buffers with a parallel loop that has the same shape
  and schedule as the generation loop. Under `static`, the thread that later computes rows
  [a, b) is the one that first writes their pages, so on a multi-node machine those pages
  are placed on its NUMA node. `GOL_OMP_FIRST_TOUCH=0` fills them from the main thread
  instead, which puts every page on node 0.

## The sweep

```
bench/omp_sweep.sh                         # all omp_* impls, N = 1024 4096 16384
bench/summarize_omp.py                     # speedup, efficiency, % of STREAM per point
FIRST_TOUCH="1 0" bench/omp_sweep.sh       # multi-NUMA machines only: first-touch comparison
```

- Schedules: `static`, `dynamic,1`, `dynamic,16` (`CHUNKS` overrides the chunk sizes) and
  `guided`.
- Threads: 1, 2, 4, ... up to `nproc`. Every point runs with
  `OMP_PROC_BIND=close OMP_PLACES=cores`.
- Each point is one `bench/run.sh` call: a checksum check against byte, gens chosen for at
  least 1 s, and the median of 5 runs. Results go to `bench/results/omp/<host>.csv`.
- With `OMP_PLACES=cores`, a place is a physical core. Beyond the core count (16 threads on
  an 8-core/16-thread CPU), `close` puts two threads on each core's place, so the top point
  measures SMT.

## Why static and dynamic perform as they do

The work per row is uniform. Every row has the same number of cells or words, and the
kernels are branch-free in the cell state: the byte kernel's sum and the bit-sliced adder
tree execute the same instructions whatever the cells hold. Static scheduling therefore
already balances the load. Thread t gets one contiguous block of about N/T rows, and the
blocks differ by at most one row. Dynamic scheduling cannot improve the balance. It only
adds costs:

1. **Scheduling overhead.** Every chunk is an atomic fetch-and-add on a counter that all
   threads share. Its cache line bounces between cores, at tens to a few hundred ns per grab
   under contention. Compare that with the work in one row: one bit-packed row at N=1024 is
   16 words, and at the single-thread simd rate on the dev laptop (about 2.5e10 cells/s)
   that is about 40 ns. So `dynamic,1` on `omp_simd` at N=1024 could spend as long fetching
   rows as computing them, and may not beat one thread. Larger chunks (`dynamic,16`) and
   larger N (more work per row) shrink this cost.
2. **Lost locality inside a generation.** Row r needs rows r-1, r and r+1. With a static
   block, a thread loads each input row into its own L1/L2 once and uses it for three
   consecutive output rows. With `dynamic,1`, consecutive rows go to different threads, so
   each input row is pulled into up to three cores' caches, which can triple the read
   traffic from the shared L3 or DRAM.
3. **Lost locality across generations.** Under static, thread t computes the same rows in
   every generation. When a thread's share of both buffers fits in its private L2 (about
   1 MiB per core on Zen 5 and Alder Lake P-cores), the next generation reads its input
   from L2. With N=1024 and 8 threads, `omp_byte` has 2 x 128 KiB per thread and
   `omp_bitpack` 2 x 16 KiB. Dynamic and guided hand out different rows each generation,
   so the input usually sits in another core's cache.
4. **guided** grabs few, large chunks at first (about remaining/T), so its overhead is
   small, but it still loses (3). It should land between static and dynamic.

Expected order: `static` >= `guided` > `dynamic,16` > `dynamic,1`. The gap should be
largest where the work per row is smallest: small N and the fastest kernel, `omp_simd`.
It should shrink at N=16384 for `omp_byte`, which becomes DRAM-bound (2 x 256 MiB of
grid), so all schedules wait on the same memory bandwidth. Dynamic scheduling only pays
off when work per iteration varies, for example a sparse-grid kernel that skips empty
words. None of the kernels here does that.

Other expectations to check:

- **Scaling with threads.** `omp_bitpack` is compute-bound (step 2 runs flat across N on
  one core), so it should scale close to linearly up to the core count. `omp_simd` scales
  until it hits the cache or DRAM bandwidth of the level its grid lives in. `omp_byte`
  moves 8x more bytes per cell, so at N=16384 it should saturate near STREAM Triad well
  before all cores are busy.
- **Where the grid lives.** STREAM measures DRAM, so a grid that stays cache-resident can
  show more than 100% of STREAM. On a Ryzen 7 9800X3D (96 MiB L3), both buffers fit in L3
  for `omp_byte` up to N=4096 (2 x 16 MiB) and for `omp_bitpack`/`omp_simd` even at
  N=16384 (2 x 32 MiB). The DRAM-bound case is `omp_byte` at N=16384.
- **Barrier cost.** One barrier per generation (about 1 us) is visible only when a
  generation is short: `omp_simd` at N=1024 takes tens of microseconds per generation on
  one core.

## NUMA and first touch

Run `lscpu | grep 'NUMA node(s)'` on the benchmark machine. The dev laptop (i7-1255U) has
1 NUMA node (docs/hardware.md), and so does a single-CCD desktop part such as the Ryzen 7
9800X3D. With one node, every page is equally close to every core, so first touch should
make no difference: `GOL_OMP_FIRST_TOUCH=0` and `=1` should agree within noise. If the
testbed shows more than one NUMA node, run `FIRST_TOUCH="1 0" bench/omp_sweep.sh`. The
expected result is that serial initialization, which puts all pages on node 0, caps
DRAM-bound runs (`omp_byte`, N=16384) at roughly one node's bandwidth plus remote-access
latency, while parallel first touch under `static` scales with the number of nodes. Under
`dynamic` and `guided`, the thread that touched a page at init is usually not the one that
computes it later, so first touch helps far less there. That is another argument for
`static`.

## Results

Not measured yet. On a Linux machine with GCC and OpenMP:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure
bench/stream.sh                      # STREAM Triad, the CPU ceiling
bench/run.sh omp_byte omp_bitpack omp_simd
bench/summarize.py                   # cells/s, GiB/s, speedup vs byte, % of STREAM
bench/omp_sweep.sh && bench/summarize_omp.py
```

| N | impl | best schedule | threads | cells/s | GiB/s | speedup vs byte | % of STREAM |
|---:|---|---|---:|---:|---:|---:|---:|
| 1024 | omp_byte | | | | | | |
| 1024 | omp_bitpack | | | | | | |
| 1024 | omp_simd | | | | | | |
| 4096 | omp_byte | | | | | | |
| 4096 | omp_bitpack | | | | | | |
| 4096 | omp_simd | | | | | | |
| 16384 | omp_byte | | | | | | |
| 16384 | omp_bitpack | | | | | | |
| 16384 | omp_simd | | | | | | |
