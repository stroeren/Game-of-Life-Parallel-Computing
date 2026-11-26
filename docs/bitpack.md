# Steps 2 and 3: bit-packed and SIMD

## Step 2: `bitpack` (src/bitpack/)

- Canonical layout, N/64 `uint64_t` words per row, two buffers swapped each generation,
  so `grid_bytes = N^2 / 8`.
- For each word, 8 neighbor words are built from the rows above, same and below,
  shifted by one bit; the carried-in bit comes from the adjacent word, wrapping at the row
  ends. A bit-sliced adder tree (4 full adders + 1 half adder) produces the count bits of
  all 64 cells at once, and `next = ~t1 & (t0 ^ c0) & (s0 | alive)`, which equals
  `(count == 3) | (alive & (count == 2))`. The derivation is the comment block at the top
  of `src/bitpack/bitpack_kernel.hpp`.
- **45 bitwise ops per 64-cell word** (18 for the shifted neighbor words, 20 for four full
  adders, 2 for the half adder, 5 for the rule): `gol::bitpack::kOpsPerWord`. With 16 bytes
  of DRAM traffic per word in the CLAUDE.md model, arithmetic intensity is
  **45 / 16 = 2.8 integer ops per byte** (integer ops, not FLOPs).
- Reusable entry points for steps 4 and 6: `gol::bitpack::step_rows(in, out, rows,
  words_per_row, r0, r1)` and `step_region(..., w0, w1)`, with the same ghost-row/ghost-word
  conventions as the byte kernel.
- Auto-vectorization is switched off for this one function
  (`__attribute__((optimize("no-tree-vectorize")))`, GCC only). Measured on the dev laptop at
  N=4096: GCC -O3 -march=native vectorizes 4 loops in the kernel on its own and reaches
  2.8e10 cells/s, nearly the hand-written AVX2 speed. With it off, step 2 measures the scalar
  bit-sliced kernel and step 3 shows what SIMD adds. Worth one sentence in the report.

## Step 3: `simd` (src/simd/)

- Same algorithm, `__m256i` with one 64-bit word per lane: 4 words (256 cells) per
  instruction, using per-lane `_mm256_slli_epi64` / `_mm256_srli_epi64` shifts.
- Carries across word boundaries come from unaligned loads at offset -1 and +1 word, so
  lane i of the offset vector is exactly the left/right neighbor word of lane i. That was
  chosen over `permute4x64` + blend: the permutes are 12 extra shuffle-port uops per output
  vector, all on one port, while the offset loads hit L1 (the same three rows are being
  streamed) and the core issues 2-3 loads per cycle. Full reasoning in `src/simd/simd.cpp`.
- Word 0, the last word and any tail words go through the scalar
  `gol::bitpack::next_word_at`, which handles the torus wrap; N=64 is entirely scalar.
- Registered only when `__builtin_cpu_supports("avx2")` is true at runtime; the AVX2 code is
  compiled with `__attribute__((target("avx2")))`, so the runtime check, not the build flags,
  decides whether it is used.
- `simd512` is not implemented: the dev laptop (i7-1255U, Alder Lake) has no `avx512f` in
  lscpu, and the prompt asks for it only when lscpu shows it. If the testbed has
  `avx512f`, it is the same code with `__m512i`, `_mm512_*` and 8 words per vector.

## Results (dev laptop, not the testbed)

i7-1255U under WSL2, single thread, seed 42, median of 5 runs of at least 1 s each, from
`bench/run.sh byte bitpack simd` and `bench/summarize.py`. Every impl's checksum was checked
against byte at each N before it was benchmarked.

| N | impl | cells/s | GiB/s | speedup vs byte |
|---:|---|---:|---:|---:|
| 1024 | byte | 1.63e9 | 3.04 | 1.00x |
| 1024 | bitpack | 1.26e10 | 2.94 | 7.7x |
| 1024 | simd | 2.32e10 | 5.41 | 14.2x |
| 4096 | byte | 1.65e9 | 3.08 | 1.00x |
| 4096 | bitpack | 1.20e10 | 2.79 | 7.3x |
| 4096 | simd | 2.64e10 | 6.14 | 16.0x |
| 16384 | byte | 1.52e9 | 2.83 | 1.00x |
| 16384 | bitpack | 1.18e10 | 2.76 | 7.8x |
| 16384 | simd | 2.37e10 | 5.51 | 15.6x |

Observations:

- bitpack runs at about 1.2e10 cells/s = 1.9e8 words/s = 8.4e9 bitwise ops/s on one core,
  flat across N whether or not the grid fits in the LLC: it is compute-bound. Its 2.8 GiB/s
  is far below STREAM Triad (18.1 GiB/s with all threads).
- simd is 1.9-2.2x faster than bitpack, not the 4x the vector width suggests. Likely
  reasons, not yet confirmed with hardware counters because `perf` is not available under
  WSL: the core has fewer vector ALU ports than scalar integer ALU ports, so the scalar
  kernel already executes several independent ops per cycle; each vector also needs 9
  loads (3 of them offset and partly line-splitting) and one store; and the
  scalar row-end words are a fixed cost per row. Run
  `perf stat -e cycles,instructions,uops_dispatched.port_0,... ./build/gol --impl simd ...` on
  the testbed to confirm.
- simd at N=16384 (grid 32 MiB, larger than the 12 MiB LLC) is about 10% slower than at
  N=4096 (2 MiB, fits in cache): at 5.5 GiB/s it is starting to feel memory latency and
  bandwidth for a single core, but it is not yet bandwidth-bound.
- Laptop turbo makes run-to-run variance around +/-10% (a quick run gave simd 3.0e10 at
  N=4096). Rerun on the testbed for the report.
