# Step 5: CUDA

`src/cuda/`, impls `cuda_naive` and `cuda_shared`, plus the bandwidth tool `build/gpu_copy`.
All of it builds only when CMake finds a CUDA compiler. The backends register only when
`cudaGetDeviceCount` reports a device at runtime.

> **Status: not built or measured yet.** This was written on a machine with an
> RTX 4070 Ti but no CUDA toolkit or host compiler, so nothing here has been compiled,
> ctest has not been run, and the default block size has not been chosen by measurement
> (see "Block size").

## Files

| File | What |
|---|---|
| `cuda_naive.cu` | `cuda_naive` kernel and its registration |
| `cuda_shared.cu` | `cuda_shared` kernel (4 block-size instantiations) and its registration |
| `byte_device_backend.cuh` | the backend around a kernel: device buffers, load/run/store, copy timing |
| `cuda_util.cuh` | error checks, owning `DeviceBuffer`, `EventTimer` (no kernels) |
| `gpu_copy.cu` | `build/gpu_copy`: device-to-device copy bandwidth, for docs/hardware.md |

## What the backends share (`byte_device_backend.cuh`)

- **Layout.** One byte per cell, the step 1 layout, so `grid_bytes = N^2`.
- **load.** Unpacks the canonical grid on the host, copies it to the device and zeroes the
  second buffer. The H2D copy is timed with cudaEvents and printed to stderr, for example
  `cuda_shared: H2D copy 256.0 MiB in 21.000 ms (11.90 GiB/s), not included in seconds`.
  The source buffer is pageable host memory, so this is below the PCIe peak; it is not part
  of `seconds` either way.
- **run.** The grid stays on the device for all generations. Each generation is one kernel
  launch on the default stream, followed by a swap of the two device pointers (never the
  data). Kernels on one stream run in order, so no synchronization is needed between
  generations. `seconds` is a cudaEvent pair around the launches, so it covers the GPU time
  of the generation loop only. Launch errors are checked once after the loop.
- **store.** D2H copy (timed, stderr), then `gol::pack` to the canonical layout for the
  checksum.

## cuda_naive

One thread per cell, blocks of 32x8 threads (a warp covers 32 consecutive cells of a row).
Each thread reads its 8 neighbors and itself straight from global memory and wraps both
coordinates with `% n`. The loads overlap heavily between neighboring threads, so most hit
L1/L2. The cost is mostly instructions: 9 load instructions per cell, plus integer modulo by
a runtime `n`. That is a multi-instruction sequence on the GPU, needed for 3 row indices and
3 column indices per cell after the compiler hoists what it can.

## cuda_shared

Each BX x BY block owns a BX x BY tile:

1. The block's threads cooperatively load the (BX+2) x (BY+2) tile plus its one-cell halo
   into shared memory, walking the tile in row-major order so consecutive threads read
   consecutive bytes. Only halo cells can fall outside the grid, and they wrap with a
   compare-and-add instead of a modulo.
2. `__syncthreads()`.
3. Each thread counts its 8 neighbors from shared memory and writes its one cell.

Each input byte is read from global memory once per block that needs it. Per output cell
that is (BX+2)(BY+2) / (BX*BY) reads, against up to 9 load instructions per cell in
cuda_naive.

### Block size

The prompt asks to sweep 16x16, 32x8, 32x16 and 32x32 and keep the best. All four are
template instantiations of the same kernel, and `GOL_CUDA_BLOCK=<BX>x<BY>` picks one at
runtime (stderr shows which):

```
bench/cuda_block_sweep.sh     # runs all four via bench/run.sh, prints the median table and the best per N
```

The winner goes into `kDefaultBlock` in `cuda_shared.cu`. **Until that sweep has run, the
default is 32x16, chosen by reasoning, not measurement.**

| Block | Threads | Halo overhead (BX+2)(BY+2)/(BX*BY) | Resident blocks/SM (CC 8.9: 1536 threads, 24 blocks) | Occupancy |
|---|---:|---:|---:|---:|
| 16x16 | 256 | 1.27 | 6 | 100% |
| 32x8 | 256 | 1.33 | 6 | 100% |
| 32x16 | 512 | 1.20 | 3 | 100% |
| 32x32 | 1024 | 1.13 | 1 | 67% |

- **16x16**: a warp's 32 output stores span two rows of 16 bytes, so each warp store
  touches two 32-byte sectors, each only half used. The tile loads are unaffected, since
  they walk the tile linearly whatever the block shape.
- **32x32** has the least halo, but only one block fits per SM (1024 of the 1536 thread
  slots). Every `__syncthreads()` then idles the whole SM until the slowest warp arrives,
  because no second block is resident to fill the gap.
- **32x16** keeps full 32-byte rows per warp, full occupancy and three blocks per SM to
  overlap one block's load phase with another's compute, at 1.20 halo overhead.
- Expected ranking: 32x16 >= 32x8 > 16x16 > 32x32. Shared memory is not a limit: the
  largest tile is 34 x 34 = 1156 bytes.

## Bandwidth ceiling: `build/gpu_copy`

`bench/gpu_copy.sh` runs `build/gpu_copy`, the GPU counterpart of STREAM. For buffer sizes
16, 64, 256 and 1024 MiB it times, with cudaEvents, the best of 20 runs after a warm-up:

- `vec16`: a grid-stride copy kernel with one 16-byte `uint4` per thread per iteration. This
  is the ceiling.
- `byte`: the same kernel with one `uint8_t` per thread. That is the access size the
  byte-grid kernels use, which shows how much the narrow accesses cost by themselves.
- `memcpy`: `cudaMemcpy` DeviceToDevice, for reference.

Bandwidth is `2 * bytes / s` (read + write), the same model as `gib_per_s`. The last line,
`GPU copy kernel: X GiB/s (vec16, 1024 MiB per buffer)`, is what `bench/hwinfo.sh` writes
into docs/hardware.md and what `bench/summarize.py` divides `cuda_*` GiB/s by.

For the RTX 4070 Ti, NVIDIA's spec is 504 GB/s (192-bit GDDR6X at 21 Gbps), which is
504 x 0.931 = **469.4 GiB/s** theoretical. Pass it to hwinfo as `GPU_SPEC_GBS=504`. The card
has a 48 MB L2, so the 16 MiB copy (32 MiB moved) mostly measures L2.

## What to expect

- **The DRAM-resident case is N=16384:** 2 x 256 MiB of grid. Against a copy ceiling of
  roughly 400+ GiB/s, the byte layout's 2 bytes moved per cell per generation caps the rate
  near 2e11 cells/s. Reaching that needs the kernel to be bandwidth-bound rather than
  instruction-bound. cuda_naive probably is not, because of the modulos and 9 loads per
  cell. cuda_shared should get closer.
- **N=4096** (2 x 16 MiB) fits in the 48 MB L2, so its GiB/s can exceed the DRAM copy
  ceiling. Read that as L2 bandwidth, not an error.
- **N=1024** is 1024 blocks of 32x16 per generation, a few microseconds of work, so launch
  overhead (a few us per kernel) is a visible part of each generation.
- The byte layout spends 8x the bandwidth of a bit-packed one. A `cuda_bitpack` kernel (one
  thread per 64-bit word, the step 2 adder tree) is the stretch goal and is **not**
  implemented.

## Results

Not measured yet. On a Linux machine with an NVIDIA GPU and the CUDA toolkit:

```
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j && ctest --test-dir build --output-on-failure
bench/gpu_copy.sh                       # copy-kernel ceiling -> bench/results/gpucopy_<host>.txt
GPU_SPEC_GBS=504 bench/hwinfo.sh        # docs/hardware.md with both GPU rows
bench/cuda_block_sweep.sh               # then set kDefaultBlock, rebuild
bench/run.sh cuda_naive cuda_shared
bench/summarize.py                      # cells/s, GiB/s, speedup vs byte, % of GPU copy kernel
```

| N | impl | cells/s | GiB/s | speedup vs byte | % of GPU copy kernel |
|---:|---|---:|---:|---:|---:|
| 1024 | cuda_naive | | | | |
| 1024 | cuda_shared | | | | |
| 4096 | cuda_naive | | | | |
| 4096 | cuda_shared | | | | |
| 16384 | cuda_naive | | | | |
| 16384 | cuda_shared | | | | |
