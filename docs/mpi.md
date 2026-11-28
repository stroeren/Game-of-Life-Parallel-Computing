# Step 6: MPI halo exchange (`gol_mpi`)

```
mpirun -n P build/gol_mpi --n N --gens K --seed S [--decomp strip|block]
                          [--mode blocking|nonblocking] [--kernel byte|bitpack] [--csv on|off] [--check]
```

All ranks are processes (one thread each; `--threads` is ignored). stdout is the CSV row of
CLAUDE.md (`impl` = `mpi-<decomp>-<mode>-<kernel>`, `ranks` = P, `seconds` = the generation
loop only, max over ranks). stderr carries one `gol_mpi stats:` line with the compute/comm
split, the measured halo volume and the scatter/gather times (the setup that is not in
`seconds`). `--csv off` suppresses the CSV row (the halo tests use it so that mpirun cannot interleave
stdout into the stats line). `--check` re-runs the serial byte backend on rank 0 and fails (exit 3) on a
checksum mismatch.

gol_mpi links the existing kernels (`gol::byte::step_region`, `gol::bitpack::step_region`) and
calls them over a row/column range; nothing in `gol` depends on MPI.

## Data layout

Rank 0 builds the canonical grid, converts it to the kernel's own layout (bytes, or the
canonical words as they are) and distributes it with `MPI_Scatterv`; the final grid comes back
with `MPI_Gatherv` and is checksummed on rank 0 in the canonical layout. Block mode packs each
rank's block contiguously on rank 0 first (the strip case is already contiguous), so both
decompositions use the same two collectives. Splits are balanced (`N/p` rows, the first
`N mod p` ranks get one more), so any `p <= N` works.

| | local buffer (elements) | ghosts |
|---|---|---|
| strip | `(h+2) x W` | one row above and below; columns wrap inside the kernel (`c0=0, c1=W`) |
| block | `(h+2) x (w+2)` | rows and columns, corners included |

`W` is `N` bytes or `N/64` words. Updating `[1,h+1)` (strip) or `[1,h+1) x [1,w+1)` (block) never
takes the kernel's own wrap, so the ghosts supply every neighbor.

## Halo exchange

**Strip.** Periodic ring: rank 0's upper neighbor is `p-1`. Each generation:
`MPI_Sendrecv(first row -> up, last-row ghost <- down)`, then the mirror image. Nonblocking:
`Irecv`/`Isend` for both ghost rows, update rows `2..h-1` (they need no ghost), `Waitall`,
update rows `1` and `h`.

**Block.** `MPI_Dims_create` + `MPI_Cart_create(periods = {1,1}, reorder = 0)`, neighbors from
`MPI_Cart_shift`. Rows are contiguous; a column is `MPI_Type_vector(h+2, 1, pitch, T)`.

### Corners: two-phase (rows first, then columns including the ghost corners)

Phase 1 swaps the first and last owned rows (interior columns only). Phase 2 swaps the first and
last owned *columns over all h+2 local rows*, i.e. including the two ghost-row cells that phase 1
just filled. Those cells are the neighbor's diagonal cells, so the corners arrive with the
column message and no diagonal message is needed.

Why not eight messages (four sides + four diagonals):

* 4 messages per rank per generation instead of 8, so half the latency terms; the 1-cell corner
  messages would be pure latency.
* Only the 4 Cartesian neighbors are needed; with diagonals you need 4 more ranks, and on tiny
  process grids (2x1, 2x2) diagonal neighbors coincide with side neighbors or with yourself,
  which needs special cases.
* Cost: the phases are dependent (phase 2 reads phase 1's result), so they serialize, and the
  column messages are `h+2` instead of `h` long (2 extra elements per message, +2 per column
  vs the model). Nonblocking mode hides both phases behind computation (below).

Message tags name the direction of travel (up/down/left/right), because with `p = 2` the upper
and lower neighbor are the same rank, and in a 1-wide process grid left/right are the rank itself.

### Nonblocking schedule (block)

```
post row Irecv/Isend      -> update deep interior (rows 2..h-1, cols 2..w-1; needs no ghost)
Waitall(rows)
post column Irecv/Isend   -> update first/last row except their end cells (needs ghost rows only)
Waitall(columns)          -> update first/last column
```

The regions are disjoint and cover the block for every `h, w >= 1`.

## Bit-packed kernel (`--kernel bitpack`)

Ghost rows are packed (`N/64` words per row, 8 bytes per 64 cells). Block boundaries are in
*words*, so every block width is a multiple of 64 cells; if `N/64` is not divisible by the number
of process columns, blocks differ by one word (still a multiple of 64). `bitpack::step_region`
reads bit 63 of the left ghost word and bit 0 of the right ghost word, so the corner words and
`west`/`east` carries are right.

**Cost of word-aligned block columns:** a ghost *column* has to be a whole word column, 64 cells
wide, although only 1 bit of it is used. Column traffic is therefore 64x the model's cell count
(see the table). For bit-packed grids strips are the cheap decomposition; sending only the edge
bit (packing the ghost column into `h/64` words) would fix it but is not implemented.

## What is timed

Every generation is timed with `MPI_Wtime` into `(compute, comm)`:

* blocking: `comm` = the exchange, `compute` = the kernel call;
* nonblocking: `comm` = posting + time blocked in `Waitall` (the part overlap did **not** hide),
  `compute` = all kernel calls.

The per-generation records are reduced with `MPI_Reduce(MPI_MAX)` after the loop (no
communication inside the timed region), then summed over generations: `compute_s` and `comm_s`
are *sums of per-generation maxima over ranks*, so `compute_s + comm_s >= seconds`. `seconds` is
the max over ranks of the loop wall time, started after an `MPI_Barrier`.

## Halo volume

Counted at the send sites: elements this rank posts to *other* ranks per generation (messages to
itself, which occur when a process-grid dimension is 1, cost no network and are excluded),
averaged over ranks. This is what the code asks MPI to send, not wire bytes (no protocol
headers). `tests/CMakeLists.txt` asserts these counts at N = 4096 against hand-derived geometry
for p = 2, 4, 8, 16.

Model from the guide: strips `2N` cells per rank per generation, blocks `4N/sqrt(p)`.
N = 4096, byte kernel (cells = bytes):

| decomp | p | grid | measured | model | measured / model | why |
|---|--:|---|--:|--:|--:|---|
| strip | 2, 4, 8, 16 | px1 | 8192 | 8192 | 1.00 | two full rows, independent of p |
| block | 2 | 2x1 | 8192 | 11585 | 0.71 | no column neighbors, rows only; model assumes a square grid |
| block | 4 | 2x2 | 8196 | 8192 | 1.00 | 2 rows of 2048 + 2 columns of 2050 (corners ride along) |
| block | 8 | 4x2 | 6148 | 5793 | 1.06 | 4x2 is not square: rows 2*2048 + columns 2*1026 |
| block | 16 | 4x4 | 4100 | 4096 | 1.00 | 2 rows of 1024 + 2 columns of 1026 |

So the model holds for square process grids (the `+2` per column message is the corners); it
over-predicts at p = 2 (no column traffic) and under-predicts at p = 8 (rectangular grid). Strip
volume does not shrink with p, block volume falls as `1/sqrt(p)`: blocks win asymptotically
(at p = 16: 4100 vs 8192 cells), strips send fewer, larger messages (2 vs 4).

Bit-packed (N = 4096, cells = 64 per word): strips send 128 words = 8192 cells = 1 KiB, exactly
the model. Blocks send (words) 128 at p = 2, 4164 at p = 4, 2116 at p = 8, 2084 at p = 16, i.e.
33x / 23x / 33x the model in cell-equivalents, because of the whole-word ghost columns described
above (bytes: 33 KB at p = 4 vs 8 KB for byte-kernel blocks).

## Tests

`ctest` runs `gol_mpi --check` (checksum equals the serial byte backend after 100 generations,
seed 42) for strip and block x blocking and nonblocking x byte and bitpack at p = 1, 2, 4
(N = 256 and 1024), plus uneven splits (p = 3, 6; bitpack blocks of unequal word width at
p = 12; N = 64 on 16 ranks) and the halo-volume assertions above. Deliberately breaking the
column exchange or the corner cells makes the block tests fail.

mpirun refuses to start more ranks than slots. If it does, CMake detects that at configure time
and adds `--oversubscribe` **to the tests only** (`-DGOL_MPI_OVERSUBSCRIBE=ON|OFF|AUTO`). On a
machine with enough cores nothing is added. `bench/mpi_sweep.sh` never uses it.

## Strong-scaling sweep

`bench/mpi_sweep.sh` (see its header for environment overrides) runs p = 1, 2, 4, 8, 16 (only
`p <= nproc`) at N = 4096 and 16384 for strip/block x blocking/nonblocking, median of 5 runs,
gens chosen for >= 1 s, first run of each configuration with `--check`. It writes the raw rows to
`bench/results/mpi_<host>.csv` and the table (speedup vs the serial kernel, efficiency, speedup
vs p = 1, compute/comm split, measured vs modelled halo) to `bench/results/mpi_<host>.md`.

Scaling results on the testbed: TODO(run `bench/mpi_sweep.sh` on the multicore testbed and
paste `bench/results/mpi_<host>.md` here). The development container used for the implementation
has one core, so only p = 1 was timed there (MPI overhead at p = 1: 1-8% vs the serial byte
backend); it says nothing about scaling.
