// Step 4: OpenMP over blocks of rows. Three backends, one per existing kernel, none of
// which is copied here: each generation is a `#pragma omp for` over the rows, and every
// iteration calls the kernel's row-range entry point (gol::byte::step_rows,
// gol::bitpack::step_rows, gol::simd::step_rows_avx2) for its row. The kernels read only
// `in` and write only their own rows of `out`, so rows are independent within a
// generation and the only synchronization needed is the barrier between generations.
//
//   omp_byte     byte kernel     (1 byte per cell)
//   omp_bitpack  bitpack kernel  (64 cells per uint64_t, scalar adder tree)
//   omp_simd     simd kernel     (AVX2, 4 words per instruction); registered only with AVX2
//
// Schedule: the loop uses schedule(runtime), so bench/omp_sweep.sh can pick static,
// dynamic,<chunk> or guided through OMP_SCHEDULE without rebuilding. With static, thread t
// gets one contiguous block of about N/T rows; with dynamic,c threads grab blocks of c
// rows from a shared counter; guided starts with large blocks and shrinks them. When
// OMP_SCHEDULE is not set, the backend uses static. (libgomp's own default for
// schedule(runtime) is dynamic,1, which is the worst case here; see docs/omp.md.)
//
// One parallel region covers all generations: threads are created once per run() and
// each generation costs one barrier (the implicit one at the end of the `omp for`), not a
// fork/join. Each thread keeps its own copy of the in/out pointers and swaps them after
// the barrier, so no `single` is needed.
//
// First touch: the two buffers are allocated without initialization (aligned_alloc leaves
// the pages untouched) and then filled by a parallel loop with the same schedule as the
// generation loop, so on a multi-socket / multi-NUMA-node machine each page is placed on
// the node of the thread that will later compute those rows. GOL_OMP_FIRST_TOUCH=0 fills
// them from the main thread instead, to measure the difference (docs/omp.md).
#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

#include "common/grid.hpp"
#include "common/registry.hpp"
#include "common/timer.hpp"
#include "src/bitpack/bitpack_kernel.hpp"
#include "src/byte/byte_kernel.hpp"
#include "src/simd/simd_kernel.hpp"

namespace gol::omp {

namespace {

// ---------------------------------------------------------------------------------------
// Kernel adapters: cell type, row pitch, one call into the existing kernel for rows
// [r0, r1), and conversion of one canonical row in and the whole grid out.
// ---------------------------------------------------------------------------------------

struct ByteKernel {
    using Cell = uint8_t;
    static std::size_t pitch(int n) { return static_cast<std::size_t>(n); }
    static void step(const Cell* in, Cell* out, int n, int r0, int r1) {
        gol::byte::step_rows(in, out, n, n, r0, r1);
    }
    static void load_row(const Grid& g, int r, Cell* dst) {
        const uint64_t* src = g.row(r);
        for (int c = 0; c < g.n; ++c) dst[c] = static_cast<Cell>((src[c >> 6] >> (c & 63)) & 1u);
    }
    static Grid store(const Cell* cells, int n) { return pack(cells, n); }
};

struct BitpackKernel {
    using Cell = uint64_t;
    static std::size_t pitch(int n) { return static_cast<std::size_t>(n / 64); }
    static void step(const Cell* in, Cell* out, int n, int r0, int r1) {
        gol::bitpack::step_rows(in, out, n, n / 64, r0, r1);
    }
    static void load_row(const Grid& g, int r, Cell* dst) {
        std::memcpy(dst, g.row(r), pitch(g.n) * sizeof(Cell));
    }
    static Grid store(const Cell* words, int n) {
        Grid g(n);
        std::memcpy(g.words.data(), words, g.words.size() * sizeof(Cell));
        return g;
    }
};

// Same layout as bitpack, AVX2 kernel.
struct SimdKernel : BitpackKernel {
    static void step(const Cell* in, Cell* out, int n, int r0, int r1) {
        gol::simd::step_rows_avx2(in, out, n, n / 64, r0, r1);
    }
};

// Grid buffers: uninitialized (so the first write decides page placement, see above) and
// 64-byte aligned. A row is a multiple of 64 bytes (byte: N bytes; bit-packed: N/8 bytes
// for N >= 512), so with an aligned base every row starts on its own cache line and two
// threads writing neighboring rows never write the same line (no false sharing at block
// or chunk boundaries). Large new[]/malloc allocations are only 16-byte aligned.
struct FreeDeleter {
    void operator()(void* p) const { std::free(p); }
};
template <class T>
using Buffer = std::unique_ptr<T[], FreeDeleter>;

template <class T>
Buffer<T> allocate_uninitialized(std::size_t count) {
    const std::size_t bytes = (count * sizeof(T) + 63) / 64 * 64;  // aligned_alloc wants a multiple
    void* p = std::aligned_alloc(64, bytes);
    if (p == nullptr) throw std::bad_alloc();
    return Buffer<T>(static_cast<T*>(p));
}

// schedule(runtime) reads the run-sched-var ICV, which the OpenMP runtime initializes
// from OMP_SCHEDULE at startup. Without OMP_SCHEDULE, set it to static.
void default_to_static_schedule() {
    if (std::getenv("OMP_SCHEDULE") == nullptr) omp_set_schedule(omp_sched_static, 0);
}

bool first_touch_enabled() {
    const char* v = std::getenv("GOL_OMP_FIRST_TOUCH");
    return v == nullptr || std::strcmp(v, "0") != 0;
}

template <class K>
class OmpBackend : public Backend {
    using Cell = typename K::Cell;

public:
    void load(const Grid& initial, const RunConfig& cfg) override {
        n_ = initial.n;
        threads_ = cfg.threads > 0 ? cfg.threads : omp_get_max_threads();
        default_to_static_schedule();

        const int n = n_;
        const std::size_t pitch = K::pitch(n);
        cells_ = pitch * static_cast<std::size_t>(n);
        cur_ = allocate_uninitialized<Cell>(cells_);  // first touch happens below
        next_ = allocate_uninitialized<Cell>(cells_);
        Cell* cur = cur_.get();
        Cell* next = next_.get();

        // Same loop shape and schedule as run(), so each row's pages are first written by
        // the thread that computes that row (under static, the same thread every time).
#pragma omp parallel for schedule(runtime) num_threads(threads_) if (first_touch_enabled())
        for (int r = 0; r < n; ++r) {
            const std::size_t off = static_cast<std::size_t>(r) * pitch;
            K::load_row(initial, r, cur + off);
            std::memset(next + off, 0, pitch * sizeof(Cell));
        }
    }

    double run(int gens) override {
        default_to_static_schedule();
        const int n = n_;
        Cell* a = cur_.get();
        Cell* b = next_.get();

        Timer t;
#pragma omp parallel num_threads(threads_) firstprivate(a, b)
        {
            for (int g = 0; g < gens; ++g) {
#pragma omp for schedule(runtime)
                for (int r = 0; r < n; ++r) K::step(a, b, n, r, r + 1);
                // Implicit barrier above: every row of generation g is written before any
                // thread starts reading it as generation g + 1.
                std::swap(a, b);
            }
        }
        const double seconds = t.seconds();

        if (gens % 2 != 0) std::swap(cur_, next_);  // the result is in next_ after an odd count
        return seconds;
    }

    Grid store() override { return K::store(cur_.get(), n_); }

    std::size_t grid_bytes() const override { return cells_ * sizeof(Cell); }

    int threads() const override { return threads_; }

private:
    int n_ = 0;
    int threads_ = 1;
    std::size_t cells_ = 0;
    Buffer<Cell> cur_, next_;
};

}  // namespace

}  // namespace gol::omp

GOL_REGISTER_BACKEND("omp_byte", gol::omp::OmpBackend<gol::omp::ByteKernel>);
GOL_REGISTER_BACKEND("omp_bitpack", gol::omp::OmpBackend<gol::omp::BitpackKernel>);
GOL_REGISTER_BACKEND_IF("omp_simd", gol::omp::OmpBackend<gol::omp::SimdKernel>,
                        gol::simd::avx2_supported);
