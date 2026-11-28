// Step 6: MPI halo exchange. Binary: gol_mpi.
//
//   mpirun -n P gol_mpi --n <N> --gens <K> --seed <S> [--decomp strip|block]
//                       [--mode blocking|nonblocking] [--kernel byte|bitpack] [--check]
//
// stdout: the CSV row of CLAUDE.md "Driver" (printed by rank 0, impl = mpi-<decomp>-<mode>-<kernel>).
// stderr: one "gol_mpi stats:" line with the compute/communication split and the measured
//         halo volume (parsed by bench/mpi_sweep.sh). See docs/mpi.md for the design notes.
//
// Layout of the local buffer (rows x pitch, in elements: 1 byte per cell for --kernel byte,
// one uint64_t = 64 cells for --kernel bitpack):
//
//   strip:   (h + 2) x W       one ghost row above and below, full width, no ghost columns
//                              (the kernel wraps the columns itself, c0 = 0 and c1 = W)
//   block:   (h + 2) x (w + 2) ghost rows and ghost columns, corners included
//
// The cell update is the existing kernel, called over a row/column range:
//   gol::byte::step_region / gol::bitpack::step_region.
#include <mpi.h>

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/cli.hpp"
#include "common/grid.hpp"
#include "common/registry.hpp"
#include "common/report.hpp"
#include "src/bitpack/bitpack_kernel.hpp"
#include "src/byte/byte_kernel.hpp"

namespace {

// --------------------------------------------------------------------------------------
// Kernel policies: element type, MPI type, conversion to/from the canonical grid, step.
// --------------------------------------------------------------------------------------
struct ByteKernel {
    using T = uint8_t;
    static constexpr int kCellsPerElem = 1;
    static constexpr const char* kName = "byte";
    static MPI_Datatype mpi_type() { return MPI_UINT8_T; }
    static std::vector<T> from_canonical(const gol::Grid& g) { return gol::unpack(g); }
    static gol::Grid to_canonical(const std::vector<T>& v, int n) { return gol::pack(v.data(), n); }
    static void step(const T* in, T* out, int rows, int cols, int r0, int r1, int c0, int c1) {
        gol::byte::step_region(in, out, rows, cols, r0, r1, c0, c1);
    }
};

struct BitpackKernel {
    using T = uint64_t;
    static constexpr int kCellsPerElem = 64;
    static constexpr const char* kName = "bitpack";
    static MPI_Datatype mpi_type() { return MPI_UINT64_T; }
    static std::vector<T> from_canonical(const gol::Grid& g) { return g.words; }
    static gol::Grid to_canonical(const std::vector<T>& v, int n) {
        gol::Grid g(n);
        g.words = v;
        return g;
    }
    static void step(const T* in, T* out, int rows, int cols, int r0, int r1, int c0, int c1) {
        gol::bitpack::step_region(in, out, rows, cols, r0, r1, c0, c1);
    }
};

// Message tags name the direction the data travels, so that up == down (p = 2) and
// left == right == self (a 1-wide process grid) never mix messages up.
constexpr int kTagUp = 0;     // sent to the upper neighbor
constexpr int kTagDown = 1;   // sent to the lower neighbor
constexpr int kTagLeft = 2;   // sent to the left neighbor
constexpr int kTagRight = 3;  // sent to the right neighbor

// Balanced split of `total` items over `parts`: the first total % parts parts get one more.
void split(int total, int parts, int idx, int& start, int& size) {
    const int base = total / parts, rem = total % parts;
    size = base + (idx < rem ? 1 : 0);
    start = idx * base + std::min(idx, rem);
}

struct Config {
    int n = 0;
    uint64_t seed = 42;
    bool block = false;
    bool nonblocking = false;
};

struct Extent {
    int r0, h, c0, w;  // global row/column offset and size, in elements
};

struct RunStats {  // valid on rank 0 only
    double loop_s = 0;        // max over ranks of the generation loop
    double compute_s = 0;     // sum over generations of the max over ranks of compute time
    double comm_s = 0;        // same for communication
    double halo_avg_elems = 0;  // elements sent per rank per generation, averaged over ranks
    double halo_max_elems = 0;
};

// --------------------------------------------------------------------------------------
template <class K>
class Life {
public:
    using T = typename K::T;

    Life(const Config& cfg, MPI_Comm world) : cfg_(cfg), n_(cfg.n) {
        int p = 0;
        MPI_Comm_size(world, &p);
        p_ = p;
        cols_el_ = n_ / K::kCellsPerElem;  // elements per global row

        if (cfg.block) {
            // 2-D process grid, periodic in both directions.
            dims_[0] = dims_[1] = 0;
            MPI_Dims_create(p_, 2, dims_);
            int periods[2] = {1, 1};
            MPI_Cart_create(world, 2, dims_, periods, /*reorder=*/0, &comm_);
            MPI_Comm_rank(comm_, &rank_);
            MPI_Cart_shift(comm_, 0, 1, &up_, &down_);     // source = upper, dest = lower
            MPI_Cart_shift(comm_, 1, 1, &left_, &right_);  // source = left,  dest = right
            gc_ = 1;
        } else {
            // 1-D strips: p x 1 process grid; the column wrap stays inside the kernel.
            dims_[0] = p_;
            dims_[1] = 1;
            MPI_Comm_dup(world, &comm_);
            MPI_Comm_rank(comm_, &rank_);
            up_ = (rank_ + p_ - 1) % p_;  // periodic: rank 0's upper neighbor is p - 1
            down_ = (rank_ + 1) % p_;
            left_ = right_ = rank_;
            gc_ = 0;
        }
        if (dims_[0] > n_ || dims_[1] > cols_el_)
            throw std::invalid_argument("grid too small for " + std::to_string(p_) + " ranks (" +
                                        std::to_string(dims_[0]) + "x" + std::to_string(dims_[1]) +
                                        " process grid, N=" + std::to_string(n_) + ")");

        const Extent e = extent_of(rank_);
        r0_ = e.r0;
        c0_ = e.c0;
        h_ = e.h;
        w_ = e.w;
        pitch_ = w_ + 2 * gc_;
        cur_.assign(static_cast<std::size_t>(h_ + 2) * pitch_, 0);
        nxt_.assign(cur_.size(), 0);

        if (cfg.block) {
            // One column of the local buffer including both ghost rows: h + 2 elements,
            // stride = row pitch.
            MPI_Type_vector(h_ + 2, 1, pitch_, K::mpi_type(), &coltype_);
            MPI_Type_commit(&coltype_);
        }
    }

    ~Life() {
        if (cfg_.block) MPI_Type_free(&coltype_);
        MPI_Comm_free(&comm_);
    }
    Life(const Life&) = delete;
    Life& operator=(const Life&) = delete;

    int rank() const { return rank_; }
    int size() const { return p_; }
    std::string dims_string() const { return std::to_string(dims_[0]) + "x" + std::to_string(dims_[1]); }
    int dims0() const { return dims_[0]; }
    int dims1() const { return dims_[1]; }

    // Scatter. `initial` is read on rank 0 only. Not timed by the caller's loop timer.
    void load(const gol::Grid* initial) {
        const double t0 = MPI_Wtime();
        std::vector<int> counts, displs;
        layout(counts, displs);

        std::vector<T> global;  // rank 0: whole grid in elements, row-major
        if (rank_ == 0) global = K::from_canonical(*initial);

        std::vector<T> packed;  // rank 0: blocks one after another in rank order
        const T* sendbuf = nullptr;
        if (rank_ == 0) {
            if (dims_[1] == 1) {
                sendbuf = global.data();  // full-width row blocks are already contiguous
            } else {
                packed.resize(global.size());
                for (int r = 0; r < p_; ++r) {
                    const Extent e = extent_of(r);
                    T* dst = packed.data() + displs[r];
                    for (int i = 0; i < e.h; ++i)
                        std::memcpy(dst + static_cast<std::size_t>(i) * e.w,
                                    global.data() + static_cast<std::size_t>(e.r0 + i) * cols_el_ + e.c0,
                                    static_cast<std::size_t>(e.w) * sizeof(T));
                }
                std::vector<T>().swap(global);
                sendbuf = packed.data();
            }
        }

        std::vector<T> mine(static_cast<std::size_t>(h_) * w_);
        MPI_Scatterv(sendbuf, counts.data(), displs.data(), K::mpi_type(), mine.data(), h_ * w_,
                     K::mpi_type(), 0, comm_);
        for (int i = 0; i < h_; ++i)
            std::memcpy(at(1 + i, gc_), mine.data() + static_cast<std::size_t>(i) * w_,
                        static_cast<std::size_t>(w_) * sizeof(T));
        scatter_s_ = MPI_Wtime() - t0;
    }

    // Advance `gens` generations. Collective; the returned stats are valid on rank 0.
    RunStats run(int gens) {
        // Per-generation (compute, communication) seconds on this rank, allocated up front.
        std::vector<double> rec(static_cast<std::size_t>(gens) * 2, 0.0);
        sent_elems_ = 0;

        MPI_Barrier(comm_);
        const double t0 = MPI_Wtime();
        for (int g = 0; g < gens; ++g) {
            double comp = 0, comm = 0;
            if (!cfg_.nonblocking) gen_blocking(comp, comm);
            else if (!cfg_.block) gen_nonblocking_strip(comp, comm);
            else gen_nonblocking_block(comp, comm);
            cur_.swap(nxt_);
            rec[2 * static_cast<std::size_t>(g)] = comp;
            rec[2 * static_cast<std::size_t>(g) + 1] = comm;
        }
        const double local_loop = MPI_Wtime() - t0;

        RunStats s;
        double loop_max = 0;
        MPI_Reduce(&local_loop, &loop_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
        // Per-generation maximum over ranks, for compute and communication separately.
        std::vector<double> gmax(rec.size(), 0.0);
        if (!rec.empty())
            MPI_Reduce(rec.data(), gmax.data(), static_cast<int>(rec.size()), MPI_DOUBLE, MPI_MAX, 0, comm_);
        const double sent = static_cast<double>(sent_elems_);
        double sent_sum = 0, sent_max = 0;
        MPI_Reduce(&sent, &sent_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);
        MPI_Reduce(&sent, &sent_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
        if (rank_ == 0) {
            s.loop_s = loop_max;
            for (int g = 0; g < gens; ++g) {
                s.compute_s += gmax[2 * static_cast<std::size_t>(g)];
                s.comm_s += gmax[2 * static_cast<std::size_t>(g) + 1];
            }
            if (gens > 0) {
                s.halo_avg_elems = sent_sum / (static_cast<double>(p_) * gens);
                s.halo_max_elems = sent_max / gens;
            }
        }
        return s;
    }

    // Gather to rank 0 and return the canonical grid there (empty Grid elsewhere).
    gol::Grid store() {
        const double t0 = MPI_Wtime();
        std::vector<int> counts, displs;
        layout(counts, displs);

        std::vector<T> mine(static_cast<std::size_t>(h_) * w_);
        for (int i = 0; i < h_; ++i)
            std::memcpy(mine.data() + static_cast<std::size_t>(i) * w_, at(1 + i, gc_),
                        static_cast<std::size_t>(w_) * sizeof(T));

        std::vector<T> recv;
        if (rank_ == 0) recv.resize(static_cast<std::size_t>(n_) * cols_el_);
        MPI_Gatherv(mine.data(), h_ * w_, K::mpi_type(), recv.data(), counts.data(), displs.data(),
                    K::mpi_type(), 0, comm_);
        gather_s_ = MPI_Wtime() - t0;

        gol::Grid out;
        if (rank_ == 0) {
            if (dims_[1] == 1) {
                out = K::to_canonical(recv, n_);
            } else {
                std::vector<T> global(recv.size());
                for (int r = 0; r < p_; ++r) {
                    const Extent e = extent_of(r);
                    const T* src = recv.data() + displs[r];
                    for (int i = 0; i < e.h; ++i)
                        std::memcpy(global.data() + static_cast<std::size_t>(e.r0 + i) * cols_el_ + e.c0,
                                    src + static_cast<std::size_t>(i) * e.w,
                                    static_cast<std::size_t>(e.w) * sizeof(T));
                }
                out = K::to_canonical(global, n_);
            }
        }
        return out;
    }

    void release() {
        std::vector<T>().swap(cur_);
        std::vector<T>().swap(nxt_);
    }

    // max over ranks of the scatter / gather wall time (collective; valid on rank 0).
    void setup_times(double& scatter_max, double& gather_max) const {
        MPI_Reduce(&scatter_s_, &scatter_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
        MPI_Reduce(&gather_s_, &gather_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm_);
    }

private:
    // Block of rank r in global element coordinates.
    Extent extent_of(int r) const {
        int c[2] = {r, 0};
        if (cfg_.block) MPI_Cart_coords(comm_, r, 2, c);
        Extent e{};
        split(n_, dims_[0], c[0], e.r0, e.h);
        split(cols_el_, dims_[1], c[1], e.c0, e.w);
        return e;
    }

    // Element counts and displacements of every rank's block in the packed (rank-ordered)
    // buffer used by Scatterv / Gatherv.
    void layout(std::vector<int>& counts, std::vector<int>& displs) const {
        const long long total = static_cast<long long>(n_) * cols_el_;
        if (total > 2147483647LL)
            throw std::invalid_argument("grid has too many elements for int counts in MPI_Scatterv");
        counts.assign(p_, 0);
        displs.assign(p_, 0);
        int off = 0;
        for (int r = 0; r < p_; ++r) {
            const Extent e = extent_of(r);
            counts[r] = e.h * e.w;
            displs[r] = off;
            off += counts[r];
        }
    }

    T* at(int r, int c) { return cur_.data() + static_cast<std::size_t>(r) * pitch_ + c; }

    // Cells this rank hands to *other* ranks. Messages to itself (a 1-wide process grid
    // wraps onto the same rank) cost no network traffic and are not counted.
    void note_sent(int peer, int elems) {
        if (peer != rank_) sent_elems_ += static_cast<uint64_t>(elems);
    }

    void upd(int r0, int r1, int c0, int c1) {
        if (r0 >= r1 || c0 >= c1) return;
        K::step(cur_.data(), nxt_.data(), h_ + 2, pitch_, r0, r1, c0, c1);
    }

    // ---- ghost rows: first row -> upper neighbor's lower ghost, last row -> lower neighbor's
    // upper ghost. Rows carry only the owned columns (no corners).
    void rows_blocking() {
        const MPI_Datatype t = K::mpi_type();
        MPI_Sendrecv(at(1, gc_), w_, t, up_, kTagUp, at(h_ + 1, gc_), w_, t, down_, kTagUp, comm_,
                     MPI_STATUS_IGNORE);
        MPI_Sendrecv(at(h_, gc_), w_, t, down_, kTagDown, at(0, gc_), w_, t, up_, kTagDown, comm_,
                     MPI_STATUS_IGNORE);
        note_sent(up_, w_);
        note_sent(down_, w_);
    }

    // ---- ghost columns, second phase of the two-phase scheme: whole columns including
    // the two ghost-row cells, so the corners arrive with them (see docs/mpi.md).
    void cols_blocking() {
        MPI_Sendrecv(at(0, 1), 1, coltype_, left_, kTagLeft, at(0, w_ + 1), 1, coltype_, right_,
                     kTagLeft, comm_, MPI_STATUS_IGNORE);
        MPI_Sendrecv(at(0, w_), 1, coltype_, right_, kTagRight, at(0, 0), 1, coltype_, left_,
                     kTagRight, comm_, MPI_STATUS_IGNORE);
        note_sent(left_, h_ + 2);
        note_sent(right_, h_ + 2);
    }

    void post_rows(MPI_Request* rq) {
        const MPI_Datatype t = K::mpi_type();
        MPI_Irecv(at(0, gc_), w_, t, up_, kTagDown, comm_, &rq[0]);
        MPI_Irecv(at(h_ + 1, gc_), w_, t, down_, kTagUp, comm_, &rq[1]);
        MPI_Isend(at(1, gc_), w_, t, up_, kTagUp, comm_, &rq[2]);
        MPI_Isend(at(h_, gc_), w_, t, down_, kTagDown, comm_, &rq[3]);
        note_sent(up_, w_);
        note_sent(down_, w_);
    }

    void post_cols(MPI_Request* rq) {
        MPI_Irecv(at(0, 0), 1, coltype_, left_, kTagRight, comm_, &rq[0]);
        MPI_Irecv(at(0, w_ + 1), 1, coltype_, right_, kTagLeft, comm_, &rq[1]);
        MPI_Isend(at(0, 1), 1, coltype_, left_, kTagLeft, comm_, &rq[2]);
        MPI_Isend(at(0, w_), 1, coltype_, right_, kTagRight, comm_, &rq[3]);
        note_sent(left_, h_ + 2);
        note_sent(right_, h_ + 2);
    }

    // ---- one generation per mode ------------------------------------------------------
    void gen_blocking(double& comp, double& comm) {
        const double t0 = MPI_Wtime();
        rows_blocking();
        if (cfg_.block) cols_blocking();
        const double t1 = MPI_Wtime();
        upd(1, h_ + 1, gc_, gc_ + w_);
        const double t2 = MPI_Wtime();
        comm = t1 - t0;
        comp = t2 - t1;
    }

    // Post Irecv/Isend, update the rows that need no ghost row, Waitall, update the two
    // boundary rows. "comm" is the exposed communication: posting plus the time blocked in
    // Waitall, i.e. what the overlap did not hide.
    void gen_nonblocking_strip(double& comp, double& comm) {
        MPI_Request rq[4];
        const double t0 = MPI_Wtime();
        post_rows(rq);
        const double t1 = MPI_Wtime();
        upd(2, h_, 0, pitch_);  // rows 2 .. h-1: all neighbors are owned rows
        const double t2 = MPI_Wtime();
        MPI_Waitall(4, rq, MPI_STATUSES_IGNORE);
        const double t3 = MPI_Wtime();
        upd(1, 2, 0, pitch_);                      // first owned row
        if (h_ > 1) upd(h_, h_ + 1, 0, pitch_);    // last owned row
        const double t4 = MPI_Wtime();
        comm = (t1 - t0) + (t3 - t2);
        comp = (t2 - t1) + (t4 - t3);
    }

    // Two-phase exchange with overlap in both phases:
    //   rows posted   -> update the deep interior (needs no ghost at all)
    //   rows done     -> post columns (they carry the now-valid ghost rows, hence the corners)
    //                 -> update first/last row except their two end cells (need ghost rows only)
    //   columns done  -> update the first and last column
    void gen_nonblocking_block(double& comp, double& comm) {
        MPI_Request rr[4], rc[4];
        const double t0 = MPI_Wtime();
        post_rows(rr);
        const double t1 = MPI_Wtime();
        upd(2, h_, 2, w_);  // owned rows 2..h-1 x owned columns 2..w-1 (local cols 1..w owned)
        const double t2 = MPI_Wtime();
        MPI_Waitall(4, rr, MPI_STATUSES_IGNORE);
        const double t3 = MPI_Wtime();
        post_cols(rc);
        const double t4 = MPI_Wtime();
        upd(1, 2, 2, w_);
        if (h_ > 1) upd(h_, h_ + 1, 2, w_);
        const double t5 = MPI_Wtime();
        MPI_Waitall(4, rc, MPI_STATUSES_IGNORE);
        const double t6 = MPI_Wtime();
        upd(1, h_ + 1, 1, 2);
        if (w_ > 1) upd(1, h_ + 1, w_, w_ + 1);
        const double t7 = MPI_Wtime();
        comm = (t1 - t0) + (t3 - t2) + (t4 - t3) + (t6 - t5);
        comp = (t2 - t1) + (t5 - t4) + (t7 - t6);
    }

    Config cfg_;
    int n_ = 0, p_ = 1, rank_ = 0;
    int cols_el_ = 0;
    int dims_[2] = {1, 1};
    int gc_ = 0;  // ghost columns per side (0 strip, 1 block)
    int up_ = 0, down_ = 0, left_ = 0, right_ = 0;
    int r0_ = 0, c0_ = 0, h_ = 0, w_ = 0, pitch_ = 0;
    MPI_Comm comm_ = MPI_COMM_NULL;
    MPI_Datatype coltype_ = MPI_DATATYPE_NULL;
    std::vector<T> cur_, nxt_;
    uint64_t sent_elems_ = 0;
    double scatter_s_ = 0, gather_s_ = 0;
};

// --------------------------------------------------------------------------------------
struct Summary {
    int ok = 1;  // 0 = --check failed
};

template <class K>
int run_all(const gol::Options& o, const Config& cfg, const std::string& decomp,
            const std::string& mode, int world_rank, int world_size) {
    gol::Grid initial;
    if (world_rank == 0) initial = gol::random_grid(o.n, o.seed);

    Life<K> life(cfg, MPI_COMM_WORLD);
    life.load(world_rank == 0 ? &initial : nullptr);
    const RunStats st = life.run(o.gens);
    gol::Grid fin = life.store();
    double scatter_max = 0, gather_max = 0;
    life.setup_times(scatter_max, gather_max);
    const std::string dims = life.dims_string();
    life.release();

    int ok = 1;
    uint64_t sum = 0;
    if (world_rank == 0) {
        sum = gol::checksum(fin);
        if (o.check) {
            const gol::BackendInfo* ref = gol::find_backend("byte");
            if (ref == nullptr) {
                std::fprintf(stderr, "gol_mpi: --check needs the byte backend\n");
                ok = 0;
            } else {
                auto b = ref->make();
                b->load(initial, gol::RunConfig{o.n, 1});
                b->run(o.gens);
                const uint64_t want = gol::checksum(b->store());
                if (want != sum) {
                    std::fprintf(stderr, "gol_mpi: CHECK FAILED: mpi checksum %s, byte checksum %s\n",
                                 gol::format_checksum(sum).c_str(), gol::format_checksum(want).c_str());
                    ok = 0;
                } else {
                    std::fprintf(stderr, "gol_mpi: check ok, checksum matches serial byte\n");
                }
            }
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (!ok) return 3;

    if (world_rank == 0) {
        gol::Result r;
        r.impl = "mpi-" + decomp + "-" + mode + "-" + K::kName;
        r.n = o.n;
        r.gens = o.gens;
        r.threads = 1;
        r.ranks = world_size;
        r.seconds = st.loop_s;
        r.grid_bytes = static_cast<std::size_t>(o.n) * o.n * sizeof(typename K::T) / K::kCellsPerElem;
        r.checksum = sum;
        if (o.get("csv", "on") == "on") gol::print_csv_row(stdout, r);

        // Halo model from the guide: 2N cells per rank per generation for strips,
        // 4N / sqrt(p) for blocks.
        const double model = cfg.block ? 4.0 * o.n / std::sqrt(static_cast<double>(world_size))
                                       : 2.0 * o.n;
        char line[1024];
        std::snprintf(line, sizeof line,
                      "gol_mpi stats: decomp=%s mode=%s kernel=%s p=%d dims=%s compute_s=%.6f comm_s=%.6f "
                      "halo_elems_per_rank_gen=%.1f halo_cells_per_rank_gen=%.1f "
                      "halo_bytes_per_rank_gen=%.1f halo_max_cells_per_rank_gen=%.1f "
                      "model_cells_per_rank_gen=%.1f scatter_s=%.6f gather_s=%.6f\n",
                      decomp.c_str(), mode.c_str(), K::kName, world_size, dims.c_str(), st.compute_s,
                      st.comm_s, st.halo_avg_elems, st.halo_avg_elems * K::kCellsPerElem,
                      st.halo_avg_elems * sizeof(typename K::T),
                      st.halo_max_elems * K::kCellsPerElem, model, scatter_max, gather_max);
        std::fputs(line, stderr);  // one write: the line cannot be split by our own formatting
        std::fflush(stderr);
    }
    return 0;
}

void finalize_at_exit() {
    int done = 0;
    MPI_Finalized(&done);
    if (!done) MPI_Finalize();
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    std::atexit(finalize_at_exit);  // parse_cli may exit() on --help or a bad flag
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Only rank 0 talks: silence the others while the shared CLI parser may print.
    int saved_out = -1, saved_err = -1;
    if (rank != 0) {
        saved_out = dup(1);
        saved_err = dup(2);
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, 1);
            dup2(devnull, 2);
            close(devnull);
        }
    }
    gol::Options o = gol::parse_cli(argc, argv, {"decomp", "mode", "kernel", "csv"});
    if (rank != 0) {
        dup2(saved_out, 1);
        dup2(saved_err, 2);
        close(saved_out);
        close(saved_err);
    }

    if (o.csv_header) {
        if (rank == 0) gol::print_csv_header(stdout);
        return 0;
    }
    if (o.list) {
        if (rank == 0) std::printf("mpi\n");
        return 0;
    }

    int rc = 0;
    try {
        if (!o.impl.empty() && o.impl != "mpi")
            throw std::invalid_argument("--impl must be 'mpi' for gol_mpi (got '" + o.impl + "')");
        const std::string decomp = o.get("decomp", "strip");
        const std::string mode = o.get("mode", "blocking");
        const std::string kernel = o.get("kernel", "byte");
        if (decomp != "strip" && decomp != "block")
            throw std::invalid_argument("--decomp must be strip or block (got '" + decomp + "')");
        if (mode != "blocking" && mode != "nonblocking")
            throw std::invalid_argument("--mode must be blocking or nonblocking (got '" + mode + "')");
        if (kernel != "byte" && kernel != "bitpack")
            throw std::invalid_argument("--kernel must be byte or bitpack (got '" + kernel + "')");

        const std::string csv = o.get("csv", "on");
        if (csv != "on" && csv != "off")
            throw std::invalid_argument("--csv must be on or off (got '" + csv + "')");

        Config cfg;
        cfg.n = o.n;
        cfg.seed = o.seed;
        cfg.block = decomp == "block";
        cfg.nonblocking = mode == "nonblocking";

        if (kernel == "byte") rc = run_all<ByteKernel>(o, cfg, decomp, mode, rank, size);
        else rc = run_all<BitpackKernel>(o, cfg, decomp, mode, rank, size);
    } catch (const std::invalid_argument& e) {
        // Raised identically on every rank before any rank can block in a collective.
        if (rank == 0) std::fprintf(stderr, "gol_mpi: %s\n", e.what());
        rc = 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gol_mpi: rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return rc;
}
