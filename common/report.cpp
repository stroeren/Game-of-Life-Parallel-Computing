#include "common/report.hpp"

#include <cinttypes>

namespace gol {

double Result::cells_per_sec() const {
    if (seconds <= 0.0) return 0.0;
    return static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(gens) / seconds;
}

double Result::gib_per_s() const {
    if (seconds <= 0.0) return 0.0;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    return 2.0 * static_cast<double>(grid_bytes) * static_cast<double>(gens) / gib / seconds;
}

void print_csv_header(std::FILE* out) {
    std::fputs("impl,n,gens,threads,ranks,seconds,cells_per_sec,grid_bytes,gib_per_s,checksum\n", out);
}

std::string format_checksum(uint64_t c) {
    char buf[24];
    std::snprintf(buf, sizeof buf, "0x%016" PRIx64, c);
    return buf;
}

void print_csv_row(std::FILE* out, const Result& r) {
    std::fprintf(out, "%s,%d,%d,%d,%d,%.6f,%.6e,%zu,%.4f,%s\n", r.impl.c_str(), r.n, r.gens,
                 r.threads, r.ranks, r.seconds, r.cells_per_sec(), r.grid_bytes, r.gib_per_s(),
                 format_checksum(r.checksum).c_str());
    std::fflush(out);
}

}  // namespace gol
