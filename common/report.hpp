// The CSV row every driver prints (CLAUDE.md "Driver").
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace gol {

struct Result {
    std::string impl;
    int n = 0;
    int gens = 0;
    int threads = 1;
    int ranks = 1;
    double seconds = 0.0;        // generation loop only
    std::size_t grid_bytes = 0;  // one grid in the implementation's representation
    uint64_t checksum = 0;

    double cells_per_sec() const;
    // 2 * grid_bytes * gens / 1024^3 / seconds: each generation reads and writes the grid.
    double gib_per_s() const;
};

void print_csv_header(std::FILE* out);
void print_csv_row(std::FILE* out, const Result& r);
// Checksums print as 16 lowercase hex digits with a 0x prefix.
std::string format_checksum(uint64_t c);

}  // namespace gol
