#include "common/grid.hpp"

#include <stdexcept>
#include <string>

namespace gol {

void validate_n(int n) {
    if (n <= 0 || n % 64 != 0)
        throw std::invalid_argument("N must be a positive multiple of 64, got " + std::to_string(n));
}

Grid::Grid(int n_) : n(n_) {
    validate_n(n_);
    words.assign(static_cast<std::size_t>(n_) * (n_ / 64), 0);
}

std::size_t Grid::population() const {
    std::size_t pop = 0;
    for (uint64_t w : words) pop += static_cast<std::size_t>(__builtin_popcountll(w));
    return pop;
}

Grid random_grid(int n, uint64_t seed) {
    Grid g(n);
    uint64_t state = seed;
    for (uint64_t& w : g.words) w = splitmix64(state);
    return g;
}

void unpack_into(const Grid& g, uint8_t* cells) {
    const int n = g.n;
    for (int r = 0; r < n; ++r) {
        const uint64_t* src = g.row(r);
        uint8_t* dst = cells + static_cast<std::size_t>(r) * n;
        for (int c = 0; c < n; ++c) dst[c] = static_cast<uint8_t>((src[c >> 6] >> (c & 63)) & 1u);
    }
}

std::vector<uint8_t> unpack(const Grid& g) {
    std::vector<uint8_t> cells(static_cast<std::size_t>(g.n) * g.n);
    unpack_into(g, cells.data());
    return cells;
}

Grid pack(const uint8_t* cells, int n) {
    Grid g(n);
    for (int r = 0; r < n; ++r) {
        const uint8_t* src = cells + static_cast<std::size_t>(r) * n;
        uint64_t* dst = g.row(r);
        for (int w = 0; w < n / 64; ++w) {
            uint64_t word = 0;
            for (int j = 0; j < 64; ++j) word |= static_cast<uint64_t>(src[64 * w + j] != 0) << j;
            dst[w] = word;
        }
    }
    return g;
}

uint64_t fnv1a64(const uint8_t* data, std::size_t len, uint64_t h) {
    for (std::size_t i = 0; i < len; ++i) {
        h ^= data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t checksum(const Grid& g) {
    // Serialize each word explicitly as little-endian so the result does not depend on
    // the host's byte order.
    uint64_t h = 0xcbf29ce484222325ULL;
    for (uint64_t w : g.words) {
        uint8_t bytes[8];
        for (int b = 0; b < 8; ++b) bytes[b] = static_cast<uint8_t>(w >> (8 * b));
        h = fnv1a64(bytes, 8, h);
    }
    return h;
}

}  // namespace gol
