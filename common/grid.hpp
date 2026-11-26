// Canonical grid, initial state, pack/unpack and checksum (CLAUDE.md "Contracts").
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gol {

// The canonical layout: N x N torus, N a multiple of 64, N/64 uint64_t words per row,
// row-major. Bit j of word w in row r is cell (r, 64*w + j).
struct Grid {
    int n = 0;
    std::vector<uint64_t> words;

    Grid() = default;
    explicit Grid(int n_);  // all dead

    int words_per_row() const { return n / 64; }
    uint64_t* row(int r) { return words.data() + static_cast<std::size_t>(r) * words_per_row(); }
    const uint64_t* row(int r) const { return words.data() + static_cast<std::size_t>(r) * words_per_row(); }

    bool get(int r, int c) const { return (row(r)[c >> 6] >> (c & 63)) & 1u; }
    void set(int r, int c, bool alive) {
        uint64_t mask = uint64_t{1} << (c & 63);
        uint64_t& w = row(r)[c >> 6];
        w = alive ? (w | mask) : (w & ~mask);
    }
    std::size_t population() const;

    bool operator==(const Grid& o) const { return n == o.n && words == o.words; }
    bool operator!=(const Grid& o) const { return !(*this == o); }
};

// Throws std::invalid_argument unless n > 0 and n % 64 == 0.
void validate_n(int n);

// splitmix64 (Steele, Lea, Flood 2014). Advances state and returns the next output.
inline uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// The contract's initial state: splitmix64 seeded with `seed` fills the canonical grid
// one word at a time in row-major order.
Grid random_grid(int n, uint64_t seed);

// Byte grid <-> canonical grid. A byte grid is n*n bytes, row-major, 0 = dead, 1 = alive
// (any nonzero byte packs as alive).
std::vector<uint8_t> unpack(const Grid& g);
void unpack_into(const Grid& g, uint8_t* cells);
Grid pack(const uint8_t* cells, int n);

// FNV-1a 64 over arbitrary bytes.
uint64_t fnv1a64(const uint8_t* data, std::size_t len, uint64_t h = 0xcbf29ce484222325ULL);

// The contract's checksum: FNV-1a 64 over the canonical words as little-endian bytes.
uint64_t checksum(const Grid& g);

}  // namespace gol
