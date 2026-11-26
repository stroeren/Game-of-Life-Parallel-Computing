// CTest entry point. Usage: gol_tests common | blinker | glider | checksum <N>
//
// blinker, glider and checksum run against every available registered backend, so a
// backend added with one source file and one registry line is tested automatically.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "common/grid.hpp"
#include "common/registry.hpp"
#include "common/report.hpp"

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("  FAIL %s\n", what.c_str());
    ++failures;
}

void pass(const std::string& what) { std::printf("  ok   %s\n", what.c_str()); }

int wrap(int x, int n) { return ((x % n) + n) % n; }

// Grid with the given cells (row, col) alive, coordinates taken modulo n.
gol::Grid with_cells(int n, const std::vector<std::pair<int, int>>& cells, int dr = 0, int dc = 0) {
    gol::Grid g(n);
    for (auto [r, c] : cells) g.set(wrap(r + dr, n), wrap(c + dc, n), true);
    return g;
}

std::string describe(const gol::Grid& g, std::size_t max_cells = 12) {
    std::string s = "pop " + std::to_string(g.population()) + " {";
    std::size_t shown = 0;
    for (int r = 0; r < g.n && shown < max_cells; ++r)
        for (int c = 0; c < g.n && shown < max_cells; ++c)
            if (g.get(r, c)) {
                s += " (" + std::to_string(r) + "," + std::to_string(c) + ")";
                ++shown;
            }
    return s + " }";
}

// Independent reference: the rule written directly on the canonical grid with modulo
// indexing. Slow, but shares no code with any backend.
gol::Grid reference_step(const gol::Grid& g) {
    gol::Grid out(g.n);
    const int n = g.n;
    for (int r = 0; r < n; ++r)
        for (int c = 0; c < n; ++c) {
            int count = 0;
            for (int dr = -1; dr <= 1; ++dr)
                for (int dc = -1; dc <= 1; ++dc)
                    if (dr != 0 || dc != 0) count += g.get(wrap(r + dr, n), wrap(c + dc, n));
            bool alive = g.get(r, c);
            out.set(r, c, count == 3 || (alive && count == 2));
        }
    return out;
}

// --------------------------------------------------------------------------------------
// common: grid helpers and checksum
// --------------------------------------------------------------------------------------
void test_common() {
    // FNV-1a 64 published test vectors.
    const char* a = "a";
    if (gol::fnv1a64(nullptr, 0) != 0xcbf29ce484222325ULL) fail("fnv1a64(\"\")");
    else pass("fnv1a64(\"\")");
    if (gol::fnv1a64(reinterpret_cast<const uint8_t*>(a), 1) != 0xaf63dc4c8601ec8cULL) fail("fnv1a64(\"a\")");
    else pass("fnv1a64(\"a\")");
    const char* foobar = "foobar";
    if (gol::fnv1a64(reinterpret_cast<const uint8_t*>(foobar), 6) != 0x85944171f73967e8ULL) fail("fnv1a64(\"foobar\")");
    else pass("fnv1a64(\"foobar\")");

    // splitmix64 reference outputs for seed 1234567 (from the reference C implementation).
    uint64_t s = 1234567;
    const uint64_t expect[3] = {6457827717110365317ULL, 3203168211198807973ULL, 9817491932198370423ULL};
    bool ok = true;
    for (uint64_t e : expect) ok = ok && gol::splitmix64(s) == e;
    if (ok) pass("splitmix64 reference sequence");
    else fail("splitmix64 reference sequence");

    // Initial grid is splitmix64 words in row-major order.
    gol::Grid g = gol::random_grid(128, 42);
    uint64_t st = 42;
    ok = true;
    for (uint64_t w : g.words) ok = ok && w == gol::splitmix64(st);
    if (ok) pass("random_grid fills words in row-major order");
    else fail("random_grid fills words in row-major order");

    // Bit j of word w in row r is cell (r, 64*w + j).
    gol::Grid one(128);
    one.row(3)[1] = uint64_t{1} << 5;
    auto cells = gol::unpack(one);
    ok = cells[3 * 128 + 64 + 5] == 1 && one.population() == 1 && one.get(3, 69);
    if (ok) pass("bit layout: bit j of word w in row r is cell (r, 64w+j)");
    else fail("bit layout");

    if (gol::pack(gol::unpack(g).data(), g.n) == g) pass("pack(unpack(g)) == g");
    else fail("pack(unpack(g)) == g");

    // Checksum hashes words as little-endian bytes.
    gol::Grid tiny(64);
    tiny.words[0] = 0x0102030405060708ULL;
    std::vector<uint8_t> bytes(tiny.words.size() * 8, 0);
    const uint8_t le[8] = {8, 7, 6, 5, 4, 3, 2, 1};
    std::memcpy(bytes.data(), le, 8);
    if (gol::checksum(tiny) == gol::fnv1a64(bytes.data(), bytes.size())) pass("checksum is little-endian FNV-1a 64");
    else fail("checksum is little-endian FNV-1a 64");
}

// Runs `gens` generations on a fresh backend instance and returns the result.
gol::Grid evolve(const gol::BackendInfo& b, const gol::Grid& start, int gens) {
    auto be = b.make();
    be->load(start, gol::RunConfig{start.n, 0});
    be->run(gens);
    return be->store();
}

// --------------------------------------------------------------------------------------
// blinker: period 2, in the middle and straddling the wrap
// --------------------------------------------------------------------------------------
void test_blinker() {
    for (int n : {64, 256}) {
        for (bool on_edge : {false, true}) {
            int r = on_edge ? 0 : n / 2, c = on_edge ? 0 : n / 2;
            gol::Grid horiz = with_cells(n, {{r, c - 1}, {r, c}, {r, c + 1}});
            gol::Grid vert = with_cells(n, {{r - 1, c}, {r, c}, {r + 1, c}});
            for (const auto* b : gol::available_backends()) {
                std::string what = b->name + " N=" + std::to_string(n) + (on_edge ? " across wrap" : " center");
                try {
                    gol::Grid g1 = evolve(*b, horiz, 1);
                    gol::Grid g2 = evolve(*b, horiz, 2);
                    if (g1 != vert) fail(what + ": gen 1 should be vertical, got " + describe(g1));
                    else if (g2 != horiz) fail(what + ": gen 2 should equal gen 0, got " + describe(g2));
                    else pass(what);
                } catch (const std::exception& e) {
                    fail(what + ": threw " + e.what());
                }
            }
        }
    }
}

// --------------------------------------------------------------------------------------
// glider: after every 4 generations it equals itself shifted (+1, +1), including while it
// crosses the corner of the torus; after 4N generations it is back where it started.
// --------------------------------------------------------------------------------------
void test_glider() {
    // Moves down-right: +1 row, +1 column every 4 generations.
    const std::vector<std::pair<int, int>> glider = {{0, 1}, {1, 2}, {2, 0}, {2, 1}, {2, 2}};
    for (int n : {64, 256}) {
        const int start = n - 3;  // starts 3 cells before the corner, so it crosses both edges
        gol::Grid g0 = with_cells(n, glider, start, start);
        for (const auto* b : gol::available_backends()) {
            std::string what = b->name + " N=" + std::to_string(n);
            try {
                auto be = b->make();
                be->load(g0, gol::RunConfig{n, 0});
                bool ok = true;
                for (int k = 1; k <= 8 && ok; ++k) {  // 8 shifts: before, across and after the corner
                    be->run(4);
                    gol::Grid got = be->store();
                    gol::Grid want = with_cells(n, glider, start + k, start + k);
                    if (got != want) {
                        fail(what + ": after " + std::to_string(4 * k) + " gens expected " + describe(want) +
                             ", got " + describe(got));
                        ok = false;
                    }
                }
                if (ok && n == 64) {  // full lap: 4N generations brings it home
                    be->run(4 * n - 32);
                    gol::Grid got = be->store();
                    if (got != g0) {
                        fail(what + ": after a full lap (" + std::to_string(4 * n) + " gens) got " + describe(got));
                        ok = false;
                    }
                }
                if (ok) pass(what + (n == 64 ? " (incl. full lap)" : ""));
            } catch (const std::exception& e) {
                fail(what + ": threw " + e.what());
            }
        }
    }
}

// --------------------------------------------------------------------------------------
// checksum: every backend equals byte after 100 generations, seed 42. byte itself is
// checked against the independent reference above.
// --------------------------------------------------------------------------------------
void test_checksum(int n) {
    const int gens = 100;
    const uint64_t seed = 42;
    const gol::BackendInfo* byte = gol::find_backend("byte");
    if (byte == nullptr) {
        fail("byte backend is not registered");
        return;
    }
    gol::Grid start = gol::random_grid(n, seed);
    const uint64_t want = gol::checksum(evolve(*byte, start, gens));
    std::printf("  byte checksum N=%d gens=%d seed=%llu: %s\n", n, gens,
                static_cast<unsigned long long>(seed), gol::format_checksum(want).c_str());

    if (n <= 256) {
        gol::Grid ref = start;
        for (int g = 0; g < gens; ++g) ref = reference_step(ref);
        if (gol::checksum(ref) != want)
            fail("byte vs independent reference: reference " + gol::format_checksum(gol::checksum(ref)));
        else
            pass("byte matches independent reference");
    }

    for (const auto* b : gol::available_backends()) {
        if (b->name == "byte") continue;
        std::string what = b->name + " N=" + std::to_string(n);
        try {
            uint64_t got = gol::checksum(evolve(*b, start, gens));
            if (got != want) fail(what + ": checksum " + gol::format_checksum(got) + " != byte " + gol::format_checksum(want));
            else pass(what);
        } catch (const std::exception& e) {
            fail(what + ": threw " + e.what());
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s common | blinker | glider | checksum <N>\n", argv[0]);
        return 2;
    }
    std::string which = argv[1];
    std::printf("backends:");
    for (const auto* b : gol::available_backends()) std::printf(" %s", b->name.c_str());
    for (const auto& b : gol::all_backends())
        if (!b.is_available()) std::printf(" (%s: skipped, not supported here)", b.name.c_str());
    std::printf("\n");

    if (which == "common") test_common();
    else if (which == "blinker") test_blinker();
    else if (which == "glider") test_glider();
    else if (which == "checksum" && argc >= 3) test_checksum(std::atoi(argv[2]));
    else {
        std::fprintf(stderr, "unknown test '%s'\n", which.c_str());
        return 2;
    }
    std::printf("%s: %d failure(s)\n", which.c_str(), failures);
    return failures == 0 ? 0 : 1;
}
