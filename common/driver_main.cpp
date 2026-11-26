// gol: run one registered backend and print one CSV row (CLAUDE.md "Driver").
#include <cstdio>
#include <exception>

#include "common/cli.hpp"
#include "common/grid.hpp"
#include "common/registry.hpp"
#include "common/report.hpp"

namespace {

struct Outcome {
    double seconds;
    std::size_t grid_bytes;
    int threads;
    uint64_t checksum;
};

Outcome run_backend(const gol::BackendInfo& info, const gol::Options& o) {
    // Not timed: initial state, load (layout conversion, device copies), store, checksum.
    gol::Grid initial = gol::random_grid(o.n, o.seed);
    auto backend = info.make();
    backend->load(initial, gol::RunConfig{o.n, o.threads});
    double seconds = backend->run(o.gens);
    gol::Grid final_state = backend->store();
    return Outcome{seconds, backend->grid_bytes(), backend->threads(), gol::checksum(final_state)};
}

}  // namespace

int main(int argc, char** argv) {
    gol::Options o = gol::parse_cli(argc, argv);

    if (o.csv_header) {
        gol::print_csv_header(stdout);
        return 0;
    }
    if (o.list) {
        for (const auto* b : gol::available_backends()) std::printf("%s\n", b->name.c_str());
        return 0;
    }
    if (o.impl.empty()) {
        std::fprintf(stderr, "gol: --impl is required (see --list)\n");
        return 2;
    }
    const gol::BackendInfo* info = gol::find_backend(o.impl);
    if (info == nullptr || !info->is_available()) {
        std::fprintf(stderr, "gol: backend '%s' is %s. Available:", o.impl.c_str(),
                     info ? "not supported on this machine" : "unknown");
        for (const auto* b : gol::available_backends()) std::fprintf(stderr, " %s", b->name.c_str());
        std::fprintf(stderr, "\n");
        return 2;
    }

    try {
        Outcome out = run_backend(*info, o);

        if (o.check) {
            const gol::BackendInfo* ref = gol::find_backend("byte");
            if (ref == nullptr) {
                std::fprintf(stderr, "gol: --check needs the byte backend\n");
                return 2;
            }
            Outcome expect = run_backend(*ref, o);
            if (expect.checksum != out.checksum) {
                std::fprintf(stderr, "gol: CHECK FAILED: %s checksum %s, byte checksum %s\n",
                             o.impl.c_str(), gol::format_checksum(out.checksum).c_str(),
                             gol::format_checksum(expect.checksum).c_str());
                return 3;
            }
            std::fprintf(stderr, "gol: check ok, checksum matches byte\n");
        }

        gol::Result r;
        r.impl = o.impl;
        r.n = o.n;
        r.gens = o.gens;
        r.threads = out.threads;
        r.ranks = 1;
        r.seconds = out.seconds;
        r.grid_bytes = out.grid_bytes;
        r.checksum = out.checksum;
        gol::print_csv_row(stdout, r);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gol: %s: %s\n", o.impl.c_str(), e.what());
        return 1;
    }
    return 0;
}
