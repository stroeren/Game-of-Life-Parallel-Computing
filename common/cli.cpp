#include "common/cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace gol {

std::string Options::get(const std::string& key, const std::string& fallback) const {
    auto it = extra.find(key);
    return it == extra.end() ? fallback : it->second;
}

namespace {

void usage(std::FILE* out, const char* prog, const std::vector<std::string>& extra_keys) {
    std::fprintf(out,
                 "usage: %s --impl <name> --n <N> --gens <K> --seed <S> [--threads T] [--check]",
                 prog);
    for (const auto& k : extra_keys) std::fprintf(out, " [--%s <value>]", k.c_str());
    std::fprintf(out,
                 "\n       %s --list | --csv-header | --help\n"
                 "  N must be a positive multiple of 64. Defaults: --n 1024 --gens 100 --seed 42.\n"
                 "  Prints one CSV row: impl,n,gens,threads,ranks,seconds,cells_per_sec,"
                 "grid_bytes,gib_per_s,checksum\n",
                 prog);
}

long long to_int(const std::string& flag, const std::string& v, long long lo, long long hi) {
    std::size_t pos = 0;
    long long x = 0;
    try {
        x = std::stoll(v, &pos, 0);
    } catch (const std::exception&) {
        pos = 0;
    }
    if (pos != v.size() || v.empty() || x < lo || x > hi)
        throw std::invalid_argument(flag + " expects an integer in [" + std::to_string(lo) + ", " +
                                    std::to_string(hi) + "], got '" + v + "'");
    return x;
}

}  // namespace

Options parse_cli(int argc, char** argv, const std::vector<std::string>& extra_keys) {
    Options o;
    const char* prog = argc > 0 ? argv[0] : "gol";
    try {
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto value = [&]() -> std::string {
                if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
                return argv[++i];
            };
            if (a == "--help" || a == "-h") {
                usage(stdout, prog, extra_keys);
                std::exit(0);
            } else if (a == "--impl") {
                o.impl = value();
            } else if (a == "--n") {
                o.n = static_cast<int>(to_int(a, value(), 64, 1 << 20));
                if (o.n % 64 != 0) throw std::invalid_argument("--n must be a multiple of 64");
            } else if (a == "--gens") {
                o.gens = static_cast<int>(to_int(a, value(), 0, 2000000000));
            } else if (a == "--seed") {
                std::string v = value();
                std::size_t pos = 0;
                try {
                    o.seed = std::stoull(v, &pos, 0);
                } catch (const std::exception&) {
                    pos = 0;
                }
                if (v.empty() || pos != v.size() || v[0] == '-')
                    throw std::invalid_argument("--seed expects an unsigned integer, got '" + v + "'");
            } else if (a == "--threads") {
                o.threads = static_cast<int>(to_int(a, value(), 0, 4096));
            } else if (a == "--check") {
                o.check = true;
            } else if (a == "--list") {
                o.list = true;
            } else if (a == "--csv-header") {
                o.csv_header = true;
            } else {
                bool matched = false;
                for (const auto& k : extra_keys) {
                    if (a == "--" + k) {
                        o.extra[k] = value();
                        matched = true;
                        break;
                    }
                }
                if (!matched) throw std::invalid_argument("unknown argument '" + a + "'");
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", prog, e.what());
        usage(stderr, prog, extra_keys);
        std::exit(2);
    }
    return o;
}

}  // namespace gol
