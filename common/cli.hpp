// Command-line parsing shared by gol and gol_mpi.
//
//   gol --impl <name> --n <N> --gens <K> --seed <S> [--threads T] [--check]
//       [--list] [--csv-header] [--help]
//
// gol_mpi passes its extra flags (decomp, mode, kernel) as `extra_keys`; their values end
// up in Options::extra.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace gol {

struct Options {
    std::string impl;
    int n = 1024;
    int gens = 100;
    uint64_t seed = 42;
    int threads = 0;  // 0 = backend default (OpenMP: omp_get_max_threads())
    bool check = false;       // also run "byte" and fail if checksums differ
    bool list = false;        // print available backends and exit
    bool csv_header = false;  // print the CSV header and exit
    std::map<std::string, std::string> extra;  // --key value for each extra key

    // Value of an extra flag, or `fallback` if it was not given.
    std::string get(const std::string& key, const std::string& fallback) const;
};

// Parses argv. On --help prints usage to stdout and exits 0; on a bad flag prints the
// error and usage to stderr and exits 2.
Options parse_cli(int argc, char** argv, const std::vector<std::string>& extra_keys = {});

}  // namespace gol
