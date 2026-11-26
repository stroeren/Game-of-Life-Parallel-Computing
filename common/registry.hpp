// Implementation registry.
//
// Adding a backend = one new source file + one registry line:
//
//   // src/foo/foo.cpp
//   #include "common/registry.hpp"
//   class FooBackend : public gol::Backend { ... };
//   GOL_REGISTER_BACKEND("foo", FooBackend);
//
//   # src/CMakeLists.txt (the registry line)
//   gol_backend(foo/foo.cpp)
//
// The driver, the tests and bench/run.sh then pick it up automatically. Backends that need
// a CPU feature or a GPU use GOL_REGISTER_BACKEND_IF with a predicate checked at runtime;
// when it returns false the backend is hidden from --list, tests and benchmarks.
#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "common/grid.hpp"

namespace gol {

struct RunConfig {
    int n = 0;
    int threads = 0;  // 0 = backend default
};

class Backend {
public:
    virtual ~Backend() = default;

    // Take the canonical grid as the initial state. Allocation, layout conversion and
    // host-to-device copies happen here; they are not timed.
    virtual void load(const Grid& initial, const RunConfig& cfg) = 0;

    // Advance `gens` generations and return the seconds spent in the generation loop
    // only (CPU: gol::Timer around the loop; GPU: cudaEvent around the kernels). May be
    // called several times; each call continues from the current state.
    virtual double run(int gens) = 0;

    // Current state in the canonical layout. Device-to-host copies happen here.
    virtual Grid store() = 0;

    // Bytes of one grid in this backend's own representation (for gib_per_s).
    virtual std::size_t grid_bytes() const = 0;

    // Threads actually used, reported in the CSV.
    virtual int threads() const { return 1; }
};

using BackendFactory = std::unique_ptr<Backend> (*)();
using BackendPredicate = bool (*)();

struct BackendInfo {
    std::string name;
    BackendFactory make;
    BackendPredicate available;  // nullptr = always available
    bool is_available() const { return available == nullptr || available(); }
};

// All registered backends, sorted by name (including unavailable ones).
const std::vector<BackendInfo>& all_backends();
// Registered backends whose predicate passes, sorted by name.
std::vector<const BackendInfo*> available_backends();
// nullptr if no backend with that name is registered.
const BackendInfo* find_backend(const std::string& name);

struct Registrar {
    Registrar(const char* name, BackendFactory make, BackendPredicate available = nullptr);
};

}  // namespace gol

#define GOL_CONCAT_INNER(a, b) a##b
#define GOL_CONCAT(a, b) GOL_CONCAT_INNER(a, b)

#define GOL_REGISTER_BACKEND_IF(NAME, CLASS, PREDICATE)                                    \
    static const ::gol::Registrar GOL_CONCAT(gol_registrar_, __LINE__)(                    \
        NAME, []() -> std::unique_ptr<::gol::Backend> { return std::make_unique<CLASS>(); }, \
        PREDICATE)

#define GOL_REGISTER_BACKEND(NAME, CLASS) GOL_REGISTER_BACKEND_IF(NAME, CLASS, nullptr)
