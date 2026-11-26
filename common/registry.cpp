#include "common/registry.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace gol {

namespace {
// Function-local static: safe to use from other translation units' static initializers.
std::vector<BackendInfo>& registry() {
    static std::vector<BackendInfo> r;
    return r;
}
}  // namespace

Registrar::Registrar(const char* name, BackendFactory make, BackendPredicate available) {
    auto& r = registry();
    for (const auto& b : r) {
        if (b.name == name) {
            std::fprintf(stderr, "gol: backend '%s' registered twice\n", name);
            std::abort();
        }
    }
    r.push_back(BackendInfo{name, make, available});
    std::sort(r.begin(), r.end(),
              [](const BackendInfo& a, const BackendInfo& b) { return a.name < b.name; });
}

const std::vector<BackendInfo>& all_backends() { return registry(); }

std::vector<const BackendInfo*> available_backends() {
    std::vector<const BackendInfo*> out;
    for (const auto& b : registry())
        if (b.is_available()) out.push_back(&b);
    return out;
}

const BackendInfo* find_backend(const std::string& name) {
    for (const auto& b : registry())
        if (b.name == name) return &b;
    return nullptr;
}

}  // namespace gol
