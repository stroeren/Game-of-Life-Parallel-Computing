// Step 5b: CUDA, shared-memory tiles. Each BX x BY thread block owns a BX x BY tile of
// the byte grid. It first loads the tile plus a one-cell halo, (BX+2) x (BY+2) cells,
// into shared memory (every thread loads one or two cells, consecutive threads loading
// consecutive bytes of a row so the loads coalesce), calls __syncthreads(), and then each
// thread counts its 8 neighbors from shared memory and writes its one interior cell.
//
// Compared to cuda_naive:
//   - each input cell is read from global memory (1 + halo) times per block instead of up
//     to 9 times per cell (the halo overhead is (BX+2)(BY+2) / (BX*BY): 1.27 for 16x16,
//     1.33 for 32x8, 1.20 for 32x16, 1.13 for 32x32);
//   - the torus wrap is needed only for halo cells at the grid edge, and it is a
//     compare-and-add instead of a modulo by N.
//
// Block size: BX x BY is a template parameter so the tile dimensions, the cooperative
// load loop and the shared array are compile-time constants. The block sizes the prompt
// asks to compare, 16x16, 32x8, 32x16 and 32x32, are all instantiated; GOL_CUDA_BLOCK
// (e.g. GOL_CUDA_BLOCK=32x8) selects one at runtime and bench/cuda_block_sweep.sh runs all
// four. kDefaultBlock is the default when GOL_CUDA_BLOCK is unset. It must be set to the
// sweep's winner; until the sweep has run on a GPU it is 32x16, the size with the
// lowest halo overhead that still allows several resident blocks per SM (docs/cuda.md).
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "src/cuda/byte_device_backend.cuh"

namespace gol::cuda {

namespace {

template <int BX, int BY>
__global__ void step_shared(const uint8_t* __restrict__ in, uint8_t* __restrict__ out, int n) {
    constexpr int TX = BX + 2;  // tile width including halo
    constexpr int TY = BY + 2;
    __shared__ uint8_t tile[TY][TX];

    // Global coordinates of tile[0][0], the top-left halo cell.
    const int row0 = static_cast<int>(blockIdx.y) * BY - 1;
    const int col0 = static_cast<int>(blockIdx.x) * BX - 1;

    // Cooperative load: the block's BX*BY threads walk the TY*TX tile in row-major order.
    for (int i = threadIdx.y * BX + threadIdx.x; i < TY * TX; i += BX * BY) {
        const int ty = i / TX;
        const int tx = i % TX;
        int gr = row0 + ty;  // only the halo can fall outside [0, n): wrap it (torus)
        int gc = col0 + tx;
        gr += gr < 0 ? n : 0;
        gr -= gr >= n ? n : 0;
        gc += gc < 0 ? n : 0;
        gc -= gc >= n ? n : 0;
        tile[ty][tx] = in[static_cast<std::size_t>(gr) * n + gc];
    }
    __syncthreads();

    // Interior: tile coordinates are thread coordinates + 1.
    const int y = threadIdx.y + 1;
    const int x = threadIdx.x + 1;
    const int count = tile[y - 1][x - 1] + tile[y - 1][x] + tile[y - 1][x + 1] +
                      tile[y][x - 1] + tile[y][x + 1] +
                      tile[y + 1][x - 1] + tile[y + 1][x] + tile[y + 1][x + 1];
    const int alive = tile[y][x];

    // n is a multiple of 64 and BX, BY divide 64, so every thread owns a real cell.
    const std::size_t r = static_cast<std::size_t>(blockIdx.y) * BY + threadIdx.y;
    const std::size_t c = static_cast<std::size_t>(blockIdx.x) * BX + threadIdx.x;
    out[r * n + c] = static_cast<uint8_t>((count == 3) | (alive & (count == 2)));
}

template <int BX, int BY>
void launch_tiles(const uint8_t* in, uint8_t* out, int n) {
    static_assert(64 % BX == 0 && 64 % BY == 0, "block dimensions must divide 64 (N is a multiple of 64)");
    const dim3 block(BX, BY);
    const dim3 grid(n / BX, n / BY);
    step_shared<BX, BY><<<grid, block>>>(in, out, n);
}

enum class Block { k16x16, k32x8, k32x16, k32x32 };

constexpr Block kDefaultBlock = Block::k32x16;  // provisional: set to the sweep's winner

const char* block_name(Block b) {
    switch (b) {
        case Block::k16x16: return "16x16";
        case Block::k32x8: return "32x8";
        case Block::k32x16: return "32x16";
        case Block::k32x32: return "32x32";
    }
    return "?";
}

// GOL_CUDA_BLOCK, parsed once per process.
Block block_config() {
    static const Block chosen = [] {
        const char* env = std::getenv("GOL_CUDA_BLOCK");
        Block b = kDefaultBlock;
        if (env != nullptr && *env != '\0') {
            const std::string s = env;
            if (s == "16x16") b = Block::k16x16;
            else if (s == "32x8") b = Block::k32x8;
            else if (s == "32x16") b = Block::k32x16;
            else if (s == "32x32") b = Block::k32x32;
            else throw std::invalid_argument("GOL_CUDA_BLOCK must be 16x16, 32x8, 32x16 or 32x32, got '" + s + "'");
        }
        std::fprintf(stderr, "cuda_shared: block %s%s\n", block_name(b),
                     env != nullptr && *env != '\0' ? " (GOL_CUDA_BLOCK)" : " (default)");
        return b;
    }();
    return chosen;
}

struct SharedStep {
    static constexpr const char* kName = "cuda_shared";
    static void prepare() { block_config(); }  // parse GOL_CUDA_BLOCK before the timed loop
    static void launch(const uint8_t* in, uint8_t* out, int n) {
        switch (block_config()) {
            case Block::k16x16: launch_tiles<16, 16>(in, out, n); break;
            case Block::k32x8: launch_tiles<32, 8>(in, out, n); break;
            case Block::k32x16: launch_tiles<32, 16>(in, out, n); break;
            case Block::k32x32: launch_tiles<32, 32>(in, out, n); break;
        }
    }
};

}  // namespace

}  // namespace gol::cuda

GOL_REGISTER_BACKEND_IF("cuda_shared", gol::cuda::ByteDeviceBackend<gol::cuda::SharedStep>,
                        gol::cuda::device_available);
