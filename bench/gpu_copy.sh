#!/usr/bin/env bash
# Measure achievable GPU memory bandwidth with the device-to-device copy kernel
# (build/gpu_copy, src/cuda/gpu_copy.cu), the GPU counterpart of bench/stream.sh.
#
#   bench/gpu_copy.sh [size_MiB ...]     default sizes: 16 64 256 1024
#
# Saves the output to bench/results/gpucopy_<hostname>.txt. Its last line,
# "GPU copy kernel: X GiB/s ...", is what bench/hwinfo.sh puts in docs/hardware.md and
# what bench/summarize.py divides the cuda_* GiB/s by.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BIN=${GPU_COPY:-$ROOT/build/gpu_copy}
HOST=$(hostname -s 2>/dev/null || hostname)
OUT=$ROOT/bench/results/gpucopy_$HOST.txt

if [[ ! -x $BIN ]]; then
  echo "gpu_copy.sh: $BIN not found; it is built only when CMake finds CUDA" >&2
  exit 1
fi
mkdir -p "$(dirname "$OUT")"
"$BIN" "$@" | tee "$OUT"
