#!/usr/bin/env bash
# Measure sustainable CPU memory bandwidth with STREAM (McCalpin).
#
# Downloads stream.c from the official STREAM site (falling back to its GitHub mirror),
# builds it with OpenMP and an array size of at least 4x the last-level cache, runs it,
# saves the output to bench/results/stream_<hostname>.txt and prints Triad in GiB/s.
# stream.c is downloaded, not committed (see .gitignore), and is run unmodified.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DIR=$ROOT/bench/stream
HOST=$(hostname -s 2>/dev/null || hostname)
OUT=$ROOT/bench/results/stream_$HOST.txt
mkdir -p "$DIR" "$ROOT/bench/results"

if [[ ! -s $DIR/stream.c ]]; then
  curl -fsSL -o "$DIR/stream.c" https://www.cs.virginia.edu/stream/FTP/Code/stream.c ||
    curl -fsSL -o "$DIR/stream.c" https://raw.githubusercontent.com/jeffhammond/STREAM/master/stream.c
fi

# Last-level cache size in bytes (total over all instances): largest "Lx cache" in lscpu -B.
llc=$(lscpu -B 2>/dev/null | awk '/^L[1-4][a-z]* cache:/ { if ($3 + 0 > m) m = $3 + 0 } END { print m + 0 }')
if [[ -z $llc || $llc -eq 0 ]]; then llc=$((64 * 1024 * 1024)); fi

# Each of the three arrays >= 4x LLC (8-byte doubles), at least 10M elements, rounded up
# to a multiple of 1M.
elems=$(((4 * llc + 7) / 8))
((elems < 10000000)) && elems=10000000
elems=$(((elems + 999999) / 1000000 * 1000000))

gcc -O3 -march=native -fopenmp -mcmodel=medium \
  -DSTREAM_ARRAY_SIZE="$elems" -DNTIMES=20 "$DIR/stream.c" -o "$DIR/stream"

threads=${OMP_NUM_THREADS:-$(nproc)}
echo "STREAM: LLC=$llc bytes, STREAM_ARRAY_SIZE=$elems ($((elems * 8 / 1024 / 1024)) MiB per array), OMP_NUM_THREADS=$threads" | tee "$OUT"
OMP_NUM_THREADS=$threads OMP_PROC_BIND=${OMP_PROC_BIND:-spread} OMP_PLACES=${OMP_PLACES:-cores} \
  "$DIR/stream" | tee -a "$OUT"

# STREAM reports MB/s = 10^6 bytes/s; convert to GiB/s.
awk '/^(Copy|Scale|Add|Triad):/ { printf "%s %.2f GiB/s (best rate)\n", $1, $2 * 1e6 / 1024 / 1024 / 1024 }' "$OUT" | tee -a "$OUT"
