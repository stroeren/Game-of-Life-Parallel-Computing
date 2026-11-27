#!/usr/bin/env bash
# Step 4: OpenMP schedule x thread-count sweep (docs/omp.md).
#
#   bench/omp_sweep.sh                    omp_byte omp_bitpack omp_simd (those available)
#   bench/omp_sweep.sh omp_bitpack        only that one
#
# Schedules: static, dynamic,<c> for each c in CHUNKS, guided. They are passed through
# OMP_SCHEDULE: the omp backends use schedule(runtime) and default to static when it is
# unset. Threads: 1, 2, 4, ... and finally MAX_THREADS (default nproc), with
# OMP_PROC_BIND=close OMP_PLACES=cores. Each (schedule, threads) point is one
# bench/run.sh call (checksum check against byte, gens for >= MIN_SECONDS, REPS runs), and
# its rows are appended to bench/results/omp/<hostname>.csv with
# schedule,chunk,proc_bind,places,first_touch in front of the usual gol columns.
# bench/summarize_omp.py turns that file into the speedup / efficiency / %STREAM table.
#
# The results go in a subdirectory so bench/summarize.py (results/*.csv) does not mix
# schedules into its per-(impl, n, threads) medians.
#
# Environment: GOL, SIZES, REPS, MIN_SECONDS, SEED, CHECK (passed to run.sh), CHUNKS
# (default "1 16"), MAX_THREADS (default nproc), FIRST_TOUCH (default 1; "1 0" also runs
# every point with serial initialization, for the NUMA first-touch comparison), OUT.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GOL=${GOL:-$ROOT/build/gol}
HOST=$(hostname -s 2>/dev/null || hostname)
OUT=${OUT:-$ROOT/bench/results/omp/$HOST.csv}
CHUNKS=${CHUNKS:-"1 16"}
MAX_THREADS=${MAX_THREADS:-$(nproc)}
FIRST_TOUCH=${FIRST_TOUCH:-1}
export GOL
export OMP_PROC_BIND=close OMP_PLACES=cores

if [[ ! -x $GOL ]]; then
  echo "omp_sweep.sh: $GOL not found; build first (see CLAUDE.md)" >&2
  exit 1
fi

if (($#)); then
  IMPLS=("$@")
else
  IMPLS=()
  available=$("$GOL" --list)
  for impl in omp_byte omp_bitpack omp_simd; do
    if grep -qx "$impl" <<<"$available"; then IMPLS+=("$impl"); fi
  done
fi
if ((${#IMPLS[@]} == 0)); then
  echo "omp_sweep.sh: no omp backends available" >&2
  exit 1
fi

SCHEDULES=(static)
for c in $CHUNKS; do SCHEDULES+=("dynamic,$c"); done
SCHEDULES+=(guided)

THREAD_COUNTS=()
for ((t = 1; t < MAX_THREADS; t *= 2)); do THREAD_COUNTS+=("$t"); done
THREAD_COUNTS+=("$MAX_THREADS")

mkdir -p "$(dirname "$OUT")"
[[ -s $OUT ]] || echo "schedule,chunk,proc_bind,places,first_touch,$("$GOL" --csv-header)" >"$OUT"

tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

echo "host=$HOST out=$OUT impls=${IMPLS[*]} schedules=${SCHEDULES[*]} threads=${THREAD_COUNTS[*]} first_touch=$FIRST_TOUCH" >&2
for ft in $FIRST_TOUCH; do
  for sched in "${SCHEDULES[@]}"; do
    kind=${sched%%,*}
    chunk=0
    if [[ $sched == *,* ]]; then chunk=${sched#*,}; fi
    for t in "${THREAD_COUNTS[@]}"; do
      echo "== OMP_SCHEDULE=$sched threads=$t GOL_OMP_FIRST_TOUCH=$ft" >&2
      : >"$tmp"
      OMP_SCHEDULE=$sched GOL_OMP_FIRST_TOUCH=$ft THREADS=$t OUT=$tmp \
        "$ROOT/bench/run.sh" "${IMPLS[@]}" >/dev/null
      tail -n +2 "$tmp" | sed "s/^/$kind,$chunk,$OMP_PROC_BIND,$OMP_PLACES,$ft,/" >>"$OUT"
    done
  done
done
echo "omp_sweep.sh: wrote $OUT; summarize with bench/summarize_omp.py" >&2
