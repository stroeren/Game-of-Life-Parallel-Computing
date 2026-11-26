#!/usr/bin/env bash
# Benchmark registered backends (CLAUDE.md "Benchmarks").
#
#   bench/run.sh                 every backend in `gol --list`
#   bench/run.sh byte bitpack    only those
#
# For each impl and N it first verifies the checksum against byte (--check, 10 gens), then
# picks gens so one run takes at least MIN_SECONDS, then appends REPS rows to
# bench/results/<hostname>.csv. Calibration runs are not recorded.
#
# Environment overrides: GOL (binary), SIZES, REPS, MIN_SECONDS, SEED, THREADS, OUT,
# CHECK=0 to skip the checksum check.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GOL=${GOL:-$ROOT/build/gol}
SIZES=${SIZES:-"1024 4096 16384"}
REPS=${REPS:-5}
MIN_SECONDS=${MIN_SECONDS:-1.0}
SEED=${SEED:-42}
THREADS=${THREADS:-}
CHECK=${CHECK:-1}
HOST=$(hostname -s 2>/dev/null || hostname)
OUT=${OUT:-$ROOT/bench/results/$HOST.csv}

if [[ ! -x $GOL ]]; then
  echo "run.sh: $GOL not found; build first (see CLAUDE.md)" >&2
  exit 1
fi

if (($#)); then
  IMPLS=("$@")
else
  mapfile -t IMPLS < <("$GOL" --list)
fi

mkdir -p "$(dirname "$OUT")"
[[ -s $OUT ]] || "$GOL" --csv-header >"$OUT"

run_once() { # impl n gens -> CSV row
  "$GOL" --impl "$1" --n "$2" --gens "$3" --seed "$SEED" ${THREADS:+--threads "$THREADS"}
}

seconds_of() { cut -d, -f6 <<<"$1"; }

# Smallest gens (found by geometric growth) for which one run takes >= MIN_SECONDS.
pick_gens() { # impl n
  local gens=1 row s
  while :; do
    row=$(run_once "$1" "$2" "$gens")
    s=$(seconds_of "$row")
    if awk -v s="$s" -v m="$MIN_SECONDS" 'BEGIN { exit !(s >= m) }'; then
      echo "$gens"
      return
    fi
    # Aim 20% past the target; grow at least 2x, at most 100x per probe.
    gens=$(awk -v g="$gens" -v s="$s" -v m="$MIN_SECONDS" 'BEGIN {
      f = (s > 0) ? 1.2 * m / s : 100; if (f < 2) f = 2; if (f > 100) f = 100;
      printf "%d", int(g * f + 0.999) }')
  done
}

echo "host=$HOST out=$OUT impls=${IMPLS[*]} sizes=$SIZES reps=$REPS" >&2
for impl in "${IMPLS[@]}"; do
  for n in $SIZES; do
    if [[ $CHECK == 1 && $impl != byte ]]; then
      if ! "$GOL" --impl "$impl" --n "$n" --gens 10 --seed "$SEED" ${THREADS:+--threads "$THREADS"} --check >/dev/null; then
        echo "run.sh: $impl N=$n checksum does not match byte; not benchmarking it" >&2
        continue
      fi
    fi
    gens=$(pick_gens "$impl" "$n")
    echo "$impl N=$n gens=$gens" >&2
    for ((rep = 1; rep <= REPS; rep++)); do
      row=$(run_once "$impl" "$n" "$gens")
      echo "$row" | tee -a "$OUT"
    done
  done
done
