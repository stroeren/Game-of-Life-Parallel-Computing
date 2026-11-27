#!/usr/bin/env bash
# Strong-scaling sweep for gol_mpi (step 6).
#
#   bench/mpi_sweep.sh
#
# For each N in SIZES, decomposition in DECOMPS (strip block), mode in MODES (blocking
# nonblocking) and rank count p in P_LIST (only p <= CORES), it
#   1. times the serial baseline (build/gol --impl $KERNEL) once per N,
#   2. picks gens so one run takes at least MIN_SECONDS (as bench/run.sh does),
#   3. runs REPS repetitions; the first one adds --check (serial byte checksum at the same
#      gens). A configuration whose checksum does not match is dropped, never reported.
# Raw rows (one per run) go to bench/results/mpi_<hostname>.csv. The summary (speedup,
# efficiency, compute/communication split, measured vs modelled halo volume) is printed
# by bench/mpi_summarize.py and saved next to the CSV as .md.
#
# Speedup = serial seconds per generation / MPI seconds per generation, medians of REPS.
# The compute/comm split is the sum over generations of the per-generation max over ranks.
#
# --oversubscribe is NEVER used here (it is for ctest only: timing with more ranks than
# cores measures the scheduler, not the algorithm). A p larger than CORES is skipped. If
# mpirun complains about slots on a machine with hyperthreads, either lower CORES or pass
# MPIRUN_FLAGS="--use-hwthread-cpus" (counts hardware threads as slots; still no oversubscription).
#
# Environment overrides: GOL, GOL_MPI, MPIRUN, MPIRUN_FLAGS (e.g. "--bind-to core"),
# CORES (default nproc), P_LIST, SIZES, DECOMPS, MODES, KERNEL (byte|bitpack), REPS,
# MIN_SECONDS, SEED, OUT.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GOL=${GOL:-$ROOT/build/gol}
GOL_MPI=${GOL_MPI:-$ROOT/build/gol_mpi}
MPIRUN=${MPIRUN:-mpirun}
MPIRUN_FLAGS=${MPIRUN_FLAGS:-}
CORES=${CORES:-$(nproc)}
P_LIST=${P_LIST:-"1 2 4 8 16"}
SIZES=${SIZES:-"4096 16384"}
DECOMPS=${DECOMPS:-"strip block"}
MODES=${MODES:-"blocking nonblocking"}
KERNEL=${KERNEL:-byte}
REPS=${REPS:-5}
MIN_SECONDS=${MIN_SECONDS:-1.0}
SEED=${SEED:-42}
HOST=$(hostname -s 2>/dev/null || hostname)
OUT=${OUT:-$ROOT/bench/results/mpi_$HOST.csv}

for b in "$GOL" "$GOL_MPI"; do
  [[ -x $b ]] || { echo "mpi_sweep.sh: $b not found; build first (cmake --build build)" >&2; exit 1; }
done
if [[ $MPIRUN_FLAGS == *oversubscribe* ]]; then
  echo "mpi_sweep.sh: refusing --oversubscribe for a benchmark (tests only)" >&2
  exit 1
fi

HEADER="impl,n,gens,threads,ranks,seconds,cells_per_sec,grid_bytes,gib_per_s,checksum,decomp,mode,kernel,compute_s,comm_s,halo_cells_per_rank_gen,halo_bytes_per_rank_gen,model_cells_per_rank_gen,dims"
mkdir -p "$(dirname "$OUT")"
[[ -s $OUT ]] || echo "$HEADER" >"$OUT"
ERR=$(mktemp)
trap 'rm -f "$ERR"' EXIT

stat_of() { grep -o "$1=[^ ]*" "$ERR" | head -1 | cut -d= -f2; }
seconds_of() { cut -d, -f6 <<<"$1"; }
meets() { awk -v s="$1" -v m="$MIN_SECONDS" 'BEGIN { exit !(s >= m) }'; }

serial_run() { "$GOL" --impl "$KERNEL" --n "$1" --gens "$2" --seed "$SEED"; }

# mpi_run <p> <extra gol_mpi args...>: CSV row on stdout, stats in $ERR. Non-zero on failure.
mpi_run() {
  local p=$1; shift
  # shellcheck disable=SC2086
  "$MPIRUN" -np "$p" $MPIRUN_FLAGS "$GOL_MPI" "$@" --seed "$SEED" --kernel "$KERNEL" 2>"$ERR"
}

# Smallest gens (geometric growth, as bench/run.sh) for which one run takes >= MIN_SECONDS.
pick_gens() { # runner-function args...: uses "$@" then appends --gens
  local gens=1 row s
  while :; do
    row=$("$@" "$gens") || return 1
    s=$(seconds_of "$row")
    if meets "$s"; then echo "$gens"; return; fi
    gens=$(awk -v g="$gens" -v s="$s" -v m="$MIN_SECONDS" 'BEGIN {
      f = (s > 0) ? 1.2 * m / s : 100; if (f < 2) f = 2; if (f > 100) f = 100;
      printf "%d", int(g * f + 0.999) }')
  done
}

echo "host=$HOST out=$OUT cores=$CORES kernel=$KERNEL sizes=$SIZES p=$P_LIST reps=$REPS" >&2

for n in $SIZES; do
  # ---- serial baseline --------------------------------------------------------------
  sgens=$(pick_gens bash -c '"$0" --impl "$1" --n "$2" --seed "$3" --gens "$4"' "$GOL" "$KERNEL" "$n" "$SEED") || exit 1
  echo "serial $KERNEL N=$n gens=$sgens" >&2
  for ((rep = 1; rep <= REPS; rep++)); do
    row=$(serial_run "$n" "$sgens")
    sec=$(seconds_of "$row")
    echo "$row,serial,-,$KERNEL,$sec,0,0,0,0,1x1" >>"$OUT"
  done

  for decomp in $DECOMPS; do
    for mode in $MODES; do
      for p in $P_LIST; do
        if ((p > CORES)); then
          echo "skip $decomp/$mode p=$p N=$n: p exceeds CORES=$CORES (no oversubscription in benchmarks)" >&2
          continue
        fi
        args=(--n "$n" --decomp "$decomp" --mode "$mode")
        run_cfg() { mpi_run "$p" "${args[@]}" --gens "$1"; }
        if ! gens=$(pick_gens run_cfg); then
          if grep -qi "slots" "$ERR"; then
            echo "skip $decomp/$mode p=$p N=$n: mpirun reports not enough slots (see MPIRUN_FLAGS in the header)" >&2
          else
            echo "skip $decomp/$mode p=$p N=$n: gol_mpi failed:" >&2; head -5 "$ERR" >&2
          fi
          continue
        fi
        echo "$decomp/$mode p=$p N=$n gens=$gens" >&2
        first_sum=""
        valid=1
        rows=()
        for ((rep = 1; rep <= REPS; rep++)); do
          extra=()
          ((rep == 1)) && extra=(--check)
          if ! row=$(mpi_run "$p" "${args[@]}" --gens "$gens" "${extra[@]}"); then
            echo "CHECKSUM/RUN FAILURE $decomp/$mode p=$p N=$n; dropping this configuration" >&2
            head -5 "$ERR" >&2
            valid=0; break
          fi
          sum=$(cut -d, -f10 <<<"$row")
          [[ -z $first_sum ]] && first_sum=$sum
          if [[ $sum != "$first_sum" ]]; then
            echo "CHECKSUM DIFFERS between repetitions for $decomp/$mode p=$p N=$n; dropping" >&2
            valid=0; break
          fi
          rows+=("$row,$decomp,$mode,$KERNEL,$(stat_of compute_s),$(stat_of comm_s),$(stat_of halo_cells_per_rank_gen),$(stat_of halo_bytes_per_rank_gen),$(stat_of model_cells_per_rank_gen),$(stat_of dims)")
        done
        ((valid)) && printf '%s\n' "${rows[@]}" >>"$OUT"
      done
    done
  done
done

SUMMARY=${OUT%.csv}.md
python3 "$ROOT/bench/mpi_summarize.py" "$OUT" | tee "$SUMMARY"
echo "summary written to $SUMMARY" >&2
