#!/usr/bin/env bash
# Step 5: cuda_shared block-size sweep (docs/cuda.md).
#
#   bench/cuda_block_sweep.sh
#
# Runs cuda_shared once per block size in BLOCKS (default 16x16 32x8 32x16 32x32, chosen
# with GOL_CUDA_BLOCK) through bench/run.sh (checksum check, gens for >= MIN_SECONDS, REPS
# runs) and appends to bench/results/cuda_blocks/<hostname>.csv with a `block` column in
# front of the usual gol columns. Then prints the median cells/s per (N, block) and the
# best block per N. Put the overall winner in kDefaultBlock in src/cuda/cuda_shared.cu.
#
# Environment: GOL, SIZES, REPS, MIN_SECONDS, SEED, CHECK (passed to run.sh), BLOCKS, OUT.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GOL=${GOL:-$ROOT/build/gol}
HOST=$(hostname -s 2>/dev/null || hostname)
OUT=${OUT:-$ROOT/bench/results/cuda_blocks/$HOST.csv}
BLOCKS=${BLOCKS:-"16x16 32x8 32x16 32x32"}
export GOL

if [[ ! -x $GOL ]]; then
  echo "cuda_block_sweep.sh: $GOL not found; build first (see CLAUDE.md)" >&2
  exit 1
fi
if ! grep -qx cuda_shared <<<"$("$GOL" --list)"; then
  echo "cuda_block_sweep.sh: cuda_shared is not available (no CUDA build or no GPU)" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT")"
[[ -s $OUT ]] || echo "block,$("$GOL" --csv-header)" >"$OUT"
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

for block in $BLOCKS; do
  echo "== GOL_CUDA_BLOCK=$block" >&2
  : >"$tmp"
  GOL_CUDA_BLOCK=$block OUT=$tmp "$ROOT/bench/run.sh" cuda_shared >/dev/null
  tail -n +2 "$tmp" | sed "s/^/$block,/" >>"$OUT"
done

# Median cells/s of the last REPS rows per (n, block), and the best block per n.
python3 - "$OUT" "${REPS:-5}" <<'EOF'
import csv, statistics, sys
from collections import OrderedDict
path, last = sys.argv[1], int(sys.argv[2])
groups = OrderedDict()
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        groups.setdefault((int(row["n"]), row["block"]), []).append(float(row["cells_per_sec"]))
best = {}
print("| N | block | median cells/s |")
print("|---:|---|---:|")
for (n, block), vals in sorted(groups.items()):
    med = statistics.median(vals[-last:])
    print(f"| {n} | {block} | {med:.3e} |")
    if n not in best or med > best[n][1]:
        best[n] = (block, med)
for n, (block, med) in sorted(best.items()):
    print(f"best at N={n}: {block} ({med:.3e} cells/s)")
EOF
