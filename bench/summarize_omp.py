#!/usr/bin/env python3
"""Table for the OpenMP sweep (bench/omp_sweep.sh -> bench/results/omp/<host>.csv).

    bench/summarize_omp.py [omp_results.csv ...] [--last K]

Groups rows by (host, impl, n, schedule, chunk, first_touch, threads), keeps the last K
rows of each group (default 5) and prints the median cells/s and GiB/s, the speedup over
the same impl/N/schedule at 1 thread, the parallel efficiency (speedup / threads) and GiB/s
as a fraction of the host's STREAM Triad (ceilings.py). A final table gives the best
schedule and thread count per impl and N.
"""
import argparse
import csv
import glob
import os
import statistics
import sys
from collections import OrderedDict

import ceilings

HERE = os.path.dirname(os.path.abspath(__file__))


def sched_label(schedule, chunk):
    return schedule if chunk in ("", "0") else f"{schedule},{chunk}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="CSV files (default: bench/results/omp/*.csv)")
    ap.add_argument("--last", type=int, default=5, help="rows per group to use (default 5)")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(HERE, "results", "omp", "*.csv")))
    if not files:
        sys.exit("summarize_omp.py: no CSV files found (run bench/omp_sweep.sh)")

    groups = OrderedDict()
    for path in files:
        host = ceilings.host_of(path)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (host, row["impl"], int(row["n"]), sched_label(row["schedule"], row["chunk"]),
                       row["first_touch"], int(row["threads"]))
                groups.setdefault(key, []).append(row)

    stats = {}
    for key, rows in groups.items():
        rows = rows[-args.last:]
        stats[key] = {
            "runs": len(rows),
            "cells": statistics.median(float(r["cells_per_sec"]) for r in rows),
            "gib": statistics.median(float(r["gib_per_s"]) for r in rows),
            "consistent": len({(r["gens"], r["checksum"]) for r in rows}) == 1,
        }

    print("| host | impl | N | schedule | first touch | threads | runs | cells/s | GiB/s | "
          "speedup vs 1 thread | efficiency | % of STREAM |")
    print("|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    order = sorted(stats, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5]))
    for key in order:
        host, impl, n, sched, ft, threads = key
        s = stats[key]
        one = stats.get((host, impl, n, sched, ft, 1))
        if not s["consistent"]:
            speedup = eff = "checksum/gens differ"
        elif one and one["consistent"]:
            sp = s["cells"] / one["cells"]
            speedup, eff = f"{sp:.2f}x", f"{100 * sp / threads:.0f}%"
        else:
            speedup = eff = "n/a (no 1-thread run)"
        peak = ceilings.stream_triad(host)
        frac = f"{100 * s['gib'] / peak:.0f}%" if peak else "n/a (no STREAM)"
        print(f"| {host} | {impl} | {n} | {sched} | {ft} | {threads} | {s['runs']} | {s['cells']:.3e} | "
              f"{s['gib']:.2f} | {speedup} | {eff} | {frac} |")

    best = {}
    for key in order:
        if not stats[key]["consistent"]:
            continue
        host, impl, n = key[:3]
        if (host, impl, n) not in best or stats[key]["cells"] > stats[best[(host, impl, n)]]["cells"]:
            best[(host, impl, n)] = key
    print()
    print("| host | impl | N | best schedule | first touch | threads | cells/s | GiB/s |")
    print("|---|---|---:|---|---|---:|---:|---:|")
    for (host, impl, n), key in sorted(best.items()):
        s = stats[key]
        print(f"| {host} | {impl} | {n} | {key[3]} | {key[4]} | {key[5]} | {s['cells']:.3e} | {s['gib']:.2f} |")


if __name__ == "__main__":
    main()
