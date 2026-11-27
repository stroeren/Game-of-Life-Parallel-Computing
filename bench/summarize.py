#!/usr/bin/env python3
"""Median table from bench/results/*.csv (CLAUDE.md: median of 5 runs).

    bench/summarize.py [results.csv ...] [--last K]

Groups rows by (host, impl, n, threads, ranks), where host is the CSV file name
(bench/results/<host>.csv), keeps the last K rows of each group (default 5, i.e. the most
recent bench/run.sh invocation), and prints a Markdown table of median seconds, cells/s
and GiB/s with the speedup over byte on the same host at the same N, and GiB/s as a
fraction of the host's measured bandwidth: STREAM Triad for CPU impls, the GPU copy
kernel for cuda_* (see ceilings.py). Groups whose checksums disagree across their rows
are flagged and get no speedup.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="CSV files (default: bench/results/*.csv)")
    ap.add_argument("--last", type=int, default=5, help="rows per group to use (default 5)")
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(HERE, "results", "*.csv")))
    if not files:
        sys.exit("summarize.py: no CSV files found")

    groups = OrderedDict()
    for path in files:
        host = ceilings.host_of(path)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (host, row["impl"], int(row["n"]), int(row["threads"]), int(row["ranks"]))
                groups.setdefault(key, []).append(row)

    stats = {}
    for key, rows in groups.items():
        rows = rows[-args.last:]
        stats[key] = {
            "runs": len(rows),
            "gens": rows[-1]["gens"],
            "seconds": statistics.median(float(r["seconds"]) for r in rows),
            "cells": statistics.median(float(r["cells_per_sec"]) for r in rows),
            "gib": statistics.median(float(r["gib_per_s"]) for r in rows),
            "consistent": len({(r["gens"], r["checksum"]) for r in rows}) == 1,
            "checksum": rows[-1]["checksum"],
        }

    byte = {(k[0], k[2]): v["cells"] for k, v in stats.items() if k[1] == "byte" and v["consistent"]}

    print("| host | impl | N | threads | ranks | gens | runs | median s | cells/s | GiB/s | "
          "speedup vs byte | % of measured BW |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(stats, key=lambda k: (k[0], k[2], k[1], k[3], k[4])):
        host, impl, n, threads, ranks = key
        s = stats[key]
        if not s["consistent"]:
            speedup = "checksum/gens differ across runs"
        elif (host, n) in byte:
            speedup = f"{s['cells'] / byte[(host, n)]:.2f}x"
        else:
            speedup = "n/a (no byte run)"
        name, peak = ceilings.ceiling(host, impl)
        frac = f"{100 * s['gib'] / peak:.0f}% of {name}" if peak else f"n/a (no {name} for {host})"
        print(f"| {host} | {impl} | {n} | {threads} | {ranks} | {s['gens']} | {s['runs']} | "
              f"{s['seconds']:.3f} | {s['cells']:.3e} | {s['gib']:.2f} | {speedup} | {frac} |")


if __name__ == "__main__":
    main()
