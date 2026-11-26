#!/usr/bin/env python3
"""Median table from bench/results/*.csv (CLAUDE.md: median of 5 runs).

    bench/summarize.py [results.csv ...] [--last K]

Groups rows by (impl, n, threads, ranks), keeps the last K rows of each group (default 5,
i.e. the most recent bench/run.sh invocation), and prints a Markdown table of median
seconds, cells/s and GiB/s with the speedup over byte at the same N. Groups whose
checksums disagree across their rows are flagged and get no speedup.
"""
import argparse
import csv
import glob
import os
import statistics
import sys
from collections import OrderedDict

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
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["impl"], int(row["n"]), int(row["threads"]), int(row["ranks"]))
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

    byte = {k[1]: v["cells"] for k, v in stats.items() if k[0] == "byte" and v["consistent"]}

    print("| impl | N | threads | ranks | gens | runs | median s | cells/s | GiB/s | speedup vs byte |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key in sorted(stats, key=lambda k: (k[1], k[0], k[2], k[3])):
        impl, n, threads, ranks = key
        s = stats[key]
        if not s["consistent"]:
            speedup = "checksum/gens differ across runs"
        elif n in byte:
            speedup = f"{s['cells'] / byte[n]:.2f}x"
        else:
            speedup = "n/a (no byte run)"
        print(f"| {impl} | {n} | {threads} | {ranks} | {s['gens']} | {s['runs']} | {s['seconds']:.3f} | "
              f"{s['cells']:.3e} | {s['gib']:.2f} | {speedup} |")


if __name__ == "__main__":
    main()
