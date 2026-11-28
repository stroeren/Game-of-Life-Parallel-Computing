#!/usr/bin/env python3
"""Summarize bench/results/mpi_<host>.csv written by bench/mpi_sweep.sh.

Per (N, decomp, mode, kernel, p) the median over repetitions of seconds per generation.
  speedup    = serial seconds/gen (same N, same kernel) / MPI seconds/gen
  efficiency = speedup / p
  vs p=1     = MPI p=1 seconds/gen (same config) / MPI seconds/gen
  compute %, comm % = sum over generations of the per-generation max over ranks of the
                      compute / communication time, as shares of their sum. In nonblocking
                      mode "comm" is the exposed part: posting plus time blocked in Waitall.
Halo table: measured cells sent to other ranks per rank per generation (average over
ranks; a bit-packed word counts as 64 cells) against the guide's model, 2N for strips and
4N/sqrt(p) for blocks.
"""
import csv
import statistics as st
import sys
from collections import defaultdict


def f(x, d=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main(path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        print("no rows in", path)
        return
    serial = defaultdict(list)
    mpi = defaultdict(list)
    for r in rows:
        n, gens, sec = int(r["n"]), int(r["gens"]), float(r["seconds"])
        if r["decomp"] == "serial":
            serial[(n, r["kernel"])].append(sec / gens)
        else:
            mpi[(n, r["decomp"], r["mode"], r["kernel"], int(r["ranks"]))].append(r)

    def med(rs, key):
        return st.median(f(r[key], 0.0) / int(r["gens"]) for r in rs)

    cfgs = defaultdict(dict)
    for (n, decomp, mode, kernel, p), rs in mpi.items():
        spg = st.median(float(r["seconds"]) / int(r["gens"]) for r in rs)
        comp, comm = med(rs, "compute_s"), med(rs, "comm_s")
        sums = {r["checksum"] for r in rs}
        cfgs[(n, kernel, decomp, mode)][p] = dict(
            spg=spg, comp=comp, comm=comm, reps=len(rs), dims=rs[0]["dims"],
            halo=f(rs[0]["halo_cells_per_rank_gen"], 0.0),
            halo_bytes=f(rs[0]["halo_bytes_per_rank_gen"], 0.0),
            model=f(rs[0]["model_cells_per_rank_gen"], 0.0), ok=len(sums) == 1)

    for n, kernel in sorted({(k[0], k[1]) for k in cfgs}):
        base = statistics_median(serial.get((n, kernel)))
        print(f"\n### Strong scaling, N = {n}, kernel = {kernel}\n")
        print("serial", kernel, "baseline:",
              "n/a" if base is None else f"{base * 1e3:.3f} ms/generation "
              f"({n * n / base / 1e9:.3f} Gcell/s)")
        print("\n| decomp | mode | p | grid | ms/gen | speedup | efficiency | vs p=1 | compute % | comm % | reps |")
        print("|---|---|--:|---|--:|--:|--:|--:|--:|--:|--:|")
        for (cn, ck, decomp, mode), byp in sorted(cfgs.items()):
            if (cn, ck) != (n, kernel):
                continue
            p1 = byp.get(1, {}).get("spg")
            for p in sorted(byp):
                d = byp[p]
                sp = base / d["spg"] if base else None
                tot = d["comp"] + d["comm"]
                print(f"| {decomp} | {mode} | {p} | {d['dims']} | {d['spg'] * 1e3:.3f} | "
                      f"{fmt(sp, '{:.2f}')} | {fmt(sp / p if sp else None, '{:.0%}')} | "
                      f"{fmt(p1 / d['spg'] if p1 else None, '{:.2f}')} | "
                      f"{100 * d['comp'] / tot if tot else 0:.0f} | {100 * d['comm'] / tot if tot else 0:.0f} | "
                      f"{d['reps']}{'' if d['ok'] else ' CHECKSUM MISMATCH'} |")

    print("\n### Halo volume per rank per generation (cells sent to other ranks)\n")
    print("| N | kernel | decomp | p | grid | measured cells | model cells | measured / model | measured bytes |")
    print("|--:|---|---|--:|---|--:|--:|--:|--:|")
    seen = set()
    for (n, kernel, decomp, mode), byp in sorted(cfgs.items()):
        for p in sorted(byp):
            if (n, kernel, decomp, p) in seen:
                continue
            seen.add((n, kernel, decomp, p))
            d = byp[p]
            ratio = d["halo"] / d["model"] if d["model"] else 0
            print(f"| {n} | {kernel} | {decomp} | {p} | {d['dims']} | {d['halo']:.0f} | {d['model']:.0f} | "
                  f"{ratio:.2f} | {d['halo_bytes']:.0f} |")


def statistics_median(xs):
    return st.median(xs) if xs else None


def fmt(x, spec):
    return "n/a" if x is None else spec.format(x)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: mpi_summarize.py bench/results/mpi_<host>.csv")
    main(sys.argv[1])
