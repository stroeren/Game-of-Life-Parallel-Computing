"""Measured bandwidth ceilings per host, read from bench/results/.

CPU impls are compared with STREAM Triad (bench/stream.sh -> results/stream_<host>.txt),
GPU impls (cuda_*) with the device copy kernel (bench/gpu_copy.sh ->
results/gpucopy_<host>.txt). Both are base-1024 GiB/s, the same unit as gib_per_s.

The ceilings are DRAM numbers. A grid that fits in the last-level cache (CPU L3, GPU L2)
can run above 100% of them; that means the run is cache-resident, not that the model is
wrong.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")


def _first_float(path, pattern):
    try:
        with open(path) as f:
            for line in f:
                m = re.match(pattern, line)
                if m:
                    return float(m.group(1))
    except OSError:
        pass
    return None


def stream_triad(host):
    """STREAM Triad GiB/s for host, or None if bench/stream.sh has not been run there."""
    return _first_float(os.path.join(RESULTS, f"stream_{host}.txt"), r"^Triad:\s+([0-9.]+) GiB/s")


def gpu_copy(host):
    """GPU copy kernel GiB/s for host, or None if bench/gpu_copy.sh has not been run there."""
    return _first_float(os.path.join(RESULTS, f"gpucopy_{host}.txt"), r"^GPU copy kernel:\s+([0-9.]+) GiB/s")


def is_gpu(impl):
    return impl.startswith("cuda")


def ceiling(host, impl):
    """(name, GiB/s or None) of the measured ceiling impl's gib_per_s is compared with."""
    if is_gpu(impl):
        return "GPU copy kernel", gpu_copy(host)
    return "STREAM Triad", stream_triad(host)


def host_of(csv_path):
    """bench/results/<host>.csv -> <host>."""
    return os.path.splitext(os.path.basename(csv_path))[0]
