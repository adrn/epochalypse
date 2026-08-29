#!/usr/bin/env python
"""Time the harv fit, one system at a time, so settings can be compared.

Answers three questions that the production driver cannot, because a work unit
there is hours long and reports only its total:

1. **Which CPU?** Run it on one exclusive node per architecture. Every rank
   fits the *same* systems in the same order, so the numbers are directly
   comparable across `-C genoa` / `-C icelake` / `-C rome`.
2. **How much does packing a node cost?** Same script, same node, different
   `--ntasks-per-node`. Every rank does identical work, so per-rank time versus
   the one-rank baseline *is* the contention factor -- and the aggregate line
   says whether more ranks still buys throughput.
3. **Where does the time go?** The first system carries the XLA compile and is
   reported separately from the warm ones.

    mpirun python scripts/bench_harv.py --catalog-root $OUT_ROOT
    mpirun python scripts/bench_harv.py --n-prior-samples 100000 --batch-size 10000

The last line of output converts the measured rate straight into core-hours for
the whole catalog, which is the number that decides whether a setting is
affordable.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np

from epochalypse import mpi
from epochalypse.harv import config as C
from epochalypse.harv import library as L
from epochalypse.harv.unit import fit_system  # also enables x64, quiets harv
from epochalypse.periodogram.shards import ShardReader, discover_shards


def cpu_model():
    """The CPU's own name for itself, so a log says which node it ran on."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog-root", type=Path, help="the catalog to read")
    parser.add_argument(
        "--population", default="1_companion", choices=list(C.POPULATIONS)
    )
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument(
        "--n-systems", type=int, default=5, help="systems to time (the first is cold)"
    )
    parser.add_argument("--n-prior-samples", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--sigma-a0",
        type=float,
        default=None,
        help="pin sigma_a0 rather than deriving it from each host's mass",
    )
    args = parser.parse_args(argv)

    comm, rank, size = mpi.mpi_context()
    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)
    if args.n_prior_samples is not None:
        C.N_PRIOR_SAMPLES = args.n_prior_samples
    if args.top_k is not None:
        C.TOP_K = args.top_k
    if args.batch_size is not None:
        C.BATCH_SIZE = args.batch_size
    if args.sigma_a0 is not None:
        C.SIGMA_A0_AU = args.sigma_a0

    # Every rank under `mpirun` believes it is rank 0 of 1 when mpi4py is
    # missing, which silently turns a contention measurement into N copies of
    # the one-rank case. SLURM knows the truth, so compare.
    allocated = int(os.environ.get("SLURM_NTASKS") or size)
    if size != allocated:
        print(
            f"*** MPI reports {size} rank(s) but SLURM_NTASKS={allocated} -- "
            "mpi4py is probably missing, so these ranks are not talking. "
            "The aggregate line below is wrong.",
            flush=True,
        )

    _, n_shards = discover_shards(args.population)
    if rank == 0:
        per_node = os.environ.get("SLURM_NTASKS_PER_NODE", "?")
        n_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", "1"))
        print(f"host        : {platform.node()}  ({cpu_model()})")
        print(f"ranks       : {size}  ({per_node} per node, {n_nodes} node(s))")
        print(
            f"settings    : M={C.N_PRIOR_SAMPLES:,}  top_k={C.TOP_K}  "
            f"batch_size={C.BATCH_SIZE:,}"
        )
        print(
            f"work        : {args.population} shard {args.shard}, first "
            f"{args.n_systems} systems -- identical on every rank",
            flush=True,
        )

    started = time.time()
    prior_samples = L.draw()
    library_seconds = time.time() - started
    # Per system unless pinned, exactly as production does it.
    prior = L.prior() if C.SIGMA_A0_AU is not None else None

    timings, buckets = [], []
    with ShardReader(args.population, args.shard, n_shards) as reader:
        systems_per_shard = len(reader.truths)
        for count, (_, truth, t, psi, pf, y, yerr) in enumerate(reader.iter_systems()):
            if count >= args.n_systems:
                break
            t0 = time.time()
            record, _ = fit_system(
                t,
                psi,
                pf,
                y,
                yerr,
                prior=prior,
                prior_samples=prior_samples,
                seed=1,
                m_star_msun=truth["mass_st_msun"],
            )
            elapsed = time.time() - t0
            timings.append(elapsed)
            buckets.append((len(t), record["n_padded"]))
            if rank == 0:
                tag = "compile+run" if count == 0 else "warm       "
                print(
                    f"system {count:>3}  : N={len(t):>3} bucket={record['n_padded']:>3}  "
                    f"{tag}  {elapsed:7.2f} s",
                    flush=True,
                )

    warm = timings[1:] or timings  # the first carries the XLA compile
    mine = {
        "rank": rank,
        "library_seconds": library_seconds,
        "first_seconds": timings[0],
        "warm_median": float(np.median(warm)),
        "warm_total": float(np.sum(warm)),
        "n_warm": len(warm),
    }
    everyone = mpi.gather(comm, mine)

    if rank == 0:
        medians = np.array([s["warm_median"] for s in everyone])
        firsts = np.array([s["first_seconds"] for s in everyone])
        libraries = np.array([s["library_seconds"] for s in everyone])
        # Every rank fitted the same systems, so the slowest rank sets the pace a
        # real run would see: throughput is n_ranks x n_warm / slowest.
        slowest = max(s["warm_total"] for s in everyone)
        throughput = size * mine["n_warm"] / slowest
        n_nodes = max(int(os.environ.get("SLURM_JOB_NUM_NODES", "1")), 1)

        print(f"\nlibrary     : {libraries.mean():6.1f} s   (drawn once per rank)")
        print(
            f"compile     : {firsts.mean() - medians.mean():+6.1f} s   "
            "(first system minus the warm median)"
        )
        print("\n--- warm seconds per system, per rank ---")
        print(
            f"  min {medians.min():7.2f}   median {np.median(medians):7.2f}   "
            f"max {medians.max():7.2f}"
        )
        print("\n--- aggregate ---")
        print(
            f"  throughput  : {throughput:7.3f} systems/s over {size} rank(s)  "
            f"=  {throughput / n_nodes:7.3f} per node"
        )
        # The number that decides whether a setting is affordable. Derived from
        # this catalog's own shard size, so it follows a 250 pc / 500 pc switch.
        catalog = systems_per_shard * n_shards * len(C.POPULATIONS)
        core_hours = catalog * np.median(medians) / 3600.0
        print(
            f"  catalog     : {catalog:,} systems  ->  {core_hours:,.0f} core-h "
            f"at this rate"
        )
        print(
            f"                {core_hours / 24:,.0f} cores for 24 h, or "
            f"{core_hours / 12:,.0f} for 12 h"
        )
        buckets_seen = sorted({b for _, b in buckets})
        print(
            f"\n  NOTE: timed only bucket(s) {buckets_seen}. Cost scales with the\n"
            "  padded epoch count, so the catalog projection above is only right\n"
            "  if these systems are typical. Compare architectures with the same\n"
            "  --population/--shard, which fits the same systems every time."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
