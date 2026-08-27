#!/usr/bin/env python
"""Simulate the catalog across MPI ranks -- the only simulation entry point.

`mpirun -n 1024 python run_mpi.py` launches 1024 copies of this script. Each
copy asks MPI two questions -- which rank am I, and how many of us are there --
takes the corresponding contiguous slice of the source list, and simulates it.
Ranks never communicate except for one gather at the end to print a summary,
and no two ranks write the same file, so nothing here needs MPI-IO or parallel
HDF5. MPI is being used purely as a launcher.

Why contiguous slices rather than round-robin: at ~4M stars the scan law is tens
of GB and memory-mapped, so a rank reading a contiguous block of sources streams
a contiguous region of the file, while round-robin would scatter reads across
the whole thing. Per-source cost varies little, so I/O locality is worth more
than load balancing.

mpi4py is optional, and that fallback is how you run this on a laptop:

    python run_mpi.py --limit 200                    # one process, no MPI
    mpirun -n 8 python run_mpi.py                    # 8 local processes
    srun -n 1024 python run_mpi.py                   # a cluster allocation

Each rank pays a one-off JAX warm-up of roughly 90 seconds (the simulator
recompiles per distinct epoch count, and there are ~200 of them), then runs at a
few ms per source. Give each rank tens of thousands of sources so that warm-up
is amortized; a rank with only a few hundred sources is almost all compilation.

Set OMP_NUM_THREADS=1 in the job script: with hundreds of ranks per node, the
per-rank BLAS threads would otherwise oversubscribe the cores.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


from epochalypse import astrometry as astro
from epochalypse import config as C
from epochalypse.sources import ScanLawStore, SourceCatalog


def mpi_context():
    """(comm, rank, size). Falls back to a single rank when mpi4py is absent."""
    try:
        from mpi4py import MPI
    except ImportError:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def slice_for_rank(n_sources, rank, size):
    """This rank's contiguous [start, stop) of the source list.

    The remainder is spread over the first few ranks so the largest and smallest
    slices differ by at most one source.
    """
    base, extra = divmod(n_sources, size)
    start = rank * base + min(rank, extra)
    stop = start + base + (1 if rank < extra else 0)
    return start, stop


def run_rank(populations, rank, size, *, limit=None, skip_existing=False):
    """Simulate this rank's slice, writing one parquet pair per population."""
    catalog = SourceCatalog()
    scanlaw = ScanLawStore()

    ids = catalog.ids()
    start, stop = slice_for_rank(len(ids), rank, size)
    mine = ids[start:stop]
    if limit:
        mine = mine[:limit]

    results, skipped = [], []
    for population in populations:
        if skip_existing and C.shard_epochs(population, rank, size).exists():
            print(f"[rank {rank:05d}] {population:<14} already done, skipping", flush=True)
            continue

        started = time.time()
        with astro.ShardWriter(population, rank, size) as writer:
            for gaia_id in mine:
                try:
                    epochs, truth = astro.simulate_source(
                        population, gaia_id, catalog=catalog, scanlaw=scanlaw)
                except Exception as error:
                    # One unusable source must not take down a rank of millions.
                    skipped.append({"gaia_source_id": gaia_id,
                                    "population": population, "reason": str(error)})
                    continue
                writer.add(epochs, truth)
        elapsed = time.time() - started
        results.append(writer.n_systems)
        rate = writer.n_systems / elapsed if elapsed else 0
        print(f"[rank {rank:05d}] {population:<14} {writer.n_systems:>8,} systems "
              f"in {elapsed:7.1f} s ({rate:6.1f}/s)", flush=True)

    if skipped:
        import pandas as pd

        path = C.skipped_dir() / f"rank_{rank:05d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped).to_csv(path, index=False)

    return {"rank": rank, "n_systems": sum(results), "n_skipped": len(skipped)}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--populations", nargs="+", choices=list(C.POPULATIONS),
                        default=list(C.POPULATIONS))
    parser.add_argument("--output-root", type=Path,
                        help="write shards here instead of <repo>/outputs")
    parser.add_argument("--limit", type=int,
                        help="cap sources per rank (smoke tests only)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip a (population, rank) whose output already exists, "
                             "so a rerun only redoes the ranks that died")
    args = parser.parse_args(argv)

    comm, rank, size = mpi_context()
    if args.output_root:
        C.set_output_root(args.output_root)

    if rank == 0:
        n_sources = len(SourceCatalog())
        print(f"ranks       : {size}" + ("" if comm else "  (mpi4py not found -- "
                                                         "running as a single rank)"))
        print(f"sources     : {n_sources:,}  ->  ~{n_sources // size:,} per rank")
        print(f"populations : {', '.join(args.populations)}")
        print(f"threads/rank: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}"
              + ("" if os.environ.get("OMP_NUM_THREADS") == "1"
                 else "   <- set this to 1 to avoid oversubscription"), flush=True)

    started = time.time()
    summary = run_rank(args.populations, rank, size,
                       limit=args.limit, skip_existing=args.skip_existing)
    summary["seconds"] = time.time() - started

    all_summaries = comm.gather(summary, root=0) if comm else [summary]
    if rank == 0:
        systems = sum(s["n_systems"] for s in all_summaries)
        skipped = sum(s["n_skipped"] for s in all_summaries)
        print(f"\ndone: {systems:,} systems across {size} rank(s)")
        print(f"  slowest rank : {max(s['seconds'] for s in all_summaries) / 60:.1f} min")
        if skipped:
            print(f"  skipped      : {skipped:,} sources (see {C.skipped_dir()})")
        print("  next         : python scripts/generate_catalog.py "
              "--stages merge select figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
