#!/usr/bin/env python
"""Generate the catalog across MPI ranks -- the cluster entry point.

`mpirun -n 1000 python run_mpi.py` launches 1000 copies of this script. Each
copy asks MPI two questions -- which rank am I, and how many of us are there --
takes the corresponding contiguous slice of the frozen source list, and
simulates it. Ranks never communicate except for one gather at the end to print
a summary, and no two ranks write the same file, so nothing here needs MPI-IO or
parallel HDF5. MPI is being used purely as a launcher.

Why contiguous slices rather than round-robin: at ~4M stars the scan law is tens
of GB and memory-mapped, so a rank reading a contiguous block of sources streams
a contiguous region of the file, while round-robin would scatter reads across
the whole thing. Per-source cost varies little, so I/O locality is worth more
than load balancing.

mpi4py is optional. Without it the script runs as a single rank, which is what
makes it usable on a laptop and for debugging:

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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_catalog import POPULATIONS, build_config     # noqa: E402
from pipeline import astrometry as astro                   # noqa: E402
from pipeline.sources import ScanLawStore, SourceCatalog    # noqa: E402
from simulate_source import CHOICES, resolve_population     # noqa: E402


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


def run_rank(config, populations, rank, size, *, limit=None, skip_existing=False,
             verbose=True):
    """Simulate this rank's slice, writing one parquet part per population."""
    catalog = SourceCatalog(config)
    catalog.verify_checksum()      # every rank proves it holds the same list
    scanlaw = ScanLawStore(config)

    ids = catalog.ids()
    start, stop = slice_for_rank(len(ids), rank, size)
    mine = ids[start:stop]
    if limit:
        mine = mine[:limit]

    results, skipped = [], []
    for spec in populations:
        epochs_path = config.paths.shard_epochs(spec.name, rank, size, tag="rank")
        if skip_existing and epochs_path.exists():
            if verbose:
                print(f"[rank {rank:05d}] {spec.name:<14} already done, skipping",
                      flush=True)
            results.append({"population": spec.name, "rank": rank,
                            "n_systems": 0, "n_epochs": 0, "skipped_existing": True})
            continue

        started = time.time()
        with astro.ShardWriter(config, spec, rank, size, tag="rank") as writer:
            for gaia_id in mine:
                try:
                    epochs, truth = astro.simulate_source(
                        config, spec, gaia_id, catalog=catalog, scanlaw=scanlaw)
                except Exception as error:
                    # One unusable source must not take down a rank of millions.
                    skipped.append({"gaia_source_id": gaia_id,
                                    "population": spec.name, "reason": str(error)})
                    continue
                writer.add(epochs, truth)
        elapsed = time.time() - started
        results.append({"population": spec.name, "rank": rank,
                        "n_systems": writer.n_systems, "n_epochs": writer.n_epochs,
                        "seconds": elapsed})
        if verbose:
            rate = writer.n_systems / elapsed if elapsed else 0
            print(f"[rank {rank:05d}] {spec.name:<14} {writer.n_systems:>8,} systems "
                  f"in {elapsed:7.1f} s ({rate:6.1f}/s)", flush=True)

    if skipped:
        import pandas as pd

        path = config.paths.data_dir / "skipped" / f"rank_{rank:05d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(skipped).to_csv(path, index=False)

    return {"rank": rank, "sources": len(mine), "range": (start, stop),
            "results": results, "n_skipped": len(skipped)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--populations", nargs="+", default=None, choices=CHOICES,
                        help="populations to simulate (default: all generated ones)")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--limit", type=int,
                        help="cap sources per rank (smoke tests only)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="skip a (population, rank) whose output already exists, "
                             "so a rerun only redoes the ranks that died")
    parser.add_argument("--dry-run", action="store_true",
                        help="print each rank's slice and exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    comm, rank, size = mpi_context()
    config = build_config(args.output_root.resolve() if args.output_root else None)

    names = args.populations or [spec.name for spec in config.generated]
    populations = [config.population(resolve_population(n)) for n in names]
    derived = [spec.name for spec in populations if not spec.is_generated]
    if derived:
        raise SystemExit(f"{derived} are derived populations: they are selected from a "
                         "generated population by the `select` stage, not simulated")

    if rank == 0:
        n_sources = len(SourceCatalog(config))
        print(f"ranks       : {size}" + ("" if comm else "  (mpi4py not found -- "
                                                        "running as a single rank)"))
        print(f"sources     : {n_sources:,}  ->  ~{n_sources // size:,} per rank")
        print(f"populations : {', '.join(spec.name for spec in populations)}")
        print(f"threads/rank: OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}"
              + ("" if os.environ.get("OMP_NUM_THREADS") == "1"
                 else "   <- set this to 1 to avoid oversubscription"), flush=True)

    if args.dry_run:
        n_sources = len(SourceCatalog(config))
        start, stop = slice_for_rank(n_sources, rank, size)
        print(f"[rank {rank:05d}] would simulate sources {start:,}..{stop - 1:,} "
              f"({stop - start:,} of them)", flush=True)
        return 0

    astro.configure_jax(config.astrometry)
    started = time.time()
    summary = run_rank(config, populations, rank, size,
                       limit=args.limit, skip_existing=args.skip_existing)
    summary["seconds"] = time.time() - started

    all_summaries = comm.gather(summary, root=0) if comm else [summary]
    if rank == 0:
        systems = sum(r["n_systems"] for s in all_summaries for r in s["results"])
        skipped = sum(s["n_skipped"] for s in all_summaries)
        slowest = max(s["seconds"] for s in all_summaries)
        print(f"\ndone: {systems:,} systems across {size} rank(s)")
        print(f"  slowest rank : {slowest / 60:.1f} min")
        if skipped:
            print(f"  skipped      : {skipped:,} sources (see outputs/data/skipped/)")
        print(f"  next         : python catalog_generation/generate_catalog.py "
              f"--stages merge select figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
