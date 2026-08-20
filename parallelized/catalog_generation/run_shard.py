#!/usr/bin/env python
"""Simulate one shard of the source list -- the unit of parallel work.

A source belongs to shard ``blake2s(gaia_source_id) % n_shards``, so the
partition needs no coordination: launch the shards in any order, on any number
of machines, restart any one of them, and the result is identical. Each shard
writes its own parquet pair per population and touches no other shard's files.

Usage
-----
    # one shard, locally
    python catalog_generation/run_shard.py --shard 0 --n-shards 512

    # a range of shards with a local process pool
    python catalog_generation/run_shard.py --shards 0-31 --n-shards 512 --workers 8

    # a specific list of sources, ignoring the partition
    python catalog_generation/run_shard.py --id-file my_targets.txt

    # what a scheduler needs (SLURM array, etc.)
    python catalog_generation/run_shard.py --n-shards 512 --print-commands

Requires the per-source indices (`python catalog_generation/build_indices.py`).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_catalog import POPULATIONS, build_config      # noqa: E402

GENERATED = [spec.name for spec in POPULATIONS if spec.is_generated]
from pipeline import astrometry as astro                    # noqa: E402
from pipeline.sources import ScanLawStore, SourceCatalog, select_shard  # noqa: E402


def run_one_shard(shard, n_shards, population_names, output_root, *,
                  id_file=None, limit=None, verbose=True):
    """Simulate every source assigned to `shard`. Returns a per-population summary."""
    config = build_config(output_root)
    astro.configure_jax(config.astrometry)
    catalog = SourceCatalog(config)
    scanlaw = ScanLawStore(config)

    derived = [n for n in population_names if not config.population(n).is_generated]
    if derived:
        raise ValueError(f"{derived} are derived populations: they are selected from a "
                         "generated population by the `select` stage, not simulated")

    if id_file:
        ids = [line.strip() for line in Path(id_file).read_text().split("\n") if line.strip()]
    else:
        ids = select_shard(catalog.ids(), shard, n_shards)
    if limit:
        ids = ids[:limit]

    started = time.time()
    summaries, skipped = [], []
    for name in population_names:
        spec = config.population(name)
        with astro.ShardWriter(config, spec, shard, n_shards) as writer:
            for gaia_id in ids:
                try:
                    epochs, truth = astro.simulate_source(
                        config, spec, gaia_id, catalog=catalog, scanlaw=scanlaw)
                except Exception as error:
                    # One unusable source must not take down a shard of millions.
                    skipped.append({"gaia_source_id": gaia_id, "population": name,
                                    "reason": str(error)})
                    continue
                writer.add(epochs, truth)
        summaries.append(writer.close.__self__ and {
            "population": name, "shard": shard, "n_systems": writer.n_systems,
            "n_epochs": writer.n_epochs})
        if verbose:
            print(f"  [shard {shard:>5}] {name:<14} {writer.n_systems:>7,} systems, "
                  f"{writer.n_epochs:>9,} epochs")

    if skipped:
        skipped_path = config.paths.data_dir / "skipped" / f"shard_{shard:05d}.csv"
        skipped_path.parent.mkdir(parents=True, exist_ok=True)
        import pandas as pd

        pd.DataFrame(skipped).to_csv(skipped_path, index=False)
        if verbose:
            print(f"  [shard {shard:>5}] {len(skipped)} sources skipped -> {skipped_path}")

    return {"shard": shard, "n_sources": len(ids), "populations": summaries,
            "seconds": time.time() - started, "n_skipped": len(skipped)}


def _worker(payload):
    return run_one_shard(**payload)


def parse_shards(text):
    """'0-31' or '0,5,7' or '3' -> a list of shard indices."""
    shards = []
    for piece in str(text).split(","):
        if "-" in piece:
            lo, hi = piece.split("-")
            shards.extend(range(int(lo), int(hi) + 1))
        else:
            shards.append(int(piece))
    return shards


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shard", type=int, help="a single shard index")
    parser.add_argument("--shards", help="a range or list, e.g. 0-31 or 0,4,9")
    parser.add_argument("--n-shards", type=int, help="total shards (default: config)")
    parser.add_argument("--populations", nargs="+", default=None, choices=GENERATED,
                        help="populations to simulate (default: all generated ones). "
                             "The high-SNR populations are derived by selection "
                             "after generation and are never simulated here.")
    parser.add_argument("--id-file", type=Path,
                        help="simulate exactly these source ids, one per line")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="local processes to run shards with")
    parser.add_argument("--limit", type=int, help="cap sources per shard (smoke tests)")
    parser.add_argument("--print-commands", action="store_true",
                        help="print one command per shard and exit (for a scheduler)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args.output_root.resolve() if args.output_root else None)
    n_shards = args.n_shards or config.sharding.n_shards
    # derived (high-SNR) populations are selections over a generated one; they
    # are produced by the `select` stage, not simulated
    populations = args.populations or [spec.name for spec in config.generated]

    if args.print_commands:
        script = Path(__file__).resolve()
        for shard in range(n_shards):
            print(f"{sys.executable} {script} --shard {shard} --n-shards {n_shards}")
        return 0

    if args.id_file:
        shards = [0]
    elif args.shards:
        shards = parse_shards(args.shards)
    elif args.shard is not None:
        shards = [args.shard]
    else:
        raise SystemExit("pass --shard, --shards, --id-file, or --print-commands")

    output_root = args.output_root.resolve() if args.output_root else None
    payloads = [{"shard": s, "n_shards": n_shards, "population_names": populations,
                 "output_root": output_root, "id_file": args.id_file,
                 "limit": args.limit, "verbose": True} for s in shards]

    print(f"shards      : {len(shards)} of {n_shards}")
    print(f"populations : {', '.join(populations)}")
    print(f"workers     : {args.workers}")

    started = time.time()
    if args.workers > 1 and len(payloads) > 1:
        # spawn, not fork: JAX and a forked process do not mix
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            results = pool.map(_worker, payloads)
    else:
        results = [_worker(p) for p in payloads]

    systems = sum(s["n_systems"] for r in results for s in r["populations"] if s)
    skipped = sum(r["n_skipped"] for r in results)
    elapsed = time.time() - started
    print(f"\ndone: {systems:,} systems across {len(shards)} shard(s) "
          f"in {elapsed / 60:.1f} min"
          + (f" ({skipped:,} sources skipped)" if skipped else ""))
    if systems:
        print(f"rate: {systems / max(elapsed, 1e-9):.0f} systems/s with {args.workers} worker(s)"
              f"  ->  extrapolated {3 * 4_000_000 / max(systems / elapsed, 1e-9) / 3600:.1f} h "
              "for 3 populations x 4M stars at this rate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
