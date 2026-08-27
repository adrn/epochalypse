#!/usr/bin/env python
"""Characterize one (population, shard) work unit -- the whole of the compute.

`run_mpi.py` is a loop over `run_unit` below and a `gather` at the end; there is
no other work in this pipeline. Running this script directly is how a single
shard is tested, timed, or redone after a failure:

    python scripts/characterize_shard.py 1_companion 7          # one shard
    python scripts/characterize_shard.py 1_companion 7 --limit 200
    python scripts/characterize_shard.py 1_companion 7 --part 0 --n-parts 4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from epochalypse_periodograms import config as C
from epochalypse_periodograms.grid import frequency_segments, segment_periods
from epochalypse_periodograms.periodogram import characterize_system
from epochalypse_periodograms.shards import ShardReader
from epochalypse_periodograms.writers import (CharacterizationWriter, PowerWriter,
                                              write_period_grid)


def run_unit(population, shard, n_shards, part=0, n_parts=1, *, segments=None,
             limit=None, skip_existing=False, power_mode=None, progress_every=2000,
             verbose=True):
    """Search every system in one work unit; write its two parquet files.

    Returns a summary dict. A system that raises is recorded and skipped rather
    than taken as fatal: one unusable star must not cost a rank its shard, and
    at 17 M systems a per-system exception that happens once in a million still
    happens seventeen times.
    """
    segments = frequency_segments() if segments is None else segments
    periods = segment_periods(segments)
    chars_path = C.chars_shard(population, shard, n_shards, part, n_parts)
    power_path = C.power_shard(population, shard, n_shards, part, n_parts)

    if skip_existing and chars_path.exists():
        if verbose:
            print(f"[{population} {shard:05d}.{part}] already done, skipping", flush=True)
        return {"population": population, "shard": shard, "part": part,
                "n_systems": 0, "n_failed": 0, "skipped": True, "seconds": 0.0}

    started = time.time()
    failures = []
    with ShardReader(population, shard, n_shards) as reader:
        power = PowerWriter(power_path, len(periods), mode=power_mode)
        # `wants` is evaluated for the whole shard at once: the subsample mode
        # is a blake2s per id and doing it per system inside the loop would put
        # a hash in the hot path for no reason.
        wanted = power.wants(reader.truths["gaia_source_id"].to_numpy())
        n_unit = reader.n_systems(part, n_parts)

        with CharacterizationWriter(chars_path, population, shard, reader.truths) as chars, power:
            for count, (index, truth, t, psi, pf, y, yerr) in enumerate(
                    reader.iter_systems(part, n_parts)):
                if limit and count >= limit:
                    break
                try:
                    record, curve = characterize_system(
                        t, psi, pf, y, yerr, truth=truth, segments=segments,
                        want_power=bool(wanted[index]))
                except Exception as error:
                    failures.append({"population": population, "shard": shard,
                                     "shard_row": index,
                                     "gaia_source_id": truth["gaia_source_id"],
                                     "reason": repr(error)})
                    continue
                chars.add(index, record)
                power.add(truth["gaia_source_id"], index, curve)
                if verbose and progress_every and (count + 1) % progress_every == 0:
                    rate = (count + 1) / (time.time() - started)
                    print(f"[{population} {shard:05d}.{part}] {count + 1:,}/{n_unit:,} "
                          f"({rate:.1f}/s)", flush=True)

    if failures:
        import pandas as pd

        path = C.failed_dir() / f"{population}_shard{shard:05d}_part{part:02d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(path, index=False)

    elapsed = time.time() - started
    summary = {"population": population, "shard": shard, "part": part,
               "n_systems": chars.n_systems, "n_power": power.n_systems,
               "n_failed": len(failures), "skipped": False, "seconds": elapsed}
    if verbose:
        rate = chars.n_systems / elapsed if elapsed else 0.0
        print(f"[{population} {shard:05d}.{part}] {chars.n_systems:>7,} systems in "
              f"{elapsed / 60:6.1f} min ({rate:5.1f}/s), {power.n_systems:,} curves stored"
              + (f", {len(failures)} FAILED" if failures else ""), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("population", choices=list(C.POPULATIONS))
    parser.add_argument("shard", type=int)
    parser.add_argument("--n-shards", type=int, default=None,
                        help="defaults to the count in the shard file names")
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--n-parts", type=int, default=1)
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--power", choices=("all", "subsample", "none"), default=None,
                        help=f"which systems keep a raw curve (default {C.POWER_MODE})")
    parser.add_argument("--limit", type=int, help="cap systems (smoke tests only)")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)

    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)
    if args.output_root:
        C.set_output_root(args.output_root)

    n_shards = args.n_shards
    if n_shards is None:
        from epochalypse_periodograms.shards import discover_shards

        _, n_shards = discover_shards(args.population)

    segments = frequency_segments()
    periods = segment_periods(segments)
    write_period_grid(periods)
    print(f"grid: {len(periods):,} trial periods, {periods[0]:.2e} - {periods[-1]:.0f} yr, "
          f"{len(segments)} segments", flush=True)

    run_unit(args.population, args.shard, n_shards, args.part, args.n_parts,
             segments=segments, limit=args.limit, skip_existing=args.skip_existing,
             power_mode=args.power)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
