#!/usr/bin/env python
"""Search one star and print the result -- for inspection, not for production.

    python scripts/periodogram_source.py 1_companion 5484066448309985152
    python scripts/periodogram_source.py 1_companion 5484066448309985152 --plot out.pdf

Finds the source in the shards by its id (a scan of 320 truth shards' id column,
a few seconds), runs the same `characterize_system` the production run calls, and
prints the record beside the injected truth. This is the thing to reach for when
a row in the output table looks wrong: it is the same code path, on one star,
with the curve available to look at.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from epochalypse.periodogram import config as C
from epochalypse.periodogram.grid import frequency_segments, segment_periods
from epochalypse.periodogram.periodogram import characterize_system
from epochalypse.periodogram.shards import ShardReader, discover_shards


def locate(population, gaia_source_id):
    """`(shard, n_shards, shard_row)` of one source, by scanning the truth shards.

    Only the id column of each truth shard is read -- 17,890 values per file, so
    the whole scan is a few tens of MB. There is a lookup index in the generated
    catalog (`data/index/`) that would answer this without a scan, but it maps
    ids to *scan-law* rows, not to shard positions, and building a second index
    to save five seconds in a debugging tool is not worth the file.
    """
    numbers, n_shards = discover_shards(population)
    target = str(int(gaia_source_id))
    for shard in numbers:
        ids = pd.read_parquet(C.shard_truths(population, shard, n_shards),
                              columns=["gaia_source_id"])["gaia_source_id"].to_numpy()
        hit = np.flatnonzero(ids.astype(str) == target)
        if hit.size:
            return shard, n_shards, int(hit[0])
    raise KeyError(f"{gaia_source_id} is not in {population}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("population", choices=list(C.POPULATIONS))
    parser.add_argument("gaia_source_id", type=int)
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--plot", type=Path, help="save the Delta-chi^2 curve here")
    args = parser.parse_args(argv)

    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)

    shard, n_shards, row = locate(args.population, args.gaia_source_id)
    print(f"{args.gaia_source_id} -> shard {shard:05d} of {n_shards:05d}, row {row}\n")

    segments = frequency_segments()
    periods = segment_periods(segments)
    with ShardReader(args.population, shard, n_shards) as reader:
        truth = reader.truths.iloc[row]
        # iter_systems walks the part containing this row; ask for the single
        # row rather than filtering afterwards, so only its row group is read.
        n = len(reader.truths)
        for index, tr, t, psi, pf, y, yerr in reader.iter_systems(row, n):
            break

    record, power = characterize_system(t, psi, pf, y, yerr, truth=truth,
                                        segments=segments, want_power=True)

    print("injected truth")
    for key in ("n_transits_dr4", "parallax_mas", "mass_st_msun", "sigma_single_mas",
                "n_planets", "period_1", "mass_pl_1", "ecc_1", "inc_1", "alpha_mas_1",
                "snr_total_1", "period_2", "mass_pl_2", "snr_total_2"):
        if key in truth.index and pd.notna(truth[key]):
            print(f"  {key:22s} {truth[key]}")
    print("\ncharacterization")
    for key, value in record.items():
        print(f"  {key:22s} {value}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(periods, np.clip(power, 1e-3, None), lw=0.8, color="#332288")
        for k in (1, 2):
            if f"period_{k}" in truth.index and pd.notna(truth[f"period_{k}"]):
                ax.axvline(float(truth[f"period_{k}"]), color="#2e6f95", lw=1.4)
        ax.axvline(C.DR4_BASELINE_YEARS, color="k", ls="--", lw=1.2)
        ax.set(xscale="log", yscale="log", xlabel="period [yr]",
               ylabel=r"$\Delta\chi^2$", xlim=(periods[0], periods[-1]),
               title=f"{args.population}  {args.gaia_source_id}")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=200)
        print(f"\nwrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
