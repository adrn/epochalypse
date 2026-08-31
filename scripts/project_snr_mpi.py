#!/usr/bin/env python
"""Detectable SNR for every system in the catalog (MPI).

    mpirun python scripts/project_snr_mpi.py --catalog-root $OUT_ROOT

Writes `$OUT_ROOT/detectability/<population>/*.parquet`, one row per system:
`retained_k`, `snr_detectable_k` and `snr_expected_k` for each companion `k`,
joined onto the harv analysis on `gaia_source_id` by
`harv.census.with_detectability`.

WHY THIS EXISTS. `snr_total` is not a detectability measure. Position, proper
motion and parallax are free parameters, so whatever part of a companion's
signal those five columns reproduce is subtracted along with them and can never
be detected. Measured over the real catalog, the median orbit keeps only ~60% of
its amplitude and 18.6% of the nominally high-SNR sample keeps under 25% --
with one measured case at 5.4%, where a recorded `snr_total` of 21.5 is a
detectable 1.85. Every recovery figure was binning on that. See `SNR.md`.

THREE QUANTITIES, and they answer different questions:

  snr_total       what the catalog records: plausibly observable
  snr_detectable  exact, this system, this orientation: observable HERE. The
                  right axis for "given a signal this strong is present, does
                  the fit find it?"
  snr_expected    marginalized over the inclination, node, argument of
                  periastron and phase that no real survey knows: observable IN
                  EXPECTATION. The right input to an occurrence-rate correction,
                  which cannot condition on the true orientation.

COST. One reflex reconstruction and one 5-column least-squares per companion:
~50 systems/s per core, so ~95 core-hours over 17.2 M systems. Small against
harv's ~2,100. Per-system cost is flat, so `mpi.slice_for_rank` is right here
and the cost-aware `mpi.balance` harv needs is not.

`snr_expected` would be ~20x that if drawn per system. It is not: `E[retained]`
is a property of the ORBIT, not of the star -- `check_snr.py --across-stars`
measures the star-to-star spread at 0.8-6.5%, no larger than its own Monte Carlo
noise -- so rank 0 builds one `(log10 P/T, e)` table, broadcasts it, and every
rank interpolates. **Rerun `--across-stars` on any new catalog before trusting
that**: the table stands or falls on the scan law, and the measurement behind it
was taken on uniformly distributed scan angles.

FOR A FUTURE CATALOG THIS BELONGS IN `simulate_mpi.py`. The reflex is already in
hand there -- it is what gets added to the astrometric model -- so the marginal
cost of the projection is a single least-squares per system and nothing has to
be read back off disk. This standalone stage exists to backfill a catalog that
already exists; do not rebuild it from scratch if the generator gains the column.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from epochalypse import detectability as D
from epochalypse import mpi
from epochalypse.periodogram import config as PG
from epochalypse.periodogram.shards import ShardReader, work_units
from epochalypse.shardio import BufferedParquetWriter

FLUSH_EVERY = 5000


def output_path(population, shard, n_shards):
    root = PG.CATALOG_ROOT / "detectability" / population
    return root / f"detectability_{PG._piece(shard, n_shards, 0, 1)}.parquet"


class DetectabilityWriter(BufferedParquetWriter):
    """One row per system: the ids, and the per-companion projections."""

    def _table(self, rows):
        return pa.Table.from_pandas(
            pd.DataFrame.from_records(rows), preserve_index=False
        )


def run_unit(population, shard, n_shards, table, skip_existing=False, verbose=True):
    """Project every system in one shard. Returns a summary dict."""
    path = output_path(population, shard, n_shards)
    if skip_existing and path.exists():
        return {"population": population, "shard": shard, "n": 0, "skipped": True}

    n_companions = PG.POPULATIONS[population]
    started = time.time()
    failures = []
    with (
        ShardReader(population, shard, n_shards) as reader,
        DetectabilityWriter(path, FLUSH_EVERY, PG.PARQUET_COMPRESSION) as writer,
    ):
        for index, truth, t, psi, pf, _y, yerr in reader.iter_systems():
            # One unusable star must not cost a rank its shard. `harv.unit`
            # learned this already -- at 17.2 M systems a once-in-a-million
            # exception still happens seventeen times -- and this stage was
            # written without it, so a single bad uncertainty took down a
            # production rank and every shard it had left.
            try:
                record = {
                    "gaia_source_id": int(truth["gaia_source_id"]),
                    "shard": int(shard),
                    "shard_row": int(index),
                    "n_epochs": len(t),
                }
                sigma_single = float(truth["sigma_single_mas"])
                for j, reflex in enumerate(
                    D.per_companion_reflex(truth, t, psi, pf, n_companions), start=1
                ):
                    snr, retained = D.snr_detectable(
                        reflex, t, psi, pf, yerr, sigma_single
                    )
                    record[f"retained_{j}"] = retained
                    record[f"snr_detectable_{j}"] = snr
                    expected = D.expected_retained(
                        table, float(truth[f"period_{j}"]), float(truth[f"ecc_{j}"])
                    )
                    record[f"snr_expected_{j}"] = float(
                        np.sqrt(len(t))
                        * expected
                        * float(truth[f"alpha_mas_{j}"])
                        / sigma_single
                    )
            except Exception as error:  # noqa: BLE001
                failures.append(
                    {
                        "population": population,
                        "shard": shard,
                        "shard_row": index,
                        "gaia_source_id": int(truth["gaia_source_id"]),
                        "reason": repr(error),
                    }
                )
                continue
            writer.add(record)
        n_systems = writer.n_rows

    if failures:
        failed = PG.CATALOG_ROOT / "detectability" / "failed"
        failed.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(
            failed / f"{population}_shard{shard:05d}.csv", index=False
        )

    elapsed = time.time() - started
    if verbose:
        print(
            f"[{population} {shard:05d}] {n_systems:>7,} systems in "
            f"{elapsed / 60:6.1f} min ({n_systems / max(elapsed, 1e-9):5.1f}/s)"
            + (f", {len(failures)} FAILED" if failures else ""),
            flush=True,
        )
    return {
        "population": population,
        "shard": shard,
        "n": n_systems,
        "n_failed": len(failures),
        "skipped": False,
        "seconds": elapsed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument(
        "--populations",
        nargs="+",
        choices=list(PG.POPULATIONS),
        default=list(PG.POPULATIONS),
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-units", type=int)
    parser.add_argument(
        "--table-stars",
        type=int,
        default=40,
        help="real stars averaged into the orientation table (default 40); the "
        "star-to-star spread is a few percent, so this converges fast",
    )
    parser.add_argument(
        "--table-draws",
        type=int,
        default=40,
        help="orientations per table cell (default 40)",
    )
    args = parser.parse_args(argv)

    if args.catalog_root:
        PG.set_catalog_root(args.catalog_root)

    comm, rank, size = mpi.mpi_context()
    units = [
        (population, shard, n_shards)
        for population, shard, n_shards, _part, _n_parts in work_units(
            args.populations, 1
        )
    ]
    if args.max_units:
        units = units[: args.max_units]

    # Rank 0 builds the orientation table and broadcasts it. It is a
    # property of the scan law, identical for every rank, and takes a minute
    # -- so building it once beats building it 1,536 times.
    table = None
    if rank == 0:
        started = time.time()
        stars = D.sample_stars(args.populations[-1], args.table_stars)
        table = D.retained_table(stars, n_draws=args.table_draws)
        print(
            f"orientation table: {table.shape[0]}x{table.shape[1]} cells from "
            f"{len(stars)} stars x {args.table_draws} draws in "
            f"{time.time() - started:.0f} s",
            flush=True,
        )
    table = mpi.broadcast(comm, table, root=0)

    if rank == 0:
        mpi.banner(
            comm,
            size,
            len(units),
            item="shards",
            catalog=PG.CATALOG_ROOT,
            populations=", ".join(args.populations),
            table=f"E[retained] over {D.TABLE_LOG_RATIO.size} x {D.TABLE_ECC.size}"
            " in (log10 P/T, e)",
        )

    start, stop = mpi.slice_for_rank(len(units), rank, size)
    summaries = [
        run_unit(*units[i], table, skip_existing=args.skip_existing)
        for i in range(start, stop)
    ]
    gathered = mpi.gather(comm, summaries)

    if rank == 0:
        # gather returns one summary LIST per rank, not one summary
        total = sum(item["n"] for group in gathered for item in group)
        print(f"\ndone: {total:,} systems across {size} rank(s)")
        print(f"  written to {PG.CATALOG_ROOT / 'detectability'}")
        print("  next: harv_finish.py --stages figures  (now binned on it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
