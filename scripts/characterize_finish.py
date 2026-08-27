#!/usr/bin/env python
"""Post-processing: calibrate the thresholds, count the classes, merge the shards.

Serial, cheap, and none of it needs a cluster -- every stage reads a handful of
columns out of the parquet dataset rather than the whole table.

    python scripts/finish.py --stages calibrate census
    python scripts/finish.py --stages merge --populations 1_companion
    python scripts/finish.py --stages subsample-cutoff

Stages:

  calibrate         thresholds from `0_companion` at TARGET_FP -> calibration.json
  census            class counts per population, and for the high-SNR subsets
  merge             each population's 320 shards -> one parquet, plus the
                    high-SNR views (small enough to be worth materializing)
  subsample-cutoff  re-derive `config.SUBSAMPLE_RANK_CUTOFF` from the merged
                    truth table; only needed if the parent sample, the seed or
                    the subsample size changes
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from epochalypse.periodogram import calibrate as cal
from epochalypse.periodogram import config as C

STAGES = ("calibrate", "census", "merge", "subsample-cutoff")

# Columns worth keeping in a merged file: everything the three paper figures
# read, and nothing that only exists to trace a row back to its epochs.


def stage_calibrate(args):
    calibration = cal.calibrate(target_fp=None)
    path = cal.write_calibration(calibration)
    print(f"thresholds from {calibration['n_null_systems']:,} companion-free systems "
          f"@ FP={calibration['target_fp']:.1%}")
    print(f"  orbit Delta-chi2 > {calibration['thr_orbit']:10.1f}   "
          f"(null rate {calibration['realized_fp_peak']:.3%})")
    print(f"  accel Delta-chi2 > {calibration['thr_accel']:10.1f}   "
          f"(null rate {calibration['realized_fp_accel']:.3%})")
    print(f"  either channel   : {calibration['realized_fp']:.3%} of the null")
    print(f"-> {path}")
    return calibration


def stage_census(args):
    calibration = cal.read_calibration()
    print(f"{'population':<26}{'n':>12}{'undetected':>14}{'localized':>12}{'not localized':>16}")
    for population in args.populations:
        counts = cal.census(population, calibration["thr_orbit"], calibration["thr_accel"])
        for label, key in (("", "all"), (" (high-SNR)", "high_snr")):
            if key not in counts:
                continue
            c = counts[key]
            print(f"{population + label:<26}{c['n']:>12,}{c['undet']:>14,}"
                  f"{c['narrow']:>12,}{c['broad']:>16,}")


def stage_merge(args):
    for population in args.populations:
        for high_snr in (False, True):
            if high_snr and C.N_COMPANIONS[population] == 0:
                continue
            path, n = cal.merge(population, high_snr=high_snr, columns=None)
            size = path.stat().st_size / 1e9
            print(f"  {path.name:<52} {n:>10,} rows  {size:6.2f} GB", flush=True)


def stage_subsample_cutoff(args):
    """Re-derive the rank quantile `writers.in_paper_subsample` compares against."""
    from epochalypse.periodogram.writers import source_id_ranks

    merged = C.catalog_data_dir() / "injected_solutions_0_companion.parquet"
    ids = pd.read_parquet(merged, columns=["gaia_source_id"])["gaia_source_id"].to_numpy()
    ranks = np.sort(source_id_ranks(ids))
    cutoff = int(ranks[C.SUBSAMPLE_SIZE - 1])
    print(f"parent sample        : {len(ids):,} source ids  ({merged.name})")
    print(f"subsample            : {C.SUBSAMPLE_SIZE:,} at seed {C.SUBSAMPLE_SEED}")
    print(f"SUBSAMPLE_RANK_CUTOFF = {cutoff}")
    print(f"  next rank up       = {int(ranks[C.SUBSAMPLE_SIZE])}  (the gap is the margin)")
    if cutoff != C.SUBSAMPLE_RANK_CUTOFF or len(ids) != C.SUBSAMPLE_PARENT_SIZE:
        print("\n  !! config.py disagrees -- update SUBSAMPLE_RANK_CUTOFF and "
              "SUBSAMPLE_PARENT_SIZE")
    else:
        print("\n  config.py agrees.")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=["calibrate", "census"])
    parser.add_argument("--populations", nargs="+", choices=list(C.POPULATIONS),
                        default=list(C.POPULATIONS))
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--target-fp", type=float, default=None,
                        help=f"null false-positive rate (default {C.TARGET_FP})")
    args = parser.parse_args(argv)

    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)
    if args.output_root:
        C.set_output_root(args.output_root)

    if C.manifest_path().exists():
        manifest = json.loads(C.manifest_path().read_text())
        print(f"run: {manifest['written']}  grid {manifest['grid']['n_periods']:,} periods  "
              f"fit_jitter={manifest['fit_jitter']}  curves={manifest['power']['mode']}\n")

    for stage in args.stages:
        print(f"=== {stage} ===")
        {"calibrate": stage_calibrate, "census": stage_census, "merge": stage_merge,
         "subsample-cutoff": stage_subsample_cutoff}[stage](args)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
