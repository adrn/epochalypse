#!/usr/bin/env python
"""Post-processing: calibrate the thresholds, count the classes, merge the shards.

Serial, cheap, and none of it needs a cluster -- every stage reads a handful of
columns out of the parquet dataset rather than the whole table.

    python scripts/characterize_finish.py --stages calibrate census
    python scripts/characterize_finish.py --stages merge --populations 1_companion

Stages:

  calibrate         thresholds from `0_companion` at TARGET_FP -> calibration.json
  census            class counts per population, and for the high-SNR subsets
  merge             each population's 320 shards -> one parquet, plus the
                    high-SNR views (small enough to be worth materializing)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


from epochalypse.periodogram import calibrate as cal
from epochalypse.periodogram import config as C

STAGES = ("calibrate", "census", "merge")


def stage_calibrate(args):
    calibration = cal.calibrate(target_fp=None)
    path = C.calibration_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2) + "\n")
    print(
        f"thresholds from {calibration['n_null_systems']:,} companion-free systems "
        f"@ FP={calibration['target_fp']:.1%}"
    )
    print(
        f"  orbit Delta-chi2 > {calibration['thr_orbit']:10.1f}   "
        f"(null rate {calibration['realized_fp_peak']:.3%})"
    )
    print(
        f"  accel Delta-chi2 > {calibration['thr_accel']:10.1f}   "
        f"(null rate {calibration['realized_fp_accel']:.3%})"
    )
    print(f"  either channel   : {calibration['realized_fp']:.3%} of the null")
    print(f"-> {path}")
    return calibration


def stage_census(args):
    calibration = json.loads(C.calibration_path().read_text())
    print(
        f"{'population':<26}{'n':>12}{'undetected':>14}{'localized':>12}{'not localized':>16}"
    )
    for population in args.populations:
        counts = cal.census(
            population, calibration["thr_orbit"], calibration["thr_accel"]
        )
        for label, key in (("", "all"), (" (high-SNR)", "high_snr")):
            if key not in counts:
                continue
            c = counts[key]
            print(
                f"{population + label:<26}{c['n']:>12,}{c['undet']:>14,}"
                f"{c['narrow']:>12,}{c['broad']:>16,}"
            )


def stage_merge(args):
    for population in args.populations:
        for high_snr in (False, True):
            if high_snr and C.POPULATIONS[population] == 0:
                continue
            path, n = cal.merge(population, high_snr=high_snr, columns=None)
            size = path.stat().st_size / 1e9
            print(f"  {path.name:<52} {n:>10,} rows  {size:6.2f} GB", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--stages", nargs="+", choices=STAGES, default=["calibrate", "census"]
    )
    parser.add_argument(
        "--populations",
        nargs="+",
        choices=list(C.POPULATIONS),
        default=list(C.POPULATIONS),
    )
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)
    if args.output_root:
        C.set_output_root(args.output_root)

    if C.manifest_path().exists():
        manifest = json.loads(C.manifest_path().read_text())
        print(
            f"run: {manifest['written']}  grid {manifest['grid']['n_periods']:,} periods  "
            f"curves={manifest['power']['mode']}\n"
        )

    for stage in args.stages:
        print(f"=== {stage} ===")
        {"calibrate": stage_calibrate, "census": stage_census, "merge": stage_merge}[
            stage
        ](args)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
