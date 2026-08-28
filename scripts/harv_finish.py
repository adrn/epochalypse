#!/usr/bin/env python
"""Post-processing: count what came back, merge the per-system shards.

Serial, cheap, and none of it needs a cluster -- the census reads five columns
out of the parquet dataset rather than the whole table.

    python scripts/harv_finish.py --stages census
    python scripts/harv_finish.py --stages merge --populations 1_companion

Stages:

  census   the ESS and weight_captured distributions, and period recovery
           against the injected truth -- the two diagnostics that decide
           whether a second, MCMC pass is needed and how big it would be
  merge    each population's per-system shards -> one parquet

**The samples are never merged.** They are ~850 GB across the catalog; read them
with `pyarrow.dataset` over `config.samples_dir(population)`, which memory-maps
row groups instead of materializing the table. The per-system table is ~2 GB and
is what most analysis wants anyway.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from epochalypse.harv import config as C
from epochalypse.periodogram import config as PG

STAGES = ("census", "merge")

CENSUS_COLUMNS = ["ess", "weight_captured", "period_best_yr"]


def _systems(population, columns=None):
    """The per-system rows of one population, as one Arrow table."""
    directory = C.systems_dir(population)
    if not directory.exists():
        raise FileNotFoundError(f"no harv output in {directory}")
    return ds.dataset(directory, format="parquet").to_table(columns=columns)


def _high_snr_mask(table, population):
    """Every injected companion clears the SNR floor -- the generator's rule."""
    n = C.POPULATIONS[population]
    if n == 0:
        return None
    snr = np.column_stack(
        [np.asarray(table[f"snr_total_{k}"], float) for k in range(1, n + 1)]
    )
    return np.isfinite(snr).all(axis=1) & (snr >= PG.HIGH_SNR_MIN).all(axis=1)


def stage_census(args):
    header = (
        f"{'population':<26}{'n':>10}{'ESS med':>10}{'ESS<10':>9}"
        f"{'wcap med':>10}{'wcap low':>10}{'P recovered':>13}"
    )
    print(header)
    print("-" * len(header))
    for population in args.populations:
        n_companions = C.POPULATIONS[population]
        columns = list(CENSUS_COLUMNS)
        if n_companions:
            columns += [f"snr_total_{k}" for k in range(1, n_companions + 1)]
            columns += ["period_1"]
        table = _systems(population, columns)
        high_snr = _high_snr_mask(table, population)

        for label, mask in (("", None), (" (high-SNR)", high_snr)):
            # The control population has no companions, so it has no high-SNR
            # subset -- without this it prints the same row twice.
            if label and (mask is None or not mask.any()):
                continue
            rows = table if mask is None else table.filter(mask)
            n = rows.num_rows
            if not n:
                continue
            ess = np.asarray(rows["ess"], float)
            wcap = np.asarray(rows["weight_captured"], float)
            if n_companions:
                # |ln(P_best / P_true)| < ln(tol): the periodogram stage's rule,
                # imported so the two analyses cannot recover to different bars.
                ratio = np.asarray(rows["period_best_yr"], float) / np.asarray(
                    rows["period_1"], float
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    recovered = np.abs(np.log(ratio)) < np.log(PG.PERIOD_RECOVER_TOL)
                recovered_txt = f"{np.nanmean(recovered):>12.1%}"
            else:
                recovered_txt = f"{'--':>12}"
            print(
                f"{population + label:<26}{n:>10,}{np.median(ess):>10.1f}"
                f"{np.mean(ess < C.ESS_RESOLVED):>9.1%}{np.median(wcap):>10.4f}"
                f"{np.mean(wcap < C.WEIGHT_CAPTURED_MIN):>10.1%}{recovered_txt}"
            )
    print(
        f"\nESS<10 is the share whose posterior the {C.N_PRIOR_SAMPLES:,}-sample "
        "library did not resolve --\nthose systems have a good point estimate and "
        "an unmeasured uncertainty, and are the\ncandidates for an MCMC second "
        f"pass. wcap low is the share where TOP_K={C.TOP_K} truncated\nmore than "
        f"{1 - C.WEIGHT_CAPTURED_MIN:.0%} of the posterior mass."
    )


def stage_merge(args):
    for population in args.populations:
        table = _systems(population)
        path = C.merged_systems(population)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(
            table,
            tmp,
            compression=C.PARQUET_COMPRESSION,
            compression_level=C.PARQUET_COMPRESSION_LEVEL,
        )
        tmp.replace(path)
        print(
            f"  {path.name:<44} {table.num_rows:>10,} rows  "
            f"{path.stat().st_size / 1e9:6.3f} GB",
            flush=True,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=["census"])
    parser.add_argument(
        "--populations",
        nargs="+",
        choices=list(C.POPULATIONS),
        default=list(C.POPULATIONS),
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)

    if args.output_root:
        C.set_output_root(args.output_root)

    if C.manifest_path().exists():
        manifest = json.loads(C.manifest_path().read_text())
        library = manifest["library"]
        # Describe the output as it was produced, not as the config now reads:
        # a census run after someone edited config.py would otherwise quote
        # settings this data was never fitted under.
        C.N_PRIOR_SAMPLES = library["n_prior_samples"]
        C.TOP_K = library["top_k"]
        print(
            f"run: {manifest['written']}  {library['n_prior_samples']:,} prior "
            f"samples  top {library['top_k']}  {library['fingerprint']}\n"
        )

    for stage in args.stages:
        print(f"=== {stage} ===")
        {"census": stage_census, "merge": stage_merge}[stage](args)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
