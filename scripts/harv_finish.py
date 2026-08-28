#!/usr/bin/env python
"""Post-processing: count what came back, merge the per-system shards.

Serial, cheap, and none of it needs a cluster -- the census reads five columns
out of the parquet dataset rather than the whole table.

    python scripts/harv_finish.py --stages census recovery
    python scripts/harv_finish.py --stages merge --populations 1_companion

Stages:

  census   the ESS and weight_captured distributions, and period recovery
           against the injected truth -- the two diagnostics that decide
           whether a second, MCMC pass is needed and how big it would be
  recovery period recovery binned by injected period and eccentricity -- the
           two things that actually limit it, and the breakdown that tells you
           whether a low number is the data's fault or the prior's
  figures  four diagnostic PNGs -- see epochalypse.harv.figures for what each
           one answers and the order to read them in
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
import pyarrow.parquet as pq

from epochalypse.harv import census
from epochalypse.harv import config as C
from epochalypse.harv import figures as F

STAGES = ("census", "recovery", "figures", "merge")

CENSUS_COLUMNS = ["ess", "weight_captured", "period_best_yr"]


# One authority for all four, in `epochalypse.harv.census`, because the figures
# need exactly these rules -- see that module for why.
_systems = census.read_systems
_high_snr_mask = census.high_snr_mask
_recovered = census.recovered
_period_columns = census.period_columns


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
            columns += _period_columns(population)
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
                recovered = _recovered(rows, population)
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


def stage_recovery(args):
    """Period recovery binned by injected period and eccentricity.

    A single recovery percentage is nearly uninterpretable, because two
    unrelated things limit it and they compound:

    * **the baseline.** DR4 is 5.5 yr with ~80 transits, so only roughly
      0.1-10 yr is constrainable at all. Below that the signal aliases; above
      it the orbit is absorbed into proper motion. The injected prior spans
      7.8 decades, so most of it is unrecoverable *by construction* -- that is
      the data's limit, not the method's, and no amount of prior samples fixes
      it.
    * **the prior.** Coverage of the injected eccentricity distribution. This
      one IS fixable, and `config.ECC_LOC` records what it cost.

    Only the joint breakdown separates them. Read the period profile for the
    shape of the window and the eccentricity profile for prior mismatch; if the
    grid shows recovery is high in the sweet spot *and* flat in eccentricity,
    the remaining shortfall is library resolution and more `N_PRIOR_SAMPLES`
    is the answer.

    ESS is reported per period bin because it runs the *opposite* way to
    recovery -- a well-constrained period gives a sharp posterior, which a
    fixed-size library resolves with fewer effective samples. Low ESS where
    recovery is high is correct, not a warning.
    """
    import pandas as pd

    log_bins = [-5, -2, -1, -0.5, 0, 0.5, 1, 4]
    ecc_bins = [0, 0.3, 0.5, 0.7, 0.9, 1.0]
    for population in args.populations:
        n_companions = C.POPULATIONS[population]
        if not n_companions:
            continue
        columns = (
            ["period_best_yr", "ess", "weight_captured"]
            + _period_columns(population)
            + [f"snr_total_{k}" for k in range(1, n_companions + 1)]
            + [f"ecc_{k}" for k in range(1, n_companions + 1)]
        )
        table = _systems(population, columns)
        high_snr = _high_snr_mask(table, population)
        frame = table.to_pandas()
        frame["recovered"] = _recovered(table, population)
        frame = frame[high_snr] if high_snr is not None else frame
        if frame.empty:
            continue
        # Companion 1 labels the bins. For 2_companion that is a simplification
        # -- recovery is scored against either orbit but binned by the first.
        frame["log_period"] = np.log10(frame["period_1"])
        print(f"\n--- {population}, high-SNR only ({len(frame):,} systems) ---")

        for label, key, bins in (
            ("injected period (log10 yr)", "log_period", log_bins),
            ("injected eccentricity", "ecc_1", ecc_bins),
        ):
            grouped = frame.groupby(pd.cut(frame[key], bins), observed=False).agg(
                n=("recovered", "size"),
                recovered=("recovered", "mean"),
                ess=("ess", "median"),
            )
            print(f"\n  by {label}:")
            for name, row in grouped.iterrows():
                if not row["n"]:
                    continue
                print(
                    f"    {name!s:>16}  n={int(row['n']):>6,}  "
                    f"recovered {row['recovered']:>6.1%}  ESS med {row['ess']:>6.1f}"
                )

        # Where do the failures land? With ESS ~ 1 the answer is a single prior
        # draw, and an astrometric likelihood is multi-modal: the annual
        # parallax term puts aliases at 1/P +- 1/yr, and 2P / P/2 also compete.
        # If the misses cluster at particular ratios they are aliases -- more
        # samples make the true mode reliably win. If they are spread flat, the
        # data simply does not constrain those systems.
        missed = frame[~frame["recovered"].astype(bool)]
        if len(missed):
            ratio = np.log10(
                np.asarray(missed["period_best_yr"], float)
                / np.asarray(missed["period_1"], float)
            )
            ratio = ratio[np.isfinite(ratio)]
            edges = [-9, -3, -1, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1, 3, 9]
            counts, _ = np.histogram(ratio, bins=edges)
            print(f"\n  where the {len(missed):,} misses landed, log10(P_best/P_true):")
            for lo, hi, n in zip(edges[:-1], edges[1:], counts):
                if not n:
                    continue
                bar = "#" * round(40 * n / counts.max())
                print(f"    {lo:>5.1f} to {hi:>4.1f}  {n:>6,}  {bar}")
            print(
                f"    railed at the prior edge (P_best < 1e-3 yr): "
                f"{(np.asarray(missed['period_best_yr'], float) < 1e-3).sum():,}"
            )

        # The grid is the point: it says whether a low headline number is the
        # baseline's fault (bad everywhere except the sweet spot) or the
        # prior's (bad at high e even in the sweet spot).
        grid = frame.pivot_table(
            index=pd.cut(frame["log_period"], log_bins),
            columns=pd.cut(frame["ecc_1"], ecc_bins),
            values="recovered",
            aggfunc="mean",
            observed=False,
        )
        print("\n  recovered, period (rows) x eccentricity (columns):")
        header = "".join(f"{c.right!s:>9}" for c in grid.columns)
        print(f"    {'e <=':>16}{header}")
        for name, row in grid.iterrows():
            cells = "".join(
                "        -" if np.isnan(v) else f"{v:>8.0%} " for v in row.to_numpy()
            )
            print(f"    {name!s:>16}{cells}")


def stage_figures(args):
    """Draw the diagnostics. Reads the shards, so it does not need `merge` first."""
    for path in F.make_figures(args.figures):
        del path


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
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=list(F.FIGURES),
        default=None,
        help="which diagnostics to draw (default: all of them)",
    )
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
        {
            "census": stage_census,
            "recovery": stage_recovery,
            "figures": stage_figures,
            "merge": stage_merge,
        }[stage](args)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
