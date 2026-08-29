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
  figures  six diagnostic PNGs -- see epochalypse.harv.figures for what each
           one answers and the order to read them in. harv_amplitude is the one
           that says whether the amplitude PRIOR, rather than the data, is
           setting the detection threshold
  gallery  per-system diagnostics -- the data, the model and the posterior
           samples for a few systems from each (SNR, period) cell, stratified
           across recovered / railed / wrong-period so the failures are always
           represented. Needs --catalog-root, because it reads the epochs.
           Start with the 0.79-1.26 yr cells: a one-year orbit is degenerate
           with parallax
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
from epochalypse.harv import gallery as G
from epochalypse.periodogram import config as PG

STAGES = ("census", "recovery", "figures", "gallery", "merge")

CENSUS_COLUMNS = ["ess", "weight_captured", "period_best_yr"]


# One authority for all four, in `epochalypse.harv.census`, because the figures
# need exactly these rules -- see that module for why.
_systems = census.read_systems
_high_snr_mask = census.high_snr_mask
_recovered = census.recovered
_period_columns = census.period_columns


def stage_census(args):
    header = (
        f"{'population':<26}{'n':>10}{'ESS med':>10}{'ESS<10':>9}{'ESS bad':>9}"
        f"{'wcap med':>10}{'wcap low':>10}{'railed':>9}{'P recovered':>13}"
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
            # A system whose log-likelihoods are ALL non-finite gives ESS = NaN.
            # np.median then poisons the whole column, and `nan < ESS_RESOLVED`
            # is False so it would also be silently counted as *resolved*. Both
            # were live bugs. The count is a diagnostic in its own right: it
            # means bad input reached the likelihood without anything raising,
            # so the `failed/` CSVs never saw it.
            finite = np.isfinite(ess)
            n_bad = int((~finite).sum())
            if n_companions:
                # Quoted over the searched period range: a system injected
                # outside it cannot be recovered by construction, and counting
                # those as failures understates the method.
                searchable = census.in_search_range(rows, population)
                recovered = _recovered(rows, population)[searchable]
                recovered_txt = (
                    f"{np.nanmean(recovered):>12.1%}"
                    if recovered.size
                    else f"{'--':>12}"
                )
                railed_txt = f"{census.railed(rows).mean():>8.1%}"
            else:
                recovered_txt = f"{'--':>12}"
                railed_txt = f"{census.railed(rows).mean():>8.1%}"
            print(
                f"{population + label:<26}{n:>10,}{np.nanmedian(ess):>10.1f}"
                f"{np.mean(ess[finite] < C.ESS_RESOLVED):>9.1%}{n_bad:>9,}"
                f"{np.nanmedian(wcap):>10.4f}"
                f"{np.mean(wcap < C.WEIGHT_CAPTURED_MIN):>10.1%}"
                f"{railed_txt}{recovered_txt}"
            )
    print(
        f"\nESS<10   share the {C.N_PRIOR_SAMPLES:,}-sample library did not resolve. Those"
        "\n         have a good point estimate and an unmeasured uncertainty, and are"
        "\n         the candidates for an MCMC second pass."
        f"\nESS bad  systems whose log-likelihoods were ALL non-finite. Should be 0;"
        "\n         anything else means bad input reached the likelihood silently."
        f"\nwcap low share where TOP_K={C.TOP_K} truncated more than "
        f"{1 - C.WEIGHT_CAPTURED_MIN:.0%} of the posterior mass."
        f"\nrailed   share whose best sample sits at the prior floor "
        f"({C.PERIOD_MIN_YR * C.RAIL_FACTOR:g} yr) -- a NON-detection,"
        "\n         not a wrong period. On the first production run this was 65% of"
        "\n         all recovery failures, so the two are reported apart."
        f"\nP recov  over the searched range only ({C.PERIOD_MIN_YR:g}-{C.PERIOD_MAX_YR:g} yr);"
        "\n         systems injected outside it are unrecoverable by construction."
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

    log_bins = census.LOG_PERIOD_BINS
    ecc_bins = census.ECC_BINS
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
        frame["railed"] = census.railed(table)
        frame["searchable"] = census.in_search_range(table, population)
        # of whichever companion `recovered()` matched, so the SNR binning below
        # describes the orbit that was actually found
        frame["snr_total_best"] = census.best_truth(table, population, "snr_total")
        frame = frame[high_snr] if high_snr is not None else frame
        if frame.empty:
            continue
        # Companion 1 labels the bins. For 2_companion that is a simplification
        # -- recovery is scored against either orbit but binned by the first.
        frame["log_period"] = np.log10(frame["period_1"])
        outside = int((~frame["searchable"]).sum())
        print(f"\n--- {population}, high-SNR only ({len(frame):,} systems) ---")
        print(
            f"  {outside:,} ({outside / max(len(frame), 1):.1%}) injected outside the "
            f"searched range {C.PERIOD_MIN_YR:g}-{C.PERIOD_MAX_YR:g} yr and cannot be\n"
            "  recovered by construction. Every recovery number below is over the "
            f"remaining {len(frame) - outside:,}."
        )
        frame = frame[frame["searchable"].astype(bool)]
        if frame.empty:
            continue

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

        # Split the failures. Railing is a NON-detection -- the fit collapsed to
        # the prior floor where the amplitude is forced to zero -- and it is a
        # different thing from finding the wrong period. On the first production
        # run it was 65% of all misses, so reporting one number for both said
        # very little.
        missed = frame[~frame["recovered"].astype(bool)]
        if len(missed):
            n_rail = int(missed["railed"].sum())
            wrong = missed[~missed["railed"].astype(bool)]
            print(f"\n  {len(missed):,} misses:")
            print(
                f"    railed (no detection)  {n_rail:>6,}  "
                f"{n_rail / len(missed):>6.1%}  best sample at the prior floor"
            )
            print(
                f"    wrong period           {len(wrong):>6,}  "
                f"{len(wrong) / len(missed):>6.1%}"
            )
            if len(wrong):
                ratio = np.log10(
                    np.asarray(wrong["period_best_yr"], float)
                    / np.asarray(wrong["period_1"], float)
                )
                ratio = ratio[np.isfinite(ratio)]
                edges = [-4, -1, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 1, 4]
                counts, _ = np.histogram(ratio, bins=edges)
                print("\n  wrong-period misses, log10(P_best/P_true) -- clustered on")
                print("  the 2x (0.30) or annual tracks means an alias, flat means")
                print("  the data does not constrain them:")
                for lo, hi, n in zip(edges[:-1], edges[1:], counts):
                    if not n:
                        continue
                    bar = "#" * round(40 * n / counts.max())
                    print(f"    {lo:>5.1f} to {hi:>4.1f}  {n:>6,}  {bar}")

        # Rail fraction against SNR. This is the curve that says whether the
        # amplitude prior is setting the detection threshold: sigma_a0 controls
        # the Occam penalty on a real orbit relative to the null, so if railing
        # falls off a cliff at some SNR and is near zero above it, that cliff IS
        # the threshold -- and it should sit at HIGH_SNR_MIN, not above it.
        snr = np.asarray(frame["snr_total_best"], float)
        frame["log_snr"] = np.log10(np.maximum(snr, 1e-3))
        bins = np.linspace(np.log10(PG.HIGH_SNR_MIN), np.nanmax(frame["log_snr"]), 9)
        print(f"\n  rail fraction vs SNR (HIGH_SNR_MIN = {PG.HIGH_SNR_MIN:g}):")
        index = census.bin_index(frame["log_snr"], bins)
        for b in range(len(bins) - 1):
            sel = index == b
            if not sel.sum():
                continue
            rail = frame["railed"].to_numpy()[sel].mean()
            rec = frame["recovered"].to_numpy()[sel].mean()
            bar = "#" * round(30 * rail)
            print(
                f"    SNR {10 ** bins[b]:>7.1f}-{10 ** bins[b + 1]:<7.1f} "
                f"n={int(sel.sum()):>6,}  railed {rail:>6.1%}  recovered {rec:>6.1%}  {bar}"
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


def stage_gallery(args):
    """Per-system figures. Reads epochs, so it needs the catalog as well.

    Skips a population with no output rather than raising: this stage writes as
    it goes, and losing the figures already drawn because a later population was
    never simulated would be the wrong trade for a diagnostic.
    """
    for population in args.populations:
        if not C.POPULATIONS[population]:
            continue  # the control has no injected period to bin on
        print(f"{population}:")
        try:
            G.make_gallery(population, args.per_bin)
        except FileNotFoundError as error:
            print(f"  skipping: {error}")


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
        "--catalog-root",
        type=Path,
        help="the catalog to read epochs from (gallery only)",
    )
    parser.add_argument(
        "--per-bin",
        type=int,
        default=None,
        help=f"gallery systems per (SNR, period) cell (default {C.GALLERY_PER_BIN})",
    )
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
    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)

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
            "gallery": stage_gallery,
            "merge": stage_merge,
        }[stage](args)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
