#!/usr/bin/env python
"""Self-check: the grid, the classifier, the shard reader, and the round trip.

    python tests/test_periodograms.py                 # synthetic only, no catalog
    python tests/test_periodograms.py --catalog-root <path>   # + real shards

The synthetic half needs nothing but the package and `kepmodel`; it builds its
own epoch series and checks the invariants that the output format rests on. The
catalog half runs one small work unit end to end against real shards and checks
that what comes back out of the parquet is what went in.

Everything here is an assertion about something the run would otherwise get
silently wrong: a period axis that does not line up with the stored curves, a
part split that drops or duplicates systems, a subsample that is not the paper's.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from epochalypse.periodogram import calibrate as cal
from epochalypse.periodogram import config as C
from epochalypse.periodogram import fitting, grid
from epochalypse.periodogram import periodogram as pg
from epochalypse.periodogram import writers

PASSED = []


def check(name, condition, detail=""):
    mark = "ok  " if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f"   {detail}" if detail else ""))
    PASSED.append(bool(condition))


# ==========================================================================
# Synthetic epochs
# ==========================================================================
def fake_system(n_epochs=90, period=2.3, alpha=0.6, sigma=0.08, seed=0):
    """A five-parameter star plus one circular reflex, on a DR4-like time span."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(-C.DR4_BASELINE_YEARS / 2, C.DR4_BASELINE_YEARS / 2, n_epochs))
    psi = rng.uniform(0, 2 * np.pi, n_epochs)
    pf = np.sin(2 * np.pi * t)                       # a stand-in parallax factor
    phase = 2 * np.pi * t / period
    y = (3.0 * pf + 0.4 * np.sin(psi) + 0.2 * np.cos(psi)
         + 1.1 * t * np.sin(psi) - 0.7 * t * np.cos(psi)
         + alpha * (np.cos(phase) * np.cos(psi) + np.sin(phase) * np.sin(psi))
         + rng.normal(0, sigma, n_epochs))
    return t, psi, pf, y, np.full(n_epochs, sigma)


def test_grid():
    print("grid")
    segments = grid.frequency_segments()
    periods = grid.segment_periods(segments)
    dlog = np.diff(np.log10(periods))

    check("brackets the injected prior",
          periods[0] <= C.P_MIN * 1.001 and periods[-1] >= C.P_MAX * 0.999,
          f"{periods[0]:.3e} .. {periods[-1]:.1f} yr")
    check("no coarser than the baseline grid anywhere",
          dlog.max() <= grid.target_dlog() * 1.001,
          f"max dlogP = {dlog.max():.3e} <= {grid.target_dlog():.3e}")
    check("strictly ascending, no duplicates", bool((dlog > 0).all()))
    check("~35% more samples than the baseline's trial count",
          1.2 < len(periods) / C.BASELINE_N_PERIODS < 1.6,
          f"{len(periods):,} vs {C.BASELINE_N_PERIODS:,}")

    # The one identity the whole storage format rests on: the periods the
    # search visits are exactly `segment_periods(segments)`, so a stored power
    # array needs no period axis of its own.
    t, psi, pf, y, yerr = fake_system()
    visited, power, _ = pg.kepmodel_power(t, psi, pf, y, yerr, segments=segments)
    check("kepmodel visits exactly segment_periods(segments)",
          visited.shape == periods.shape and np.allclose(visited, periods, rtol=1e-12),
          f"{len(visited):,} periods, max rel diff "
          f"{np.abs(visited / periods - 1).max():.1e}")
    check("power is aligned to it", power.shape == periods.shape)
    return segments, periods


def test_periodogram(segments):
    print("\nperiodogram + classifier")
    period = 2.3
    t, psi, pf, y, yerr = fake_system(period=period, alpha=0.6, sigma=0.05)
    periods, power, info = pg.kepmodel_power(t, psi, pf, y, yerr, segments=segments)
    res = pg.classify_periodogram(periods, power, len(y))

    check("recovers a strong injected period",
          abs(np.log(res["best_period"] / period)) < np.log(1.2),
          f"best {res['best_period']:.4f} yr vs {period} yr injected")
    check("calls it localized", res["klass"] == "unimodal", res["klass"])
    check("power is Delta-chi^2, not normalised power",
          0.0 < power.max() <= info["chi2_base"] * (1 + 1e-9),
          f"peak {power.max():.1f}, chi2_base {info['chi2_base']:.1f}")
    check("truth lands in the competitive region",
          fitting.period_in_competitive_region(periods, power, period) is True)

    # A noise-only system must not be called localized, and its peak must be
    # far below a strong signal's -- this is the whole basis of the calibration.
    t, psi, pf, y, yerr = fake_system(alpha=0.0, sigma=0.05, seed=11)
    _, null_power, _ = pg.kepmodel_power(t, psi, pf, y, yerr, segments=segments)
    check("the null peak is far below the signal peak",
          null_power.max() < 0.2 * power.max(),
          f"{null_power.max():.1f} vs {power.max():.1f}")

    # The grid-agnostic width metric must reduce to the baseline definition on
    # a uniform log grid: (competitive fraction) x (total log range).
    uniform = np.exp(np.linspace(np.log(C.P_MIN), np.log(C.P_MAX), 4001))
    curve = np.exp(-0.5 * ((np.log10(uniform) - 0.5) / 0.02) ** 2) * 100.0
    width, _, _ = pg.period_constraint(uniform, curve)
    logp = np.log10(uniform)
    baseline_width = ((curve > curve.max() - C.WIDTH_DELTA).mean()
                      * (logp[-1] - logp[0]))
    check("width metric matches the baseline on a uniform log grid",
          abs(width - baseline_width) < 2 * np.diff(logp).max(),
          f"{width:.5f} vs {baseline_width:.5f} dex")


def test_subsample():
    print("\npaper subsample")
    # The cutoff is a quantile of the parent sample; a single id can be tested
    # against it with no table, which is the property the ranks depend on.
    ids = np.array([5484066448309985152, 424187226612669312, 1, 2, 3], dtype="int64")
    ranks = writers.source_id_ranks(ids)
    keep = writers.in_paper_subsample(ids)
    check("rank threshold agrees with the rank values",
          bool((keep == (ranks <= np.uint64(C.SUBSAMPLE_RANK_CUTOFF))).all()))
    check("the cutoff is the SIZE/PARENT quantile it claims to be",
          abs(C.SUBSAMPLE_RANK_CUTOFF / 2.0**64
              - C.SUBSAMPLE_SIZE / C.SUBSAMPLE_PARENT_SIZE) < 1e-4,
          f"{C.SUBSAMPLE_RANK_CUTOFF / 2.0**64:.6f} vs "
          f"{C.SUBSAMPLE_SIZE / C.SUBSAMPLE_PARENT_SIZE:.6f}")


def test_calibration():
    print("\ncalibration")
    rng = np.random.default_rng(4)
    top = rng.chisquare(4, 200_000) * 30.0
    accel = rng.chisquare(2, 200_000) * 10.0
    thr_orbit, thr_accel = cal.thresholds_from_null(top, accel)
    realized = ((top > thr_orbit) | (accel > thr_accel)).mean()
    check("the union lands near the target FP",
          abs(realized - C.TARGET_FP) < 0.15 * C.TARGET_FP,
          f"{realized:.4%} vs {C.TARGET_FP:.2%}")

    frame = pd.DataFrame({"top_power": top[:1000], "accel_delta_chi2": accel[:1000],
                          "klass": "unimodal", "best_period": 1.0})
    cal.apply_calibration(frame, thr_orbit, thr_accel)
    check("apply_calibration adds all four flags",
          all(c in frame for c in ("peak_significant_cal", "accel_significant_cal",
                                   "detected_cal", "period_reliable_cal")))
    check("period_reliable_cal implies detected_cal",
          bool((~frame["period_reliable_cal"] | frame["detected_cal"]).all()))

    high = pd.DataFrame({"snr_total_1": [6.0, 4.0, 9.0, np.nan],
                         "snr_total_2": [7.0, 9.0, 3.0, 9.0]})
    check("high-SNR needs EVERY companion over the floor",
          list(cal.select_high_snr(high).index) == [0])


# ==========================================================================
# Against the real catalog
# ==========================================================================
def test_shards(segments, periods, catalog_root, population="1_companion", n_systems=12):
    print(f"\nshards  ({catalog_root})")
    from epochalypse.periodogram.unit import run_unit

    from epochalypse.periodogram.shards import ShardReader, discover_shards, work_units

    C.set_catalog_root(catalog_root)
    numbers, n_shards = discover_shards(population)
    shard = numbers[0]

    with ShardReader(population, shard, n_shards) as reader:
        n_total = len(reader.truths)
        # Splitting must partition: every system exactly once, no gaps, no
        # repeats. This is the assumption --n-parts rests on.
        seen = []
        for part in range(4):
            seen += [i for i, *_ in reader.iter_systems(part, 4)]
        check("a 4-way split partitions the shard exactly",
              sorted(seen) == list(range(n_total)) and len(seen) == n_total,
              f"{len(seen):,} systems, {n_total:,} in the truth shard")

        # And the epochs a system yields must be its own, in time order.
        index, truth, t, psi, pf, y, yerr = next(iter(reader.iter_systems(0, n_total)))
        check("epochs come back sorted in time", bool((np.diff(t) >= 0).all()))
        check("epoch count matches the truth row's n_transits_dr4",
              len(t) == int(truth["n_transits_dr4"]),
              f"{len(t)} epochs, n_transits_dr4 = {truth['n_transits_dr4']}")

    check("work_units covers every shard of every population",
          len(work_units()) == 3 * n_shards, f"{len(work_units())} units")

    # One small unit, end to end, and read back out of the parquet.
    out = Path(tempfile.mkdtemp(prefix="epgram-test-"))
    try:
        C.set_output_root(out)
        writers.write_period_grid(periods)
        summary = run_unit(population, shard, n_shards, segments=segments,
                           limit=n_systems, power_mode="all", verbose=False)
        chars = pd.read_parquet(C.chars_shard(population, shard, n_shards))
        power = pd.read_parquet(C.power_shard(population, shard, n_shards))
        stored_grid = pd.read_parquet(C.period_grid_path())["period_yr"].to_numpy()

        check("wrote one row per system", len(chars) == n_systems == summary["n_systems"])
        check("no system failed", summary["n_failed"] == 0)
        check("truth columns are joined on",
              all(c in chars for c in ("gaia_source_id", "period_1", "snr_total_1",
                                       "mass_st_msun", "parallax_mas", "n_transits_dr4")))
        check("analysis-schema aliases are present",
              all(c in chars for c in ("a_1_au", "e_1", "Mp_1_msun", "i_1_rad",
                                       "alpha_1_mas")))
        check("Mp_1_msun is mass_pl_1 converted, not copied",
              np.allclose(chars["Mp_1_msun"], chars["mass_pl_1"] * C.MJUP_IN_MSUN))
        check("every row carries a klass",
              chars["klass"].isin(["undetected", "broad", "multimodal", "unimodal"]).all(),
              ", ".join(f"{k}={v}" for k, v in chars["klass"].value_counts().items()))
        check("stored curves match the stored grid",
              len(power) == n_systems and len(power["power"].iloc[0]) == len(stored_grid),
              f"{len(power)} curves x {len(power['power'].iloc[0]):,} points")
        check("curve ids line up with the table's",
              sorted(power["gaia_source_id"]) == sorted(chars["gaia_source_id"]))

        # The stored curve must reproduce the summary it was measured beside.
        row = chars.iloc[0]
        curve = np.asarray(power.loc[power["gaia_source_id"] == row["gaia_source_id"],
                                     "power"].iloc[0], float)
        check("the stored curve's peak is the table's top_power",
              abs(curve.max() - row["top_power"]) < 1e-3 * max(abs(row["top_power"]), 1.0),
              f"{curve.max():.3f} vs {row['top_power']:.3f}")
        check("its argmax is the table's best_period",
              abs(np.log(stored_grid[curve.argmax()] / row["best_period"])) < 1e-9
              or row["klass"] == "undetected",
              f"{stored_grid[curve.argmax()]:.4g} vs {row['best_period']:.4g} yr")

        # A stored curve is useless if the record cannot be traced back to the
        # epochs it came from -- (shard, shard_row) is that address.
        with ShardReader(population, shard, n_shards) as reader:
            for index, truth, t, psi, pf, y, yerr in reader.iter_systems(
                    int(row["shard_row"]), len(reader.truths)):
                break
        check("(shard, shard_row) addresses the right star",
              int(truth["gaia_source_id"]) == int(row["gaia_source_id"])
              and len(t) == int(row["n_epochs"]))
    finally:
        shutil.rmtree(out, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-root", type=Path, default=None,
                        help="run the shard tests against this catalog too")
    args = parser.parse_args(argv)

    segments, periods = test_grid()
    test_periodogram(segments)
    test_subsample()
    test_calibration()
    if args.catalog_root:
        test_shards(segments, periods, args.catalog_root)
    else:
        print("\nshards  (skipped -- pass --catalog-root to run them)")

    print(f"\n{sum(PASSED)}/{len(PASSED)} checks passed")
    return 0 if all(PASSED) else 1


if __name__ == "__main__":
    raise SystemExit(main())
