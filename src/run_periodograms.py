"""Batch-run the fast astrometric period periodogram + characterization over whole populations.

Usage:
    python run_periodograms.py <population> [<population> ...]

Writes one CSV per population to outputs/data/characterization_<population>.csv, with one row per
system (periodogram summary + acceleration test + data-only detection/characterization flags +
truth-based period recovery, joined to the injected-truth columns). See
``epochalypse_fitting.characterize_population`` for the columns.
"""
import sys
import time
from pathlib import Path

import numpy as np

import epochalypse_fitting as ef

# Trial-period grid: 0.01 yr .. the longest injected period in any population (3228 yr: the outer
# companion of 2_companion_agnostic; period_1 alone reaches 2712 yr). The grid must bracket every
# truth at BOTH ends, otherwise a system rails at a grid edge and "is the truth inside the
# competitive region?" is unanswerable-by-construction rather than measured.
#
# The lower bound is set by the innermost orbit the generator can inject. That is no longer the
# flat semi-major axis floor: PLANET_PRIORS.a_min_au is 0.001 AU, but the binding limit is the
# per-star Roche-lobe screen (the star must fit inside its own lobe), a > R*/ell(M*/Mp), which is
# 1.2-2.6 R* depending on the mass ratio. Over outputs/data/stars.csv the tightest allowed orbit is
# 0.001 AU and the shortest possible period is 8.4e-5 yr (44 minutes), median 0.088 d, so
# P_MIN = 5e-5 yr brackets the prior with 1.7x margin. This is not a cosmetic bound: trials left
# outside the grid are scored "no significant peak" for the wrong reason. If the sma floor or the
# Roche screen changes, re-derive with
#     min(sqrt(max(a_min, roche_lobe_min_separation(R*, M*, M_mars))**3 / mass_interp)).
#
# Beyond ~9x the DR4 baseline the orbit is under-sampled and a "period" is not a real constraint --
# the signal is an astrometric acceleration, which the acceleration test carries. Those trials are
# kept anyway so the competitive region can extend to where the truth actually lives; expect them to
# show up as broad, edge-free plateaus rather than peaks. At the short end, a period of days is
# sampled by only ~90 DR4 transits over 5.5 yr, so expect those peaks to be heavily aliased against
# the scan law: a narrow peak at 20 d is not the same evidence as a narrow peak at 3 yr.
#
# Density is held at ~688 trials per e-fold in period (6.32e-4 dex), unchanged across every revision
# of these bounds, so width_dex (an absolute width in log-period) stays comparable with earlier runs
# and the grid-convergence tests carry over. Cost is linear in the number of trials: 1.42x the
# 0.01 yr grid. The look-elsewhere effect grows with the trial count, so the null thresholds MUST be
# recalibrated on 0_companion (notebook Step 2.5 does this automatically from the new control run) --
# the published Dchi2_orbit > 728 / Dchi2_accel > 76 were calibrated on the 8,739-trial grid and do
# NOT carry over. Note also that a period of hours is sampled by ~90 DR4 transits spread over 5.5 yr,
# so the short end of this grid is aliasing-dominated by construction.
P_MIN, P_MAX = 5.0e-5, 3300.0
PERIODS = ef.period_grid(P_MIN, P_MAX, 12383)
OUT_DIR = ef.DATA_ROOT

ALL_POPULATIONS = [
    "0_companion",
    "1_companion_agnostic",
    "1_companion_detectable",
    "2_companion_agnostic",
    "2_companion_detectable",
]


def run_one(population, sample_size=None, seed=0, out_suffix="", eccentric_refine=False):
    """Characterize a population and write outputs/data/characterization_<pop><suffix>.csv.

    If ``sample_size`` is given, a random subset of that many systems is processed (for a quick
    sanity run); otherwise all systems are processed. ``eccentric_refine`` adds the stage-2 eccentric
    (P, e, T_p) refinement of the top period for detected systems. Returns the output path.
    """
    h5 = ef.systems_h5_path(population)
    if not h5.exists():
        print(f"[skip] {population}: {h5} not found", flush=True)
        return None
    out = Path(OUT_DIR) / f"characterization_{population}{out_suffix}.csv"
    n_total = len(ef.load_truths(h5))
    rows = None
    if sample_size is not None and sample_size < n_total:
        rows = np.random.default_rng(seed).choice(n_total, int(sample_size), replace=False)
    n = n_total if rows is None else len(rows)
    print(f"[{population}] {n}/{n_total} systems -> {out}", flush=True)
    t0 = time.time()
    df = ef.characterize_population(h5, PERIODS, row_indices=rows,
                                   eccentric_refine=eccentric_refine, progress_every=2000)
    df.insert(0, "population", population)
    df.to_csv(out, index=False)
    dt = time.time() - t0
    det = df["detected"].mean()
    rel = df["period_reliable"].mean()
    rec_vals = df["period_recovered"].to_numpy(float)
    rec = np.nanmean(rec_vals) if np.isfinite(rec_vals).any() else np.nan
    print(f"[{population}] done in {dt/60:.1f} min | detected={det:.2f} "
          f"period_reliable={rel:.2f} period_recovered(truth)={rec:.2f}", flush=True)
    return str(out)


def _parse_args(argv):
    sample_size, seed, suffix, pops = None, 0, "", []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--sample":
            i += 1; sample_size = int(argv[i])
        elif a == "--seed":
            i += 1; seed = int(argv[i])
        elif a == "--suffix":
            i += 1; suffix = argv[i]
        else:
            pops.append(a)
        i += 1
    return (pops or ALL_POPULATIONS), sample_size, seed, suffix


if __name__ == "__main__":
    pops, sample_size, seed, suffix = _parse_args(sys.argv[1:])
    for pop in pops:
        run_one(pop, sample_size=sample_size, seed=seed, out_suffix=suffix)
