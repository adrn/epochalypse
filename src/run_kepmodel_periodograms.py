"""Batch-run the ``kepmodel`` astrometric periodogram over the epochalypse populations.

This mirrors :mod:`run_periodograms` / :func:`epochalypse_fitting.characterize_population` and swaps
exactly one thing: the period search. J.-B. Delisle's ``kepmodel.astro.AstroModel.periodogram`` --
the periodogram used in the Gaia/OHP tutorial (``Gaia_OHP_sim_astrometry/gaia-astrometry-tutorial.ipynb``)
-- replaces the in-house profile-likelihood circular Thiele-Innes periodogram. The acceleration test,
the competitive-region test and the truth joins are the *same code* in both pipelines, so any
difference between the two output tables is attributable to the period search rather than to the
analysis built around it.

Usage:
    python run_kepmodel_periodograms.py [--jitter] [--sample 500] [<population> ...]

Writes ``outputs/data/characterization_kepmodel[_jitter]_<population>.csv`` with the same schema as
``characterization_<population>.csv``, so the shared figure machinery in ``epochalypse_figures``
consumes either without modification.

Five conventions are worth stating explicitly.

**Model.** The five linear columns added here (parallax factor, the two position offsets and the two
proper-motion terms, each projected on the scan angle) span exactly the same subspace as
``epochalypse_fitting.astrometric_design_matrix``, and ``kepmodel``'s four periodogram columns
(``cth cos nu t, sth cos nu t, cth sin nu t, sth sin nu t``) span exactly
``epochalypse_fitting.orbit_columns``. With ``fit_jitter=False`` the two periodograms are therefore
the *same statistic* in exact arithmetic -- which is the point: it validates the port, and any
disagreement in the tables is a conditioning, grid or noise-model effect, not a model one.

**Grid: kepmodel's own, no interpolation.** ``kepmodel`` advances the trial frequency by a *fixed*
step (it propagates ``cos nu t`` / ``sin nu t`` by a rotation), so it evaluates only on uniform
frequency grids. Every number reported here is an exact ``kepmodel`` evaluation on such a grid -- the
periodogram is *not* resampled onto the baseline's log-uniform period grid, and the classifier runs
on the trial periods ``kepmodel`` actually visited.

A single uniform frequency grid, the way the tutorial sets one up, is available with
``frequency_segments(nfreq=...)``; it is what ``kepmodel`` does out of the box and it is fine for
detection. It is a poor fit for *this* measurement, though, because the classification spans five
and a half decades in period: a uniform frequency step fine enough for the short-period end resolves
log-period ~1e-5 dex there (pointlessly fine) and ~0.1 dex at 3300 yr, leaving a handful of samples
in the last decade -- and the last decade is where the significant-but-unlocalized systems live.
Uniform in frequency is the natural sampling for a detection statistic and the wrong one for a
period *classification*.

The default therefore searches one uniform frequency grid per log-period segment
(``N_SEGMENTS`` of them) and reports the union of their sample points. Every sample is still a native
``kepmodel`` evaluation; the segments only let the frequency step be refreshed as the trial period
grows. Each segment's step is set so its coarsest log-period spacing is at most ``target_dlog``,
which defaults to the baseline's grid spacing -- so the search is at least as finely sampled in
log-period as ``run_periodograms.PERIODS`` *everywhere*, at ~35% more evaluations than the baseline
has trial periods. ``N_SEGMENTS`` is the only cost knob: the total tends to the count of a genuinely
log-uniform grid as the segments narrow (more segments, less overshoot), and the default is set to
keep each segment spanning a period ratio of ~1.5, within ~20% of that limit. It tracks
``run_periodograms.P_MIN``/``P_MAX`` automatically, so it needs revisiting only if the *span*
changes a lot: at 32 segments over 0.01-3300 yr each segment is 0.17 dex, the same as the 24
segments that covered 0.3-3300 yr.

**Classifier.** ``epochalypse_fitting.classify_periodogram`` is reused for the peak finding
(``find_peaks`` works off actual log-period values and is grid-agnostic) but *not* for the width
metric: ``epochalypse_fitting.period_constraint`` measures the competitive region as the *fraction of
grid points* within ``width_delta`` of the maximum, which is a log-period width only on a uniform log
grid. On the grid used here that would badly understate a long-period plateau. :func:`period_constraint`
below measures the same quantity as an actual log-period measure (a sum of cell widths), and
:func:`classify_periodogram` reproduces the baseline's branching verbatim on top of it. On a uniform
log grid the two agree to O(1/N); the edge test is likewise expressed as a fraction of the log-period
*range* rather than of the point count, which is the same thing on the baseline grid.

**Scale.** ``kepmodel`` returns the normalised power ``z = 1 - chi2(nu)/chi2_base`` in [0, 1]; the
classifier expects a Delta-chi^2 curve, so we report ``z * chi2_base``. ``chi2_base`` is the profiled
chi^2 of the five-parameter model in the whitened metric, i.e. exactly the baseline's ``chi2_5par``
when no jitter is fitted, so ``width_delta=4`` and ``delta_power_unimodal=10`` keep their meaning.

**Conditioning (a real difference, not a rounding one).** Where the 9-column design is well
conditioned -- P below a few tens of years -- the two periodograms agree to ~1e-7 of the peak,
evaluated at the same trial periods. Beyond P ~ 100 yr they do not, and ``kepmodel`` is the one that
is right. As P grows past the baseline the four orbit columns collapse onto the position and
proper-motion columns (cos nu t -> 1, sin nu t -> nu t) and the 4x4 normal matrix becomes
near-singular; ``epochalypse_fitting._periodogram_core`` damps that with a ``ridge=1e-8``, which
progressively suppresses the long-period tail, while ``kepmodel`` solves it unregularised. Checked
against an SVD least-squares reference on the full whitened design, ``kepmodel`` tracks the exact
profile Delta-chi^2 to ~1e-4 relative out to 3300 yr; the ridged baseline falls below it by up to
~40% there. The visible consequence in the tables: for broad, long-period systems the baseline's
ridge tilts the plateau and plants a spurious interior argmax around 100-300 yr, whereas the kepmodel
curve stays flat and rails at the grid edge. ``klass``, ``top_power`` and the detection flag are
essentially unaffected (the peak itself is short-period); ``best_period`` for already-``broad``
systems is not, and it should not be trusted in either run.

**Noise.** ``fit_jitter=False`` (the default) keeps the fixed ``1/sigma_formal^2`` weights of the
baseline, so this run is a like-for-like swap of the period search alone. ``fit_jitter=True`` fits
``kepmodel``'s excess-noise term on the *companion-free* model before the search -- the thing the
baseline notebook (Step 2.5) notes it cannot do and compensates for with an empirical null
calibration. It is a genuinely different noise model, not a refinement, and the effect is large:
because the term is fitted with no orbit in the model, it absorbs the companion's signal along with
the excess scatter, and ``chi2_base`` (hence the whole Delta-chi^2 scale) collapses by one to two
orders of magnitude on strongly-signalled systems. Detections are suppressed along with false
positives. The detection threshold is recalibrated on the companion-free control either way, so both
runs are internally self-consistent, but their ``top_power`` columns are on different scales and
must not be compared numerically.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import epochalypse_fitting as ef
import run_periodograms as rp

from kepmodel.astro import AstroModel  # noqa: E402
from spleaf import term  # noqa: E402

OUT_DIR = ef.DATA_ROOT
ALL_POPULATIONS = rp.ALL_POPULATIONS

# Search-grid controls (see the "Grid" note in the module docstring).
N_SEGMENTS = 32        # log-period segments, each searched with its own uniform frequency step
TARGET_DLOG = None     # coarsest log10-period spacing to allow; None -> the baseline grid's spacing

TWOPI = 2.0 * np.pi


def build_model(t, psi, pf, y, yerr, fit_jitter=False):
    """The five-parameter single-star ``kepmodel`` astrometric model, linear parameters fitted.

    ``t`` is in years from the DR4 reference epoch (as returned by ``ef.epoch_arrays``), so trial
    periods are in years directly. The scan-angle convention is ``kepmodel``'s: the along-scan
    abscissa is ``cth * ddelta + sth * dalpha``.

    Returns ``(model, excess_noise_mas)``. With ``fit_jitter=True`` the excess-noise term is fitted
    jointly with the linear parameters on the companion-free model; if that fit fails to converge we
    fall back to zero jitter (and report it as NaN) rather than dropping the system.
    """
    cth, sth = np.cos(psi), np.sin(psi)
    model = AstroModel(
        t, y, cth, sth,
        err=term.Error(yerr),
        excess_noise=term.Jitter(0.0),
    )
    model.add_lin(pf, 'plx')
    model.add_lin(cth, 'delta')
    model.add_lin(sth, 'alpha')
    model.add_lin(t * cth, 'mu_delta')
    model.add_lin(t * sth, 'mu_alpha')
    model.fit_lin()

    excess = 0.0
    if fit_jitter:
        model.fit_param = list(model.fit_param) + ['cov.excess_noise.sig']
        try:
            model.fit()
            excess = float(model.get_param('cov.excess_noise.sig'))
        except Exception:                      # non-convergence -> keep the fixed-weight model
            model.set_param(0.0, 'cov.excess_noise.sig')
            model.fit_lin()
            excess = np.nan
    return model, excess


def frequency_segments(p_min=None, p_max=None, n_segments=N_SEGMENTS,
                       target_dlog=None, nfreq=None):
    """The ``(nu0, dnu, nfreq)`` grids to hand to ``kepmodel.periodogram``, covering [p_min, p_max].

    With ``nfreq`` given, returns the single uniform frequency grid of the Gaia/OHP tutorial.
    Otherwise returns ``n_segments`` grids, one per equal interval of log-period, each stepped
    finely enough that its coarsest log10-period spacing is at most ``target_dlog`` (default: the
    spacing of ``run_periodograms.PERIODS``). Within a segment the log-period spacing is coarsest at
    the long-period end, which is what sets the step.
    """
    p_min = rp.P_MIN if p_min is None else float(p_min)
    p_max = rp.P_MAX if p_max is None else float(p_max)
    nu_min, nu_max = TWOPI / p_max, TWOPI / p_min

    if nfreq is not None:                       # tutorial-style: one uniform frequency grid
        return [(nu_min, (nu_max - nu_min) / (int(nfreq) - 1), int(nfreq))]

    if target_dlog is None:
        target_dlog = TARGET_DLOG
    if target_dlog is None:                     # match the baseline period grid's resolution
        target_dlog = float(np.diff(np.log10(rp.PERIODS)).max())

    edges = np.exp(np.linspace(np.log(p_max), np.log(p_min), n_segments + 1))
    segs = []
    for p_hi, p_lo in zip(edges[:-1], edges[1:]):      # descending period -> ascending frequency
        nu_lo, nu_hi = TWOPI / p_hi, TWOPI / p_lo
        # dlogP = dnu * P / (2 pi ln10), worst at the segment's long-period (low-frequency) end
        dnu_max = target_dlog * TWOPI * np.log(10.0) / p_hi
        n = max(int(np.ceil((nu_hi - nu_lo) / dnu_max)) + 1, 3)
        segs.append((float(nu_lo), float((nu_hi - nu_lo) / (n - 1)), int(n)))
    return segs


def segment_periods(segments):
    """The trial periods a set of ``frequency_segments`` visits, ascending and de-duplicated."""
    nu = np.concatenate([nu0 + np.arange(n) * dnu for nu0, dnu, n in segments])
    nu = np.unique(nu)                                  # ascending frequency, no repeated edges
    return (TWOPI / nu)[::-1]                           # -> ascending period


def kepmodel_power(t, psi, pf, y, yerr, segments=None, fit_jitter=False):
    """Run the kepmodel periodogram; return ``(periods, power, info)``.

    ``periods`` are the trial periods kepmodel visited (ascending), ``power`` the Delta-chi^2
    ``chi2_base - chi2(period)`` at each of them -- no interpolation anywhere. ``info`` carries the
    base chi^2, the fitted excess noise, the analytic FAP of the highest peak and the number of trial
    frequencies evaluated.
    """
    segments = frequency_segments() if segments is None else segments
    model, excess = build_model(t, psi, pf, y, yerr, fit_jitter=fit_jitter)

    # chi2 of the fitted five-parameter model in the whitened metric == kepmodel's periodogram
    # normalisation chi2_base (all linear parameters are fitted, so the profiled value is the fitted
    # one). Without jitter this is identical to epochalypse's chi2_5par.
    u = model.cov.solveL(model.residuals()) / model.cov.sqD()
    chi2_base = float(u @ u)

    nus, zs = [], []
    for nu0, dnu, nfreq in segments:
        nu, z = model.periodogram(nu0, dnu, nfreq)
        nus.append(nu)
        zs.append(z)
    nu = np.concatenate(nus)
    z = np.concatenate(zs)

    order = np.argsort(nu)                              # merge the segments, drop shared edges
    nu, z = nu[order], z[order]
    keep = np.concatenate([[True], np.diff(nu) > 0])
    nu, z = nu[keep], z[keep]

    try:
        fap = float(model.fap(float(z.max()), float(nu.max())))
    except Exception:
        fap = np.nan

    periods = (TWOPI / nu)[::-1]                        # ascending period
    power = (z * chi2_base)[::-1]
    return periods, power, {"chi2_base": chi2_base, "excess_noise_mas": excess,
                            "fap": fap, "n_freq": int(nu.size)}


# --------------------------------------------------------------------------------------
# Grid-agnostic versions of the two classifier pieces that assume a uniform log-period grid
# --------------------------------------------------------------------------------------
def period_constraint(periods, power, width_delta=4.0, edge_frac=0.02):
    """``ef.period_constraint`` for an arbitrary (non-log-uniform) period grid.

    The baseline measures the competitive region as (fraction of grid points within ``width_delta``
    of the maximum) x (total log range), which is a log-period width only when the grid is uniform in
    log-period. Here it is summed as an actual measure: each competitive sample contributes its own
    log-period cell width. The edge test likewise asks whether the maximum sits within ``edge_frac``
    of the log-period *range* of either end, rather than within that fraction of the point count.
    Both reduce to the baseline definitions on a uniform log grid.
    """
    logp = np.log10(periods)
    cell = np.empty_like(logp)                          # midpoint (trapezoidal) cell widths
    cell[1:-1] = 0.5 * (logp[2:] - logp[:-2])
    cell[0] = logp[1] - logp[0]
    cell[-1] = logp[-1] - logp[-2]

    gi = int(np.argmax(power))
    comp = power > power[gi] - width_delta
    width_dex = float(cell[comp].sum())
    span = logp[-1] - logp[0]
    at_edge = bool(logp[gi] - logp[0] <= edge_frac * span
                   or logp[-1] - logp[gi] <= edge_frac * span)
    return width_dex, float(periods[gi]), at_edge


def classify_periodogram(periods, power, n_epochs, n_orbit_params=4,
                         delta_bic_detect=10.0, delta_power_unimodal=10.0,
                         n_weak_max=3, min_separation_dex=0.1,
                         width_delta=4.0, width_constrained_dex=0.05):
    """``ef.classify_periodogram`` with the grid-agnostic width above; branching is verbatim.

    Peak finding is ``ef.find_peaks`` unchanged -- it works off actual log-period values and so needs
    no grid assumption. Returns the same dict, so the callers downstream cannot tell the difference.
    """
    width_dex, _, best_at_edge = period_constraint(periods, power, width_delta)

    peaks = ef.find_peaks(periods, power, n_orbit_params, min_separation_dex)
    if not peaks:
        return {"klass": "undetected", "best_period": np.nan, "n_competitive": 0,
                "top_power": np.nan, "delta_bic_best": np.nan,
                "width_dex": width_dex, "best_at_edge": best_at_edge,
                "top_periods": [], "top_powers": []}

    best = peaks[0]
    delta_bic_best = best["power"] - n_orbit_params * np.log(max(n_epochs, 2))
    competitive = [p for p in peaks if best["power"] - p["power"] < delta_power_unimodal]
    n_competitive = len(competitive)

    if delta_bic_best < delta_bic_detect:
        klass = "undetected"
    elif width_dex > width_constrained_dex or best_at_edge:
        klass = "broad"
    elif n_competitive >= 2:
        klass = "multimodal"
    else:
        klass = "unimodal"

    top = peaks[:5]
    return {"klass": klass, "best_period": best["period"],
            "n_competitive": int(n_competitive), "top_power": best["power"],
            "delta_bic_best": float(delta_bic_best),
            "width_dex": width_dex, "best_at_edge": best_at_edge,
            "top_periods": [p["period"] for p in top],
            "top_powers": [p["power"] for p in top]}


def characterize_population_kepmodel(systems_h5, segments=None, fit_jitter=False, row_indices=None,
                                     detect_delta_bic=10.0, period_recover_tol=1.2,
                                     period_reliable_baseline_frac=1.0, width_delta=4.0,
                                     progress_every=2000):
    """The kepmodel counterpart of :func:`epochalypse_fitting.characterize_population`.

    Every step other than the periodogram and its width metric calls the same epochalypse function
    used by the baseline run, so the two tables differ only through the period search.
    """
    import h5py

    systems_h5 = Path(systems_h5)
    truths = ef.load_truths(systems_h5)
    if row_indices is None:
        row_indices = np.arange(len(truths))
    segments = frequency_segments() if segments is None else segments

    keep_cols = [c for c in (
        "system_id", "gaia_source_id", "n_planets", "period_1", "period_2",
        "a_1_au", "a_2_au", "e_1", "e_2", "Mp_1_msun", "Mp_2_msun",
        "alpha_1_mas", "alpha_2_mas", "snr_single_1", "snr_single_2",
        "snr_total_1", "snr_total_2", "n_transits_dr4", "sigma_single_mas",
        "near_resonance", "coplanar",
    ) if c in truths.columns]

    records = []
    with h5py.File(systems_h5, "r") as f:
        for count, ri in enumerate(row_indices):
            ri = int(ri)
            grp = f[f"systems/{ri}"]
            if "epochs" not in grp:
                continue
            ep = ef._decode_str_cols(pd.DataFrame(grp["epochs"][:]))
            ep = ep.sort_values("obs_time_tcb").reset_index(drop=True)
            t, psi, pf, y, yerr = ef.epoch_arrays(ep)
            n = len(y)

            periods, power, info = kepmodel_power(t, psi, pf, y, yerr, segments=segments,
                                                  fit_jitter=fit_jitter)
            res = classify_periodogram(periods, power, n, width_delta=width_delta)
            accel_dchi2 = ef.acceleration_delta_chi2(t, psi, pf, y, yerr)
            accel_dbic = accel_dchi2 - 2.0 * np.log(max(n, 2))

            detected = (res["delta_bic_best"] >= detect_delta_bic) or (accel_dbic >= detect_delta_bic)
            period_reliable = bool(
                detected and res["klass"] == "unimodal"
                and res["best_period"] < period_reliable_baseline_frac * ef.DR4_BASELINE_YEARS
            )

            rec = {"row_index": ri, "n_epochs": n, "chi2_5par": info["chi2_base"],
                   "klass": res["klass"], "best_period": res["best_period"],
                   "best_period_ecc": np.nan, "e_ecc": np.nan,
                   "n_competitive": res["n_competitive"], "top_power": res["top_power"],
                   "delta_bic_best": res["delta_bic_best"],
                   "width_dex": res["width_dex"], "best_at_edge": res["best_at_edge"],
                   "accel_delta_chi2": float(accel_dchi2), "accel_delta_bic": float(accel_dbic),
                   "detected": bool(detected), "period_reliable": period_reliable,
                   # kepmodel-specific bookkeeping
                   "kepmodel_fap": info["fap"],
                   "kepmodel_excess_noise_mas": info["excess_noise_mas"],
                   "kepmodel_n_freq": info["n_freq"]}
            for k in range(2):
                rec[f"peak{k+1}_period"] = (res["top_periods"][k]
                                            if k < len(res["top_periods"]) else np.nan)
            row = truths.iloc[ri]
            for c in keep_cols:
                rec[c] = row[c]
            for k in (1, 2):
                col = f"period_{k}"
                pk = float(row[col]) if col in truths.columns and pd.notna(row[col]) else np.nan
                rec[f"period_{k}_in_bound"] = ef.period_in_competitive_region(
                    periods, power, pk, width_delta)
            records.append(rec)
            if progress_every and (count + 1) % progress_every == 0:
                print(f"  {count + 1}/{len(row_indices)} systems", flush=True)

    df = pd.DataFrame.from_records(records)

    # truth-based recovery flags, matching the baseline definitions
    for k in (1, 2):
        col = f"period_{k}"
        if col in df.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                df[f"period_{k}_recovered"] = (
                    np.abs(np.log(df["best_period"] / df[col])) < np.log(period_recover_tol))
    if "period_1" in df.columns:
        df["period_recovered"] = df["period_1_recovered"].astype(float)
    for a, b in (("snr_total", "snr_total"), ("snr_single", "snr_single")):
        cols = [c for c in (f"{a}_1", f"{a}_2") if c in df.columns and df[c].notna().any()]
        df[b] = np.nanmax(df[cols].to_numpy(float), axis=1) if cols else np.nan
    return df


def tag(fit_jitter=False):
    """Filename tag for a run: ``_kepmodel`` or ``_kepmodel_jitter``."""
    return "_kepmodel_jitter" if fit_jitter else "_kepmodel"


def run_one(population, fit_jitter=False, sample_size=None, seed=0, segments=None, out_suffix=""):
    """Characterize one population with the kepmodel periodogram; returns the output path."""
    segments = frequency_segments() if segments is None else segments
    h5 = ef.systems_h5_path(population)
    if not h5.exists():
        print(f"[skip] {population}: {h5} not found", flush=True)
        return None
    out = Path(OUT_DIR) / f"characterization{tag(fit_jitter)}_{population}{out_suffix}.csv"
    n_total = len(ef.load_truths(h5))
    rows = None
    if sample_size is not None and sample_size < n_total:
        rows = np.random.default_rng(seed).choice(n_total, int(sample_size), replace=False)
    n = n_total if rows is None else len(rows)
    print(f"[{population}] jitter={fit_jitter}  {n}/{n_total} systems, "
          f"{sum(s[2] for s in segments)} trial frequencies -> {out}", flush=True)
    t0 = time.time()
    df = characterize_population_kepmodel(h5, segments=segments, fit_jitter=fit_jitter,
                                          row_indices=rows)
    df.insert(0, "population", population)
    df.to_csv(out, index=False)
    dt = time.time() - t0
    rec = df["period_recovered"].to_numpy(float)
    print(f"[{population}] jitter={fit_jitter} done in {dt/60:.1f} min | "
          f"detected={df['detected'].mean():.2f} "
          f"period_recovered={np.nanmean(rec) if np.isfinite(rec).any() else float('nan'):.2f}",
          flush=True)
    return str(out)


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("populations", nargs="*", default=None)
    p.add_argument("--jitter", action="store_true",
                   help="fit kepmodel's excess-noise term on the companion-free model first")
    p.add_argument("--segments", type=int, default=N_SEGMENTS,
                   help="log-period segments, each with its own uniform frequency step")
    p.add_argument("--nfreq", type=int, default=None,
                   help="instead, use ONE uniform frequency grid of this size (tutorial style)")
    p.add_argument("--sample", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suffix", default="")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse(sys.argv[1:])
    segs = frequency_segments(n_segments=args.segments, nfreq=args.nfreq)
    for pop in (args.populations or ALL_POPULATIONS):
        run_one(pop, fit_jitter=args.jitter, sample_size=args.sample,
                seed=args.seed, segments=segs, out_suffix=args.suffix)
