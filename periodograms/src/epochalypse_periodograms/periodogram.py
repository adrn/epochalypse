"""The period search: `kepmodel`'s astrometric periodogram, one system at a time.

This is `src/run_kepmodel_periodograms.py` reduced to its per-system core, with
the population loop, the HDF5 readers and the CSV writer removed -- those are
`shards.py` and `writers.py` here. The statistic is unchanged, and five
conventions are worth restating because they are what the numbers mean.

**Model.** The five linear columns fitted here (parallax factor, two position
offsets, two proper-motion terms, each projected on the scan angle) span exactly
the same subspace as `fitting.astrometric_design_matrix`, and `kepmodel`'s four
periodogram columns (`cth cos nu t, sth cos nu t, cth sin nu t, sth sin nu t`)
span exactly the in-house pipeline's circular Thiele-Innes columns. With
`fit_jitter=False` the two periodograms are therefore the *same statistic* in
exact arithmetic, computed along completely different routes.

**Grid: kepmodel's own, no interpolation.** Every number reported here is an
exact `kepmodel` evaluation on a uniform frequency grid. The periodogram is not
resampled onto a log-uniform period grid, and the classifier runs on the trial
periods `kepmodel` actually visited. See `grid.py`.

**Scale.** `kepmodel` returns the normalised power `z = 1 - chi2(nu)/chi2_base`
in [0, 1]; the classifier wants a Delta-chi^2 curve, so `z * chi2_base` is what
is reported and stored. `chi2_base` is the profiled chi^2 of the five-parameter
model in the whitened metric -- exactly the in-house `chi2_5par` when no jitter
is fitted -- so `WIDTH_DELTA = 4` and `DELTA_POWER_UNIMODAL = 10` keep their
meaning.

**Conditioning (a real difference, not a rounding one).** Where the 9-column
design is well conditioned -- P below a few tens of years -- `kepmodel` and the
in-house periodogram agree to ~1e-7 of the peak at identical trial periods.
Beyond P ~ 100 yr they do not, and `kepmodel` is the one that is right: as P
grows past the baseline the four orbit columns collapse onto the position and
proper-motion columns (cos nu t -> 1, sin nu t -> nu t) and the 4x4 normal
matrix goes near-singular. The in-house code damps that with a ridge, which
progressively suppresses the long-period tail; `kepmodel` solves it
unregularised and tracks an SVD reference to ~1e-4 out to 3300 yr. The visible
consequence is confined to systems that are already classified `broad`: the
ridged version's tilted plateau plants a spurious interior argmax around
100-300 yr where the true curve is flat to the grid edge. `klass`, `top_power`
and the detection flag are unaffected; `best_period` for a `broad` system is
not, and should not be read out of either table.

**Width metric.** `find_peaks` is reused unchanged -- it works off actual
log-period values and is grid-agnostic. The competitive-width metric is not:
the in-house `period_constraint` measures the competitive region as the
*fraction of grid points* within `WIDTH_DELTA` of the maximum, which is a
log-period width only on a uniform log grid, and on the segmented grid that
would badly understate a long-period plateau. `period_constraint` below
measures the same quantity as an actual log-period measure -- a sum of cell
widths -- and reduces to the baseline definition on a uniform log grid.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from . import fitting
from .grid import TWOPI, frequency_segments


def build_model(t, psi, pf, y, yerr, fit_jitter=False):
    """The five-parameter single-star `kepmodel` astrometric model, linear terms fitted.

    The scan-angle convention is `kepmodel`'s: the along-scan abscissa is
    `cth * ddelta + sth * dalpha`. Returns `(model, excess_noise_mas)`. With
    `fit_jitter=True` the excess-noise term is fitted jointly with the linear
    parameters on the companion-free model; a fit that fails to converge falls
    back to zero jitter (reported as NaN) rather than dropping the system.

    The imports are function-local so that `grid.py`, `config.py` and the
    manifest can be read on a machine with no `kepmodel` install -- which is
    what the merge and calibrate stages do.
    """
    from kepmodel.astro import AstroModel
    from spleaf import term

    cth, sth = np.cos(psi), np.sin(psi)
    model = AstroModel(t, y, cth, sth,
                       err=term.Error(yerr),
                       excess_noise=term.Jitter(0.0))
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
        except Exception:                     # non-convergence -> the fixed-weight model
            model.set_param(0.0, 'cov.excess_noise.sig')
            model.fit_lin()
            excess = np.nan
    return model, excess


def kepmodel_power(t, psi, pf, y, yerr, segments=None, fit_jitter=False):
    """Run the periodogram; return `(periods, power, info)`.

    `periods` are the trial periods kepmodel visited (ascending), `power` the
    Delta-chi^2 `chi2_base - chi2(period)` at each of them -- no interpolation
    anywhere. `info` carries the base chi^2, the fitted excess noise, the
    analytic FAP of the highest peak, and the number of trial frequencies.

    `periods` is `grid.segment_periods(segments)` by construction: the same
    concatenate / unique / reverse, applied to the same frequencies. That
    identity is what lets the period axis live in one file per run instead of
    beside every stored curve, and `tests/test_periodograms.py` asserts it.
    """
    segments = frequency_segments() if segments is None else segments
    model, excess = build_model(t, psi, pf, y, yerr, fit_jitter=fit_jitter)

    # chi2 of the fitted five-parameter model in the whitened metric == kepmodel's
    # normalisation chi2_base (every linear parameter is fitted, so the profiled
    # value is the fitted one). Without jitter this is the in-house chi2_5par.
    u = model.cov.solveL(model.residuals()) / model.cov.sqD()
    chi2_base = float(u @ u)

    nus, zs = [], []
    for nu0, dnu, nfreq in segments:
        nu, z = model.periodogram(nu0, dnu, nfreq)
        nus.append(nu)
        zs.append(z)
    nu, z = np.concatenate(nus), np.concatenate(zs)

    order = np.argsort(nu)                          # merge segments, drop shared edges
    nu, z = nu[order], z[order]
    keep = np.concatenate([[True], np.diff(nu) > 0])
    nu, z = nu[keep], z[keep]

    try:
        fap = float(model.fap(float(z.max()), float(nu.max())))
    except Exception:
        fap = np.nan

    return (TWOPI / nu)[::-1], (z * chi2_base)[::-1], {
        "chi2_base": chi2_base, "excess_noise_mas": excess,
        "fap": fap, "n_freq": int(nu.size)}


def period_constraint(periods, power, width_delta=None, edge_frac=None):
    """How tightly the periodogram localizes the period, on an arbitrary grid.

    The competitive region is the set of trial periods within `width_delta`
    (Delta-chi^2) of the global maximum. It is summed here as an actual measure
    -- each competitive sample contributes its own log-period cell width --
    rather than as a fraction of the point count, which is a width only on a
    uniform log grid. The edge test likewise asks whether the maximum sits
    within `edge_frac` of the log-period *range* of either end. Both reduce to
    the in-house definitions on a uniform log grid.

    This measures peak WIDTH, not peak count, on purpose: the profile-likelihood
    periodogram of a long-period, under-sampled orbit goes broad and flat-topped
    (the continuous period-eccentricity-acceleration degeneracy) rather than
    splitting into discrete peaks, and peak counting mislabels that "unimodal".

    Returns `(width_dex, best_period, best_at_edge)`.
    """
    width_delta = C.WIDTH_DELTA if width_delta is None else width_delta
    edge_frac = C.EDGE_FRAC if edge_frac is None else edge_frac

    logp = np.log10(periods)
    cell = np.empty_like(logp)                      # midpoint (trapezoidal) widths
    cell[1:-1] = 0.5 * (logp[2:] - logp[:-2])
    cell[0] = logp[1] - logp[0]
    cell[-1] = logp[-1] - logp[-2]

    gi = int(np.argmax(power))
    comp = power > power[gi] - width_delta
    span = logp[-1] - logp[0]
    at_edge = bool(logp[gi] - logp[0] <= edge_frac * span
                   or logp[-1] - logp[gi] <= edge_frac * span)
    return float(cell[comp].sum()), float(periods[gi]), at_edge


def classify_periodogram(periods, power, n_epochs):
    """Classify one periodogram into a characterizability category.

      - `undetected` : the best peak does not improve BIC over the 5-par model
                       by `DELTA_BIC_DETECT`; an orbit is not even preferred.
      - `broad`      : detected, but the competitive region is wider than
                       `WIDTH_CONSTRAINED_DEX` or the argmax is railed to a
                       grid edge -- the period is not localized.
      - `multimodal` : narrow, but two or more separated competitive peaks.
      - `unimodal`   : narrow, single.

    Note that `undetected` here is the classifier's own internal threshold, far
    below the null-calibrated detection threshold in `calibrate.py`. A system
    can be `unimodal` and still fail the calibrated cut; that is exactly why
    `apply_calibration` records the peak channel separately.
    """
    width_dex, _, best_at_edge = period_constraint(periods, power)
    peaks = fitting.find_peaks(periods, power)
    if not peaks:                                   # cannot happen: argmax is always kept
        return {"klass": "undetected", "best_period": np.nan, "n_competitive": 0,
                "top_power": np.nan, "delta_bic_best": np.nan,
                "width_dex": width_dex, "best_at_edge": best_at_edge,
                "top_periods": [], "top_powers": []}

    best = peaks[0]
    # BIC relative to the 5-par model: k orbit params cost k ln N.
    delta_bic_best = best["power"] - C.N_ORBIT_PARAMS * np.log(max(n_epochs, 2))
    competitive = [p for p in peaks
                   if best["power"] - p["power"] < C.DELTA_POWER_UNIMODAL]

    if delta_bic_best < C.DELTA_BIC_DETECT:
        klass = "undetected"
    elif width_dex > C.WIDTH_CONSTRAINED_DEX or best_at_edge:
        klass = "broad"
    elif len(competitive) >= 2:
        klass = "multimodal"
    else:
        klass = "unimodal"

    top = peaks[:5]
    return {"klass": klass, "best_period": best["period"],
            "n_competitive": int(len(competitive)), "top_power": best["power"],
            "delta_bic_best": float(delta_bic_best),
            "width_dex": width_dex, "best_at_edge": best_at_edge,
            "top_periods": [p["period"] for p in top],
            "top_powers": [p["power"] for p in top]}


def characterize_system(t, psi, pf, y, yerr, truth=None, segments=None,
                        fit_jitter=None, want_power=False):
    """Search, classify, and test one system. Returns `(record, power)`.

    `record` is the one row this system contributes to the characterization
    table -- the periodogram summary, the acceleration channel, the data-only
    detection flags, and the truth-based recovery flags when `truth` is given.
    `power` is the raw Delta-chi^2 curve if `want_power`, else None; the period
    axis is not returned because it is the same for every system in a run.

    `truth` is a mapping (a pandas row, or a dict) carrying the injected
    `period_k`. Nothing else about it is read here -- the rest of the truth
    columns are joined by the writer, which has the whole shard's table.
    """
    fit_jitter = C.FIT_JITTER if fit_jitter is None else fit_jitter
    n = len(y)

    periods, power, info = kepmodel_power(t, psi, pf, y, yerr,
                                          segments=segments, fit_jitter=fit_jitter)
    res = classify_periodogram(periods, power, n)
    accel_dchi2 = fitting.acceleration_delta_chi2(t, psi, pf, y, yerr)
    accel_dbic = accel_dchi2 - 2.0 * np.log(max(n, 2))

    # The data-only flags, on the classifier's internal threshold. The
    # null-calibrated versions (`detected_cal`, ...) are added downstream by
    # `calibrate.apply_calibration`, once the control population has run and
    # the thresholds exist; these two are kept because the in-house tables
    # carry them and the two runs stay column-comparable.
    detected = (res["delta_bic_best"] >= C.DELTA_BIC_DETECT) or (accel_dbic >= C.DELTA_BIC_DETECT)
    period_reliable = bool(detected and res["klass"] == "unimodal"
                           and res["best_period"] < C.DR4_BASELINE_YEARS)

    record = {
        "n_epochs": n,
        "chi2_5par": info["chi2_base"],
        "klass": res["klass"],
        "best_period": res["best_period"],
        "n_competitive": res["n_competitive"],
        "top_power": res["top_power"],
        "delta_bic_best": res["delta_bic_best"],
        "width_dex": res["width_dex"],
        "best_at_edge": res["best_at_edge"],
        "accel_delta_chi2": float(accel_dchi2),
        "accel_delta_bic": float(accel_dbic),
        "detected": bool(detected),
        "period_reliable": period_reliable,
        "kepmodel_fap": info["fap"],
        "kepmodel_excess_noise_mas": info["excess_noise_mas"],
    }
    for k in range(2):                              # the two tallest peaks, for the truth match
        record[f"peak{k + 1}_period"] = (res["top_periods"][k]
                                         if k < len(res["top_periods"]) else np.nan)
        record[f"peak{k + 1}_power"] = (res["top_powers"][k]
                                        if k < len(res["top_powers"]) else np.nan)

    # Truth-based flags. `period_k_in_bound` is the honest question to ask of a
    # broad system -- does the competitive REGION bracket the truth -- where a
    # point-estimate comparison would call an imprecise-but-correct constraint
    # an error. `period_k_recovered` is the point estimate, for the narrow ones.
    #
    # Both are floats (1.0 / 0.0 / NaN), not bools, and that is not cosmetic.
    # `ParquetWriter` fixes the schema on the first row group, so a column whose
    # dtype depends on whether a particular batch happened to contain an
    # un-injected companion would fail the shard halfway through. NaN is also
    # what `epochalypse_figures.classes` expects -- it does `.fillna(0)` on
    # `period_k_in_bound`.
    if truth is not None:
        keys = truth.index if hasattr(truth, "index") else truth
        ln_tol = np.log(C.PERIOD_RECOVER_TOL)
        for k in (1, 2):
            if f"period_{k}" not in keys:        # population-level, so schema-stable
                continue
            record[f"period_{k}_in_bound"] = np.nan
            record[f"period_{k}_recovered"] = np.nan
            pk = float(truth[f"period_{k}"])
            if not np.isfinite(pk):              # no companion injected in this slot
                continue
            in_bound = fitting.period_in_competitive_region(periods, power, pk)
            record[f"period_{k}_in_bound"] = float(in_bound)
            with np.errstate(invalid="ignore", divide="ignore"):
                record[f"period_{k}_recovered"] = float(
                    abs(np.log(res["best_period"] / pk)) < ln_tol)

    return record, (power if want_power else None)
