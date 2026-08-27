"""The pieces of `src/epochalypse_fitting.py` this run needs, and nothing else.

Six functions, copied verbatim in behaviour from the repository's analysis
module: the along-scan design matrix, the weighted least-squares chi^2, the
acceleration test, the peak finder, and the truth check. They are duplicated
rather than imported for two reasons. `epochalypse_fitting` imports h5py and
carries the whole serial-catalog HDF5 reader stack, none of which exists on the
shard side; and `periodograms/` has to be a tree that can be copied to a cluster
on its own.

What is deliberately *not* copied is `epochalypse_fitting.astrometric_periodogram`
-- the in-house profile-likelihood period search. That is the thing this package
replaces, and `periodogram.py` explains why.

The acceleration channel, by contrast, is the *same code* in both pipelines, so
a difference between the two output tables is attributable to the period search
rather than to the analysis built around it.
"""
from __future__ import annotations

import numpy as np

from . import config as C

TWOPI = 2.0 * np.pi


def epoch_arrays(obs_time_tcb, scan_pos_angle, parallax_factor_al,
                 centroid_pos_al, centroid_pos_error_al):
    """The along-scan fit arrays, from an epoch table's five columns.

    Returns `t` (yr from the DR4 reference epoch), `psi` (rad), the parallax
    factor, the along-scan centroid `y` (mas) and its reported uncertainty
    `yerr` (mas). Because `t` is in years, trial periods are in years directly.

    Takes columns rather than a DataFrame: at 17 M systems the per-system
    DataFrame round trip that `epochalypse_fitting.epoch_arrays` does is pure
    overhead, and the shard reader already holds the columns as arrays.
    """
    t = (np.asarray(obs_time_tcb, float) - C.GAIA_EPOCH_TCB_JD) / C.DAYS_PER_YEAR
    return (t, np.asarray(scan_pos_angle, float), np.asarray(parallax_factor_al, float),
            np.asarray(centroid_pos_al, float), np.asarray(centroid_pos_error_al, float))


def astrometric_design_matrix(t, scan_angle, pf):
    """Five-parameter along-scan design: [sin psi, cos psi, sin psi t, cos psi t, pf]."""
    return np.column_stack([
        np.sin(scan_angle), np.cos(scan_angle),
        np.sin(scan_angle) * t, np.cos(scan_angle) * t, pf,
    ])


def _wls_chi2(X, w, y):
    """Weighted least squares: returns (chi^2 of the fit, coefficients)."""
    Xw = X * w[:, None]
    beta = np.linalg.pinv(X.T @ Xw) @ (Xw.T @ y)
    resid = y - X @ beta
    return float(np.sum(w * resid**2)), beta


def acceleration_delta_chi2(t, scan_angle, pf, y, yerr):
    """Delta-chi^2 of a 7-parameter (5-par + along-scan acceleration) model.

    Adds quadratic-in-time along-scan terms (sin psi t^2, cos psi t^2). This is
    the data-only signature of a long-period companion whose orbit is
    under-sampled by the baseline (P >> T): the reflex shows up as an
    astrometric acceleration rather than a resolved orbit, which a bounded
    period periodogram would otherwise miss. Returns chi2_5par - chi2_7par >= 0.

    It is the second, independent detection channel, and it is what lets a
    system be called detected with no localizable period at all.
    """
    w = 1.0 / np.square(yerr)
    X5 = astrometric_design_matrix(t, scan_angle, pf)
    chi2_5, _ = _wls_chi2(X5, w, y)
    Xacc = np.hstack([X5, np.column_stack([np.sin(scan_angle) * t**2,
                                           np.cos(scan_angle) * t**2])])
    chi2_7, _ = _wls_chi2(Xacc, w, y)
    return chi2_5 - chi2_7


def _local_maxima(power):
    """Indices of strict interior local maxima of a 1-D array."""
    if len(power) < 3:
        return np.array([], dtype=int)
    left = power[1:-1] > power[:-2]
    right = power[1:-1] > power[2:]
    return np.where(left & right)[0] + 1


def find_peaks(periods, power, min_separation_dex=None):
    """Rank periodogram peaks by height (Delta-chi^2 vs the 5-par baseline).

    Peaks closer than `min_separation_dex` in log-period to an already-accepted,
    taller peak are merged (kept as the taller one) so a single broad mode is
    not counted several times. Returns dicts sorted by descending power.

    Grid-agnostic as written: it works off actual log-period values, not off
    index distances, so it needs no assumption about how the trial periods are
    spaced. That is why it survives the move to the segmented grid unchanged.
    """
    if min_separation_dex is None:
        min_separation_dex = C.MIN_SEPARATION_DEX
    idx = _local_maxima(power)
    gmax = int(np.argmax(power))            # keep the global argmax even on a boundary
    if gmax not in idx:
        idx = np.append(idx, gmax)
    idx = idx[np.argsort(-power[idx])]

    accepted, log_p = [], np.log10(periods)
    for i in idx:
        if any(abs(log_p[i] - log_p[j]) < min_separation_dex for j in accepted):
            continue
        accepted.append(i)
    return [{"index": int(i), "period": float(periods[i]), "power": float(power[i])}
            for i in accepted]


def period_in_competitive_region(periods, power, p_target, width_delta=None):
    """TRUTH check: is the injected period inside the competitive region?

    True if the power at `p_target` is within `width_delta` of the global
    maximum -- the data do not statistically exclude the true period. False if
    the true period is excluded (a confidently *biased* localization) or lies
    outside the tested grid. NaN if no companion was injected.
    """
    if width_delta is None:
        width_delta = C.WIDTH_DELTA
    if not np.isfinite(p_target) or p_target <= 0:
        return np.nan
    if p_target < periods[0] or p_target > periods[-1]:
        return False                         # beyond the tested grid -> not bracketed
    j = int(np.abs(np.log(periods) - np.log(p_target)).argmin())
    return bool(power[j] > power.max() - width_delta)
