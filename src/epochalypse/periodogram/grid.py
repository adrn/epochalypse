"""The trial-period grid: one uniform frequency grid per log-period segment.

`kepmodel` advances the trial frequency by a *fixed* step -- it propagates
`cos nu t` / `sin nu t` by a rotation rather than recomputing them -- so it
evaluates only on uniform frequency grids. A single such grid, the way the
Gaia/OHP tutorial sets one up, is fine for detection but is the wrong sampling
for *this* measurement,
because the classification spans 7.8 decades in period: a uniform frequency
step fine enough for 44-minute orbits resolves log-period to ~1e-5 dex there
(pointlessly fine) and to ~0.1 dex at 3300 yr, leaving a handful of samples in
the last decade -- and the last decade is where the significant-but-unlocalized
systems live.

So the search uses `N_SEGMENTS` uniform frequency grids, one per equal interval
of log-period, and reports the union of their sample points. Every sample is
still a native kepmodel evaluation; the segments only let the frequency step be
refreshed as the trial period grows.

The grid depends on `config.P_MIN`, `P_MAX` and `N_SEGMENTS` and on nothing
else -- not on the star, not on its epochs -- so it is global to a run. That is
what lets the stored power arrays carry no period axis of their own.
"""
from __future__ import annotations

import numpy as np

from . import config as C

TWOPI = 2.0 * np.pi


def target_dlog(p_min=None, p_max=None, n_periods=None):
    """The coarsest log10-period spacing a segment is allowed, in dex.

    Defaults to the spacing of the in-house baseline grid
    (`run_periodograms.PERIODS`, `n_periods` log-uniform points over
    [p_min, p_max]), so this search is at least as finely sampled in log-period
    as the baseline *everywhere*. `config.TARGET_DLOG` overrides it outright.
    """
    if C.TARGET_DLOG is not None:
        return float(C.TARGET_DLOG)
    p_min = C.P_MIN if p_min is None else float(p_min)
    p_max = C.P_MAX if p_max is None else float(p_max)
    n = C.BASELINE_N_PERIODS if n_periods is None else int(n_periods)
    return float(np.log10(p_max / p_min) / (n - 1))


def frequency_segments(p_min=None, p_max=None, n_segments=None, dlog=None):
    """The `(nu0, dnu, nfreq)` grids to hand to `kepmodel.periodogram`.

    Covers [p_min, p_max] with `n_segments` uniform frequency grids, one per
    equal interval of log-period, each stepped finely enough that its coarsest
    log10-period spacing is at most `dlog`. Within a segment the log-period
    spacing is coarsest at the long-period end, which is what sets the step.
    """
    p_min = C.P_MIN if p_min is None else float(p_min)
    p_max = C.P_MAX if p_max is None else float(p_max)
    n_segments = C.N_SEGMENTS if n_segments is None else int(n_segments)
    dlog = target_dlog(p_min, p_max) if dlog is None else float(dlog)
    edges = np.exp(np.linspace(np.log(p_max), np.log(p_min), n_segments + 1))
    segments = []
    for p_hi, p_lo in zip(edges[:-1], edges[1:]):   # descending P -> ascending nu
        nu_lo, nu_hi = TWOPI / p_hi, TWOPI / p_lo
        # dlogP = dnu * P / (2 pi ln10), worst at the segment's low-frequency end
        dnu_max = dlog * TWOPI * np.log(10.0) / p_hi
        n = max(int(np.ceil((nu_hi - nu_lo) / dnu_max)) + 1, 3)
        segments.append((float(nu_lo), float((nu_hi - nu_lo) / (n - 1)), int(n)))
    return segments


def segment_periods(segments):
    """The trial periods a set of `frequency_segments` visits, ascending.

    Adjacent segments share an endpoint frequency, so the union is de-duplicated
    exactly the way `periodogram.kepmodel_power` de-duplicates the powers it
    computes on them -- the two arrays are aligned index for index, which is the
    whole reason the period axis can be stored once per run.
    """
    nu = np.concatenate([nu0 + np.arange(n) * dnu for nu0, dnu, n in segments])
    nu = np.unique(nu)                              # ascending frequency, no repeats
    return (TWOPI / nu)[::-1]                       # -> ascending period


def describe(segments):
    """A one-screen summary of a search grid, for logs and the manifest."""
    periods = segment_periods(segments)
    dlog = np.diff(np.log10(periods))
    return {"n_segments": len(segments),
            "n_freq_evaluated": int(sum(s[2] for s in segments)),
            "n_periods": int(periods.size),
            "p_min_yr": float(periods[0]), "p_max_yr": float(periods[-1]),
            "dlog_min": float(dlog.min()), "dlog_max": float(dlog.max()),
            "target_dlog": target_dlog()}
