"""Shared orbit-fitting and fast-characterization utilities for the epochalypse catalog.

This module factors the reusable pieces out of ``fit_astrometric_orbits.ipynb`` (data
loaders, the along-scan design matrix, and the weighted linear solution) so that both the
tutorial notebook and the population-scale characterization notebook
(``characterize_populations.ipynb``) share identical, tested code.

It additionally provides the *fast astrometric period periodogram* and the multimodality
classifier used to map the "detectable-but-poorly-characterized" population. At fixed period
a circular astrometric orbit is linear in four Thiele-Innes-like coefficients on top of the
linear five-parameter astrometric model, so the profile likelihood over period is one
weighted least-squares solve per trial period -- no sampling required.

The JAX/jaxoplanet/NumPyro machinery (the full sampler, used only for validation) is imported
lazily inside ``fitting_context`` so this module is usable for the periodogram work without a
JAX install.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
# Physical and mission constants live in one place, derived from astropy, and are
# re-exported here so existing callers (`ef.MJUP_TO_MSUN`, ...) keep working.
from epochalypse_constants import (  # noqa: E402,F401
    DAYS_PER_YEAR,
    DR4_BASELINE_YEARS,
    GAIA_EPOCH_TCB_JD,
    MARS_TO_MSUN,
    MAX_COMPANION_MASS_MSUN,
    MJUP_TO_MSUN,
    RSUN_TO_AU,
)

TWOPI = 2.0 * np.pi
ECC_MAX = 0.99

# Prior bounds for the full-sampler (NUTS) period parameter in epochalypse_validation. They must
# bracket the injected prior, which sim_planets.ipynb sets through A_MIN_AU/A_MAX_AU: at
# A_MIN_AU = 0.1 AU the shortest injected period is 5 d = 0.014 yr.
PERIOD_MIN_YEARS = 0.01
PERIOD_MAX_YEARS = 5000.0

DATA_ROOT = Path("outputs/data")


def systems_h5_path(population, data_root=DATA_ROOT):
    """Path to the per-population HDF5 product."""
    return Path(data_root) / f"simulated_astrometry_{population}_systems.h5"


# --------------------------------------------------------------------------------------
# HDF5 loaders (verbatim behavior from fit_astrometric_orbits.ipynb, cell 4)
# --------------------------------------------------------------------------------------
def _decode_str_cols(df):
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(
                lambda v: v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else v
            )
    return df


def load_truths(systems_h5):
    """Per-population /truths table (one row per system) as a DataFrame."""
    systems_h5 = Path(systems_h5)
    if not systems_h5.exists():
        raise FileNotFoundError(f"Missing per-population systems file: {systems_h5}")
    with h5py.File(systems_h5, "r") as f:
        truths = pd.DataFrame(f["truths"][:])
    return _decode_str_cols(truths)


def total_snr(truths):
    """Detectability per system: max SNR_tot over the injected companions."""
    cols = [c for c in ("snr_total_1", "snr_total_2") if c in truths.columns]
    if not cols:
        return np.zeros(len(truths))
    return np.nanmax(truths[cols].to_numpy(float), axis=1)


def max_injected_period_years(truths):
    """Longest injected companion period per system [yr]; inf if none injected."""
    cols = [c for c in ("period_1", "period_2") if c in truths.columns]
    if not cols:
        return np.full(len(truths), np.inf)
    with np.errstate(invalid="ignore"):
        pmax = np.nanmax(truths[cols].to_numpy(float), axis=1)
    return np.where(np.isfinite(pmax), pmax, np.inf)


def load_epochs(systems_h5, row_index):
    """Load the epoch table for the system at /truths row ``row_index``.

    Returns the epoch DataFrame sorted by observation time. Kept lightweight (no truth
    join / validation) so it can be called in a tight loop over a whole population.
    """
    with h5py.File(systems_h5, "r") as f:
        group = f[f"systems/{row_index}"]
        if "epochs" not in group:
            raise KeyError(f"No epochs for row {row_index}")
        epochs = _decode_str_cols(pd.DataFrame(group["epochs"][:]))
    return epochs.sort_values("obs_time_tcb").reset_index(drop=True)


def load_simulated_system(systems_h5, row_index):
    """Load truth row + epoch table for the system at /truths row ``row_index``."""
    systems_h5 = Path(systems_h5)
    with h5py.File(systems_h5, "r") as f:
        truths = _decode_str_cols(pd.DataFrame(f["truths"][:]))
        truth = truths.iloc[row_index]
        grp = f[f"systems/{row_index}"]
        if "epochs" not in grp:
            raise KeyError(f"No epochs for row {row_index}")
        epochs = _decode_str_cols(pd.DataFrame(grp["epochs"][:]))
    epochs = epochs.sort_values("obs_time_tcb").reset_index(drop=True)
    return {"system_id": truth["system_id"], "truth": truth, "epochs": epochs,
            "row_index": row_index, "gaia_source_id": truth["gaia_source_id"]}


def epoch_arrays(epochs):
    """Extract the along-scan fit arrays from an epoch table.

    Returns t (yr from the DR4 reference epoch), psi (rad), parallax factor, the along-scan
    centroid y (mas) and its reported uncertainty yerr (mas).
    """
    t = (epochs["obs_time_tcb"].to_numpy(float) - GAIA_EPOCH_TCB_JD) / DAYS_PER_YEAR
    psi = epochs["scan_pos_angle"].to_numpy(float)
    pf = epochs["parallax_factor_al"].to_numpy(float)
    y = epochs["centroid_pos_al"].to_numpy(float)
    yerr = epochs["centroid_pos_error_al"].to_numpy(float)
    return t, psi, pf, y, yerr


# --------------------------------------------------------------------------------------
# Five-parameter astrometric design + weighted least squares (cell 5)
# --------------------------------------------------------------------------------------
def astrometric_design_matrix(t, scan_angle, pf):
    """Five-parameter along-scan design: [sin psi, cos psi, sin psi * t, cos psi * t, pf]."""
    return np.column_stack([
        np.sin(scan_angle), np.cos(scan_angle),
        np.sin(scan_angle) * t, np.cos(scan_angle) * t, pf,
    ])


def weighted_linear_solution(t, scan_angle, pf, values, errors):
    """Weighted LSQ of the five-parameter astrometric model. Returns (beta, cov, X)."""
    X = astrometric_design_matrix(t, scan_angle, pf)
    w = 1.0 / np.square(errors)
    normal = X.T @ (w[:, None] * X)
    covariance = np.linalg.pinv(normal)
    beta = covariance @ (X.T @ (w * values))
    return beta, covariance, X


def _wls_chi2(X, w, y):
    """Weighted chi-square of the least-squares fit of y on design X with weights w."""
    normal = X.T @ (w[:, None] * X)
    beta = np.linalg.pinv(normal) @ (X.T @ (w * y))
    resid = y - X @ beta
    return float(np.sum(w * resid**2)), beta


# --------------------------------------------------------------------------------------
# Fast astrometric period periodogram (circular Thiele-Innes profile likelihood)
# --------------------------------------------------------------------------------------
def orbit_columns(t, scan_angle, period_years):
    """Four circular Thiele-Innes-like along-scan columns at a trial period.

    The circular reflex along-scan signal is a linear combination of
    {cos phi * cos psi, sin phi * cos psi, cos phi * sin psi, sin phi * sin psi}
    with phi = 2*pi*t/period; the linear coefficients absorb amplitude/phase/orientation.
    """
    phi = TWOPI * t / period_years
    c, s = np.cos(phi), np.sin(phi)
    cos_psi, sin_psi = np.cos(scan_angle), np.sin(scan_angle)
    return np.column_stack([c * cos_psi, s * cos_psi, c * sin_psi, s * sin_psi])


def _kepler_E(M, e, iters=10):
    """Solve Kepler's equation E - e sin E = M (vectorized Newton)."""
    M = np.mod(M, TWOPI)
    E = M + e * np.sin(M)
    for _ in range(iters):
        E = E - (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
    return E


def eccentric_orbit_columns(t, scan_angle, period, e, tp):
    """Four Thiele-Innes along-scan columns for a Keplerian orbit at fixed (period, e, time-of-peri)."""
    E = _kepler_E(TWOPI * (t - tp) / period, e)
    X = np.cos(E) - e
    Y = np.sqrt(max(1.0 - e * e, 0.0)) * np.sin(E)
    cpsi, spsi = np.cos(scan_angle), np.sin(scan_angle)
    return np.column_stack([X * spsi, Y * spsi, X * cpsi, Y * cpsi])


def refine_orbit(t, scan_angle, pf, y, yerr, p_seed, max_dex=0.3):
    """Stage-2 eccentric refinement: locally optimize (P, e, T_p) from a circular-periodogram seed.

    At each trial (P, e, T_p) the 5 astrometric + 4 Thiele-Innes amplitudes are solved by weighted
    least squares (variable projection); Nelder-Mead minimizes the resulting chi^2 over the 3 nonlinear
    parameters, from a few (e, T_p) starts. Period is kept within ``max_dex`` of the seed (the circular
    peak already localizes it; eccentricity mainly shifts, not relocates, it). Returns (P, e, chi2).
    """
    from scipy.optimize import minimize
    w = 1.0 / np.square(yerr)
    X5 = astrometric_design_matrix(t, scan_angle, pf)
    lp0 = np.log(p_seed)
    lp_lo, lp_hi = lp0 - max_dex * np.log(10), lp0 + max_dex * np.log(10)

    def chi2(theta):
        lp, e, tpf = theta
        lp = min(max(lp, lp_lo), lp_hi)
        e = min(max(e, 0.0), 0.95)
        P = np.exp(lp)
        tp = (tpf % 1.0) * P
        X = np.hstack([X5, eccentric_orbit_columns(t, scan_angle, P, e, tp)])
        c2, _ = _wls_chi2(X, w, y)
        return c2

    best = None
    for e0, tpf0 in [(0.1, 0.25), (0.3, 0.6), (0.5, 0.0)]:
        res = minimize(chi2, [lp0, e0, tpf0], method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-2, "maxiter": 250})
        if best is None or res.fun < best.fun:
            best = res
    lp, e, _ = best.x
    return float(np.exp(min(max(lp, lp_lo), lp_hi))), float(min(max(e, 0.0), 0.95)), float(best.fun)


def period_grid(p_min, p_max, n):
    """Log-uniform trial-period grid [yr]."""
    return np.exp(np.linspace(np.log(p_min), np.log(p_max), n))


def acceleration_delta_chi2(t, scan_angle, pf, y, yerr):
    """Delta-chi^2 of a 7-parameter (5-par + along-scan acceleration) model vs the 5-par model.

    Adds quadratic-in-time along-scan terms (sin psi * t^2, cos psi * t^2). This is the data-only
    signature of a long-period companion whose orbit is under-sampled by the baseline (P >> T): the
    reflex shows up as an astrometric acceleration rather than a resolved orbit, which a bounded
    period periodogram would otherwise miss. Returns chi2_5par - chi2_7par (>= 0).
    """
    w = 1.0 / np.square(yerr)
    X5 = astrometric_design_matrix(t, scan_angle, pf)
    chi2_5, _ = _wls_chi2(X5, w, y)
    Xacc = np.hstack([X5, np.column_stack([np.sin(scan_angle) * t**2, np.cos(scan_angle) * t**2])])
    chi2_7, _ = _wls_chi2(Xacc, w, y)
    return chi2_5 - chi2_7


def _periodogram_core(base, scan_angle, t, y, w, periods, ridge=1e-8):
    """Delta-chi^2 for adding one circular orbit (4 cols) at each trial period, on top of ``base``.

    Frisch-Waugh-Lovell: project ``base`` (any set of linear columns, in the weighted metric) out of
    both the data and the 4 orbit columns once, then each trial period is a batched 4x4 solve.
    Returns (power, chi2_base) where power[p] = chi2_base - chi2(base + orbit@period[p]).
    """
    Nb = base.T @ (w[:, None] * base)
    Ab = np.linalg.pinv(Nb) @ (base.T * w)                         # (k, N)
    ry = y - base @ (Ab @ y)
    chi2_base = float(np.sum(w * ry**2))

    phi = TWOPI * (t[None, :] / periods[:, None])                  # (P, N)
    cph, sph = np.cos(phi), np.sin(phi)
    cpsi, spsi = np.cos(scan_angle), np.sin(scan_angle)
    O = np.stack([cph * cpsi, sph * cpsi, cph * spsi, sph * spsi], axis=-1)  # (P, N, 4)
    AbO = np.einsum("kn,pnc->pkc", Ab, O)
    rO = O - np.einsum("nk,pkc->pnc", base, AbO)                   # residualized orbit cols (P, N, 4)

    wrO = w[None, :, None] * rO
    M = np.einsum("pnc,pnd->pcd", rO, wrO)                         # (P, 4, 4)
    b = np.einsum("pnc,n->pc", wrO, ry)                            # (P, 4)
    M[:, np.arange(4), np.arange(4)] += ridge
    beta = np.linalg.solve(M, b[:, :, None])[:, :, 0]             # (P, 4); b[...,None] for NumPy 1/2
    power = np.einsum("pc,pc->p", b, beta)
    return power, chi2_base


def astrometric_periodogram(t, scan_angle, pf, y, yerr, periods, ridge=1e-8):
    """Delta-chi^2 astrometric periodogram: 5-parameter model + one circular orbit at each period.

    Returns (power, chi2_5par) with power[p] = chi2_5par - chi2(5-par + orbit@period[p]).
    """
    periods = np.asarray(periods, float)
    w = 1.0 / np.square(yerr)
    X5 = astrometric_design_matrix(t, scan_angle, pf)
    return _periodogram_core(X5, scan_angle, t, y, w, periods, ridge)


def double_periodogram(t, scan_angle, pf, y, yerr, periods, ridge=1e-8, exclude_dex=0.05):
    """Iterative ("CLEAN") two-planet astrometric periodogram.

    Pass 1: the standard 1-planet periodogram -> the strongest period P1. Pass 2: a *conditional*
    periodogram with an orbit fixed at P1 also in the design, scanning P2 (its amplitudes and all
    linear terms re-fit jointly at each P2); power2[p] = chi2(5-par + orbit@P1) - chi2(5-par + orbit@P1
    + orbit@P2). P2 near P1 (within ``exclude_dex``) is masked out as degenerate. Returns both periods,
    both power spectra, and the peak powers -- power1 vs the single-star model, power2 vs the 1-planet
    model (calibrate the latter on the 1-companion population for a second-planet false-positive rate).
    """
    periods = np.asarray(periods, float)
    w = 1.0 / np.square(yerr)
    X5 = astrometric_design_matrix(t, scan_angle, pf)
    power1, chi2_5par = _periodogram_core(X5, scan_angle, t, y, w, periods, ridge)
    i1 = int(np.argmax(power1))
    P1 = float(periods[i1])

    base2 = np.hstack([X5, orbit_columns(t, scan_angle, P1)])      # 5-par + fixed orbit at P1
    power2, chi2_1planet = _periodogram_core(base2, scan_angle, t, y, w, periods, ridge)
    p2 = power2.copy()
    p2[np.abs(np.log10(periods / P1)) < exclude_dex] = -np.inf     # mask degenerate P2 ~ P1
    i2 = int(np.argmax(p2))
    P2 = float(periods[i2])
    return {"P1": P1, "P2": P2, "power1": power1, "power2": power2,
            "top_power1": float(power1[i1]), "top_power2": float(power2[i2]),
            "chi2_5par": chi2_5par, "chi2_1planet": chi2_1planet, "i1": i1, "i2": i2}


# --------------------------------------------------------------------------------------
# Multimodality classification from a periodogram
# --------------------------------------------------------------------------------------
def _local_maxima(power):
    """Indices of strict interior local maxima of a 1-D array."""
    if len(power) < 3:
        return np.array([], dtype=int)
    left = power[1:-1] > power[:-2]
    right = power[1:-1] > power[2:]
    return np.where(left & right)[0] + 1


def find_peaks(periods, power, n_orbit_params=4, min_separation_dex=0.1):
    """Rank periodogram peaks by height (Delta-chi^2 vs the 5-par baseline).

    Peaks closer than ``min_separation_dex`` in log-period to an already-accepted, taller
    peak are merged (kept as the taller one) so that a single broad mode is not counted
    multiple times. Returns a list of dicts sorted by descending power, each with
    ``period``, ``power`` (Delta-chi^2), and ``delta_bic`` relative to the 5-par model
    (delta_bic = power - k*ln N, k = n_orbit_params; more negative BIC = preferred).
    """
    idx = _local_maxima(power)
    # always include the global argmax even if it sits on a boundary
    gmax = int(np.argmax(power))
    if gmax not in idx:
        idx = np.append(idx, gmax)
    idx = idx[np.argsort(-power[idx])]

    accepted = []
    log_p = np.log10(periods)
    for i in idx:
        if any(abs(log_p[i] - log_p[j]) < min_separation_dex for j in accepted):
            continue
        accepted.append(i)

    peaks = [{"index": int(i), "period": float(periods[i]), "power": float(power[i])}
             for i in accepted]
    return peaks


def period_constraint(periods, power, width_delta=4.0, edge_frac=0.02):
    """How tightly the periodogram localizes the period (peak WIDTH, not peak count).

    Because the profile-likelihood periodogram of a long-period, under-sampled orbit goes broad and
    flat-topped (the continuous period-eccentricity-acceleration degeneracy) rather than splitting
    into discrete peaks, peak counting mislabels it "unimodal." Instead we measure the log-period
    extent of the competitive region -- grid points within ``width_delta`` (Delta-chi^2) of the global
    maximum. On the uniform log grid this is (competitive fraction) x (total log range), and it counts
    disconnected competitive regions too.

    Returns (width_dex, best_period, best_at_edge): the competitive width in dex, the best-fit period,
    and whether the global maximum sits against a grid edge (period pinned only by the grid bound).
    """
    logp = np.log10(periods)
    gi = int(np.argmax(power))
    comp = power > power[gi] - width_delta
    width_dex = float(comp.mean() * (logp[-1] - logp[0]))
    at_edge = bool(gi <= edge_frac * len(periods) or gi >= (1 - edge_frac) * len(periods))
    return width_dex, float(periods[gi]), at_edge


def period_in_competitive_region(periods, power, p_target, width_delta=4.0):
    """TRUTH check: does the injected period fall within the periodogram's competitive region?

    Returns True if the periodogram power at ``p_target`` is within ``width_delta`` (Delta-chi^2) of
    the global maximum -- i.e. the data do not statistically exclude the true period. False if the
    true period is excluded (a confidently *biased* localization) or lies outside the tested grid.
    Returns np.nan if ``p_target`` is not finite/positive (no injected companion).
    """
    if not np.isfinite(p_target) or p_target <= 0:
        return np.nan
    if p_target < periods[0] or p_target > periods[-1]:
        return False  # true period beyond the tested grid -> not bracketed
    j = int(np.abs(np.log(periods) - np.log(p_target)).argmin())  # nearest grid point in log-period
    return bool(power[j] > power.max() - width_delta)


def classify_periodogram(periods, power, n_epochs, n_orbit_params=4,
                         delta_bic_detect=10.0, delta_power_unimodal=10.0,
                         n_weak_max=3, min_separation_dex=0.1,
                         width_delta=4.0, width_constrained_dex=0.05):
    """Classify a system's periodogram into a characterizability category.

    Categories:
      - ``undetected``      : best peak does not improve BIC over the 5-par model
                              (delta_bic_best > -delta_bic_detect); an orbit is not
                              even preferred, so 'characterizability' is moot.
      - ``unimodal``        : one dominant peak; the next distinct peak is weaker by more
                              than ``delta_power_unimodal`` in Delta-chi^2.
      - ``weakly_multimodal``: 2..n_weak_max competitive peaks within the threshold.
      - ``very_multimodal`` : more than n_weak_max competitive peaks (period essentially
                              unconstrained / aliased).

    Returns a dict with the class, the best-fit period, the number of competitive peaks,
    the top peak powers, and the best-peak delta_bic.
    """
    width_dex, best_period_edge, best_at_edge = period_constraint(periods, power, width_delta)

    peaks = find_peaks(periods, power, n_orbit_params, min_separation_dex)
    if not peaks:
        return {"klass": "undetected", "best_period": np.nan, "n_competitive": 0,
                "top_power": np.nan, "delta_bic_best": np.nan,
                "width_dex": width_dex, "best_at_edge": best_at_edge,
                "top_periods": [], "top_powers": []}

    best = peaks[0]
    # BIC relative to the 5-par model: adding k orbit params costs k*ln(N).
    delta_bic_best = best["power"] - n_orbit_params * np.log(max(n_epochs, 2))

    # competitive = within delta_power_unimodal (Delta-chi^2) of the tallest peak
    competitive = [p for p in peaks if best["power"] - p["power"] < delta_power_unimodal]
    n_competitive = len(competitive)

    # WIDTH-based constraint class (captures broad plateaus that peak-counting misses):
    #   undetected -> broad (unconstrained: wide competitive region or railed to a grid edge)
    #   -> multimodal (narrow but >=2 separated competitive peaks) -> unimodal (narrow, single).
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


# --------------------------------------------------------------------------------------
# Population-scale batch driver
# --------------------------------------------------------------------------------------
def characterize_population(systems_h5, periods, row_indices=None, classify_kwargs=None,
                           detect_delta_bic=10.0, period_recover_tol=1.2,
                           period_reliable_baseline_frac=1.0, width_delta=4.0,
                           eccentric_refine=False, progress_every=2000):
    """Run the periodogram + acceleration test + classifier over a population's systems.

    Opens the HDF5 file once and iterates ``/systems/<i>/epochs``. Returns a DataFrame with one
    row per system, joining the periodogram/acceleration summary to the injected truth columns
    needed for the period-vs-signal map. If ``row_indices`` is None, all systems are processed.

    Adds, per system:
      - ``accel_delta_chi2`` and ``accel_delta_bic`` (5-par vs 7-par acceleration);
      - ``detected``: data-only flag -- orbit OR acceleration decisively beats the single-star
        5-par model (delta_bic >= ``detect_delta_bic``);
      - ``period_reliable``: data-only flag -- detected, unimodal, and best period well within the
        baseline (best_period < ``period_reliable_baseline_frac`` * DR4 baseline);
      - ``period_recovered``: TRUTH-based flag -- best period within ``period_recover_tol`` of the
        injected period_1 (single-companion populations); NaN if no injected companion.
    """
    systems_h5 = Path(systems_h5)
    classify_kwargs = dict(classify_kwargs or {})
    classify_kwargs.setdefault("width_delta", width_delta)  # keep in_bound + klass width consistent
    truths = load_truths(systems_h5)
    if row_indices is None:
        row_indices = np.arange(len(truths))

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
            ep = _decode_str_cols(pd.DataFrame(grp["epochs"][:]))
            ep = ep.sort_values("obs_time_tcb").reset_index(drop=True)
            t, psi, pf, y, yerr = epoch_arrays(ep)
            n = len(y)
            power, chi2_5par = astrometric_periodogram(t, psi, pf, y, yerr, periods)
            res = classify_periodogram(periods, power, n, **classify_kwargs)
            accel_dchi2 = acceleration_delta_chi2(t, psi, pf, y, yerr)
            accel_dbic = accel_dchi2 - 2.0 * np.log(max(n, 2))  # 2 extra params

            detected = (res["delta_bic_best"] >= detect_delta_bic) or (accel_dbic >= detect_delta_bic)
            period_reliable = bool(
                detected and res["klass"] == "unimodal"
                and res["best_period"] < period_reliable_baseline_frac * DR4_BASELINE_YEARS
            )

            # Stage-2 eccentric refinement of the top circular period (only for detected systems).
            best_period_ecc, e_ecc = np.nan, np.nan
            if eccentric_refine and detected and np.isfinite(res["best_period"]):
                best_period_ecc, e_ecc, _ = refine_orbit(t, psi, pf, y, yerr, res["best_period"])

            rec = {"row_index": ri, "n_epochs": n, "chi2_5par": chi2_5par,
                   "klass": res["klass"], "best_period": res["best_period"],
                   "best_period_ecc": best_period_ecc, "e_ecc": e_ecc,
                   "n_competitive": res["n_competitive"], "top_power": res["top_power"],
                   "delta_bic_best": res["delta_bic_best"],
                   "width_dex": res["width_dex"], "best_at_edge": res["best_at_edge"],
                   "accel_delta_chi2": float(accel_dchi2), "accel_delta_bic": float(accel_dbic),
                   "detected": bool(detected), "period_reliable": period_reliable}
            for k in range(2):  # top-2 peak periods for aliasing diagnostics
                rec[f"peak{k+1}_period"] = (res["top_periods"][k]
                                            if k < len(res["top_periods"]) else np.nan)
            row = truths.iloc[ri]
            for c in keep_cols:
                rec[c] = row[c]
            # TRUTH: is each injected period inside the periodogram's competitive region?
            for k in (1, 2):
                col = f"period_{k}"
                pk = float(row[col]) if col in truths.columns and pd.notna(row[col]) else np.nan
                rec[f"period_{k}_in_bound"] = period_in_competitive_region(
                    periods, power, pk, width_delta)
            records.append(rec)
            if progress_every and (count + 1) % progress_every == 0:
                print(f"  {count + 1}/{len(row_indices)} systems", flush=True)

    df = pd.DataFrame.from_records(records)

    def _row_nanmax(cols):
        cols = [c for c in cols if c in df.columns and df[c].notna().any()]
        if not cols:
            return np.full(len(df), np.nan)
        return np.nanmax(df[cols].to_numpy(float), axis=1)

    df["snr_total"] = _row_nanmax(["snr_total_1", "snr_total_2"])
    df["snr_single"] = _row_nanmax(["snr_single_1", "snr_single_2"])
    df["arc_snr"] = np.sqrt(df["n_transits_dr4"].to_numpy(float)) * df["snr_single"]

    # TRUTH-based period recovery (single-companion populations): best circular period vs injected.
    p_inj = df["period_1"].to_numpy(float) if "period_1" in df.columns else np.full(len(df), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = df["best_period"].to_numpy(float) / p_inj
        recovered = np.abs(np.log(ratio)) < np.log(period_recover_tol)
    df["period_recovered"] = np.where(np.isfinite(p_inj), recovered, np.nan)
    if "best_period_ecc" in df.columns:   # eccentric-refined period vs injected
        with np.errstate(invalid="ignore", divide="ignore"):
            rec_e = np.abs(np.log(df["best_period_ecc"].to_numpy(float) / p_inj)) < np.log(period_recover_tol)
        df["period_recovered_ecc"] = np.where(np.isfinite(p_inj), rec_e, np.nan)

    # cast in-bound flags to float (1.0 / 0.0 / NaN) for clean CSV round-trip
    for k in (1, 2):
        c = f"period_{k}_in_bound"
        if c in df.columns:
            df[c] = df[c].map({True: 1.0, False: 0.0}).astype(float)
    return df


# --------------------------------------------------------------------------------------
# Characterizability class (shared by the figure notebook and the MCMC-validation notebook)
# --------------------------------------------------------------------------------------
def characterize_population_double(systems_h5, periods, row_indices=None, progress_every=2000):
    """Run the iterative two-planet (double) periodogram over a 2-companion population.

    For each system records the two recovered periods (P1_pdg dominant, P2_pdg second), their peak
    powers (top_power1 vs single-star, top_power2 vs the 1-planet model), the peak widths, and -- using
    the injected truth -- whether the inner and outer companions are each recovered (their period within
    25% of P1_pdg or P2_pdg). Returns one row per system, joined to the truth columns needed for the map.
    """
    systems_h5 = Path(systems_h5)
    truths = load_truths(systems_h5)
    if row_indices is None:
        row_indices = np.arange(len(truths))
    keep = [c for c in ("system_id", "gaia_source_id", "n_planets", "period_1", "period_2",
                        "a_1_au", "a_2_au", "Mp_1_msun", "Mp_2_msun", "alpha_1_mas", "alpha_2_mas",
                        "snr_total_1", "snr_total_2", "near_resonance", "coplanar") if c in truths.columns]
    lt = np.log(1.25)
    records = []
    with h5py.File(systems_h5, "r") as f:
        for count, ri in enumerate(row_indices):
            ri = int(ri)
            grp = f[f"systems/{ri}"]
            if "epochs" not in grp:
                continue
            ep = _decode_str_cols(pd.DataFrame(grp["epochs"][:])).sort_values("obs_time_tcb")
            t, psi, pf, y, yerr = epoch_arrays(ep)
            d = double_periodogram(t, psi, pf, y, yerr, periods)
            w1, _, edge1 = period_constraint(periods, d["power1"])
            w2, _, edge2 = period_constraint(periods, d["power2"])
            row = truths.iloc[ri]
            rec = {"row_index": ri, "n_epochs": len(y),
                   "P1_pdg": d["P1"], "P2_pdg": d["P2"],
                   "top_power1": d["top_power1"], "top_power2": d["top_power2"],
                   "width1_dex": w1, "width2_dex": w2, "edge1": edge1, "edge2": edge2}
            recovered_periods = [d["P1"], d["P2"]]
            for k in (1, 2):
                pk = float(row[f"period_{k}"]) if pd.notna(row.get(f"period_{k}", np.nan)) else np.nan
                if np.isfinite(pk):
                    rec[f"recovered_{k}"] = bool(any(abs(np.log(rp / pk)) < lt for rp in recovered_periods))
                else:
                    rec[f"recovered_{k}"] = np.nan
            for c in keep:
                rec[c] = row[c]
            records.append(rec)
            if progress_every and (count + 1) % progress_every == 0:
                print(f"  {count + 1}/{len(row_indices)} systems", flush=True)
    df = pd.DataFrame.from_records(records)
    if {"recovered_1", "recovered_2"}.issubset(df.columns):
        r1 = df["recovered_1"].fillna(False).astype(bool)
        r2 = df["recovered_2"].fillna(False).astype(bool)
        df["n_recovered"] = r1.astype(int) + r2.astype(int)   # 0 / 1 / 2 companions recovered
    return df


def detection_thresholds(null_df, target_fp=0.01):
    """Data-driven detection thresholds calibrated on the companion-free control.

    Returns (thr_orbit, thr_accel): the (1 - target_fp/2) percentile of the periodogram peak power
    and of the acceleration Delta-chi^2 among companion-free (0_companion) systems, so the two
    channels together give a ~``target_fp`` null false-positive rate.
    """
    q = 100.0 * (1.0 - target_fp / 2.0)
    thr_orbit = float(np.nanpercentile(null_df["top_power"].to_numpy(float), q))
    thr_accel = float(np.nanpercentile(null_df["accel_delta_chi2"].to_numpy(float), q))
    return thr_orbit, thr_accel


def is_primary_companion(df, k):
    """Boolean mask: is companion k the dominant (higher snr_total) one in its system?

    A single periodogram characterizes at most one companion; the dominant one is attributed the
    result and the other is left undetected. One-companion populations: every companion is primary.
    """
    other = 2 if k == 1 else 1
    oc = f"snr_total_{other}"
    if oc not in df.columns or not df[oc].notna().any():
        return np.ones(len(df), bool)
    stk = np.nan_to_num(df[f"snr_total_{k}"].to_numpy(float), nan=-np.inf)
    sto = np.nan_to_num(df[oc].to_numpy(float), nan=-np.inf)
    return stk >= sto


def characterizability_class(df, k, thr_orbit, thr_accel, period_tol=1.25, period_cols=None):
    """Per-companion characterizability label from a characterization DataFrame.

    Returns an array over rows with values in {"undet", "ok", "biased", "uncon"}:
      - undet  : not data-detected, or the non-dominant companion of a two-companion system;
      - ok     : detected, narrow (unimodal) periodogram peak AT the true period (characterized);
      - biased : detected, narrow peak at the WRONG period (within period_tol miss);
      - uncon  : detected, broad/multimodal periodogram (period unconstrained).
    Detection is data-driven (periodogram top_power > thr_orbit OR accel_delta_chi2 > thr_accel).
    ``period_cols`` are the periodogram period estimates checked for "found"; default is the circular
    peaks. Pass ``("best_period_ecc",)`` to use the stage-2 eccentric-refined period instead (which
    converts eccentricity-driven "biased" systems into "characterized").
    """
    if period_cols is None:
        period_cols = ("peak1_period", "peak2_period", "best_period")
    detected = (df["top_power"].to_numpy(float) > thr_orbit) | \
               (df["accel_delta_chi2"].to_numpy(float) > thr_accel)
    active = detected & is_primary_companion(df, k)
    narrow = df["klass"].to_numpy(object) == "unimodal"
    p_inj = df[f"period_{k}"].to_numpy(float)
    found = np.zeros(len(df), bool)
    lt = np.log(period_tol)
    for col in period_cols:
        if col in df.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                found |= np.abs(np.log(df[col].to_numpy(float) / p_inj)) < lt
    out = np.full(len(df), "undet", object)
    out[active & ~narrow] = "uncon"
    out[active & narrow & found] = "ok"
    out[active & narrow & ~found] = "biased"
    return out
