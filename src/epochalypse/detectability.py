"""How much of an injected orbit a fit could ever see.

`snr_total` is not a detectability measure, and treating it as one is what sent
a week of this project chasing a catalog bug that did not exist. The reason is
geometric and unavoidable: **position, proper motion and parallax are free
parameters**, so whatever part of a companion's signal those five columns can
reproduce is subtracted along with them and can never be detected — no matter
how large `alpha` is. An orbit whose period is comparable to the mission span is
mostly a straight line plus a curve across the data, and a straight line is
exactly what proper motion is.

`planets.draw_companions` charges for this with
`snr_eff = snr_single / (1 + (sma/a_crit)^3)`, which is exactly `1 + (P/T)^2`.
That heuristic has the right period scaling — it agrees with the exact
projection to 10-30% over `0.2 <= P/T <= 10` — but it omits the along-scan
projection entirely, so it is optimistic by ~1.8x at short period, and it has no
eccentricity term at all. No better formula exists, either: at fixed `(P/T, e)`
the true retained fraction spreads by 1.9x at `e = 0` and 6.4x at `e = 0.8`,
because the variable that dominates at long period is *where periastron falls
relative to the observing window*, which no function of `(P, a)` can see.

So this module measures it instead. Three quantities, and they answer different
questions:

| quantity | knows orientation? | answers |
| --- | --- | --- |
| `snr_total` (catalog) | no | **plausibly** observable |
| `snr_detectable` | yes, exactly | observable **for this system** |
| `snr_expected` | no, marginalized | observable **in expectation** |

`snr_detectable` is the right axis for a *method* question — given a signal this
strong is genuinely present, does the fit find it? It is the **wrong** input to
an occurrence-rate correction, because it conditions on the true inclination and
phase, which no real survey knows. `snr_expected` marginalizes those away and is
the survey-facing one. See `SNR.md`.

**The reflex is reconstructed by calling the generator, not by reimplementing
it.** `injected_reflex` runs `astrometry.simulate_along_scan` twice — once with
the companions and once without — and differences the results. Degrees versus
radians, sign conventions and the phase definition therefore cannot drift out of
agreement with the data, which is a real risk here: the Thiele-Innes and
Campbell conventions differ, and a silent mismatch would look exactly like a
missing signal.
"""

from __future__ import annotations

import numpy as np

# The per-companion truth columns `simulate_along_scan` needs, mapped to the
# names it takes. `periodogram.config.TRUTH_COLUMNS_COMPANION` is the authority
# on what the shard stores; this is the subset that defines the orbit.
ORBIT_COLUMNS = ("mass_pl", "period", "ecc", "inc", "omega", "Omega", "M_anom")


def astrometric_design(t, psi, pf, n_columns=5):
    """The along-scan astrometric basis, straight from the simulator's own model.

    `simulate_along_scan` writes
    `al = sin(psi) (ra0 + mu_a t) + cos(psi) (dec0 + mu_d t) + parallax * pf`,
    so these five columns span exactly the motion a Gaia five-parameter solution
    can absorb. Any basis spanning the same space gives the same residual, so
    there is nothing to get wrong here and no need for harv's design matrix.
    """
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)
    columns = [sin_psi, cos_psi, t * sin_psi, t * cos_psi, pf]
    return np.column_stack(columns[:n_columns])


def _shared(truth, pf):
    """The star-level arguments `simulate_along_scan` takes, from a truth row."""
    return {
        "mstar": float(truth["mass_st_msun"]),
        "rstar": float(truth["radius_st_rsun"]),
        "parallax": float(truth["parallax_mas"]),
        "mu_alpha": float(truth["pmra_mas_yr"]),
        "mu_delta": float(truth["pmdec_mas_yr"]),
        "parallax_factor": pf,
        "sigma_ueva": 0.0,
        "seed": 0,
    }


def companion(truth, index):
    """One companion's orbital elements, in `simulate_along_scan`'s own dict form."""
    return {name: float(truth[f"{name}_{index}"]) for name in ORBIT_COLUMNS}


def reflex_of(truth, t, psi, pf, companions):
    """The noise-free along-scan signal of `companions`, and nothing else.

    Differences two calls to the generator so the five-parameter astrometric
    motion cancels exactly, leaving only the companions' contribution. `[]`
    returns zeros, which is the honest answer for a control system.
    """
    from .astrometry import simulate_along_scan

    if not companions:
        return np.zeros_like(np.asarray(t, dtype=np.float64))
    shared = _shared(truth, pf)
    _, with_orbit = simulate_along_scan(t, psi, list(companions), **shared)
    _, without = simulate_along_scan(t, psi, [], **shared)
    return np.asarray(with_orbit) - np.asarray(without)


def injected_reflex(truth, t, psi, pf, n_companions):
    """The signal of every injected companion together."""
    return reflex_of(
        truth, t, psi, pf, [companion(truth, j) for j in range(1, n_companions + 1)]
    )


def per_companion_reflex(truth, t, psi, pf, n_companions):
    """One reflex per companion, each computed on its own.

    Per companion rather than per system because the recorded SNRs are per
    companion (`snr_total_1`, `snr_total_2`) and `harv.census.best_truth` scores
    a two-companion system against whichever orbit the fit actually matched.
    A single summed number could not be attributed to either.

    They are additive -- the photocentre traces the sum -- so these sum to
    `injected_reflex`, which `tests/test_harv.py` asserts.
    """
    return [
        reflex_of(truth, t, psi, pf, [companion(truth, j)])
        for j in range(1, n_companions + 1)
    ]


def retained_fraction(reflex, t, psi, pf, yerr):
    """How much of THIS orbit survives the five-parameter astrometric fit.

    Projects the reflex onto the orthogonal complement of the astrometric basis,
    inverse-variance weighted because that is the metric the fit minimizes in.
    What is left is the entire detectable signal.

    Measured on the actual injected reflex rather than on the orbit's design
    columns. An earlier version projected the four Thiele-Innes columns
    independently and averaged their Frobenius norms, which is not the retained
    fraction of any real orbit: a partial arc has one well-retained direction in
    that 4-D space and three nearly-killed ones, so the average described nothing
    and read ~60% where the truth was ~10%.

    A half-cycle orbit is close to a quadratic in time, and constant plus linear
    is exactly what position and proper motion are. Expect single-digit
    percentages for a period near twice the observing span.

    Returns `(fraction, rms_left)`, the second in units of the REPORTED
    uncertainty -- which is what the chi-square of a fit weighted by those
    uncertainties would show.
    """
    yerr = np.asarray(yerr, dtype=np.float64)
    reflex = np.asarray(reflex, dtype=np.float64)
    design = astrometric_design(t, psi, pf, 5)

    # A zero or non-finite uncertainty divides to inf and LAPACK's SVD then
    # fails to converge outright -- `LinAlgError: SVD did not converge in Linear
    # Least Squares`, which killed a production rank and with it the whole
    # shard. The catalog is known to contain such rows: `harv_finish`'s census
    # counts systems whose log-likelihoods are all non-finite for the same
    # reason. Drop them rather than propagating a crash.
    usable = (
        np.isfinite(yerr)
        & (yerr > 0)
        & np.isfinite(reflex)
        & np.isfinite(design).all(axis=1)
    )
    if usable.sum() <= design.shape[1]:
        return float("nan"), float("nan")

    scaled = design[usable] / yerr[usable, None]
    target = reflex[usable] / yerr[usable]
    try:
        fitted, *_ = np.linalg.lstsq(scaled, target, rcond=None)
    except np.linalg.LinAlgError:
        # A degenerate scan geometry can leave the basis rank-deficient even
        # with every row finite. The pseudo-inverse of the 5x5 normal matrix is
        # well defined there, and 5x5 is small enough that the extra work is
        # free at this scale.
        fitted = np.linalg.pinv(scaled.T @ scaled) @ (scaled.T @ target)
    left = target - scaled @ fitted
    total = float(np.sum(target**2))
    return (
        float(np.sqrt(np.sum(left**2) / total)) if total > 0 else 0.0,
        float(np.sqrt(np.mean(left**2))),
    )


def snr_detectable(reflex, t, psi, pf, yerr, sigma_single):
    """Accumulated SNR of the part of `reflex` a fit could actually use.

    `sqrt(N) * rms(reflex orthogonal to the astrometric basis) / sigma_single`,
    divided by the INJECTED noise scale so it is directly comparable with the
    catalog's `snr_total`, which divides by the same thing. `retained_fraction`
    works in units of the reported uncertainty, so the ratio converts.

    Returns `(snr, retained_fraction)`.
    """
    yerr = np.asarray(yerr, dtype=np.float64)
    fraction, rms_left = retained_fraction(reflex, t, psi, pf, yerr)
    reported = float(np.median(yerr))
    if not np.isfinite(sigma_single) or sigma_single <= 0:
        return float("nan"), fraction
    return float(np.sqrt(len(yerr)) * rms_left * reported / sigma_single), fraction


def snr_expected(truth, index, t, psi, pf, yerr, sigma_single, n_draws=20, seed=0):
    """`snr_detectable` marginalized over the orientation nobody knows.

    The occurrence-rate quantity. `snr_detectable` conditions on the true
    inclination, argument of periastron, node and mean anomaly; a real survey
    knows none of those, so a selection function expressed in terms of them is
    not usable. This redraws all four isotropically at fixed period,
    eccentricity and mass and returns the median, which is what a completeness
    correction should integrate over.

    `cos i` is drawn uniform, not `i` -- an isotropic orbit is uniform in
    `cos i`, and drawing the angle instead would over-weight edge-on
    configurations and understate detectability.

    Returns `(median, 16th, 84th)`. The spread is not decoration: at `e = 0.8`
    and `P/T = 2` it is a factor of 6, so a single expected value badly
    misrepresents whether any individual system was findable.
    """
    rng = np.random.default_rng(seed)
    base = companion(truth, index)
    values = []
    for _ in range(int(n_draws)):
        drawn = dict(base)
        drawn["inc"] = float(np.degrees(np.arccos(rng.uniform(-1.0, 1.0))))
        drawn["omega"] = float(rng.uniform(0.0, 360.0))
        drawn["Omega"] = float(rng.uniform(0.0, 360.0))
        drawn["M_anom"] = float(rng.uniform(0.0, 360.0))
        reflex = reflex_of(truth, t, psi, pf, [drawn])
        values.append(snr_detectable(reflex, t, psi, pf, yerr, sigma_single)[0])
    return tuple(float(v) for v in np.nanpercentile(values, [50, 16, 84]))


# ==========================================================================
# The orientation-marginalized table
# ==========================================================================
# `snr_expected` needs E[retained] marginalized over inclination, node, argument
# of periastron and phase. Doing that per system costs ~20 reflex evaluations
# each, which is ~1,900 core-hours over the catalog.
#
# It does not have to be per system. `check_snr.py --across-stars` measures the
# star-to-star spread of E[retained] with common random numbers, against a
# within-star Monte Carlo control, and finds the between-star part is 0.8-6.5%
# and no larger than the noise. E[retained] is a property of the ORBIT -- of
# `(P/T, e)` -- so one table serves the catalog and the marginalization becomes
# an interpolation.
#
# Grid in log10(P/T) because that is the variable the physics is smooth in: a
# half-cycle orbit and a hundred-cycle orbit differ by orders of magnitude in
# retention, not by a factor.
TABLE_LOG_RATIO = np.linspace(-1.5, 1.5, 25)  # P/T from 0.032 to 32
TABLE_ECC = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.95])


def sample_stars(population, n_stars, spread=True):
    """`n_stars` real stars, spread ACROSS shards rather than taken from one.

    A shard is a contiguous block of the source list, so it is a contiguous
    region of sky -- and both things `retained` depends on, the transit count
    and the scan-angle distribution, vary with ecliptic latitude. Taking the
    first `n_stars` in shard order therefore samples one patch: measured on the
    real catalog it returned epoch counts spanning 68-87 out of a catalog range
    of 44-298, which silently understates any star-to-star variation.

    Taking a few from every shard costs one open per shard and covers the sky.
    """
    from .periodogram.shards import ShardReader, discover_shards

    numbers, n_shards = discover_shards(population)
    per_shard = max(1, -(-int(n_stars) // len(numbers))) if spread else int(n_stars)
    stars = []
    for shard in numbers:
        taken = 0
        with ShardReader(population, shard, n_shards) as reader:
            for _index, truth, t, psi, pf, _y, yerr in reader.iter_systems():
                stars.append((truth, t, psi, pf, yerr))
                taken += 1
                if taken >= per_shard or len(stars) >= n_stars:
                    break
        if len(stars) >= n_stars:
            break
    return stars


def retained_table(stars, n_draws=40, seed=0, baseline_years=None):
    """E[retained] over a `(log10 P/T, e)` grid, averaged over stars and orientation.

    `stars` is a sequence of `(truth, t, psi, pf, yerr)`. A few dozen is plenty
    -- the star-to-star spread is a few percent, so the average converges far
    faster than the orientation marginalization does.

    Common random numbers: one orientation set per cell, reused for every star.
    That removes the Monte Carlo error from the star average rather than letting
    it accumulate into it.
    """
    from . import constants as k

    baseline = k.DR4_BASELINE_YEARS if baseline_years is None else baseline_years
    rng = np.random.default_rng(seed)
    table = np.zeros((len(TABLE_LOG_RATIO), len(TABLE_ECC)))
    for i, log_ratio in enumerate(TABLE_LOG_RATIO):
        for j, ecc in enumerate(TABLE_ECC):
            draws = [
                {
                    "mass_pl": 10.0,
                    "period": float(10.0**log_ratio * baseline),
                    "ecc": float(ecc),
                    "inc": float(np.degrees(np.arccos(rng.uniform(-1.0, 1.0)))),
                    "omega": float(rng.uniform(0.0, 360.0)),
                    "Omega": float(rng.uniform(0.0, 360.0)),
                    "M_anom": float(rng.uniform(0.0, 360.0)),
                }
                for _ in range(int(n_draws))
            ]
            values = [
                retained_fraction(
                    reflex_of(truth, t, psi, pf, [drawn]), t, psi, pf, yerr
                )[0]
                for truth, t, psi, pf, yerr in stars
                for drawn in draws
            ]
            table[i, j] = float(np.median(values))
    return table


def expected_retained(table, period_yr, ecc, baseline_years=None):
    """Interpolate `retained_table` at one orbit, or at millions of them.

    Bilinear in `(log10 P/T, e)`, clamped at the grid edges -- beyond
    `P/T = 32` the retained fraction is already ~1e-3 and its exact value stops
    mattering against everything else in a selection function.

    Written out rather than handed to `scipy.interpolate`: this runs once per
    companion over 17.2 M systems, so it has to be vectorized over the whole
    column, and a per-point interpolator call would dominate the stage.
    """
    from . import constants as k

    baseline = k.DR4_BASELINE_YEARS if baseline_years is None else baseline_years
    scalar = np.ndim(period_yr) == 0 and np.ndim(ecc) == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio = np.log10(np.asarray(period_yr, dtype=np.float64) / baseline)
    ecc = np.asarray(ecc, dtype=np.float64)
    log_ratio, ecc = np.broadcast_arrays(np.atleast_1d(log_ratio), np.atleast_1d(ecc))

    i = np.clip(
        np.searchsorted(TABLE_LOG_RATIO, log_ratio) - 1, 0, TABLE_LOG_RATIO.size - 2
    )
    j = np.clip(np.searchsorted(TABLE_ECC, ecc) - 1, 0, TABLE_ECC.size - 2)
    wl = np.clip(
        (log_ratio - TABLE_LOG_RATIO[i])
        / (TABLE_LOG_RATIO[i + 1] - TABLE_LOG_RATIO[i]),
        0.0,
        1.0,
    )
    we = np.clip((ecc - TABLE_ECC[j]) / (TABLE_ECC[j + 1] - TABLE_ECC[j]), 0.0, 1.0)
    value = (
        (1 - wl) * (1 - we) * table[i, j]
        + wl * (1 - we) * table[i + 1, j]
        + (1 - wl) * we * table[i, j + 1]
        + wl * we * table[i + 1, j + 1]
    )
    # a non-finite period (a control system has none) interpolates to nothing
    value = np.where(np.isfinite(log_ratio), value, np.nan)
    return float(value[0]) if scalar else value
