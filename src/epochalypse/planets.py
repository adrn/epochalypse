"""Stage 2: draw the companions for ONE star.

Two changes from the serial pipeline, both required by the move to ~4 million
stars:

1. **Per-source RNG.** The stream is seeded from the Gaia source id, not from a
   position in a list, so a star's companions depend only on that star. Any
   subset can be drawn, in any order, in any process.

2. **No detectability rejection.** Every population is drawn from the unbiased
   prior; the S/N metrics are computed and stored per companion so a high-S/N
   sample is selected downstream by cutting on ``snr_total_*``. This removes the
   rejection loop that dominated the cost of the old high-SNR populations and
   turns the threshold into an analysis choice.

The rejection that remains is over *physical possibility* only: the star must
fit inside its Roche lobe, and a two-companion pair must be non-crossing and
Hill-stable.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import config as C

# Bail-out for a star with no physically allowed orbit anywhere in the prior
# (the Roche floor rejects ~15-25% of draws, so reaching this means unusable).
MAX_DRAWS = 10_000


# --------------------------------------------------------------------------
# Seeding: keyed on the source id, never on a row index
# --------------------------------------------------------------------------
def system_seed(master_seed: int, population: str, gaia_source_id) -> int:
    """Stable uint32 seed for one (population, source) pair."""
    payload = f"{int(master_seed)}:{population}:{gaia_source_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


# --------------------------------------------------------------------------
# Orbital bookkeeping (unchanged physics)
# --------------------------------------------------------------------------
def semimajor_axis_to_period(a_au, mtot_msun):
    """Kepler's third law: P [yr] from a [AU] and M_star + M_p [Msun]."""
    return np.sqrt(a_au**3 / mtot_msun)


def near_first_order_resonance(a1, a2, j, tol):
    """Within a fractional tolerance of the (j+1):j commensurability?"""
    return abs(((a2 / a1) ** 1.5) / ((j + 1) / j) - 1) < tol


def eggleton_lobe_fraction(q):
    """Roche-lobe radius of body 1 in units of the separation (Eggleton 1983)."""
    q13 = np.cbrt(q)
    q23 = q13 * q13
    return 0.49 * q23 / (0.6 * q23 + np.log1p(q13))


def roche_lobe_min_separation(radius_rsun, mstar_msun, mp_msun):
    """Smallest separation [AU] at which the star still fits inside its lobe."""
    return (C.ROCHE_SAFETY_FACTOR * (radius_rsun * C.RSUN_IN_AU)
            / eggleton_lobe_fraction(mstar_msun / mp_msun))


def classify_with_resonance(mu1, mu2, a1, a2, e1, e2):
    """unstable / likely_unstable / resonant_stable_possible / stable."""
    if a1 * (1 + e1) >= a2 * (1 - e2):
        return "unstable"
    hill_radius = ((mu1 + mu2) / 3) ** (1 / 3) * (a1 + a2) / 2
    delta = (a2 - a1) / hill_radius
    near = any(near_first_order_resonance(a1, a2, j=j, tol=C.RESONANCE_TOLERANCE)
               for j in C.RESONANCE_ORDERS)
    if delta < C.HILL_STABILITY_FACTOR * np.sqrt(3):
        return "resonant_stable_possible" if near else "likely_unstable"
    return "stable"


# --------------------------------------------------------------------------
# Per-star noise: one definition, used for both the S/N metric and the epochs
# --------------------------------------------------------------------------
def single_datum_sigma(n_good, n_fov, n_dof, sigma_calib, sigma_al):
    """Per-single-transit AL uncertainty [mas] implied by the DR3 solution."""
    if (not np.isfinite(n_good) or not np.isfinite(n_fov)
            or n_fov <= 0 or n_good <= n_dof):
        return np.nan
    n_al_ave = n_good / n_fov
    mu_ueva_single = (n_al_ave / (n_good - n_dof)
                      * ((n_fov - n_dof) * sigma_calib**2 + n_fov * sigma_al**2))
    return np.sqrt(mu_ueva_single / n_al_ave)


def star_noise_terms(star):
    """(sigma_single [mas], n_dof) for one star row."""
    n_dof = (C.N_DOF_FIVE_PARAM
             if star["astrometric_params_solved_dr3"] == C.PARAMS_SOLVED_FIVE_PARAM
             else C.N_DOF_OTHER)
    sigma = single_datum_sigma(float(star["astrometric_n_good_obs_al_dr3"]),
                               float(star["astrometric_matched_transits_dr3"]),
                               n_dof, float(star["sig_cal"]), float(star["sig_AL"]))
    return sigma, n_dof


# --------------------------------------------------------------------------
# Stage 2, for one source
# --------------------------------------------------------------------------
def draw_companions(population, star, *, n_transits, sigma_single):
    """Draw this star's companions. Returns a list of dicts (empty for the control).

    Raises RuntimeError if no physically allowed configuration is found within
    the draw budget, which the caller records as a skipped source.
    """
    n_companions = C.POPULATIONS[population]
    if n_companions == 0:
        return []

    rng = np.random.default_rng(
        system_seed(C.SEED_PLANETS, population, star["gaia_source_id"]))

    mstar = float(star["mass_interp"])
    rstar = float(star["radius_interp"])
    parallax = float(star["parallax"])

    log_a = (np.log10(C.A_MIN_AU), np.log10(C.A_MAX_AU))
    log_m = (np.log10(C.MASS_MIN_MJUP), np.log10(C.MASS_MAX_MJUP))
    a_crit = (C.BASELINE_YEARS**2 * mstar) ** (1.0 / 3.0)

    def draw_one():
        """One companion from the prior, rejecting Roche-lobe-violating draws."""
        for _ in range(MAX_DRAWS):
            sma = 10 ** rng.uniform(*log_a)
            mass = 10 ** rng.uniform(*log_m)
            mass_msun = mass * C.MJUP_IN_MSUN
            if sma < roche_lobe_min_separation(rstar, mstar, mass_msun):
                continue
            # S/N is recorded, not used to accept or reject
            alpha = mass_msun / (mstar + mass_msun) * sma * parallax
            snr_single = (alpha / sigma_single
                          if sigma_single and np.isfinite(sigma_single) else np.nan)
            snr_eff = snr_single / (1.0 + (sma / a_crit) ** 3)
            snr_total = np.sqrt(n_transits) * snr_eff if n_transits else np.nan
            return dict(sma=float(sma), mass_pl=float(mass),
                        alpha_mas=float(alpha), snr_single=float(snr_single),
                        snr_eff=float(snr_eff), snr_total=float(snr_total))
        raise RuntimeError(f"no Roche-allowed companion within {MAX_DRAWS} draws")

    def draw_angles():
        return dict(ecc=float(rng.uniform(C.ECC_MIN, C.ECC_MAX)),
                    inc=float(np.degrees(np.arccos(rng.uniform(-1, 1)))),
                    Omega=float(rng.uniform(0.0, 360.0)),
                    omega=float(rng.uniform(0.0, 360.0)),
                    M_anom=float(rng.uniform(0.0, 360.0)))

    def with_period(companion):
        companion["period"] = float(semimajor_axis_to_period(
            companion["sma"], mstar + companion["mass_pl"] * C.MJUP_IN_MSUN))
        return companion

    if n_companions == 1:
        return [with_period({**draw_one(), **draw_angles()})]

    # --- two companions: redraw until the pair is non-crossing and Hill-stable ---
    for _attempt in range(C.MAX_STABILITY_RETRIES):
        pair = [{**draw_one(), **draw_angles()} for _ in range(2)]
        coplanar = bool(rng.random() < C.COPLANAR_PROBABILITY)
        if coplanar:
            pair[1]["inc"], pair[1]["Omega"] = pair[0]["inc"], pair[0]["Omega"]

        pair.sort(key=lambda c: c["sma"])           # companion 1 is the inner one
        mu = [c["mass_pl"] * C.MJUP_IN_MSUN / mstar for c in pair]
        label = classify_with_resonance(mu[0], mu[1], pair[0]["sma"], pair[1]["sma"],
                                        pair[0]["ecc"], pair[1]["ecc"])
        if label in ("unstable", "likely_unstable"):
            continue
        for companion in pair:
            companion["coplanar"] = coplanar
            with_period(companion)
        return pair

    raise RuntimeError(f"no stable pair within {C.MAX_STABILITY_RETRIES} attempts")


def companion_columns(companions):
    """Flatten companions into the per-system truth columns (_1, _2 suffixes)."""
    row = {"n_planets": len(companions)}
    for index, companion in enumerate(companions, start=1):
        for key in ("sma", "ecc", "mass_pl", "inc", "Omega", "omega", "M_anom",
                    "period", "alpha_mas", "snr_single", "snr_eff", "snr_total"):
            row[f"{key}_{index}"] = companion[key]
    if len(companions) == 2:
        inner, outer = companions
        row["coplanar"] = bool(inner["coplanar"])
        row["P_ratio"] = outer["period"] / inner["period"]
        for j in C.RESONANCE_ORDERS:
            row[f"near_{j + 1}_{j}"] = bool(near_first_order_resonance(
                inner["sma"], outer["sma"], j=j, tol=C.RESONANCE_TOLERANCE))
        row["near_resonance"] = any(row[f"near_{j + 1}_{j}"] for j in C.RESONANCE_ORDERS)
    return row
