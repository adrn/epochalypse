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

The rejection loop that remains is over *physical possibility* only: the star
must fit inside its Roche lobe, and a two-companion pair must be non-crossing
and Hill-stable.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .config import CatalogConfig, PopulationSpec, PlanetPriors


# --------------------------------------------------------------------------
# Seeding: keyed on the source id, never on a row index
# --------------------------------------------------------------------------
def system_seed(master_seed: int, population: str, gaia_source_id) -> int:
    """Stable uint32 seed for one (population, source) pair."""
    payload = f"{int(master_seed)}:{population}:{gaia_source_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


def source_rng(master_seed: int, population: str, gaia_source_id):
    return np.random.default_rng(system_seed(master_seed, population, gaia_source_id))


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


def roche_lobe_min_separation(radius_rsun, mstar_msun, mp_msun, *, factor, rsun_in_au):
    """Smallest separation [AU] at which the star still fits inside its lobe."""
    return factor * (radius_rsun * rsun_in_au) / eggleton_lobe_fraction(mstar_msun / mp_msun)


def classify_with_resonance(mu1, mu2, a1, a2, e1, e2, priors: PlanetPriors):
    """unstable / likely_unstable / resonant_stable_possible / stable."""
    if a1 * (1 + e1) >= a2 * (1 - e2):
        return "unstable"
    hill_radius = ((mu1 + mu2) / 3) ** (1 / 3) * (a1 + a2) / 2
    delta = (a2 - a1) / hill_radius
    near = any(near_first_order_resonance(a1, a2, j=j, tol=priors.resonance_tolerance)
               for j in priors.resonance_orders)
    if delta < priors.hill_stability_factor * np.sqrt(3):
        return "resonant_stable_possible" if near else "likely_unstable"
    return "stable"


# --------------------------------------------------------------------------
# Per-star noise, used for the recorded S/N metric
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


def star_noise_terms(star, astrometry):
    """(sigma_single [mas], n_dof) for one star row."""
    n_dof = (astrometry.n_dof_five_param
             if star["astrometric_params_solved_dr3"] == astrometry.params_solved_five_param
             else astrometry.n_dof_other)
    sigma = single_datum_sigma(float(star["astrometric_n_good_obs_al_dr3"]),
                               float(star["astrometric_matched_transits_dr3"]),
                               n_dof, float(star["sig_cal"]), float(star["sig_AL"]))
    return sigma, n_dof


# --------------------------------------------------------------------------
# Stage 2, for one source
# --------------------------------------------------------------------------
def draw_companions(config: CatalogConfig, spec: PopulationSpec, star, *,
                    n_transits, sigma_single):
    """Draw this star's companions. Returns a list of dicts (empty for the control).

    Raises RuntimeError if no physically allowed configuration is found within
    the draw budget, which the caller records as a skipped source.
    """
    if spec.n_companions == 0:
        return []

    priors = config.priors
    rng = source_rng(config.seeds.planets, spec.name, star["gaia_source_id"])

    mstar = float(star["mass_interp"])
    rstar = float(star["radius_interp"])
    parallax = float(star["parallax"])
    rsun_in_au = config.astrometry.rsun_to_au

    log_a = (np.log10(priors.a_min_au), np.log10(priors.a_max_au))
    log_m = (np.log10(priors.mass_min_mjup), np.log10(priors.mass_max_mjup))
    a_crit = (priors.baseline_years**2 * mstar) ** (1.0 / 3.0)

    def draw_one():
        """One companion from the prior, rejecting Roche-lobe-violating draws."""
        drawn = 0
        while drawn < priors.max_draws:
            batch = priors.draw_batch
            sma = 10 ** rng.uniform(*log_a, size=batch)
            mass = 10 ** rng.uniform(*log_m, size=batch)
            mass_msun = mass * priors.mjup_in_msun

            accept = np.ones(batch, dtype=bool)
            if priors.enforce_roche_lobe:
                accept &= sma >= roche_lobe_min_separation(
                    rstar, mstar, mass_msun,
                    factor=priors.roche_lobe_safety_factor, rsun_in_au=rsun_in_au)
            hits = np.where(accept)[0]
            if hits.size:
                i = hits[0]
                # S/N is recorded, not used to accept or reject
                alpha = mass_msun[i] / (mstar + mass_msun[i]) * sma[i] * parallax
                snr_single = alpha / sigma_single if sigma_single and np.isfinite(sigma_single) else np.nan
                snr_eff = snr_single / (1.0 + (sma[i] / a_crit) ** 3)
                snr_total = np.sqrt(n_transits) * snr_eff if n_transits else np.nan
                return dict(sma=float(sma[i]), mass_pl=float(mass[i]),
                            alpha_mas=float(alpha), snr_single=float(snr_single),
                            snr_eff=float(snr_eff), snr_total=float(snr_total))
            drawn += batch
        raise RuntimeError(f"no Roche-allowed companion within {priors.max_draws} draws")

    def draw_angles():
        ecc = rng.uniform(priors.ecc_min, priors.ecc_max)
        inc = (np.degrees(np.arccos(rng.uniform(-1, 1))) if priors.isotropic_inclination
               else rng.uniform(priors.angle_min_deg, priors.angle_max_deg))
        return dict(ecc=float(ecc), inc=float(inc),
                    Omega=float(rng.uniform(priors.angle_min_deg, priors.angle_max_deg)),
                    omega=float(rng.uniform(priors.angle_min_deg, priors.angle_max_deg)),
                    M_anom=float(rng.uniform(priors.angle_min_deg, priors.angle_max_deg)))

    if spec.n_companions == 1:
        companion = {**draw_one(), **draw_angles()}
        companion["period"] = float(semimajor_axis_to_period(
            companion["sma"], mstar + companion["mass_pl"] * priors.mjup_in_msun))
        return [companion]

    # --- two companions: redraw until the pair is non-crossing and Hill-stable ---
    for _attempt in range(priors.max_stability_retries):
        first, second = draw_one(), draw_one()
        angles = [draw_angles(), draw_angles()]
        coplanar = (bool(rng.integers(2)) if priors.coplanar_probability == 0.5
                    else bool(rng.random() < priors.coplanar_probability))
        if coplanar:
            angles[1]["inc"], angles[1]["Omega"] = angles[0]["inc"], angles[0]["Omega"]

        pair = [{**first, **angles[0]}, {**second, **angles[1]}]
        pair.sort(key=lambda c: c["sma"])           # companion 1 is the inner one

        mu = [c["mass_pl"] * priors.mjup_in_msun / mstar for c in pair]
        label = classify_with_resonance(mu[0], mu[1], pair[0]["sma"], pair[1]["sma"],
                                        pair[0]["ecc"], pair[1]["ecc"], priors)
        if label in ("unstable", "likely_unstable"):
            continue
        for companion in pair:
            companion["coplanar"] = coplanar
            companion["period"] = float(semimajor_axis_to_period(
                companion["sma"], mstar + companion["mass_pl"] * priors.mjup_in_msun))
        return pair

    raise RuntimeError(f"no stable pair within {priors.max_stability_retries} attempts")


def companion_columns(companions, priors):
    """Flatten companions into the per-system truth columns (_1, _2 suffixes)."""
    row = {"n_planets": len(companions)}
    for index, companion in enumerate(companions, start=1):
        for key in ("sma", "ecc", "mass_pl", "inc", "Omega", "omega", "M_anom",
                    "period", "alpha_mas", "snr_single", "snr_eff", "snr_total"):
            row[f"{key}_{index}"] = companion[key]
    if len(companions) == 2:
        inner, outer = companions
        row["coplanar"] = bool(inner.get("coplanar", False))
        row["P_ratio"] = outer["period"] / inner["period"]
        for j in priors.resonance_orders:
            row[f"near_{j + 1}_{j}"] = bool(near_first_order_resonance(
                inner["sma"], outer["sma"], j=j, tol=priors.resonance_tolerance))
        row["near_resonance"] = any(row[f"near_{j + 1}_{j}"] for j in priors.resonance_orders)
    return row
