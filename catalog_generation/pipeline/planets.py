"""Stage 2: synthesize companion populations around the parent stellar sample.

Ported from `sim_planets.ipynb`. Every prior and threshold arrives via
`config.priors`; nothing scientific is hard-coded here. The order of RNG calls
is preserved exactly as in the notebook, so a run with the same master seed
reproduces the released draws.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import CatalogConfig, PopulationSpec, PlanetPriors


# --------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------
def stable_population_seed(master_seed: int, seed_key: str) -> int:
    """Stable uint32 seed, independent of Python's hash randomization."""
    payload = f"{int(master_seed)}:{seed_key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


# --------------------------------------------------------------------------
# Orbital bookkeeping
# --------------------------------------------------------------------------
def semimajor_axis_to_period(a_au, mtot_msun):
    """Kepler's third law: P [yr] from a [AU] and M_star + M_p [Msun].

    The companion mass is not negligible at the top of the mass prior (an
    80 Mjup brown dwarf is ~15% of a 0.5 Msun star), so it is retained.
    """
    return np.sqrt(a_au**3 / mtot_msun)


def near_first_order_resonance(a1, a2, j, tol):
    """Flag proximity to the (j+1):j first-order resonance.

    A selection criterion, not a dynamical stability condition: it asks whether
    P2/P1 sits within a fractional tolerance `tol` of (j+1)/j.
    """
    period_ratio = (a2 / a1) ** 1.5
    target = (j + 1) / j
    return abs(period_ratio / target - 1) < tol


def classify_with_resonance(mu1, mu2, a1, a2, e1, e2, priors: PlanetPriors):
    """Label a two-companion pair: unstable / likely_unstable /
    resonant_stable_possible / stable."""
    # 1. Orbit crossing is always unstable.
    if a1 * (1 + e1) >= a2 * (1 - e2):
        return "unstable"

    # 2. Hill spacing.
    hill_radius = ((mu1 + mu2) / 3) ** (1 / 3) * (a1 + a2) / 2
    delta = (a2 - a1) / hill_radius

    # 3. Proximity to a first-order MMR can stabilize a tight pair.
    near_resonance = any(
        near_first_order_resonance(a1, a2, j=j, tol=priors.resonance_tolerance)
        for j in priors.resonance_orders
    )

    if delta < priors.hill_stability_factor * np.sqrt(3):
        return "resonant_stable_possible" if near_resonance else "likely_unstable"
    return "stable"


# --------------------------------------------------------------------------
# Per-star noise: the single-transit AL uncertainty
# --------------------------------------------------------------------------
def single_datum_sigma(n_good, n_fov, n_dof, sigma_calib, sigma_al):
    """Per-single-transit AL uncertainty [mas] implied by the DR3 solution."""
    if (not np.isfinite(n_good) or not np.isfinite(n_fov)
            or n_fov <= 0 or n_good <= n_dof):
        return np.nan
    n_al_ave = n_good / n_fov
    mu_ueva_single = (
        n_al_ave / (n_good - n_dof)
        * ((n_fov - n_dof) * sigma_calib**2 + n_fov * sigma_al**2)
    )
    return np.sqrt(mu_ueva_single / n_al_ave)


def dr4_transit_counts(scanlaw_path, gaia_source_ids):
    """Number of DR4 FoV transits per star, counted from the scan-law table.

    This is the N in the sqrt(N) accumulation of single-transit S/N. It comes
    from the per-star DR4 scan law (one row per FoV transit) rather than a
    sky-position estimate; the DR3 matched-transit column is the wrong data
    release, and n_good_obs counts CCD measurements (~9 per transit).
    """
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(scanlaw_path), "r") as handle:
        ids = ipc.open_file(handle).read_all().column("gaia_source_id").to_numpy()
    unique_ids, counts = np.unique(ids, return_counts=True)
    by_id = dict(zip(unique_ids.astype(str), counts.astype(float)))
    return np.array([by_id.get(str(sid), np.nan) for sid in gaia_source_ids], dtype=float)


# --------------------------------------------------------------------------
# Stage 2
# --------------------------------------------------------------------------
def generate_population(config: CatalogConfig, spec: PopulationSpec, *,
                        overwrite=True, limit=None) -> pd.DataFrame:
    """Draw one companion population and write `outputs/data/<name>.csv`.

    `limit` truncates the parent sample (quick smoke runs); it changes the
    output, so never use it for a release run.
    """
    priors, paths = config.priors, config.paths
    output_path = paths.population_csv(spec.name)

    if spec.n_companions == 0:
        raise ValueError("the 0-companion control has no population CSV; "
                         "it is drawn straight from stars.csv in stage 3")
    if output_path.exists() and not overwrite:
        print(f"  {output_path} exists; reusing (pass --overwrite to redraw)")
        return pd.read_csv(output_path, dtype={"gaia_source_id": str,
                                               "source_id_dr2": str})

    seed = stable_population_seed(config.seeds.planets, spec.seed_key)
    rng = np.random.default_rng(seed)
    print(f"  seed key {spec.seed_key!r} -> seed {seed}")

    catalog = pd.read_csv(paths.stars_csv,
                          dtype={"gaia_source_id": str, "source_id_dr2": str},
                          low_memory=False)

    if config.stars.require_sigma_al:
        n_before = len(catalog)
        catalog = catalog[np.isfinite(catalog["sig_AL"].to_numpy(dtype=float))]
        catalog = catalog.reset_index(drop=True)
        if len(catalog) < n_before:
            print(f"  dropped {n_before - len(catalog)} stars with missing sig_AL "
                  "(no noise model)")
    if limit is not None:
        catalog = catalog.head(limit).reset_index(drop=True)
        print(f"  LIMIT: parent sample truncated to {len(catalog)} stars")

    n_stars = len(catalog)
    gaia_source_ids = catalog["gaia_source_id"].values
    source_id_dr2s = catalog["source_id_dr2"].values
    mstar_arr = catalog["mass_interp"].values
    parallax_arr = catalog["parallax"].to_numpy(dtype=float)
    n_good_arr = catalog["astrometric_n_good_obs_al_dr3"].to_numpy(dtype=float)
    n_fov_arr = catalog["astrometric_matched_transits_dr3"].to_numpy(dtype=float)
    sigma_al_arr = catalog["sig_AL"].to_numpy(dtype=float)
    sigma_calib_arr = catalog["sig_cal"].to_numpy(dtype=float)
    params_solved_arr = catalog["astrometric_params_solved_dr3"].to_numpy()

    n_dr4_arr = dr4_transit_counts(paths.scanlaw_dr4, gaia_source_ids)

    astro = config.astrometry
    sigma_single_arr = np.array([
        single_datum_sigma(
            n_good_arr[i], n_fov_arr[i],
            astro.n_dof_five_param if params_solved_arr[i] == astro.params_solved_five_param
            else astro.n_dof_other,
            sigma_calib_arr[i], sigma_al_arr[i],
        )
        for i in range(n_stars)
    ])

    log_a_min, log_a_max = np.log10(priors.a_min_au), np.log10(priors.a_max_au)
    log_m_min, log_m_max = np.log10(priors.mass_min_mjup), np.log10(priors.mass_max_mjup)

    def draw_one_companion(mstar, parallax_mas, sigma_single_mas, n_dr4):
        """Draw from the mass/sma prior.

        With `filter_snr` set, keep only companions whose period-suppressed
        *total* detectability clears the threshold, accumulating single-transit
        S/N over the star's DR4 FoV transits:

            SNR_total = sqrt(N_DR4) * (alpha / sigma_single) / (1 + (a/a_crit)^3)

        with a_crit the semi-major axis whose period equals the DR4 baseline, so
        the suppression turns on once the orbit outruns the observing window.
        Returns None if no acceptable companion is found within `snr_max_draws`
        (i.e. this star hosts no detectable companion).
        """
        if spec.filter_snr and (not np.isfinite(sigma_single_mas) or sigma_single_mas <= 0
                                or not np.isfinite(n_dr4) or n_dr4 <= 0):
            return None
        a_crit = (priors.baseline_years**2 * mstar) ** (1.0 / 3.0)
        drawn = 0
        while drawn < priors.snr_max_draws:
            batch = priors.snr_draw_batch
            sma = 10**rng.uniform(log_a_min, log_a_max, size=batch)
            mass = 10**rng.uniform(log_m_min, log_m_max, size=batch)
            mass_msun = mass * priors.mjup_in_msun_prior
            alpha_mas = mass_msun / (mstar + mass_msun) * sma * parallax_mas
            snr_single = alpha_mas / sigma_single_mas
            snr_eff = snr_single / (1.0 + (sma / a_crit) ** 3)
            snr_total = np.sqrt(n_dr4) * snr_eff

            accept = np.ones(batch, dtype=bool)
            if spec.filter_snr:
                accept &= np.isfinite(snr_total) & (snr_total >= priors.snr_total_min)
            hits = np.where(accept)[0]
            if hits.size:
                i = hits[0]
                return (sma[i], mass[i], alpha_mas[i],
                        snr_single[i], snr_eff[i], snr_total[i])
            drawn += batch
        return None

    def draw_angles():
        ecc = rng.uniform(priors.ecc_min, priors.ecc_max)
        if priors.isotropic_inclination:
            inc = np.degrees(np.arccos(rng.uniform(-1, 1)))
        else:
            inc = rng.uniform(priors.angle_min_deg, priors.angle_max_deg)
        node = rng.uniform(priors.angle_min_deg, priors.angle_max_deg)
        argument = rng.uniform(priors.angle_min_deg, priors.angle_max_deg)
        mean_anomaly = rng.uniform(priors.angle_min_deg, priors.angle_max_deg)
        return ecc, inc, node, argument, mean_anomaly

    def draw_coplanar():
        # 0.5 keeps the historical integer draw so the released populations
        # reproduce bit-for-bit; any other probability uses a uniform draw.
        if priors.coplanar_probability == 0.5:
            return bool(rng.integers(2))
        return bool(rng.random() < priors.coplanar_probability)

    rows = []
    n_failed = n_missing_sigma = n_no_detectable = 0

    for i in range(n_stars):
        mstar = mstar_arr[i]
        parallax_mas = parallax_arr[i]
        sigma_single_mas = sigma_single_arr[i]
        n_dr4 = n_dr4_arr[i]

        # The S/N filter needs a valid per-star noise estimate and transit
        # count; a few stars lack one, so drop them from the filtered
        # population rather than aborting the run.
        if spec.filter_snr and not (np.isfinite(sigma_single_mas) and sigma_single_mas > 0
                                    and np.isfinite(n_dr4) and n_dr4 > 0):
            n_missing_sigma += 1
            continue

        if spec.n_companions == 1:
            drawn = draw_one_companion(mstar, parallax_mas, sigma_single_mas, n_dr4)
            if drawn is None:
                n_no_detectable += 1
                continue
            sma, mass_pl, alpha_mas, snr_single, snr_eff, snr_total = drawn
            ecc, inc, node, argument, mean_anomaly = draw_angles()
            rows.append({
                "gaia_source_id": gaia_source_ids[i],
                "source_id_dr2": source_id_dr2s[i],
                "mass_st": mstar,
                "sma": sma, "ecc": ecc, "mass_pl": mass_pl,
                "inc": inc, "Omega": node, "omega": argument, "M_anom": mean_anomaly,
                "period": semimajor_axis_to_period(
                    sma, mstar + mass_pl * priors.mjup_in_msun_prior),
                "alpha_mas": alpha_mas,
                "sigma_single_mas": sigma_single_mas,
                "snr_single": snr_single,
                "snr_eff": snr_eff,
                "snr_total": snr_total,
                "n_transits_dr4": n_dr4,
            })

        elif spec.n_companions == 2:
            found = no_detectable = False
            for _attempt in range(priors.max_stability_retries):
                draw1 = draw_one_companion(mstar, parallax_mas, sigma_single_mas, n_dr4)
                draw2 = draw_one_companion(mstar, parallax_mas, sigma_single_mas, n_dr4)
                if draw1 is None or draw2 is None:
                    no_detectable = True
                    break
                sma1, mass1, alpha1, snr1, snr_eff1, snr_tot1 = draw1
                sma2, mass2, alpha2, snr2, snr_eff2, snr_tot2 = draw2
                ecc1, inc1, node1, arg1, anom1 = draw_angles()
                ecc2, inc2, node2, arg2, anom2 = draw_angles()

                coplanar = draw_coplanar()
                if coplanar:
                    inc2, node2 = inc1, node1

                # order the pair so companion 1 is the inner one
                if sma1 > sma2:
                    sma1, sma2 = sma2, sma1
                    mass1, mass2 = mass2, mass1
                    alpha1, alpha2 = alpha2, alpha1
                    snr1, snr2 = snr2, snr1
                    snr_eff1, snr_eff2 = snr_eff2, snr_eff1
                    snr_tot1, snr_tot2 = snr_tot2, snr_tot1
                    ecc1, ecc2 = ecc2, ecc1
                    inc1, inc2 = inc2, inc1
                    node1, node2 = node2, node1
                    arg1, arg2 = arg2, arg1
                    anom1, anom2 = anom2, anom1

                mu1 = mass1 * priors.mjup_in_msun_stability / mstar
                mu2 = mass2 * priors.mjup_in_msun_stability / mstar
                label = classify_with_resonance(mu1, mu2, sma1, sma2, ecc1, ecc2, priors)
                if label not in ("unstable", "likely_unstable"):
                    found = True
                    break

            if no_detectable:
                n_no_detectable += 1
                continue
            if not found:
                n_failed += 1
                continue

            period1 = semimajor_axis_to_period(
                sma1, mstar + mass1 * priors.mjup_in_msun_prior)
            period2 = semimajor_axis_to_period(
                sma2, mstar + mass2 * priors.mjup_in_msun_prior)
            near = {
                f"near_{j + 1}_{j}": near_first_order_resonance(
                    sma1, sma2, j=j, tol=priors.resonance_tolerance)
                for j in priors.resonance_orders
            }
            rows.append({
                "gaia_source_id": gaia_source_ids[i],
                "source_id_dr2": source_id_dr2s[i],
                "mass_st": mstar,
                "sma_1": sma1, "ecc_1": ecc1, "mass_pl_1": mass1,
                "inc_1": inc1, "Omega_1": node1, "omega_1": arg1, "M_anom_1": anom1,
                "sma_2": sma2, "ecc_2": ecc2, "mass_pl_2": mass2,
                "inc_2": inc2, "Omega_2": node2, "omega_2": arg2, "M_anom_2": anom2,
                "period_1": period1, "period_2": period2,
                "alpha_1_mas": alpha1, "alpha_2_mas": alpha2,
                "sigma_single_mas": sigma_single_mas,
                "snr_single_1": snr1, "snr_single_2": snr2,
                "snr_eff_1": snr_eff1, "snr_eff_2": snr_eff2,
                "snr_total_1": snr_tot1, "snr_total_2": snr_tot2,
                "n_transits_dr4": n_dr4,
                "P_ratio": period2 / period1,
                **near,
                "coplanar": coplanar,
            })
        else:
            raise ValueError(f"n_companions must be 1 or 2, got {spec.n_companions}")

        if (i + 1) % 500 == 0 or i == n_stars - 1:
            print(f"  processed {i + 1} / {n_stars} stars", end="\r")

    print()
    if n_failed:
        print(f"  WARNING: {n_failed} stars skipped (no stable pair in "
              f"{priors.max_stability_retries} attempts)")
    if n_missing_sigma:
        print(f"  WARNING: {n_missing_sigma} stars skipped (no valid single-datum "
              "sigma or DR4 scan law for the S/N filter)")
    if n_no_detectable:
        print(f"  WARNING: {n_no_detectable} stars skipped (no companion clears "
              f"period-suppressed total S/N >= {priors.snr_total_min:g})")

    population = pd.DataFrame(rows)
    population["generation_seed"] = int(seed)

    if spec.n_companions == 2 and len(population):
        resonance_cols = [f"near_{j + 1}_{j}" for j in priors.resonance_orders]
        population["near_resonance"] = population[resonance_cols].any(axis=1)
        n_res = int(population["near_resonance"].sum())
        detail = ", ".join(
            f"{int(population[col].sum())} near {col.split('_')[1]}:{col.split('_')[2]}"
            for col in resonance_cols
        )
        print(f"  {len(population)} systems retained | near first-order MMR: "
              f"{n_res} ({n_res / len(population):.1%}) [{detail}]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    population.to_csv(output_path, index=False)
    print(f"  wrote {len(population):,} systems -> {output_path}")
    return population
