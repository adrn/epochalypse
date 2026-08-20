"""Full-sampler (NUTS) validation of the fast periodogram characterization.

Ports the astrometric orbit model from ``fit_astrometric_orbits.ipynb`` (cells 6-7) into
data-explicit functions so it can be driven over a stratified validation set. For each system we
run a blind (non-truth-seeded) 1-planet NUTS fit with several chains initialized at dispersed
periods (the top periodogram peaks + random draws), count the number of distinct posterior period
modes, and record whether the injected period lies within the posterior. Comparing this to the
periodogram class + truth-recovery yields the confusion matrix that licenses the fast method at
scale.

Run under the JAX env, e.g.:
    /Users/daniel/opt/anaconda3/envs/epochalypse-arm64/bin/python epochalypse_validation.py
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value
from jaxoplanet.orbits.keplerian import Central, System

import epochalypse_fitting as ef

jax.config.update("jax_enable_x64", True)

# Model bounds (mirror fit_astrometric_orbits.ipynb, cell 3)
PERIOD_MIN_YEARS = ef.PERIOD_MIN_YEARS
PERIOD_MAX_YEARS = ef.PERIOD_MAX_YEARS
MJUP_TO_MSUN = ef.MJUP_TO_MSUN
MARS_TO_MSUN = ef.MARS_TO_MSUN
MAX_COMPANION_MASS_MSUN = ef.MAX_COMPANION_MASS_MSUN
RSUN_TO_AU = ef.RSUN_TO_AU
TWOPI = ef.TWOPI
ECC_MAX = ef.ECC_MAX
AL_JITTER_MIN_MAS = 1.0e-3
AL_JITTER_MAX_MAS = 5.0


def stellar_reflex_al(t, scan_angle, parallax_mas, planets, mstar, rstar):
    """JAXoplanet stellar reflex projected into Gaia along-scan coordinates (cell 6)."""
    t_days = jnp.asarray(t) * DAYS_PER_YEAR
    system = System(Central(mass=jnp.asarray(mstar), radius=jnp.asarray(rstar)))
    for planet in planets:
        period_days = planet["period_years"] * DAYS_PER_YEAR
        mean_motion = 2.0 * jnp.pi / period_days
        system = system.add_body(
            period=period_days, mass=planet["mass_msun"],
            eccentricity=planet["eccentricity"], inclination=planet["inclination"],
            omega_peri=planet["omega"], asc_node=planet["Omega"],
            time_peri=-planet["M0"] / mean_motion,
        )
    x_rsun = jnp.zeros_like(t_days)
    y_rsun = jnp.zeros_like(t_days)
    for body in system.bodies:
        x_body, y_body, _ = body.central_position(t_days)
        x_rsun = x_rsun + x_body
        y_rsun = y_rsun + y_body
    x_mas = x_rsun * RSUN_TO_AU * parallax_mas
    y_mas = y_rsun * RSUN_TO_AU * parallax_mas
    return -(x_mas * jnp.cos(scan_angle) + y_mas * jnp.sin(scan_angle))


def build_model(n_planets, beta_loc, beta_scale, mstar, rstar):
    """Along-scan NumPyro model with WLS-informed priors (cell 6), data-explicit."""
    log_period_min, log_period_max = np.log(PERIOD_MIN_YEARS), np.log(PERIOD_MAX_YEARS)
    log_mass_min = np.log(MARS_TO_MSUN / MJUP_TO_MSUN)
    log_mass_max = np.log(MAX_COMPANION_MASS_MSUN / MJUP_TO_MSUN)
    root_e_max = float(np.sqrt(ECC_MAX))
    log_jit_min, log_jit_max = np.log(AL_JITTER_MIN_MAS), np.log(AL_JITTER_MAX_MAS)

    def model(t, scan_angle, pf, reported_error, observed=None):
        alpha0 = numpyro.sample("alpha0_mas", dist.Normal(beta_loc[0], beta_scale[0]))
        delta0 = numpyro.sample("delta0_mas", dist.Normal(beta_loc[1], beta_scale[1]))
        pmra = numpyro.sample("pmra_mas_yr", dist.Normal(beta_loc[2], beta_scale[2]))
        pmdec = numpyro.sample("pmdec_mas_yr", dist.Normal(beta_loc[3], beta_scale[3]))
        parallax = numpyro.sample("parallax_mas",
                                  dist.TruncatedNormal(beta_loc[4], beta_scale[4], low=1.0e-3))
        log_al_jitter = numpyro.sample("log_al_jitter_mas", dist.Uniform(log_jit_min, log_jit_max))

        # NB: no period ordering. Sorting only the periods (while masses/eccentricities stay attached
        # to their sample index) makes the likelihood DISCONTINUOUS across P1=P2 and drives huge
        # divergences. We leave the two planets exchangeable (benign label-switching symmetry) and
        # resolve labels post-hoc by sorting each posterior sample's periods in analyze_recovery.
        period_logs = [numpyro.sample(f"log_period_{i}", dist.Uniform(log_period_min, log_period_max))
                       for i in range(1, n_planets + 1)]

        planets = []
        for idx in range(1, n_planets + 1):
            log_mass = numpyro.sample(f"log_mass_{idx}", dist.Uniform(log_mass_min, log_mass_max))
            h = numpyro.sample(f"h_{idx}", dist.Uniform(-root_e_max, root_e_max))
            k = numpyro.sample(f"k_{idx}", dist.Uniform(-root_e_max, root_e_max))
            eccentricity = h**2 + k**2
            numpyro.factor(f"ecc_bound_{idx}", jnp.where(eccentricity < ECC_MAX, 0.0, -jnp.inf))
            cos_i = numpyro.sample(f"cos_i_{idx}", dist.Uniform(-1.0, 1.0))
            Omega = numpyro.sample(f"Omega_{idx}", dist.Uniform(0.0, TWOPI))
            M0 = numpyro.sample(f"M0_{idx}", dist.Uniform(0.0, TWOPI))
            period_years = jnp.exp(period_logs[idx - 1])
            mass_mjup = jnp.exp(log_mass)
            numpyro.deterministic(f"period_years_{idx}", period_years)
            planets.append({
                "period_years": period_years, "mass_msun": mass_mjup * MJUP_TO_MSUN,
                "eccentricity": eccentricity, "inclination": jnp.arccos(cos_i),
                "omega": jnp.arctan2(k, h), "Omega": Omega, "M0": M0,
            })

        sin_psi, cos_psi = jnp.sin(scan_angle), jnp.cos(scan_angle)
        astrometry = sin_psi * (alpha0 + pmra * t) + cos_psi * (delta0 + pmdec * t) + parallax * pf
        reflex = stellar_reflex_al(t, scan_angle, parallax, planets, mstar, rstar) if n_planets else 0.0
        mu = numpyro.deterministic("mu", astrometry + reflex)
        sigma = jnp.sqrt(reported_error**2 + jnp.exp(log_al_jitter)**2)
        numpyro.sample("obs", dist.Normal(mu, sigma), obs=observed)

    return model


def init_values(beta_wls, seed_periods):
    """Blind chain init: WLS astrometry + seed period(s) (n=1 or 2), companion params neutral."""
    seeds = sorted(float(p) for p in seed_periods)
    lp_min, lp_max = np.log(PERIOD_MIN_YEARS), np.log(PERIOD_MAX_YEARS)
    lm_min, lm_max = np.log(MARS_TO_MSUN / MJUP_TO_MSUN), np.log(MAX_COMPANION_MASS_MSUN / MJUP_TO_MSUN)
    vals = {
        "alpha0_mas": float(beta_wls[0]), "delta0_mas": float(beta_wls[1]),
        "pmra_mas_yr": float(beta_wls[2]), "pmdec_mas_yr": float(beta_wls[3]),
        "parallax_mas": float(beta_wls[4]),
        "log_al_jitter_mas": float(np.log(np.sqrt(AL_JITTER_MIN_MAS * AL_JITTER_MAX_MAS))),
    }
    # Seed a small nonzero eccentricity/inclination: h=k=0 makes omega=atan2(0,0) whose gradient is
    # NaN, and cos_i=0 (edge-on) is likewise a degenerate seed -- either makes NUTS reject the init.
    e0, w0 = 0.1, 0.3
    for i, P in enumerate(seeds, start=1):
        vals[f"log_period_{i}"] = float(np.clip(np.log(P), lp_min + 1e-3, lp_max - 1e-3))
        vals[f"log_mass_{i}"] = float(0.5 * (lm_min + lm_max))
        vals[f"h_{i}"] = float(np.sqrt(e0) * np.cos(w0 + i))
        vals[f"k_{i}"] = float(np.sqrt(e0) * np.sin(w0 + i))
        vals[f"cos_i_{i}"] = 0.2
        vals[f"Omega_{i}"] = 1.0 + 0.3 * i
        vals[f"M0_{i}"] = 1.0 + 0.3 * i
    return vals


def seed_period_sets(n_planets, peaks, n_chains=4):
    """Dispersed blind seed periods for the chains, built from the periodogram peaks.

    ``peaks`` are candidate periods (e.g. [peak1_period, peak2_period]). For n=1 returns a list of
    single-period seeds; for n=2 a list of (P_inner, P_outer) pairs. Always ``n_chains`` sets, padded
    with generic dispersed periods so different chains start in different basins.
    """
    peaks = [float(p) for p in peaks if np.isfinite(p) and p > 0]
    spread = [1.0, 3.0, 10.0, 30.0, 100.0]
    if n_planets == 1:
        seeds = list(dict.fromkeys(peaks + spread))[:n_chains]
        return [[s] for s in seeds]
    pk = sorted(peaks)
    pairs = []
    if len(pk) >= 2:
        pairs.append((pk[0], pk[1]))
    pairs += [(1.0, 10.0), (2.0, 30.0), (3.0, 100.0), (1.0, 5.0), (5.0, 50.0)]
    seen, out = set(), []
    for a, b in pairs:
        a, b = sorted((a, b))
        key = (round(np.log10(a), 2), round(np.log10(b), 2))
        if key in seen:
            continue
        seen.add(key)
        out.append([a, b])
        if len(out) >= n_chains:
            break
    return out


def fit_system(t, psi, pf, y, yerr, mstar, rstar, n_planets, seeds,
               num_warmup=2000, num_samples=2000, target_accept=0.9, rng_seed=0):
    """Blind, dispersed multi-start NUTS: one chain per seed-period set in ``seeds``.

    Returns a dict with per-chain period samples per planet, total divergences, and wall time.
    """
    import time as _time
    beta_wls, beta_cov, _ = ef.weighted_linear_solution(t, psi, pf, y, yerr)
    beta_scale = np.maximum(10.0 * np.sqrt(np.diag(beta_cov)), np.array([10., 10., 10., 10., 5.]))
    model = build_model(n_planets, beta_wls, beta_scale, mstar, rstar)
    chains, n_div = [], 0
    t0 = _time.time()
    for c, sset in enumerate(seeds):
        kernel = NUTS(model, init_strategy=init_to_value(values=init_values(beta_wls, sset)),
                      target_accept_prob=target_accept, dense_mass=True)
        mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                    num_chains=1, progress_bar=False)
        mcmc.run(jax.random.key(rng_seed + c), jnp.asarray(t), jnp.asarray(psi),
                 jnp.asarray(pf), jnp.asarray(yerr), observed=jnp.asarray(y),
                 extra_fields=("diverging",))
        s = mcmc.get_samples()
        chains.append({i: np.asarray(s[f"period_years_{i}"]) for i in range(1, n_planets + 1)})
        n_div += int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return {"chains": chains, "n_planets": n_planets, "n_divergences": n_div,
            "n_chains": len(seeds), "num_warmup": num_warmup, "num_samples": num_samples,
            "wall_s": _time.time() - t0}


def analyze_recovery(fit, truth_periods, merge_dex=0.15, min_frac=0.1, recover_tol=1.25):
    """Per-planet posterior modality and truth recovery from a ``fit_system`` result.

    For two-planet systems the two planets are exchangeable (we removed period ordering in the model),
    so we resolve labels post-hoc by sorting each posterior sample's periods into inner/outer before
    counting modes. out[1] = inner (shorter-period) companion, out[2] = outer.
    """
    n = fit["n_planets"]
    lt = np.log(recover_tol)
    if n == 1:
        slots = [(1, [ch[1] for ch in fit["chains"]], float(truth_periods[0]))]
    else:
        inner = [np.minimum(ch[1], ch[2]) for ch in fit["chains"]]
        outer = [np.maximum(ch[1], ch[2]) for ch in fit["chains"]]
        ts = sorted(float(p) for p in truth_periods)   # [inner_truth, outer_truth]
        slots = [(1, inner, ts[0]), (2, outer, ts[1])]

    out = {}
    for slot, per_chain, p_true in slots:
        n_modes, modes = count_period_modes(per_chain, merge_dex, min_frac)
        recovered = any(abs(np.log(m / p_true)) < lt for m in modes) if modes else False
        out[slot] = {"n_modes": int(n_modes), "modes": [float(m) for m in modes],
                     "recovered": bool(recovered), "truth_period": p_true,
                     "post_median": float(np.median(np.concatenate(per_chain)))}
    return out


def nuts_outcome(planet_result):
    """Collapse a planet's posterior to a category comparable with the periodogram class."""
    if planet_result["n_modes"] >= 2:
        return "multimodal"
    return "unimodal_recovered" if planet_result["recovered"] else "unimodal_biased"


def fit_one_system(spec, cfg):
    """Worker: load, fit, and analyze one system. Picklable/top-level for ProcessPoolExecutor.

    ``spec`` = dict(pop, npl, row_index, pdg_class, dom_k, peaks, rng_seed).
    ``cfg``  = dict(n_chains, num_warmup, num_samples, target_accept).
    Returns a flat result dict of scalars plus the dominant companion's pooled period samples
    (``dom_period_samples``, float32) for plotting.
    """
    ds = ef.load_simulated_system(ef.systems_h5_path(spec["pop"]), spec["row_index"])
    t, psi, pf, y, yerr = ef.epoch_arrays(ds["epochs"])
    truth = ds["truth"]
    mstar, rstar = float(truth["mass_st_msun"]), float(truth["radius_st_rsun"])
    npl = int(spec["npl"])
    truthP = [float(truth[f"period_{i}"]) for i in range(1, npl + 1)]
    seeds = seed_period_sets(npl, spec["peaks"], n_chains=cfg["n_chains"])
    ta = cfg["target_accept"]
    ta = ta[npl] if isinstance(ta, dict) else ta   # allow {1: 0.95, 2: 0.98}
    fit = fit_system(t, psi, pf, y, yerr, mstar, rstar, npl, seeds,
                     num_warmup=cfg["num_warmup"], num_samples=cfg["num_samples"],
                     target_accept=ta, rng_seed=int(spec.get("rng_seed", 0)))
    rec = analyze_recovery(fit, truthP)
    domk = spec["dom_k"] if spec["dom_k"] in rec else 1
    dom = rec[domk]
    peak1 = float(spec["peaks"][0]) if np.isfinite(spec["peaks"][0]) else np.nan
    # dominant companion's period samples, post-hoc label-resolved for 2-planet (inner=min, outer=max)
    if npl == 1:
        dom_samp = np.concatenate([ch[1] for ch in fit["chains"]])
    elif domk == 1:
        dom_samp = np.concatenate([np.minimum(ch[1], ch[2]) for ch in fit["chains"]])
    else:
        dom_samp = np.concatenate([np.maximum(ch[1], ch[2]) for ch in fit["chains"]])
    out = {"system_id": ds["system_id"], "pop": spec["pop"], "npl": npl,
           "pdg_class": spec["pdg_class"], "dom_k": int(domk), "nuts_outcome": nuts_outcome(dom),
           "dom_n_modes": dom["n_modes"], "dom_recovered": dom["recovered"],
           "wall_s": round(fit["wall_s"], 1), "divergences": fit["n_divergences"],
           "dom_truth_period": dom["truth_period"], "peak1": peak1,
           "dom_period_samples": dom_samp.astype("float32")}
    for i in range(1, npl + 1):
        out[f"p{i}_truth"] = round(rec[i]["truth_period"], 3)
        out[f"p{i}_nmodes"] = rec[i]["n_modes"]
        out[f"p{i}_recovered"] = rec[i]["recovered"]
    return out


def count_period_modes(per_chain, merge_dex=0.15, min_frac=0.03, bin_dex=0.04, floor_frac=0.05):
    """Count distinct posterior period modes across chains (histogram-based, bridging-robust).

    Pools all chains' period samples and histograms them in log10-period (``bin_dex`` dex bins).
    A bin is "occupied" if it holds at least ``floor_frac`` of the tallest bin -- this ignores the
    sparse bridging samples that single-linkage on raw samples would use to merge two real humps.
    Occupied bins separated by a gap smaller than ``merge_dex`` are joined into one mode; a mode is
    kept if it holds at least ``min_frac`` of all samples. Returns (n_modes, mode_periods) sorted by
    descending posterior mass.
    """
    logp = np.log10(np.concatenate([np.asarray(c) for c in per_chain]))
    n = len(logp)
    lo, hi = float(np.min(logp)), float(np.max(logp))
    if hi - lo < bin_dex:
        return 1, [float(10 ** np.median(logp))]
    nb = int(np.ceil((hi - lo) / bin_dex))
    counts, edges = np.histogram(logp, bins=nb, range=(lo, hi))
    centers = 0.5 * (edges[:-1] + edges[1:])

    occ = np.where(counts >= max(1.0, floor_frac * counts.max()))[0]
    if len(occ) == 0:
        return 1, [float(10 ** np.median(logp))]
    gap_bins = max(1, int(round(merge_dex / bin_dex)))
    # group occupied bins, joining runs separated by fewer than gap_bins empty bins
    groups, start, prev = [], occ[0], occ[0]
    for j in occ[1:]:
        if j - prev <= gap_bins:
            prev = j
        else:
            groups.append((start, prev)); start, prev = j, j
    groups.append((start, prev))

    modes = []
    for a, b in groups:
        c = counts[a:b + 1]
        mass = c.sum()
        if mass >= min_frac * n:
            center = float(np.sum(c * centers[a:b + 1]) / mass)
            modes.append((10 ** center, mass / n))
    if not modes:
        modes = [(10 ** np.median(logp), 1.0)]
    modes.sort(key=lambda m: -m[1])
    return len(modes), [m[0] for m in modes]
