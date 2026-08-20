"""Stage 3: simulate the epoch astrometry for ONE source, and write shards.

The simulator physics is unchanged from the serial pipeline (jaxoplanet reflex +
the DR3-calibrated UEVA noise model). What changes for ~4 million stars:

* the inputs arrive per source from `sources.SourceCatalog` / `ScanLawStore`
  rather than from whole-catalog DataFrames held in every worker;
* the noise seed is derived from the Gaia source id, so a source's realization
  is independent of what else is being simulated alongside it;
* output goes to one parquet pair per (population, shard) instead of one CSV
  per system -- 12 million small files is not a workable layout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CatalogConfig, PopulationSpec
from .planets import (companion_columns, draw_companions, star_noise_terms,
                      system_seed)


def configure_jax(settings) -> None:
    """x64 is not cosmetic: at float32 the random draws differ."""
    import jax

    jax.config.update("jax_enable_x64", settings.enable_float64)


# --------------------------------------------------------------------------
# The simulator (same model as the serial pipeline)
# --------------------------------------------------------------------------
def simulate_along_scan(t, psi, companions, *, mstar, rstar, parallax, mu_alpha,
                        mu_delta, parallax_factor, sigma_ueva, settings, seed):
    """Along-scan abscissa for one system: five-parameter model + reflex + noise.

    Follows the Gaia LPC convention (Lindegren & Bastian, GAIA-C3-TN-LU-LL-061):
    w = a sin(theta) + d cos(theta).
    """
    import math

    import jax.numpy as jnp
    import jax.random as jr
    from jaxoplanet.orbits.keplerian import Central, System

    t = jnp.asarray(t)
    psi = jnp.asarray(psi)
    t_days = jnp.asarray(t * settings.days_per_year)

    system = System(Central(mass=jnp.asarray(mstar), radius=jnp.asarray(rstar)))
    for companion in companions:
        mp_sun = float(companion["mass_pl"]) * settings.mjup_to_msun
        period_days = float(companion["period"]) * settings.days_per_year
        mean_motion = 2.0 * math.pi / period_days
        system = system.add_body(
            period=jnp.asarray(period_days),
            eccentricity=jnp.asarray(float(companion["ecc"])),
            inclination=jnp.asarray(float(np.deg2rad(companion["inc"]))),
            omega_peri=jnp.asarray(float(np.deg2rad(companion["omega"]))),
            asc_node=jnp.asarray(float(np.deg2rad(companion["Omega"]))),
            time_peri=jnp.asarray(-float(np.deg2rad(companion["M_anom"])) / mean_motion),
            mass=jnp.asarray(mp_sun),
        )

    sin_psi, cos_psi = jnp.sin(psi), jnp.cos(psi)
    x_sum = y_sum = 0.0
    for body in system.bodies:
        x_rsun, y_rsun, _ = body.central_position(t_days)
        x_sum, y_sum = x_sum + x_rsun, y_sum + y_rsun
    x_reflex = jnp.asarray(x_sum) * settings.rsun_to_au * parallax
    y_reflex = jnp.asarray(y_sum) * settings.rsun_to_au * parallax
    al_reflex = -(x_reflex * cos_psi + y_reflex * sin_psi)

    al_astro = (sin_psi * settings.alpha0_mas + cos_psi * settings.delta0_mas
                + sin_psi * mu_alpha * t + cos_psi * mu_delta * t
                + parallax * parallax_factor)
    al_true = al_astro + al_reflex
    noise = jr.normal(jr.key(seed), shape=jnp.shape(t)) * sigma_ueva
    return al_true + noise, al_true


def make_noise(*, n_good, n_fov, n_dof, sigma_calib, sigma_al, sigma_att, t,
               settings, seed):
    """Per-epoch UEVA and reported AL uncertainties [mas], sharing one jitter."""
    import jax.numpy as jnp
    import jax.random as jr

    if n_fov == 0 or (n_good - n_dof) == 0:
        raise ValueError("degenerate transit counts for this source")
    n_al_ave = n_good / n_fov
    mu_ueva_single = ((n_al_ave / (n_good - n_dof))
                      * ((n_fov - n_dof) * sigma_calib**2 + n_fov * sigma_al**2))
    sigma_ueva_single = jnp.sqrt(mu_ueva_single / n_al_ave)
    sigma_reported_base = jnp.sqrt((sigma_att**2 + sigma_al**2) / n_al_ave)
    jitter = 1.0 + settings.noise_jitter_frac * jr.normal(jr.key(seed), shape=jnp.shape(t))
    return sigma_ueva_single * jitter, sigma_reported_base * jitter


# --------------------------------------------------------------------------
# One source, end to end
# --------------------------------------------------------------------------
def simulate_source(config: CatalogConfig, spec: PopulationSpec, gaia_source_id, *,
                    catalog, scanlaw):
    """Draw companions and simulate epochs for a single Gaia source.

    Returns (epochs DataFrame, truth dict). Everything is a deterministic
    function of (master seeds, population, gaia_source_id), so this call is
    reproducible on its own -- that is what makes the pipeline shardable.
    """
    settings = config.astrometry
    star = catalog.get(gaia_source_id)
    transits = scanlaw.get(gaia_source_id)

    sigma_single, n_dof = star_noise_terms(star, settings)
    if not np.isfinite(sigma_single) or sigma_single <= 0:
        # Without a noise model every epoch would come out NaN; fail loudly so
        # the source is recorded as skipped instead of writing junk.
        raise ValueError(f"gaia_source_id {gaia_source_id} has no usable AL noise "
                         f"model (sigma_single={sigma_single})")
    companions = draw_companions(config, spec, star,
                                 n_transits=len(transits), sigma_single=sigma_single)

    t_jd = transits["obs_time_tcb_jd"].to_numpy(dtype=float)
    t_years = (t_jd - settings.gaia_epoch_tcb_jd) / settings.days_per_year
    psi = transits["scan_angle_rad"].to_numpy(dtype=float)
    parallax_factor = transits["parallax_factor_al"].to_numpy(dtype=float)

    seed = system_seed(config.seeds.astrometry, spec.name, gaia_source_id)
    noise_seed, observation_seed = [
        int(v) for v in np.random.SeedSequence(seed).generate_state(2, dtype=np.uint32)]

    sigma_ueva, sigma_reported = make_noise(
        n_good=float(star["astrometric_n_good_obs_al_dr3"]),
        n_fov=float(star["astrometric_matched_transits_dr3"]),
        n_dof=n_dof, sigma_calib=float(star["sig_cal"]),
        sigma_al=float(star["sig_AL"]), sigma_att=float(star["sig_att_radec"]),
        t=t_years, settings=settings, seed=noise_seed)

    al_obs, _ = simulate_along_scan(
        t_years, psi, companions,
        mstar=float(star["mass_interp"]), rstar=float(star["radius_interp"]),
        parallax=float(star["parallax"]), mu_alpha=float(star["pmra_dr3"]),
        mu_delta=float(star["pmdec_dr3"]), parallax_factor=parallax_factor,
        sigma_ueva=sigma_ueva, settings=settings, seed=observation_seed)

    system_id = f"{spec.name}_{gaia_source_id}"
    epochs = pd.DataFrame({
        "system_id": system_id,
        "gaia_source_id": str(gaia_source_id),
        "source_id_dr2": str(star["source_id_dr2"]),
        "obs_time_tcb": t_jd,
        "centroid_pos_al": np.asarray(al_obs),
        "centroid_pos_error_al": np.asarray(sigma_reported),
        "parallax_factor_al": parallax_factor,
        "scan_pos_angle": psi,
        "field_of_view": transits["fov"].to_numpy() if "fov" in transits else "",
        "system_seed": seed,
    })

    truth = {
        "system_id": system_id,
        "population": spec.name,
        "gaia_source_id": str(gaia_source_id),
        "source_id_dr2": str(star["source_id_dr2"]),
        "n_transits_dr4": len(transits),
        "master_seed_planets": int(config.seeds.planets),
        "master_seed_astrometry": int(config.seeds.astrometry),
        "system_seed": seed,
        "noise_seed": noise_seed,
        "observation_seed": observation_seed,
        "parallax_mas": float(star["parallax"]),
        "pmra_mas_yr": float(star["pmra_dr3"]),
        "pmdec_mas_yr": float(star["pmdec_dr3"]),
        "mass_st_msun": float(star["mass_interp"]),
        "radius_st_rsun": float(star["radius_interp"]),
        "sigma_single_mas": float(sigma_single),
        **companion_columns(companions, config.priors),
    }
    return epochs, truth


# --------------------------------------------------------------------------
# Shard writer
# --------------------------------------------------------------------------
class ShardWriter:
    """Buffers systems and writes one parquet pair per (population, shard).

    Flushing on a row count keeps a worker's memory bounded regardless of how
    many sources land in its shard.
    """

    def __init__(self, config: CatalogConfig, spec: PopulationSpec, shard, n_shards):
        self.config, self.spec = config, spec
        self.shard, self.n_shards = shard, n_shards
        self.epochs_path = config.paths.shard_epochs(spec.name, shard, n_shards)
        self.truths_path = config.paths.shard_truths(spec.name, shard, n_shards)
        self.epochs_path.parent.mkdir(parents=True, exist_ok=True)
        self._epochs, self._truths = [], []
        self._epoch_writer = None
        self.n_systems = self.n_epochs = 0

    def add(self, epochs, truth):
        self._epochs.append(epochs)
        self._truths.append(truth)
        self.n_systems += 1
        self.n_epochs += len(epochs)
        if len(self._truths) >= self.config.sharding.flush_every:
            self.flush()

    def flush(self):
        """Append the buffered epochs to the shard file."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not self._epochs:
            return
        table = pa.Table.from_pandas(pd.concat(self._epochs, ignore_index=True),
                                     preserve_index=False)
        if self._epoch_writer is None:
            self._epoch_writer = pq.ParquetWriter(
                self.epochs_path, table.schema,
                compression=self.config.sharding.compression)
        self._epoch_writer.write_table(table)
        self._epochs = []

    def close(self):
        self.flush()
        if self._epoch_writer is not None:
            self._epoch_writer.close()
        pd.DataFrame(self._truths).to_parquet(
            self.truths_path, index=False,
            compression=self.config.sharding.compression)
        return {"population": self.spec.name, "shard": self.shard,
                "n_systems": self.n_systems, "n_epochs": self.n_epochs,
                "epochs": str(self.epochs_path), "truths": str(self.truths_path)}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
