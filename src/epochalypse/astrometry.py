"""Stage 3: simulate the epoch astrometry for ONE source, and write shards.

The simulator physics is unchanged from the serial pipeline (jaxoplanet reflex +
the DR3-calibrated UEVA noise model). What changes for ~4 million stars:

* the inputs arrive per source from `sources.SourceCatalog` / `ScanLawStore`
  rather than from whole-catalog DataFrames held in every rank;
* the noise seed is derived from the Gaia source id, so a source's realization
  is independent of what else is being simulated alongside it;
* output goes to one parquet pair per (population, rank) instead of one CSV per
  system -- 12 million small files is not a workable layout.
"""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd

from . import config as C
from .shardio import BufferedParquetWriter
from .planets import companion_columns, draw_companions, star_noise_terms, system_seed

# x64 is not cosmetic: at float32 the random draws, and so the injected noise
# realization, differ from the released catalog. Must run before any jax array.
jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------
# The simulator (same model as the serial pipeline)
# --------------------------------------------------------------------------
def simulate_along_scan(
    t,
    psi,
    companions,
    *,
    mstar,
    rstar,
    parallax,
    mu_alpha,
    mu_delta,
    parallax_factor,
    sigma_ueva,
    seed,
):
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
    t_days = jnp.asarray(t * C.DAYS_PER_YEAR)

    system = System(Central(mass=jnp.asarray(mstar), radius=jnp.asarray(rstar)))
    for companion in companions:
        mp_sun = float(companion["mass_pl"]) * C.MJUP_IN_MSUN
        period_days = float(companion["period"]) * C.DAYS_PER_YEAR
        mean_motion = 2.0 * math.pi / period_days
        system = system.add_body(
            period=jnp.asarray(period_days),
            eccentricity=jnp.asarray(float(companion["ecc"])),
            inclination=jnp.asarray(float(np.deg2rad(companion["inc"]))),
            omega_peri=jnp.asarray(float(np.deg2rad(companion["omega"]))),
            asc_node=jnp.asarray(float(np.deg2rad(companion["Omega"]))),
            time_peri=jnp.asarray(
                -float(np.deg2rad(companion["M_anom"])) / mean_motion
            ),
            mass=jnp.asarray(mp_sun),
        )

    sin_psi, cos_psi = jnp.sin(psi), jnp.cos(psi)
    x_sum = y_sum = 0.0
    for body in system.bodies:
        x_rsun, y_rsun, _ = body.central_position(t_days)
        x_sum, y_sum = x_sum + x_rsun, y_sum + y_rsun
    x_reflex = jnp.asarray(x_sum) * C.RSUN_IN_AU * parallax
    y_reflex = jnp.asarray(y_sum) * C.RSUN_IN_AU * parallax
    al_reflex = -(x_reflex * cos_psi + y_reflex * sin_psi)

    al_astro = (
        sin_psi * mu_alpha * t + cos_psi * mu_delta * t + parallax * parallax_factor
    )
    al_true = al_astro + al_reflex
    noise = jr.normal(jr.key(seed), shape=jnp.shape(t)) * sigma_ueva
    return al_true + noise, al_true


def make_noise(*, sigma_single, n_al_ave, sigma_al, sigma_att, t, seed):
    """Per-epoch UEVA and reported AL uncertainties [mas], sharing one jitter.

    `sigma_single` is the UEVA per-transit sigma from `planets.single_datum_sigma`
    -- the same quantity the recorded S/N metric divides by, computed once.
    """
    import jax.numpy as jnp
    import jax.random as jr

    sigma_reported_base = jnp.sqrt((sigma_att**2 + sigma_al**2) / n_al_ave)
    jitter = 1.0 + C.NOISE_JITTER_FRAC * jr.normal(jr.key(seed), shape=jnp.shape(t))
    return sigma_single * jitter, sigma_reported_base * jitter


# --------------------------------------------------------------------------
# One source, end to end
# --------------------------------------------------------------------------
def simulate_source(population, gaia_source_id, *, catalog, scanlaw):
    """Draw companions and simulate epochs for a single Gaia source.

    Returns (epochs DataFrame, truth dict). Everything is a deterministic
    function of (master seeds, population, gaia_source_id), so this call is
    reproducible on its own -- that is what makes the pipeline shardable.
    """
    star = catalog.get(gaia_source_id)
    transits = scanlaw.get(gaia_source_id)

    sigma_single, _ = star_noise_terms(star)
    if not np.isfinite(sigma_single) or sigma_single <= 0:
        # Without a noise model every epoch would come out NaN; fail loudly so
        # the source is recorded as skipped instead of writing junk.
        raise ValueError(
            f"gaia_source_id {gaia_source_id} has no usable AL noise "
            f"model (sigma_single={sigma_single})"
        )
    companions = draw_companions(
        population, star, n_transits=len(transits), sigma_single=sigma_single
    )

    t_jd = transits["obs_time_tcb_jd"].to_numpy(dtype=float)
    t_years = (t_jd - C.GAIA_EPOCH_TCB_JD) / C.DAYS_PER_YEAR
    psi = transits["scan_angle_rad"].to_numpy(dtype=float)
    parallax_factor = transits["parallax_factor_al"].to_numpy(dtype=float)

    seed = system_seed(C.SEED_ASTROMETRY, population, gaia_source_id)
    noise_seed, observation_seed = [
        int(v) for v in np.random.SeedSequence(seed).generate_state(2, dtype=np.uint32)
    ]

    sigma_ueva, sigma_reported = make_noise(
        sigma_single=sigma_single,
        n_al_ave=(
            float(star["astrometric_n_good_obs_al_dr3"])
            / float(star["astrometric_matched_transits_dr3"])
        ),
        sigma_al=float(star["sig_AL"]),
        sigma_att=float(star["sig_att_radec"]),
        t=t_years,
        seed=noise_seed,
    )

    al_obs, _ = simulate_along_scan(
        t_years,
        psi,
        companions,
        mstar=float(star["mass_interp"]),
        rstar=float(star["radius_interp"]),
        parallax=float(star["parallax"]),
        mu_alpha=float(star["pmra_dr3"]),
        mu_delta=float(star["pmdec_dr3"]),
        parallax_factor=parallax_factor,
        sigma_ueva=sigma_ueva,
        seed=observation_seed,
    )

    system_id = f"{population}_{gaia_source_id}"
    epochs = pd.DataFrame(
        {
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
        }
    )

    truth = {
        "system_id": system_id,
        "population": population,
        "gaia_source_id": str(gaia_source_id),
        "source_id_dr2": str(star["source_id_dr2"]),
        "n_transits_dr4": len(transits),
        "master_seed_planets": C.SEED_PLANETS,
        "master_seed_astrometry": C.SEED_ASTROMETRY,
        "system_seed": seed,
        "noise_seed": noise_seed,
        "observation_seed": observation_seed,
        "parallax_mas": float(star["parallax"]),
        "pmra_mas_yr": float(star["pmra_dr3"]),
        "pmdec_mas_yr": float(star["pmdec_dr3"]),
        "mass_st_msun": float(star["mass_interp"]),
        "radius_st_rsun": float(star["radius_interp"]),
        "sigma_single_mas": float(sigma_single),
        **companion_columns(companions),
    }
    return epochs, truth


# --------------------------------------------------------------------------
# Shard writer
# --------------------------------------------------------------------------
class _EpochWriter(BufferedParquetWriter):
    """The epochs half: one buffered DataFrame per system, concatenated per flush."""

    def _table(self, rows):
        import pyarrow as pa

        return pa.Table.from_pandas(
            pd.concat(rows, ignore_index=True), preserve_index=False
        )


class ShardWriter:
    """Writes one parquet pair per (population, rank).

    The epochs stream through `shardio.BufferedParquetWriter` -- flushing a row
    group every `FLUSH_EVERY` systems keeps a rank's memory bounded regardless
    of how many sources land in its slice, and both files land via `.tmp` +
    rename so a rank killed mid-write leaves no file at all rather than a
    truncated one that looks complete. That is what makes `--skip-existing`
    trustworthy.

    The truths are one row per system, so they are written whole at close
    rather than streamed.
    """

    def __init__(self, population, rank, n_ranks):
        self.population = population
        self.rank = rank
        self.epochs_path = C.shard_epochs(population, rank, n_ranks)
        self.truths_path = C.shard_truths(population, rank, n_ranks)
        self._epochs = _EpochWriter(
            self.epochs_path, C.FLUSH_EVERY, C.PARQUET_COMPRESSION
        )
        self._truths_tmp = self.truths_path.with_suffix(".parquet.tmp")
        self._truths = []
        self.n_systems = self.n_epochs = 0

    def add(self, epochs, truth):
        self._epochs.add(epochs)
        self._truths.append(truth)
        self.n_systems += 1
        self.n_epochs += len(epochs)

    def close(self):
        self._epochs.close()
        pd.DataFrame(self._truths).to_parquet(
            self._truths_tmp, index=False, compression=C.PARQUET_COMPRESSION
        )
        self._truths_tmp.replace(self.truths_path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
