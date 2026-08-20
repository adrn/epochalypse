"""Stage 3: simulate per-epoch Gaia DR4-like along-scan astrometry.

Ported from `sim_astrometry.ipynb`. The orbit model is `jaxoplanet`; the noise
model is the DR3-calibrated UEVA prescription. Everything configurable arrives
via `config.astrometry`.

Layout produced, per population:

    outputs/data/simulated_astrometry/<population>/<system_id>.csv   epoch tables
    outputs/data/injected_solutions_<population>.csv                 truth table
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .config import CatalogConfig, PopulationSpec


def configure_jax(settings) -> None:
    """Apply JAX-wide numerical settings before any array is created.

    x64 is not cosmetic here: at float32 the random draws differ, so the
    injected noise realization would not match the released catalog.
    """
    import jax

    jax.config.update("jax_enable_x64", settings.enable_float64)


def stable_system_seed(master_seed: int, system_id: str) -> int:
    """Stable uint32 seed per system, independent of hash randomization."""
    payload = f"{int(master_seed)}:{system_id}".encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "little")


# --------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------
def simulate_gaia_astrometry_multi(t, psi, planets, *, mstar, rstar, parallax,
                                   mu_alpha, mu_delta, parallax_factor,
                                   sigma_ueva, settings, seed,
                                   alpha0=0.0, delta0=0.0):
    """Along-scan astrometry for one system.

    Parameters are in Gaia-native units: `t` in years from the reference epoch,
    `psi` the scan position angle [rad], masses in Msun, radii in Rsun,
    parallax and sigmas in mas, proper motions in mas/yr.

    The five-parameter contribution follows the Gaia LPC convention (Lindegren
    & Bastian, GAIA-C3-TN-LU-LL-061-08, Eqs. 4 & 6):
        w = a sin(theta) + d cos(theta),  a ~= Delta alpha*, d ~= Delta delta.
    """
    import math

    import jax.numpy as jnp
    import jax.random as jr
    from jaxoplanet.orbits.keplerian import Central, System

    t = jnp.asarray(t)
    psi = jnp.asarray(psi)
    t_days = jnp.asarray(t * settings.days_per_year)

    system = System(Central(mass=jnp.asarray(mstar), radius=jnp.asarray(rstar)))
    for planet in planets:
        a_au = float(planet["a"])
        if "Mp_sun" in planet:
            mp_sun = float(planet["Mp_sun"])
        else:
            mp_sun = float(planet.get("Mp", 0.0)) * settings.mjup_to_msun

        if planet.get("P") is not None:
            period_years = float(planet["P"])
        else:
            period_years = math.sqrt(a_au**3 / (float(mstar) + mp_sun))
        period_days = period_years * settings.days_per_year

        mean_motion = 2.0 * math.pi / period_days
        time_peri_days = -float(planet["M0"]) / mean_motion

        system = system.add_body(
            period=jnp.asarray(period_days),
            eccentricity=jnp.asarray(float(planet["e"])),
            inclination=jnp.asarray(float(planet["i"])),
            omega_peri=jnp.asarray(float(planet["omega"])),
            asc_node=jnp.asarray(float(planet["Omega"])),
            time_peri=jnp.asarray(time_peri_days),
            mass=jnp.asarray(mp_sun),
        )

    sin_psi, cos_psi = jnp.sin(psi), jnp.cos(psi)

    # stellar reflex from every body, in Rsun -> AU -> mas
    x_sum = y_sum = 0.0
    for body in system.bodies:
        x_rsun, y_rsun, _ = body.central_position(t_days)
        x_sum = x_sum + x_rsun
        y_sum = y_sum + y_rsun
    x_reflex = jnp.asarray(x_sum) * settings.rsun_to_au * parallax
    y_reflex = jnp.asarray(y_sum) * settings.rsun_to_au * parallax
    al_reflex = -(x_reflex * cos_psi + y_reflex * sin_psi)

    al_astro = (
        sin_psi * alpha0
        + cos_psi * delta0
        + sin_psi * mu_alpha * t
        + cos_psi * mu_delta * t
        + parallax * parallax_factor
    )
    al_true = al_astro + al_reflex

    noise = jr.normal(jr.key(seed), shape=jnp.shape(t)) * sigma_ueva
    return dict(
        t=t, psi=psi,
        al_obs=al_true + noise,
        al_true=al_true,
        al_astro=al_astro,
        al_reflex=al_reflex,
        sigma_true=noise,
        truth={"planets": planets},
    )


def make_noise(*, n_good, n_fov, n_dof, sigma_calib, sigma_al, sigma_att,
               t, settings, seed):
    """Per-epoch UEVA and reported AL uncertainties [mas].

    Both carry one shared multiplicative jitter so the injected scatter and the
    reported error bar stay consistent epoch by epoch.
    """
    import jax.numpy as jnp
    import jax.random as jr

    if n_fov == 0:
        raise ValueError("N_FoV cannot be zero")
    if (n_good - n_dof) == 0:
        raise ValueError("N - gaia_n_dof cannot be zero")

    n_al_ave = n_good / n_fov
    mu_ueva_single = (
        (n_al_ave / (n_good - n_dof))
        * ((n_fov - n_dof) * sigma_calib**2 + n_fov * sigma_al**2)
    )
    sigma_ueva_single = jnp.sqrt(mu_ueva_single / n_al_ave)
    sigma_reported_base = jnp.sqrt((sigma_att**2 + sigma_al**2) / n_al_ave)

    jitter = 1.0 + settings.noise_jitter_frac * jr.normal(jr.key(seed), shape=jnp.shape(t))
    return sigma_ueva_single * jitter, sigma_reported_base * jitter


def prepare_and_simulate(gaia_source_id, source_id_dr2, scanlaw_df, sources_df,
                         planets, *, settings, seed):
    """Assemble one system's inputs from the catalogs and simulate it.

    Returns (epoch table, simulator result, source row, scan-law rows).
    """
    matches = sources_df.loc[
        (sources_df["gaia_source_id"] == gaia_source_id)
        & (sources_df["source_id_dr2"] == source_id_dr2)
    ]
    if len(matches) == 0:
        raise ValueError(f"no source row for {gaia_source_id}/{source_id_dr2}")
    if len(matches) > 1:
        raise ValueError(f"multiple source rows for {gaia_source_id}/{source_id_dr2}")
    source_row = matches.iloc[0]

    scan_rows = scanlaw_df.loc[scanlaw_df["gaia_source_id"] == gaia_source_id].copy()
    if len(scan_rows) == 0:
        raise ValueError(f"no scan-law rows for {gaia_source_id}")
    scan_rows = scan_rows.sort_values("obs_time_tcb_jd").reset_index(drop=True)

    # epochs in years, centred on the DR4 reference epoch
    t_jd = scan_rows["obs_time_tcb_jd"].to_numpy(dtype=float)
    t_years = (t_jd - settings.gaia_epoch_tcb_jd) / settings.days_per_year
    psi = scan_rows["scan_angle_rad"].to_numpy(dtype=float)
    parallax_factor = scan_rows["parallax_factor_al"].to_numpy(dtype=float)

    if pd.isna(source_row["mass_interp"]) or pd.isna(source_row["radius_interp"]):
        raise ValueError(f"missing interpolated mass/radius for {gaia_source_id}")

    n_dof = (settings.n_dof_five_param
             if source_row["astrometric_params_solved_dr3"] == settings.params_solved_five_param
             else settings.n_dof_other)

    # two independent, reproducible streams from the one system seed
    noise_seed, observation_seed = [
        int(value)
        for value in np.random.SeedSequence(int(seed)).generate_state(2, dtype=np.uint32)
    ]

    sigma_ueva, sigma_reported = make_noise(
        n_good=float(source_row["astrometric_n_good_obs_al_dr3"]),
        n_fov=float(source_row["astrometric_matched_transits_dr3"]),
        n_dof=n_dof,
        sigma_calib=float(source_row["sig_cal"]),
        sigma_al=float(source_row["sig_AL"]),
        sigma_att=float(source_row["sig_att_radec"]),
        t=t_years, settings=settings, seed=noise_seed,
    )

    result = simulate_gaia_astrometry_multi(
        t_years, psi, planets,
        mstar=float(source_row["mass_interp"]),
        rstar=float(source_row["radius_interp"]),
        parallax=float(source_row["parallax"]),
        mu_alpha=float(source_row["pmra_dr3"]),
        mu_delta=float(source_row["pmdec_dr3"]),
        parallax_factor=parallax_factor,
        sigma_ueva=sigma_ueva,
        settings=settings,
        seed=observation_seed,
        alpha0=settings.alpha0_mas,
        delta0=settings.delta0_mas,
    )
    result.update(sigma_UEVA=sigma_ueva, sigma_reported=sigma_reported,
                  noise_seed=noise_seed, observation_seed=observation_seed,
                  t_years=t_years, psi=psi, parallax_factor_al=parallax_factor)

    epochs = pd.DataFrame({
        "obs_time_tcb": scan_rows["obs_time_tcb_jd"].to_numpy(),
        "centroid_pos_al": result["al_obs"],
        "centroid_pos_error_al": sigma_reported,
        "parallax_factor_al": parallax_factor,
        "scan_pos_angle": psi,
        "field_of_view": scan_rows["fov"].to_numpy(),
        "gaia_source_id": gaia_source_id,
        "source_id_dr2": source_id_dr2,
    })
    return epochs, result, source_row, scan_rows


# --------------------------------------------------------------------------
# Turning population CSVs into system records
# --------------------------------------------------------------------------
# Per-system columns carried from the population CSV through to the truth table.
CARRIED_SYSTEM_COLUMNS = (
    "sigma_single_mas", "n_transits_dr4", "generation_seed",
    "P_ratio", "near_2_1", "near_3_2", "near_resonance", "coplanar",
)


def load_systems(config: CatalogConfig, spec: PopulationSpec) -> list[dict]:
    """One record per system: identifiers plus companions in simulator units
    (angles in radians, masses in Msun)."""
    paths, settings = config.paths, config.astrometry

    if spec.n_companions == 0:
        stars = pd.read_csv(paths.stars_csv,
                            dtype={"gaia_source_id": str, "source_id_dr2": str},
                            usecols=["gaia_source_id", "source_id_dr2", "sig_AL"])
        if config.stars.require_sigma_al:
            n_before = len(stars)
            stars = stars[np.isfinite(stars["sig_AL"].to_numpy(dtype=float))]
            stars = stars.reset_index(drop=True)
            if len(stars) < n_before:
                print(f"  dropped {n_before - len(stars)} stars with missing sig_AL")
        key = ["gaia_source_id", "source_id_dr2"]
        if stars[key].isna().any().any():
            raise ValueError("the stellar catalog contains missing Gaia identifiers")
        if stars.duplicated(key).any():
            raise ValueError("the stellar catalog contains duplicate "
                             "(gaia_source_id, source_id_dr2) keys")
        return [{"gaia_source_id": row.gaia_source_id,
                 "source_id_dr2": row.source_id_dr2,
                 "planets": []}
                for row in stars.itertuples(index=False)]

    frame = pd.read_csv(paths.population_csv(spec.name),
                        dtype={"gaia_source_id": str, "source_id_dr2": str})
    systems = []
    for _, row in frame.iterrows():
        planets = []
        if spec.n_companions == 1:
            planets.append({
                "a": row["sma"], "e": row["ecc"],
                "i": np.deg2rad(row["inc"]),
                "Omega": np.deg2rad(row["Omega"]),
                "omega": np.deg2rad(row["omega"]),
                "M0": np.deg2rad(row["M_anom"]),
                "Mp_sun": row["mass_pl"] * settings.mjup_to_msun,
                "period": row["period"], "alpha_mas": row["alpha_mas"],
                "snr_single": row["snr_single"], "snr_eff": row["snr_eff"],
                "snr_total": row["snr_total"],
            })
        else:
            for idx in range(1, spec.n_companions + 1):
                planets.append({
                    "a": row[f"sma_{idx}"], "e": row[f"ecc_{idx}"],
                    "i": np.deg2rad(row[f"inc_{idx}"]),
                    "Omega": np.deg2rad(row[f"Omega_{idx}"]),
                    "omega": np.deg2rad(row[f"omega_{idx}"]),
                    "M0": np.deg2rad(row[f"M_anom_{idx}"]),
                    "Mp_sun": row[f"mass_pl_{idx}"] * settings.mjup_to_msun,
                    "period": row[f"period_{idx}"],
                    "alpha_mas": row[f"alpha_{idx}_mas"],
                    "snr_single": row[f"snr_single_{idx}"],
                    "snr_eff": row[f"snr_eff_{idx}"],
                    "snr_total": row[f"snr_total_{idx}"],
                })

        record = {"gaia_source_id": row["gaia_source_id"],
                  "source_id_dr2": row["source_id_dr2"],
                  "planets": planets}
        for column in CARRIED_SYSTEM_COLUMNS:
            if column in frame.columns:
                record[column] = row[column]
        systems.append(record)
    return systems


def load_reference_tables(config: CatalogConfig):
    """Scan law + stellar catalog, with identifiers normalized to strings
    (never via floating point) and the shared sig_AL cut applied."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    paths = config.paths
    with pa.memory_map(str(paths.scanlaw_dr4), "r") as handle:
        scanlaw = ipc.open_file(handle).read_all().to_pandas()
    scanlaw["gaia_source_id"] = scanlaw["gaia_source_id"].astype("Int64").astype(str)

    sources = pd.read_csv(paths.stars_csv,
                          dtype={"gaia_source_id": str, "source_id_dr2": str},
                          low_memory=False)
    if config.stars.require_sigma_al:
        n_before = len(sources)
        sources = sources[np.isfinite(sources["sig_AL"].to_numpy(dtype=float))]
        sources = sources.reset_index(drop=True)
        if len(sources) < n_before:
            print(f"  dropped {n_before - len(sources)} sources with missing sig_AL")
    for column in ["gaia_source_id", "source_id_dr2"]:
        sources[column] = sources[column].astype("Int64").astype(str)

    # A Gaia source can have several DR2 cross-matches: the pair is the key.
    key = ["gaia_source_id", "source_id_dr2"]
    duplicates = sources.duplicated(key, keep=False)
    if duplicates.any():
        raise ValueError(f"stars.csv has {int(duplicates.sum())} rows with duplicate "
                         "(gaia_source_id, source_id_dr2) keys")
    return scanlaw, sources


# --------------------------------------------------------------------------
# Stage 3
# --------------------------------------------------------------------------
def simulate_population(config: CatalogConfig, spec: PopulationSpec, *,
                        scanlaw_df, sources_df, limit=None) -> pd.DataFrame:
    """Simulate every system in one population.

    Writes one epoch CSV per system plus the population's truth table, and
    returns the truth table.
    """
    from tqdm.auto import tqdm

    paths, settings = config.paths, config.astrometry
    configure_jax(settings)
    systems = load_systems(config, spec)
    if limit is not None:
        systems = systems[:limit]
        print(f"  LIMIT: simulating only the first {len(systems)} systems")

    output_dir = paths.epochs_dir / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {len(systems):,} systems -> {output_dir}")

    injected_rows = []
    for index, system in enumerate(tqdm(systems, desc=spec.name)):
        system_id = f"{spec.name}_SYS_{index:07d}"
        system_seed = stable_system_seed(config.seeds.astrometry, system_id)
        output_path = output_dir / f"{system_id}.csv"

        try:
            epochs, result, source_row, _ = prepare_and_simulate(
                str(system["gaia_source_id"]), str(system["source_id_dr2"]),
                scanlaw_df, sources_df, system["planets"],
                settings=settings, seed=system_seed,
            )
        except Exception as error:
            # Every system is expected to simulate; on the rare failure log and
            # skip rather than emit a truth row with no epochs behind it.
            tqdm.write(f"FAILED {system_id}: {error}")
            continue

        epochs["system_id"] = system_id
        epochs["system_seed"] = system_seed
        epochs.to_csv(output_path, index=False)

        row = {
            "system_id": system_id,
            "filename": output_path.name,
            "filepath": str(output_path),
            "population": spec.name,
            "gaia_source_id": str(system["gaia_source_id"]),
            "source_id_dr2": str(system["source_id_dr2"]),
            "master_seed": int(config.seeds.astrometry),
            "system_seed": system_seed,
            "noise_seed": result["noise_seed"],
            "observation_seed": result["observation_seed"],
            "n_planets": len(system["planets"]),
            "parallax_mas": source_row["parallax"],
            "pmra_mas_yr": source_row["pmra_dr3"],
            "pmdec_mas_yr": source_row["pmdec_dr3"],
            "mass_st_msun": source_row["mass_interp"],
            "radius_st_rsun": source_row["radius_interp"],
        }
        for pidx, planet in enumerate(system["planets"], start=1):
            row[f"a_{pidx}_au"] = planet["a"]
            row[f"e_{pidx}"] = planet["e"]
            row[f"i_{pidx}_rad"] = planet["i"]
            row[f"Omega_{pidx}_rad"] = planet["Omega"]
            row[f"omega_{pidx}_rad"] = planet["omega"]
            row[f"M0_{pidx}_rad"] = planet["M0"]
            row[f"Mp_{pidx}_msun"] = planet["Mp_sun"]
            row[f"period_{pidx}"] = planet.get("period")
            row[f"alpha_{pidx}_mas"] = planet.get("alpha_mas")
            row[f"snr_single_{pidx}"] = planet.get("snr_single")
            row[f"snr_eff_{pidx}"] = planet.get("snr_eff")
            row[f"snr_total_{pidx}"] = planet.get("snr_total")
        for column in CARRIED_SYSTEM_COLUMNS:
            if column in system:
                row[column] = system[column]
        injected_rows.append(row)

    truths = pd.DataFrame(injected_rows)
    truths_path = population_truths_path(config, spec.name)
    truths.to_csv(truths_path, index=False)
    print(f"  wrote {len(truths):,} truth rows -> {truths_path}")
    return truths


def population_truths_path(config: CatalogConfig, population: str):
    return config.paths.data_dir / f"injected_solutions_{population}.csv"


def combine_truth_tables(config: CatalogConfig, populations) -> pd.DataFrame:
    """Concatenate the per-population truth tables into
    `injected_solutions_all.csv`, in configured population order."""
    frames = []
    for spec in populations:
        path = population_truths_path(config, spec.name)
        if not path.exists():
            print(f"  skipping {spec.name}: {path} not found")
            continue
        frames.append(pd.read_csv(path, dtype={"gaia_source_id": str,
                                               "source_id_dr2": str},
                                  low_memory=False))
    if not frames:
        raise FileNotFoundError("no per-population truth tables to combine")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined.to_csv(config.paths.injected_solutions_csv, index=False)
    print(f"  wrote {len(combined):,} rows -> {config.paths.injected_solutions_csv}")
    return combined


# --------------------------------------------------------------------------
# Stage 4: HDF5 export
# --------------------------------------------------------------------------
def _dataframe_to_structured_array(frame):
    """DataFrame -> numpy structured array for h5py. Object/string columns
    become variable-length UTF-8 (NaN -> empty string)."""
    import h5py

    string_dtype = h5py.string_dtype(encoding="utf-8")
    dtypes, columns = [], {}
    for name in frame.columns:
        column = frame[name]
        if column.dtype == object or pd.api.types.is_string_dtype(column):
            values = column.where(column.notna(), "").astype(str).to_numpy()
            dtypes.append((name, string_dtype))
        else:
            values = column.to_numpy()
            dtypes.append((name, values.dtype))
        columns[name] = values

    array = np.empty(len(frame), dtype=dtypes)
    for name in frame.columns:
        array[name] = columns[name]
    return array


def export_population_hdf5(config: CatalogConfig, spec: PopulationSpec, *,
                           overwrite=False) -> dict:
    """Write `simulated_astrometry_<population>_systems.h5`.

        /truths                 one row per system
        /systems/<idx>/epochs   epochs for the system at /truths[<idx>]

    The integer <idx> is the row position in /truths, so truths and epochs line
    up by position.
    """
    import h5py

    from pathlib import Path

    truths_path = population_truths_path(config, spec.name)
    if not truths_path.exists():
        raise FileNotFoundError(f"{truths_path} not found; run stage 'astrometry' first")
    truths = pd.read_csv(truths_path, dtype={"gaia_source_id": str,
                                             "source_id_dr2": str}, low_memory=False)

    output_path = config.paths.systems_h5(spec.name)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace")

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    compression = config.astrometry.hdf5_compression
    n_with_epochs = 0
    try:
        with h5py.File(temporary_path, "w") as handle:
            handle.create_dataset("truths",
                                  data=_dataframe_to_structured_array(truths),
                                  compression=compression)
            handle["truths"].attrs["population"] = spec.name
            handle["truths"].attrs["n_systems"] = len(truths)

            systems = handle.create_group("systems")
            for index, row in truths.iterrows():
                group = systems.create_group(str(index))
                group.attrs["system_id"] = str(row["system_id"])
                group.attrs["gaia_source_id"] = str(row["gaia_source_id"])
                group.attrs["source_id_dr2"] = str(row["source_id_dr2"])

                epochs_path = Path(row["filepath"])
                if epochs_path.exists():
                    epochs = pd.read_csv(epochs_path, dtype={"gaia_source_id": str,
                                                             "source_id_dr2": str})
                    group.create_dataset("epochs",
                                         data=_dataframe_to_structured_array(epochs),
                                         compression=compression)
                    n_with_epochs += 1

                if (index + 1) % 500 == 0 or (index + 1) == len(truths):
                    print(f"  {spec.name}: {index + 1}/{len(truths)} systems, "
                          f"{n_with_epochs} with epochs", end="\r")
        temporary_path.replace(output_path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise

    print()
    size_mib = output_path.stat().st_size / 1024**2
    print(f"  wrote {output_path} ({size_mib:.1f} MiB)")
    return {"population": spec.name, "path": str(output_path),
            "n_systems": len(truths), "n_with_epochs": n_with_epochs}
