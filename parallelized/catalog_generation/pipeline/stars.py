"""Stage 1: build the parent stellar sample.

Ported from `load_stars.ipynb`. Reads the G23H sample subset, interpolates mass
and radius off the Pecaut & Mamajek (2013) main-sequence table in absolute Gaia
G, and writes `outputs/data/stars.csv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from .config import CatalogConfig

# Column names of the Pecaut & Mamajek table (whitespace separated, # comments).
PECAUT_COLUMNS = [
    "SpT", "Teff", "logT", "BCv", "logL", "Mbol", "R_Rsun", "Mv",
    "B-V", "Bt-Vt", "G-V", "Bp-Rp", "G-Rp", "M_G", "b-y", "U-B",
    "V-Rc", "V-Ic", "V-Ks", "J-H", "H-Ks", "M_J", "M_Ks", "Ks-W1",
    "W1-W2", "W1-W3", "W1-W4", "g-r", "i-z", "z-Y", "Msun", "SpT2",
]


def add_mass_radius_from_pecaut(sources, pecaut_path, *, parallax_col,
                                gmag_col, source_id_col, verbose=True):
    """Add absolute G magnitude plus interpolated mass and radius.

    Interpolation is linear in absolute G with `fill_value=np.nan`, so stars
    outside the table's M_G range are left NaN rather than extrapolated.
    """
    pecaut = pd.read_csv(pecaut_path, sep=r"\s+", comment="#",
                         header=None, names=PECAUT_COLUMNS)
    for column in ["M_G", "Msun", "R_Rsun"]:
        pecaut[column] = pd.to_numeric(pecaut[column], errors="coerce")
    pecaut = pecaut.dropna(subset=["M_G", "Msun", "R_Rsun"]).sort_values("M_G")

    mass_from_absg = interp1d(pecaut["M_G"], pecaut["Msun"], kind="linear",
                              bounds_error=False, fill_value=np.nan)
    radius_from_absg = interp1d(pecaut["M_G"], pecaut["R_Rsun"], kind="linear",
                                bounds_error=False, fill_value=np.nan)

    df = sources.copy()
    df[gmag_col] = pd.to_numeric(df[gmag_col], errors="coerce")
    df[parallax_col] = pd.to_numeric(df[parallax_col], errors="coerce")

    valid = df[gmag_col].notna() & df[parallax_col].notna() & (df[parallax_col] > 0)

    df["abs_G"] = np.nan
    df["mass_interp"] = np.nan
    df["radius_interp"] = np.nan

    df.loc[valid, "abs_G"] = (
        df.loc[valid, gmag_col] + 5 * np.log10(df.loc[valid, parallax_col] / 1000) + 5
    )
    df.loc[valid, "mass_interp"] = mass_from_absg(df.loc[valid, "abs_G"])
    df.loc[valid, "radius_interp"] = radius_from_absg(df.loc[valid, "abs_G"])

    if verbose:
        print(f"  rows: {len(df)} | valid parallax + G: {int(valid.sum())} | "
              f"mass NaN: {int(df['mass_interp'].isna().sum())} | "
              f"radius NaN: {int(df['radius_interp'].isna().sum())}")
    return df


def build_star_catalog(config: CatalogConfig, *, overwrite=True) -> pd.DataFrame:
    """Stage 1. Returns the stellar catalog and writes it to `paths.stars_csv`."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    paths, selection = config.paths, config.stars

    if paths.stars_csv.exists() and not overwrite:
        print(f"  {paths.stars_csv} exists; reusing (pass --overwrite to rebuild)")
        return pd.read_csv(paths.stars_csv,
                           dtype={"gaia_source_id": str, "source_id_dr2": str},
                           low_memory=False)

    with pa.memory_map(str(paths.g23h_sample), "r") as source:
        sources = ipc.open_file(source).read_all().to_pandas()
    print(f"  loaded {len(sources):,} stars from {paths.g23h_sample}")

    interpolated = add_mass_radius_from_pecaut(
        sources, paths.pecaut_mamajek,
        parallax_col=selection.parallax_col,
        gmag_col=selection.gmag_col,
        source_id_col=selection.source_id_col,
    )

    stars = interpolated
    if selection.require_mass_radius:
        keep = stars["mass_interp"].notna() & stars["radius_interp"].notna()
        print(f"  dropped {int((~keep).sum())} stars outside the Pecaut & Mamajek range")
        stars = stars[keep].copy()

    # `require_sigma_al` is applied downstream (stage 2/3) rather than here, so
    # that stars.csv stays the full mass/radius-valid sample and every
    # population applies one identical noise-model cut.
    if selection.require_sigma_al:
        n_missing = int((~np.isfinite(stars["sig_AL"].to_numpy(dtype=float))).sum())
        if n_missing:
            print(f"  note: {n_missing} stars lack sig_AL; they are dropped when "
                  "populations are built, not from stars.csv")

    paths.stars_csv.parent.mkdir(parents=True, exist_ok=True)
    stars.to_csv(paths.stars_csv, index=False)
    print(f"  wrote {len(stars):,} stars -> {paths.stars_csv}")
    return stars
