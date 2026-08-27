"""Stage 1: build the parent stellar sample.

Ported from `load_stars.ipynb`. Reads the G23H sample subset, interpolates mass
and radius off the Pecaut & Mamajek (2013) main-sequence table in absolute Gaia
G, and writes `outputs/data/stars.csv`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# Column names of the Pecaut & Mamajek table (whitespace separated, # comments).
PECAUT_COLUMNS = [
    "SpT",
    "Teff",
    "logT",
    "BCv",
    "logL",
    "Mbol",
    "R_Rsun",
    "Mv",
    "B-V",
    "Bt-Vt",
    "G-V",
    "Bp-Rp",
    "G-Rp",
    "M_G",
    "b-y",
    "U-B",
    "V-Rc",
    "V-Ic",
    "V-Ks",
    "J-H",
    "H-Ks",
    "M_J",
    "M_Ks",
    "Ks-W1",
    "W1-W2",
    "W1-W3",
    "W1-W4",
    "g-r",
    "i-z",
    "z-Y",
    "Msun",
    "SpT2",
]


def add_mass_radius_from_pecaut(sources, verbose=True):
    """Add absolute G magnitude plus interpolated mass and radius.

    Linear in absolute G; stars outside the table's M_G range come out NaN
    rather than extrapolated.
    """
    pecaut = pd.read_csv(
        C.PECAUT_MAMAJEK, sep=r"\s+", comment="#", header=None, names=PECAUT_COLUMNS
    )
    for column in ["M_G", "Msun", "R_Rsun"]:
        pecaut[column] = pd.to_numeric(pecaut[column], errors="coerce")
    pecaut = pecaut.dropna(subset=["M_G", "Msun", "R_Rsun"]).sort_values("M_G")

    df = sources.copy()
    df[C.GMAG_COL] = pd.to_numeric(df[C.GMAG_COL], errors="coerce")
    df[C.PARALLAX_COL] = pd.to_numeric(df[C.PARALLAX_COL], errors="coerce")

    valid = (
        df[C.GMAG_COL].notna() & df[C.PARALLAX_COL].notna() & (df[C.PARALLAX_COL] > 0)
    )

    df["abs_G"] = np.nan
    df["mass_interp"] = np.nan
    df["radius_interp"] = np.nan

    df.loc[valid, "abs_G"] = (
        df.loc[valid, C.GMAG_COL]
        + 5 * np.log10(df.loc[valid, C.PARALLAX_COL] / 1000)
        + 5
    )
    abs_g = df.loc[valid, "abs_G"].to_numpy(dtype=float)
    for column, table in (("mass_interp", "Msun"), ("radius_interp", "R_Rsun")):
        df.loc[valid, column] = np.interp(
            abs_g, pecaut["M_G"], pecaut[table], left=np.nan, right=np.nan
        )

    if verbose:
        print(
            f"  rows: {len(df)} | valid parallax + G: {int(valid.sum())} | "
            f"mass NaN: {int(df['mass_interp'].isna().sum())} | "
            f"radius NaN: {int(df['radius_interp'].isna().sum())}"
        )
    return df


def build_star_catalog(*, overwrite=True) -> pd.DataFrame:
    """Stage 1. Returns the stellar catalog and writes it to `stars.csv`."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    stars_csv = C.stars_csv()
    if stars_csv.exists() and not overwrite:
        print(f"  {stars_csv} exists; reusing (pass --overwrite to rebuild)")
        return pd.read_csv(
            stars_csv,
            low_memory=False,
            dtype={"gaia_source_id": str, "source_id_dr2": str},
        )

    with pa.memory_map(str(C.g23h_sample()), "r") as source:
        sources = ipc.open_file(source).read_all().to_pandas()
    print(f"  loaded {len(sources):,} stars from {C.g23h_sample()}")

    stars = add_mass_radius_from_pecaut(sources)

    # G23H repeats a star once per DR2 cross-match, so collapse to one row per
    # source. Keep the row with the smallest source_id_dr2 in each group -- a
    # rule independent of the order rows arrive in -- then restore the
    # catalog's own ordering, which is what the source list is built from.
    duplicated = stars[C.SOURCE_ID_COL].duplicated().sum()
    if duplicated:
        keep = stars.groupby(C.SOURCE_ID_COL)["source_id_dr2"].idxmin()
        stars = stars.loc[np.sort(keep.to_numpy())].copy()
        print(
            f"  collapsed {duplicated:,} duplicate DR2 cross-match rows "
            f"-> {len(stars):,} unique sources"
        )

    # Stars outside the Pecaut & Mamajek M_G range have no mass or radius.
    keep = stars["mass_interp"].notna() & stars["radius_interp"].notna()
    print(f"  dropped {int((~keep).sum())} stars outside the Pecaut & Mamajek range")
    stars = stars[keep].copy()

    # A handful of high-RUWE binaries carry no per-CCD AL noise calibration.
    # They are dropped when the source list is built (`sources.build_indices`),
    # not here, so stars.csv stays the full mass/radius-valid sample.
    n_missing = int((~np.isfinite(stars["sig_AL"].to_numpy(dtype=float))).sum())
    if n_missing:
        print(
            f"  note: {n_missing} stars lack sig_AL; they are dropped from the "
            "source list, not from stars.csv"
        )

    stars_csv.parent.mkdir(parents=True, exist_ok=True)
    stars.to_csv(stars_csv, index=False)
    print(f"  wrote {len(stars):,} stars -> {stars_csv}")
    return stars
