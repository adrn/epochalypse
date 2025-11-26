# ruff: noqa: T201
"""
Convert the Gaia commanded scan law to unique scans per sky position.

This script converts the Gaia commanded scan law (sampled at 10-second intervals) into
unique "scans" binned by sky position using a HEALPix pixelization of the sky. Unlike
the GOST (https://gaia.esac.esa.int/gost/index.jsp), this does not have a focal plane
model, so we don't resolve the CCD-level observations and we won't be extremely
precise(we only track which pixels the center of the FOV hits). Each transit is
considered as a single scan event.

For each sky position / HEALPix pixel, we store one row per transit:

- bjd_time               float32  (TCB BJD relative to Gaia reference time)
- scan_angle_deg         float32  (position angle, degrees)
- parallax_factor_al     float32

In the input data file, the healpix pixel (hpx_pixel) is level 12. The output healpix
resolution is set by the command-line "--nside" argument.

Usage
-----
python process_gaia_scanning_law_healpix_index.py \
    --input commanded_scan_law.csv.gz \
    --output-path scanlaw_by_healpix_nside64.h5 \
    --level 6
"""

from multiprocessing import shared_memory
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pandas as pd
from astropy.time import Time

# Data model / HDF5 utilities
DTYPE = np.dtype(
    [
        ("time_bjd", "<f8"),
        ("scan_angle_deg", "<f4"),
        ("parallax_factor_al", "<f4"),
        ("parallax_factor_ac", "<f4"),
        ("healpix_pixel", "<i8"),  # HEALPix pixel index at the requested nside
        ("fov", "<u1"),  # 1=preceding, 2=following
    ]
)
SAVE_COLUMNS = [
    "time_bjd",
    "scan_angle_deg",
    "parallax_factor_al",
    # "parallax_factor_ac",
]
SAVE_DTYPE = np.dtype(
    [("time_bjd", "<f4"), ("scan_angle_deg", "<f4"), ("parallax_factor_al", "<f4")]
)

# Gaia archive time origin: 2010-01-01T00:00 TCB
GAIA_TIME_ORIGIN_JD = 2455197.5

# DR3 full time range: https://www.cosmos.esa.int/web/gaia/dr3
# However, Lindegren+2021 says: data from first month "ecliptic pole scanning law" not
# used, to actually stars 22 August 2014 (21:00 UTC)
# Time(["2014-08-22 21:00:00", "2017-05-28 08:44:00"], scale="utc").tcb.jd
GAIA_DR3_BJD_RANGE = (2456892.376, 2457901.865)

# DR4 astrometry time range unknown. Full range: https://www.cosmos.esa.int/web/gaia/dr4
# Time(["2014-08-22 21:00:00", "2020-01-20 22:00:00"], scale="utc").tcb.jd
# Best guess: start of DR3 astrometry to end of range as reported
GAIA_DR4_BJD_RANGE = (2456892.376, 2458869.418)

# Presumed: start to end of observations, including EPSL data?
GAIA_DR5_BJD_RANGE = (2456863.939, 2460690.762)

dr_bjd_ranges = {
    "dr3": GAIA_DR3_BJD_RANGE,
    "dr4": GAIA_DR4_BJD_RANGE,
    "dr5": GAIA_DR5_BJD_RANGE,
}


def _pixel_worker(
    pix: int, data_meta: tuple[str, tuple[int, ...], object]
) -> tuple[int, np.ndarray] | tuple[int, None]:
    """Get unique transits (scans) a given HEALPix pixel.

    This worker function collects all unique scans for a given HEALPix pixel by
    collapsing scan samples into discrete transits. That is, the Gaia-provided scan law
    samples the fov pointing every 10 seconds, but, for a given HEALPix pixel, we
    shouldn't count another scan until it returns for another focal plane transit.
    """
    data_name, data_shape, data_dtype_spec = data_meta
    data_dtype = np.dtype(data_dtype_spec)  # type: ignore[call-overload]
    data_mem = shared_memory.SharedMemory(name=data_name)

    try:
        data = np.ndarray(data_shape, dtype=data_dtype, buffer=data_mem.buf)

        # Select all scan samples for this pixel
        mask = data["healpix_pixel"] == pix
        if not np.any(mask):
            return pix, None

        return pix, data[mask][SAVE_COLUMNS].astype(SAVE_DTYPE)

    finally:
        data_mem.close()


def process_scanning_law(
    df: pd.DataFrame, *, nside: int, bjd_range: tuple[float, float]
) -> dict[int, np.ndarray]:
    """Convert a DataFrame into per-pixel structured arrays of unique scans.

    Groups consecutive 10-second samples that belong to the same scan transit
    and outputs one representative observation per unique scan.

    Parameters
    ----------
    df : DataFrame
        Input commanded scan law data, split from two FOVs per row into separate rows
        with an 'fov' column.
    nside : int
        HEALPix nside parameter
    bjd_range : tuple[float, float]
        (min, max) BJD range to include

    Returns
    -------
    per_pixel : dict
        Maps pixel_id -> np.ndarray[DTYPE] of unique scans
    """
    time_window_mask = (df["bjd"] >= (bjd_range[0] - GAIA_TIME_ORIGIN_JD)) & (
        df["bjd"] <= (bjd_range[1] - GAIA_TIME_ORIGIN_JD)
    )
    print(
        f"Applying BJD time range filter: {bjd_range}; keeping "
        f"{np.sum(time_window_mask):,} / {len(df):,} rows"
    )
    df = df[time_window_mask].reset_index(drop=True)

    # The commanded scan law file provides healpix pixels for each scan sample, but with
    # level 12 resolution. We want to convert the pixels to the requested nside.
    base_healpix_level = 12

    new_level = hp.nside2order(nside)

    # Convert to nested ordering for fast resolution change:
    tmp_idx = df["heal_pix"].to_numpy()
    new_healpix_idx = tmp_idx >> (2 * (base_healpix_level - new_level))

    # Prepare structured array for all valid samples; per-pixel selection happens
    # inside the multiprocessing workers.
    data = np.zeros(new_healpix_idx.size, dtype=DTYPE)
    data["time_bjd"] = df["bjd"].to_numpy()
    data["scan_angle_deg"] = df["scan_angle"].to_numpy()
    data["parallax_factor_al"] = df["parallax_factor_al"].to_numpy()
    data["parallax_factor_ac"] = df["parallax_factor_ac"].to_numpy()
    data["fov"] = df["fov"].to_numpy()
    data["healpix_pixel"] = new_healpix_idx

    per_fov = []
    for fov in (1, 2):
        fov_data = data[data["fov"] == fov]
        fov_data = fov_data[np.argsort(fov_data["time_bjd"])]

        # Only keep unique scans / transits per pixel
        change_idx = np.where(np.diff(fov_data["healpix_pixel"]) != 0)[0]
        unq_mask = np.concatenate(([0], change_idx + 1))
        per_fov.append(fov_data[unq_mask])

    scan_data = np.concatenate(per_fov)
    return scan_data[np.argsort(scan_data["healpix_pixel"])]


def main(scan_law_file: Path, output_path: Path, nside: int) -> None:
    if not hp.isnsideok(nside):
        raise RuntimeError(f"nside must be a power of two >= 1: got {nside}")

    if scan_law_file.name.endswith("csv"):
        # Load entire CSV
        print(f"Loading {scan_law_file!s}...")
        df = pd.read_csv(scan_law_file, comment="#")

        print(f"...done - loaded {len(df):,} rows")

    elif scan_law_file.name.endswith("hdf5") or scan_law_file.name.endswith("h5"):
        print(f"Loading {scan_law_file!s}...")
        with h5py.File(scan_law_file, "r") as h5f:
            df = pd.DataFrame({key: h5f[key][:] for key in h5f})
        print(f"...done - loaded {len(df):,} rows")

    # Turn the DataFrame where each row is a unique time for both FOVs into a new
    # DataFrame with one row per scan sample and an FOV indicator column
    new_data = {}
    for col_name, col_data in df.items():
        if "fov1" in col_name:
            base_name = col_name[:-5]
            new_data[base_name] = np.concatenate(
                (col_data.to_numpy(), df[f"{base_name}_fov2"].to_numpy())
            )

            if "fov" not in new_data:
                new_data["fov"] = np.concatenate(
                    (
                        np.full(col_data.size, 1, dtype=np.uint8),
                        np.full(col_data.size, 2, dtype=np.uint8),
                    )
                )

        elif "fov2" in col_name:
            continue
        else:
            new_data[col_name] = np.concatenate(
                (col_data.to_numpy(), col_data.to_numpy())
            )
    df = pd.DataFrame(new_data)
    mask = np.isfinite(df["ra"]) & np.isfinite(df["dec"])
    df = df[mask].reset_index(drop=True)
    print(f"Expanded to {len(df):,} rows with both FOVs stacked (and valid ra,dec)")

    for dr_name, bjd_range in dr_bjd_ranges.items():
        output_file = output_path / f"scanlaw_nside{nside}_{dr_name}.h5"
        with h5py.File(output_file, "w") as h5f:
            # Write some root-level metadata for history:
            h5f.attrs.update(
                {
                    "created_utc": Time.now().iso,
                    "creator": "epochalypse/scripts/process_gaia_scanning_law.py",
                    "source_file": str(scan_law_file.resolve().absolute()),
                    "description": "Gaia unique scans grouped by HEALPix pixel.",
                    "dr": dr_name,
                    "gaia_time_origin_bjd": GAIA_TIME_ORIGIN_JD,
                    "bjd_min": bjd_range[0],
                    "bjd_max": bjd_range[1],
                    "healpix_level": hp.nside2order(nside),
                    "healpix_ordering": "NESTED",
                }
            )

            data_new_level = process_scanning_law(df, nside=nside, bjd_range=bjd_range)

            unq_pix, scans_per_pixel = np.unique(
                data_new_level["healpix_pixel"], return_counts=True
            )
            total_scans = np.sum(scans_per_pixel)
            print(f"Found {total_scans:,} unique scans across {len(unq_pix):,} pixels")

            print("Writing dataset to HDF5...")
            h5f.create_dataset(
                "scans",
                data=data_new_level,
                dtype=SAVE_DTYPE,
                compression="gzip",
                compression_opts=9,
            )

            # Metadata for fast indexing:
            idx_grp = h5f.create_group("index")
            idx_grp.create_dataset("pixels", data=unq_pix)

            starts_arr = np.zeros_like(scans_per_pixel)
            starts_arr[1:] = np.cumsum(scans_per_pixel[:-1])

            idx_grp.create_dataset("starts", data=starts_arr)
            idx_grp.create_dataset("counts", data=scans_per_pixel)

            ra, dec = hp.pix2ang(nside, unq_pix, lonlat=True, nest=True)
            idx_grp.create_dataset("ra_deg", data=ra.astype(np.float32))
            idx_grp.create_dataset("dec_deg", data=dec.astype(np.float32))

        print(f"Wrote {output_file!s}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-s",
        "--scan-law-file",
        required=True,
        help="Path to commanded_scan_law.csv or commanded_scan_law.hdf5",
        type=Path,
    )
    parser.add_argument(
        "--output-path", required=True, help="Path to output HDF5 file", type=Path
    )

    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--nside", type=int, default=64, help="HEALPix nside (power of 2, default 64)"
    )
    grp.add_argument("--level", type=int, default=6, help="HEALPix level (default 6)")
    args = parser.parse_args()

    nside = args.nside if args.nside else 2**args.level

    main(scan_law_file=args.scan_law_file, output_path=args.output_path, nside=nside)
