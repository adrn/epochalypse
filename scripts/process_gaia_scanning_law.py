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

- bjd_time               float64  (TCB BJD relative to Gaia reference time)
- scan_angle_deg         float32  (position angle, degrees)
- parallax_factor_al     float32
- parallax_factor_ac     float32

In the input data file, the healpix pixel (hpx_pixel) is level 12. The output healpix
resolution is set by the command-line "--nside" argument.

Usage
-----
python process_gaia_scanning_law_healpix_index.py \
    --input commanded_scan_law.csv.gz \
    --output-path scanlaw_by_healpix_nside64.h5 \
    --level 6
"""

from multiprocessing import Pool, shared_memory
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
import pandas as pd
from astropy.time import Time

# Data model / HDF5 utilities
DTYPE = np.dtype(
    [
        ("bjd_time", "<f8"),
        ("scan_angle_deg", "<f4"),
        ("parallax_factor_al", "<f4"),
        ("parallax_factor_ac", "<f4"),
    ]
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

        return pix, data[mask].copy()

    finally:
        data_mem.close()


def process_scanning_law(
    df: pd.DataFrame, *, nside: int, bjd_range: tuple[float, float], nest: bool = False
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
    nest : bool
        HEALPix ordering (False=RING, True=NESTED)

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
    # level 12 resolution.
    base_healpix_level = hp.order2nside(12)

    new_level = hp.nside2order(nside)
    tmp_idx = df.heal_pix_fov1.to_numpy()
    new_healpix_idx = tmp_idx >> (2 * (base_healpix_level - new_level))
    unq_pix = np.unique(new_healpix_idx)

    # Prepare structured array for all valid samples; per-pixel selection happens
    # inside the multiprocessing workers.
    data = np.zeros(pix.size, dtype=DTYPE)
    data["time_bjd"] = df["bjd"].to_numpy()
    data["scan_angle_deg"] = df["scan_angle"].to_numpy()
    data["parallax_factor_al"] = df["parallax_factor_al"].to_numpy()
    data["parallax_factor_ac"] = df["parallax_factor_ac"].to_numpy()
    data["healpix_pixel"] = new_healpix_idx

    # Copy data to shared memory for multiprocessing
    data_shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    data_buf = np.ndarray(data.shape, dtype=data.dtype, buffer=data_shm.buf)
    data_buf[:] = data

    data_meta = (
        data_shm.name,
        data.shape,
        data.dtype.descr,
    )

    try:
        with Pool() as pool:
            result_arrays = pool.starmap(
                _pixel_worker,
                ((int(p), data_meta) for p in unq_pix),  # tasks
            )
    finally:
        data_shm.close()
        data_shm.unlink()

    per_pixel: dict[int, np.ndarray] = {}
    for pix_id, res_arr in result_arrays:
        if res_arr is None or res_arr.size == 0:
            continue
        per_pixel[int(pix_id)] = res_arr

    return per_pixel


def main(scan_law_file: Path, output_path: Path, nside: int) -> None:
    if not hp.isnsideok(nside):
        raise RuntimeError(f"nside must be a power of two >= 1: got {nside}")

    nest = False  # Gaia uses RING by default

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
                    "nside": nside,
                    "time_scale": "TCB",
                    "units_ra": "deg",
                    "units_dec": "deg",
                    "units_scan_angle": "deg",
                }
            )

            pix_grp = h5f.require_group("pix")
            counts: dict[int, int] = {}

            print("Identifying unique scans...")
            per_pixel_arrays = process_scanning_law(
                df, nside=nside, bjd_range=bjd_range, nest=nest
            )

            total_scans = sum(len(arr) for arr in per_pixel_arrays.values())
            print(
                f"Found {total_scans:,} unique scans across {len(per_pixel_arrays):,} "
                "pixels"
            )
            print("Writing pixel datasets to HDF5...")
            # Write to HDF5 per pixel
            for p, arr in per_pixel_arrays.items():
                key = str(p)
                pix_grp.create_dataset(key, data=arr, dtype=DTYPE)
                counts[p] = arr.shape[0]

            # Write map-level info:
            idx_grp = h5f.create_group("index")
            pixels = np.array(sorted(counts.keys()), dtype="<i8")
            nrows = np.array([counts[p] for p in pixels], dtype="<i8")
            idx_grp.create_dataset("pixels", data=pixels)
            idx_grp.create_dataset("scans_per_pixel", data=nrows)

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
