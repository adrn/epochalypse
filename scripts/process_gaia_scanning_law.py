"""
Convert the Gaia commanded scan law to unique scans per sky position.

This script converts the Gaia commanded scan law (sampled at 10-second intervals) into
unique "scans" binned by sky position using a HEALPix pixelization of the sky. Unlike
the GOST (https://gaia.esac.esa.int/gost/index.jsp), this does not have a focal plane
model, so we don't resolve the CCD-level observations. Each transit is considered as a
single scan event.

For each sky position / HEALPix pixel, we store one row per scan:

- bjd_time               float64  (TCB BJD relative to Gaia reference time)
- scan_angle_deg         float32  (position angle, degrees)
- parallax_factor_al     float32
- parallax_factor_ac     float32
- fov                    uint8    (1 for FOV1/PFOV, 2 for FOV2/FFOV)
- ra_deg                 float32
- dec_deg                float32

Usage
-----
python process_gaia_scanning_law.py \
    --input commanded_scan_law.csv.gz \
    --output scanlaw_by_healpix_nside64.h5 \
    --nside 64
"""

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
        ("fov", "u1"),
        ("ra_deg", "<f4"),
        ("dec_deg", "<f4"),
    ]
)

# Gaia archive time origin: 2010-01-01T00:00 TCB
GAIA_TIME_ORIGIN_JD = 2455197.5


def find_unique_scans(
    times: np.ndarray, pixel_ids: np.ndarray, scan_gap_days: float
) -> np.ndarray:
    """Identify unique scans from consecutive samples.

    Consecutive samples within scan_gap_days and same pixel are considered part of the
    same scan transit. Returns indices of representative observations (one per unique
    scan).

    Parameters
    ----------
    times : array
        Observation times in BJD
    pixel_ids : array
        HEALPix pixel IDs for each observation
    scan_gap_days : float
        Maximum time gap (in days) to consider samples as same scan.

    Returns
    -------
    indices : array
        Boolean mask of representative observations for unique scans
    """
    if len(times) == 0:
        raise ValueError("No times provided to find_unique_scans")

    # Sort by pixel then time
    sort_idx = np.lexsort((times, pixel_ids))
    times_sorted = times[sort_idx]
    pix_sorted = pixel_ids[sort_idx]

    # Identify scan breaks - different pixel or time gap > threshold
    pixel_change = np.diff(pix_sorted) != 0
    time_gap = np.diff(times_sorted) > scan_gap_days
    scan_breaks = pixel_change | time_gap

    # First observation is always a scan start
    scan_starts = np.r_[True, scan_breaks]

    # Map back to original order
    result = np.zeros(len(times), dtype=bool)
    result[sort_idx] = scan_starts

    return result


def write_index(h5: h5py.File, counts: dict[int, int]) -> None:
    idx_grp = h5.require_group("index")
    pixels = np.array(sorted(counts.keys()), dtype="<i8")
    nrows = np.array([counts[p] for p in pixels], dtype="<i8")
    if "pixels" in idx_grp:
        del idx_grp["pixels"]
    if "scans_per_pixel" in idx_grp:
        del idx_grp["scans_per_pixel"]
    idx_grp.create_dataset("pixels", data=pixels)
    idx_grp.create_dataset("scans_per_pixel", data=nrows)


# ----------------------------
# Processing logic
# ----------------------------


def process_scanning_law(
    df: pd.DataFrame, *, nside: int, nest: bool = False, scan_gap_hours: float
) -> dict[int, np.ndarray]:
    """Convert a DataFrame into per-pixel structured arrays of unique scans.

    Groups consecutive 10-second samples that belong to the same scan transit
    and outputs one representative observation per unique scan.

    Parameters
    ----------
    df : DataFrame
        Input commanded scan law data
    nside : int
        HEALPix nside parameter
    nest : bool
        HEALPix ordering (False=RING, True=NESTED)
    scan_gap_hours : float
        Time gap threshold (hours) to separate distinct scans

    Returns
    -------
    per_pixel : dict
        Maps pixel_id -> np.ndarray[DTYPE] of unique scans
    """
    per_pixel: dict[int, list[np.ndarray]] = {}
    scan_gap_days = scan_gap_hours / 24.0

    # Process both FOVs
    for fov_num in [1, 2]:
        ra = df[f"ra_fov{fov_num}"].to_numpy()
        dec = df[f"dec_fov{fov_num}"].to_numpy()
        valid = np.isfinite(ra) & np.isfinite(dec)
        if not np.any(valid):
            continue

        # Compute HEALPix for valid observations
        pix = hp.ang2pix(nside, ra[valid], dec[valid], nest=nest, lonlat=True)
        times = df[f"bjd_fov{fov_num}"].to_numpy()[valid]

        # Identify unique scans (one per scan transit, not one per 10s sample)
        unique_scan_mask = find_unique_scans(times, pix, scan_gap_days)

        # Keep only the representative observations for unique scans
        pix_unique = pix[unique_scan_mask]
        arr = np.zeros(pix_unique.size, dtype=DTYPE)
        arr["bjd_time"] = times[unique_scan_mask]
        arr["scan_angle_deg"] = df[f"scan_angle_fov{fov_num}"].to_numpy()[valid][
            unique_scan_mask
        ]
        arr["parallax_factor_al"] = df[f"parallax_factor_al_fov{fov_num}"].to_numpy()[
            valid
        ][unique_scan_mask]
        arr["parallax_factor_ac"] = df[f"parallax_factor_ac_fov{fov_num}"].to_numpy()[
            valid
        ][unique_scan_mask]
        arr["fov"] = np.uint8(fov_num)
        arr["ra_deg"] = ra[valid][unique_scan_mask]
        arr["dec_deg"] = dec[valid][unique_scan_mask]

        # Group by pixel id
        for p in np.unique(pix_unique):
            mask = pix_unique == p
            per_pixel.setdefault(int(p), []).append(arr[mask])

    # Concatenate per-pixel lists into arrays
    out: dict[int, np.ndarray] = {}
    for p, parts in per_pixel.items():
        out[p] = np.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
    return out


def main(nside: int, scan_gap_hours: float) -> None:
    if not hp.isnsideok(nside):
        raise RuntimeError(f"nside must be a power of two >= 1: got {nside}")

    nest = False  # Gaia uses RING by default
    input_path = Path(args.input)
    output_path = Path(args.output)

    with h5py.File(output_path, "w") as h5f:
        # Write some root-level metadata for history:
        h5f.attrs.update(
            {
                "created_utc": Time.now().iso,
                "creator": "epochalypse/scripts/process_gaia_scanning_law.py",
                "source_file": str(input_path.resolve().absolute()),
                "description": "Gaia unique scans grouped by HEALPix pixel.",
                "nside": nside,
                "scan_gap_hours": scan_gap_hours,
                "time_scale": "TCB",
                "units_ra": "deg",
                "units_dec": "deg",
                "units_scan_angle": "deg",
            }
        )

        pix_grp = h5f.require_group("pix")
        counts: dict[int, int] = {}

        dtypes = {
            "bjd_fov1": "float64",
            "ra_fov1": "float32",
            "dec_fov1": "float32",
            "scan_angle_fov1": "float32",
            "parallax_factor_al_fov1": "float32",
            "parallax_factor_ac_fov1": "float32",
            "bjd_fov2": "float64",
            "ra_fov2": "float32",
            "dec_fov2": "float32",
            "scan_angle_fov2": "float32",
            "parallax_factor_al_fov2": "float32",
            "parallax_factor_ac_fov2": "float32",
        }

        # Load entire CSV
        print(f"Loading {input_path}...")
        df = pd.read_csv(
            input_path,
            usecols=list(dtypes.keys()),
            dtype=dtypes,
            comment="#",
        )
        print(f"...done - loaded {len(df):,} rows")

        print(f"Identifying unique scans (gap threshold: {scan_gap_hours}h)...")
        per_pixel_arrays = process_scanning_law(
            df, nside=nside, nest=nest, scan_gap_hours=scan_gap_hours
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
            if key not in pix_grp:
                pix_grp.create_dataset(key, shape=(0,), maxshape=(None,), dtype=DTYPE)

            # Append data from both focal planes
            n0 = pix_grp[key].shape[0]
            pix_grp[key].resize((n0 + arr.shape[0],))
            pix_grp[key][n0:] = arr

            counts[p] = arr.shape[0]

        write_index(h5f, counts)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--input", required=True, help="Path to commanded_scan_law.csv.gz"
    )
    parser.add_argument("--output", required=True, help="Path to output HDF5 file")
    parser.add_argument(
        "--nside", type=int, default=64, help="HEALPix nside (power of 2, default 64)"
    )

    parser.add_argument(
        "--scan-gap-hours",
        type=float,
        default=3.0,
        help=(
            "Time gap (hours) to separate distinct scans (default 3). Gaia spin period "
            "is ~6 hours, so this should capture individual transits."
        ),
    )
    args = parser.parse_args()

    main(nside=args.nside, scan_gap_hours=args.scan_gap_hours)
