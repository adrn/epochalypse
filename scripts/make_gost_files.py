"""
Create GOST input files: sky coordinates for healpix pixels.

This is a script to create input files of coordinates for the Gaia Observation
Forecasting Tool (GOST) for caching the Gaia scanning law on a healpix grid.

https://gaia.esac.esa.int/gost/

From GOST:
First 3 columns name, alpha, delta in degrees.
The allowed maximum number of sources is 10000

Time range:
From: 2014-07-25T10:31:26
To: 2025-01-15T06:16:32.000

NOTE: I decided to abandon this route and instead use process_gaia_scanning_law.py to
process the commanded scan law directly, but keeping this script around in case it's
useful in the future.
"""

import itertools
import pathlib

import astropy.table as at
import healpy as hp
import numpy as np


def make_input_files(nside: int, output_path: pathlib.Path) -> None:
    npix = hp.nside2npix(nside)

    # Generate central pixel coordinates:
    pix = np.arange(npix)
    ra, dec = hp.pix2ang(
        nside,
        pix,
        nest=False,
        lonlat=True,
    )
    tbl = at.Table()
    tbl["healpix_pixel"] = pix
    tbl["ra"] = ra
    tbl["dec"] = dec

    output_path.mkdir(parents=True, exist_ok=True)

    tbl_idx = np.arange(len(tbl))
    for i, batch_idx in enumerate(itertools.batched(tbl_idx, 10_000)):
        batch = tbl[np.array(batch_idx)]
        out_file = output_path / f"gost_healpix_nside{nside}_{i:02d}.csv"
        batch.write(out_file, format="ascii.no_header", delimiter=",", overwrite=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nside", type=int, default=64, help="Healpix nside for input file"
    )
    parser.add_argument(
        "--output-path",
        type=pathlib.Path,
        required=True,
        help="Output directory for GOST files",
    )
    args = parser.parse_args()
    make_input_files(args.nside, args.output_path.expanduser().resolve())
