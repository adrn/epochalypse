"""Repack the five per-population HDF5 files into one shareable bundle.

The pipeline writes one file per population in a /truths + /systems/<idx>/epochs
layout (simulated_astrometry_<population>_systems.h5). That layout costs ~24 KB
per system because it gzips 16k tiny per-system datasets independently. Packing
the same epochs into one long-form table per population and compressing that
instead costs ~3.5 KB per system -- ~1.9 GB becomes ~300 MB, with no change to
the data.

Output layout matches the older single-file bundle that used to be shared:

    /epochs/population_<population>   long-form epoch table (one row per epoch)
    /injected_solutions               truth table, all populations
    /manifest                         population, hdf_key, n_systems, n_epochs

zlib is used rather than blosc: it is a standard HDF5 filter, so plain h5py and
any generic HDF5 tool can read the result without a compression plugin. blosc
compresses marginally better but makes the file unreadable outside PyTables.

Usage:
    python repack_simulated_astrometry.py
    python repack_simulated_astrometry.py --limit 1000 --out /tmp/probe.h5
"""
import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# repo root = the directory holding src/ and outputs/, found from this file's location,
# so the script runs the same from the repo root or from catalog_generation/
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src" / "epochalypse_fitting.py").exists())
DATA = ROOT / "outputs/data"
POPULATIONS = [
    "0_companion",
    "1_companion_agnostic",
    "1_companion_detectable",
    "2_companion_agnostic",
    "2_companion_detectable",
]


def _decode(df):
    """HDF5 hands back bytes for string columns; store them as str."""
    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].str.decode("utf-8")
    return df


def _numeric_booleans(df):
    """Cast bool-with-NaN object columns to float32 (1.0/0.0/NaN).

    The 1- and 2-companion flags (near_2_1, coplanar, ...) are NaN for
    populations where they do not apply, which makes them object dtype. PyTables
    pickles object columns, which bloats the file and makes it unreadable
    outside Python.
    """
    for column in df.columns:
        if df[column].dtype != object:
            continue
        values = set(df[column].dropna().unique())
        if values and values <= {True, False}:
            df[column] = df[column].astype("boolean").astype("float32")
    return df


def _epoch_chunks(path, population, chunk_systems, limit=None):
    """Yield (start_index, DataFrame) of concatenated epoch tables."""
    with h5py.File(path, "r") as f:
        n_systems = int(f["truths"].attrs["n_systems"])
        if limit is not None:
            n_systems = min(n_systems, limit)
        systems = f["systems"]
        for start in range(0, n_systems, chunk_systems):
            stop = min(start + chunk_systems, n_systems)
            frames = []
            for idx in range(start, stop):
                key = str(idx)
                if key not in systems or "epochs" not in systems[key]:
                    continue  # system had no epoch file at export time
                frames.append(pd.DataFrame(systems[key]["epochs"][:]))
            if not frames:
                continue
            chunk = _decode(pd.concat(frames, ignore_index=True))
            chunk["population"] = population
            yield start, stop, n_systems, chunk


def repack(out_path, complevel=5, complib="zlib", chunk_systems=2000,
           limit=None, drop_filepath=False, index_system_id=True):
    out_path = Path(out_path)
    sources = {p: DATA / f"simulated_astrometry_{p}_systems.h5" for p in POPULATIONS}
    missing = [str(p) for p in sources.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing}")

    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()

    manifest = []
    started = time.time()
    with pd.HDFStore(temporary, "w", complevel=complevel, complib=complib) as store:
        for population, path in sources.items():
            key = f"epochs/population_{population}"
            n_epochs = 0
            # String columns need a declared width up front when appending in
            # chunks, otherwise later chunks are silently truncated.
            min_itemsize = {
                "system_id": 48,
                "gaia_source_id": 24,
                "source_id_dr2": 24,
                "field_of_view": 8,
                "population": 32,
            }
            for start, stop, n_systems, chunk in _epoch_chunks(
                path, population, chunk_systems, limit
            ):
                store.append(
                    key,
                    chunk,
                    format="table",
                    min_itemsize={k: v for k, v in min_itemsize.items()
                                  if k in chunk.columns},
                    data_columns=["system_id"] if index_system_id else None,
                )
                n_epochs += len(chunk)
                print(f"  {population}: {stop}/{n_systems} systems, "
                      f"{n_epochs:,} epochs", end="\r", flush=True)
            print()
            manifest.append({
                "population": population,
                "hdf_key": "/" + key,
                "n_systems": n_systems,
                "n_epochs": n_epochs,
            })

        truths = pd.read_csv(
            DATA / "injected_solutions_all.csv",
            dtype={"gaia_source_id": str, "source_id_dr2": str},
            low_memory=False,
        )
        if drop_filepath and "filepath" in truths.columns:
            # Absolute paths from the generating machine; meaningless downstream.
            truths = truths.drop(columns=["filepath"])
        store.put("injected_solutions", _numeric_booleans(truths), format="fixed")
        store.put("manifest", pd.DataFrame(manifest), format="table")

    temporary.replace(out_path)
    size = out_path.stat().st_size
    total_systems = sum(m["n_systems"] for m in manifest)
    print(f"\nWrote {out_path} ({size / 1e6:.0f} MB) in {(time.time() - started) / 60:.1f} min")
    print(f"{total_systems:,} systems, {sum(m['n_epochs'] for m in manifest):,} epochs, "
          f"{size / total_systems / 1024:.2f} KB/system")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # Deliberately not simulated_astrometry.h5: that name holds the older bundle
    # that is currently shared, and overwriting it would destroy the only local
    # copy. Rename after checking the output.
    p.add_argument("--out", default=str(DATA / "simulated_astrometry_bundle.h5"))
    p.add_argument("--complevel", type=int, default=5)
    p.add_argument("--complib", default="zlib")
    p.add_argument("--chunk-systems", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None,
                   help="only repack this many systems per population (for testing)")
    p.add_argument("--drop-filepath", action="store_true",
                   help="drop the machine-specific filepath column from the truths")
    p.add_argument("--no-index", action="store_true",
                   help="do not index system_id (smaller file, no per-system queries)")
    a = p.parse_args()
    repack(a.out, complevel=a.complevel, complib=a.complib,
           chunk_systems=a.chunk_systems, limit=a.limit,
           drop_filepath=a.drop_filepath, index_system_id=not a.no_index)


if __name__ == "__main__":
    main()
