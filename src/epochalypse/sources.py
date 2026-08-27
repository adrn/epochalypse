"""Per-source lookup: fetch one star and its scan law by Gaia source id.

At 16k stars the pipeline could read the whole stellar catalog and the whole
scan law into memory in every process. At ~4 million stars it cannot: the scan
law is one row per field-of-view transit, so it grows to O(400M) rows and tens
of GB, and every worker would pay for all of it to simulate its own slice.

This module replaces "load everything" with "look up one source":

    SourceCatalog  -- parent stellar sample, one row per star
    ScanLawStore   -- DR4 scan law, ~90 rows per star

Both are backed by a memory-mapped Arrow/Parquet file plus a small index built
once (`build_indices`) and shared read-only by every rank. Memory per rank is
the index plus one star's rows, not the catalog.

The scan-law index stores (offset, length) per source id, which requires the
file to be grouped by source id -- `build_indices` verifies this and reports the
offending ids rather than silently returning the wrong epochs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _read_arrow(path):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    with pa.memory_map(str(path), "r") as handle:
        return ipc.open_file(handle).read_all()


def _normalize_ids(array):
    """Gaia source ids as strings, never routed through float."""
    series = pd.Series(array)
    if pd.api.types.is_float_dtype(series):
        raise TypeError("source ids arrived as floats; 64-bit ids lose precision")
    if pd.api.types.is_integer_dtype(series):
        return series.astype("Int64").astype(str)
    return series.astype(str)


# --------------------------------------------------------------------------
# Index construction
# --------------------------------------------------------------------------
def build_indices(*, overwrite=False, verbose=True):
    """Build the per-source indices for the stellar catalog and the scan law.

    Run once after stage 1. Cheap relative to a full generation and read-only
    afterwards, so any number of ranks can share it.
    """
    index_dir = C.index_dir()
    index_dir.mkdir(parents=True, exist_ok=True)

    star_index = index_dir / "stars_index.parquet"
    scan_index = index_dir / "scanlaw_index.parquet"
    if star_index.exists() and scan_index.exists() and not overwrite:
        if verbose:
            print(f"  indices already built in {index_dir} (pass --overwrite to rebuild)")
        return {"stars": star_index, "scanlaw": scan_index}

    # --- stellar catalog: id -> row number ---
    stars = pd.read_csv(C.stars_csv(), dtype={"gaia_source_id": str},
                        usecols=["gaia_source_id", "sig_AL"], low_memory=False)
    rows = np.arange(len(stars), dtype=np.int64)

    # A handful of high-RUWE binaries carry no per-CCD AL noise calibration.
    # There is no noise model for them, so simulating one yields NaN epochs.
    # They are excluded from the index, which is the source list every rank
    # iterates -- so they can never reach a shard.
    usable = np.isfinite(stars["sig_AL"].to_numpy(dtype=float))
    if not usable.all():
        print(f"  excluded {int((~usable).sum())} stars with no sig_AL (no noise model)")
    stars, rows = stars[usable], rows[usable]

    ids = _normalize_ids(stars["gaia_source_id"].to_numpy())
    if pd.Series(ids).duplicated().any():
        raise ValueError("the stellar catalog has duplicate gaia_source_id values; "
                         "the per-source lookup needs them unique")
    pd.DataFrame({"gaia_source_id": ids.to_numpy(), "row": rows}) \
        .to_parquet(star_index, index=False)
    if verbose:
        print(f"  stars index   : {len(ids):,} sources -> {star_index}")

    # --- scan law: id -> (offset, length) ---
    table = _read_arrow(C.SCANLAW_DR4)
    scan_ids = _normalize_ids(table.column("gaia_source_id").to_numpy())
    codes, uniques = pd.factorize(scan_ids)          # preserves order of appearance
    boundaries = np.flatnonzero(np.diff(codes)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(codes)]])

    # each source must occupy one contiguous block, or offsets are meaningless
    seen, repeated = set(), []
    for code in codes[starts]:
        if code in seen:
            repeated.append(uniques[code])
        seen.add(code)
    if repeated:
        raise ValueError(
            f"{len(repeated)} source ids appear in more than one block of "
            f"{C.SCANLAW_DR4} (e.g. {repeated[:3]}); sort the scan law by "
            "gaia_source_id before indexing")

    pd.DataFrame({"gaia_source_id": uniques[codes[starts]],
                  "offset": starts.astype(np.int64),
                  "length": (ends - starts).astype(np.int64)}) \
        .to_parquet(scan_index, index=False)
    if verbose:
        print(f"  scanlaw index : {len(starts):,} sources, {len(codes):,} transits "
              f"-> {scan_index}")
    return {"stars": star_index, "scanlaw": scan_index}


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------
class SourceCatalog:
    """The parent stellar sample, addressable by Gaia source id."""

    # stars.csv carries ~117 columns; these are the ones the simulator reads.
    # Reading only these is what keeps a rank near 2 GB instead of 5.5 GB at 4M
    # stars, which is what sets how many ranks fit on a node. A column added to
    # the truth row must be added here too -- the failure is a loud KeyError.
    COLUMNS = ("gaia_source_id", "source_id_dr2", "parallax", "pmra_dr3", "pmdec_dr3",
               "mass_interp", "radius_interp", "sig_AL", "sig_cal", "sig_att_radec",
               "astrometric_n_good_obs_al_dr3", "astrometric_matched_transits_dr3",
               "astrometric_params_solved_dr3")

    def __init__(self):
        index = pd.read_parquet(C.index_dir() / "stars_index.parquet")
        self._row_of = dict(zip(index["gaia_source_id"], index["row"]))
        self._frame = None      # loaded lazily; see `_stars`

    @property
    def _stars(self):
        # One row per star, held once per rank -- unlike the scan law, which is
        # memory-mapped. Loaded on first use so that listing ids costs nothing.
        if self._frame is None:
            self._frame = pd.read_csv(
                C.stars_csv(), low_memory=False, usecols=list(self.COLUMNS),
                dtype={"gaia_source_id": str, "source_id_dr2": str})
            for column in ("gaia_source_id", "source_id_dr2"):
                self._frame[column] = _normalize_ids(self._frame[column].to_numpy())
        return self._frame

    def __contains__(self, gaia_source_id):
        return str(gaia_source_id) in self._row_of

    def __len__(self):
        return len(self._row_of)

    def ids(self):
        """Every source id, in catalog order. This is what ranks slice."""
        return list(self._row_of)

    def get(self, gaia_source_id):
        """One star's row as a Series. Raises KeyError if the id is unknown."""
        key = str(gaia_source_id)
        if key not in self._row_of:
            raise KeyError(f"gaia_source_id {key} is not in {C.stars_csv()}")
        return self._stars.iloc[self._row_of[key]]


class ScanLawStore:
    """The DR4 scan law, addressable by Gaia source id.

    The Arrow table is memory-mapped, so a rank touches only the pages for the
    sources it actually simulates.
    """

    COLUMNS = ("obs_time_tcb_jd", "scan_angle_rad", "parallax_factor_al", "fov")

    def __init__(self):
        index = pd.read_parquet(C.index_dir() / "scanlaw_index.parquet")
        self._span_of = dict(zip(index["gaia_source_id"],
                                 zip(index["offset"], index["length"])))
        self._table = _read_arrow(C.SCANLAW_DR4)

    def __contains__(self, gaia_source_id):
        return str(gaia_source_id) in self._span_of

    def get(self, gaia_source_id):
        """This source's transits as a DataFrame, sorted by observation time."""
        key = str(gaia_source_id)
        if key not in self._span_of:
            raise KeyError(f"no scan law for gaia_source_id {key}")
        offset, length = self._span_of[key]
        block = self._table.slice(int(offset), int(length))
        frame = block.select([c for c in self.COLUMNS if c in block.schema.names]).to_pandas()
        return frame.sort_values("obs_time_tcb_jd").reset_index(drop=True)


# --------------------------------------------------------------------------
# The high-SNR view over a generated population
# --------------------------------------------------------------------------
def select_high_snr(frame, snr_min=None):
    """Systems in which *every* injected companion clears SNR_tot >= `snr_min`.

    This is the whole of the "high-SNR population" -- there is no separate run.
    All three populations are drawn from the unbiased prior with SNR_tot
    recorded rather than rejected on, so the floor is an analysis choice applied
    after the fact. A companion whose SNR_tot is not finite never clears it.

    One caveat a post-hoc cut carries and rejection sampling did not: stars
    enter weighted by their acceptance probability, so the high-SNR populations
    come from a slightly nearer subset of the parent sample.
    """
    snr_min = C.HIGH_SNR_MIN if snr_min is None else float(snr_min)
    columns = [c for c in ("snr_total_1", "snr_total_2") if c in frame.columns]
    if not columns:
        raise KeyError("no snr_total_* columns to cut on")
    snr = frame[columns].to_numpy(float)
    keep = np.isfinite(snr).all(axis=1) & (snr >= snr_min).all(axis=1)
    selected = frame[keep]
    return selected.sort_values("gaia_source_id").reset_index(drop=True)
