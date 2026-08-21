"""Per-source lookup: fetch one star and its scan law by Gaia source id.

At 16k stars the pipeline could read the whole stellar catalog and the whole
scan law into memory in every process. At ~4 million stars it cannot: the scan
law is one row per field-of-view transit, so it grows to O(400M) rows and tens
of GB, and every worker would pay for all of it to simulate its own slice.

This module replaces "load everything" with "look up one source":

    SourceCatalog  -- parent stellar sample, one row per star
    ScanLawStore   -- DR4 scan law, ~90 rows per star

Both are backed by a memory-mapped Arrow/Parquet file plus a small index built
once (`build_indices`) and shared read-only by every worker. Memory per worker
is the index plus one star's rows, not the catalog.

The scan-law index stores (offset, length) per source id, which requires the
file to be grouped by source id -- `build_indices` verifies this and reports the
offending ids rather than silently returning the wrong epochs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


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
def build_indices(config, *, overwrite=False, verbose=True):
    """Build the per-source indices for the stellar catalog and the scan law.

    Run once after stage 1. Cheap relative to a full generation and read-only
    afterwards, so any number of workers can share it.
    """
    index_dir = config.paths.index_dir
    index_dir.mkdir(parents=True, exist_ok=True)

    star_index = index_dir / "stars_index.parquet"
    scan_index = index_dir / "scanlaw_index.parquet"
    if star_index.exists() and scan_index.exists() and not overwrite:
        if verbose:
            print(f"  indices already built in {index_dir} (pass --overwrite to rebuild)")
        return {"stars": star_index, "scanlaw": scan_index}

    # --- stellar catalog: id -> row number ---
    stars = pd.read_csv(config.paths.stars_csv,
                        dtype={"gaia_source_id": str, "source_id_dr2": str},
                        usecols=["gaia_source_id", "sig_AL"], low_memory=False)
    rows = np.arange(len(stars), dtype=np.int64)
    if config.stars.require_sigma_al:
        # A handful of high-RUWE binaries carry no per-CCD AL noise calibration.
        # There is no noise model for them, so simulating one yields NaN epochs.
        # They are excluded from the index, which is the source list every worker
        # iterates -- so they can never reach a shard.
        usable = np.isfinite(stars["sig_AL"].to_numpy(dtype=float))
        if not usable.all():
            print(f"  excluded {int((~usable).sum())} stars with no sig_AL "
                  "(no noise model)")
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
    table = _read_arrow(config.paths.scanlaw_dr4)
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
            f"{config.paths.scanlaw_dr4} (e.g. {repeated[:3]}); sort the scan law by "
            "gaia_source_id before indexing")

    pd.DataFrame({"gaia_source_id": uniques[codes[starts]],
                  "offset": starts.astype(np.int64),
                  "length": (ends - starts).astype(np.int64)}) \
        .to_parquet(scan_index, index=False)
    if verbose:
        print(f"  scanlaw index : {len(starts):,} sources, {len(codes):,} transits "
              f"-> {scan_index}")

    (index_dir / "index_manifest.json").write_text(json.dumps({
        "stars_csv": str(config.paths.stars_csv),
        "scanlaw": str(config.paths.scanlaw_dr4),
        "n_sources": int(len(ids)),
        "n_transits": int(len(codes)),
        # Integer task ids index into this list, so its content and order are
        # part of the contract between a scheduler and the catalog. The
        # checksum lets a worker assert it holds the same list the ids were
        # issued against.
        "source_list_checksum": source_list_checksum(ids.tolist()),
    }, indent=2))
    return {"stars": star_index, "scanlaw": scan_index}


def source_list_checksum(ids) -> str:
    """Fingerprint of the ordered source list."""
    import hashlib

    digest = hashlib.blake2s(digest_size=16)
    for source_id in ids:
        digest.update(str(source_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Lookup
# --------------------------------------------------------------------------
class SourceCatalog:
    """The parent stellar sample, addressable by Gaia source id."""

    def __init__(self, config):
        self.config = config
        index = pd.read_parquet(config.paths.index_dir / "stars_index.parquet")
        self._row_of = dict(zip(index["gaia_source_id"], index["row"]))
        self._frame = None      # loaded lazily; see `_stars`

    @property
    def _stars(self):
        # The stellar catalog is one row per star: 4M rows of ~100 columns is
        # large but tractable once per worker, unlike the scan law. Loaded on
        # first use so that listing ids costs nothing.
        if self._frame is None:
            self._frame = pd.read_csv(
                self.config.paths.stars_csv,
                dtype={"gaia_source_id": str, "source_id_dr2": str}, low_memory=False)
            for column in ("gaia_source_id", "source_id_dr2"):
                self._frame[column] = _normalize_ids(self._frame[column].to_numpy())
        return self._frame

    def __contains__(self, gaia_source_id):
        return str(gaia_source_id) in self._row_of

    def __len__(self):
        return len(self._row_of)

    def ids(self):
        """Every source id, in catalog order.

        This ordering IS the integer task-id mapping: `--id 7` means
        `ids()[7]`. It is fixed at index-build time and fingerprinted in
        index_manifest.json, so a worker can prove it holds the same list the
        ids were issued against.
        """
        return list(self._row_of)

    def id_at(self, index):
        """The source id for integer task `index` (what --id refers to)."""
        ids = self.ids()
        if not 0 <= int(index) < len(ids):
            raise IndexError(f"--id {index} out of range: the source list has "
                             f"{len(ids):,} entries (0..{len(ids) - 1})")
        return ids[int(index)]

    def checksum(self):
        """Fingerprint of this source list; compare against the manifest."""
        return source_list_checksum(self.ids())

    def verify_checksum(self):
        """Raise if the source list no longer matches the one indexed.

        Cheap insurance for a 1000-rank job: every rank asserts it works from
        the same list, so a stale stars.csv on one node cannot silently shift
        what `--id 7` means.
        """
        manifest = self.config.paths.index_dir / "index_manifest.json"
        if not manifest.exists():
            return None
        expected = json.loads(manifest.read_text()).get("source_list_checksum")
        actual = self.checksum()
        if expected and expected != actual:
            raise RuntimeError(
                f"source list checksum mismatch: the index was built for {expected}, "
                f"this catalog hashes to {actual}. Integer --id values from one "
                "would point at different stars in the other; rebuild the indices.")
        return actual

    def get(self, gaia_source_id):
        """One star's row as a Series. Raises KeyError if the id is unknown."""
        key = str(gaia_source_id)
        if key not in self._row_of:
            raise KeyError(f"gaia_source_id {key} is not in {self.config.paths.stars_csv}")
        return self._stars.iloc[self._row_of[key]]


class ScanLawStore:
    """The DR4 scan law, addressable by Gaia source id.

    The Arrow table is memory-mapped, so a worker touches only the pages for the
    sources it actually simulates.
    """

    COLUMNS = ("obs_time_tcb_jd", "scan_angle_rad", "parallax_factor_al", "fov")

    def __init__(self, config):
        self.config = config
        index = pd.read_parquet(config.paths.index_dir / "scanlaw_index.parquet")
        self._span_of = dict(zip(index["gaia_source_id"],
                                 zip(index["offset"], index["length"])))
        self._table = _read_arrow(config.paths.scanlaw_dr4)

    def __contains__(self, gaia_source_id):
        return str(gaia_source_id) in self._span_of

    def n_transits(self, gaia_source_id):
        """Number of FoV transits, without materializing the rows."""
        key = str(gaia_source_id)
        if key not in self._span_of:
            return 0
        return int(self._span_of[key][1])

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
# Shard assignment
# --------------------------------------------------------------------------
def shard_of(gaia_source_id, n_shards):
    """Which shard a source belongs to: a pure function of its id.

    blake2s rather than hash() so the partition is stable across processes,
    machines, and Python runs.
    """
    import hashlib

    digest = hashlib.blake2s(str(gaia_source_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(n_shards)


def select_shard(gaia_source_ids, shard, n_shards):
    """The subset of ids assigned to `shard`, in the order given."""
    return [sid for sid in gaia_source_ids if shard_of(sid, n_shards) == shard]
