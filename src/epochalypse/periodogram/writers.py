"""Two output streams per work unit: the characterization table and the curves.

**The characterization table** is one row per system -- the periodogram summary,
both detection channels, the truth-based recovery flags, and the injected truth
columns joined on. It is what every figure reads. At 17.2 M systems and ~70
columns it is ~6 GB for the whole catalog, so it is written for every system,
always.

**The raw periodograms** are one Delta-chi^2 curve per system, 16,641 float32 =
54 kB after zstd, which is 920 GB over the whole catalog. That is the honest
number and `POWER_MODE` is how it is managed:

| mode | what is stored | full catalog |
| --- | --- | --- |
| `all` | every system | ~920 GB |
| `subsample` | the paper's 10,000-star sample per population | ~1.6 GB |
| `none` | nothing | 0 |

`POWER_DECIMATE` and `POWER_DTYPE` shrink it without changing *which* systems
are kept: decimating by 4 and storing float16 is ~115 GB and still a perfectly
readable figure, because every summary statistic the classification depends on
(`width_dex`, `top_power`, `best_period`) was computed on the full-resolution
float64 curve before anything was thrown away. Decimation is a display choice,
not a measurement one.

The period axis is written once per run, to `period_grid.parquet`, and never
beside a curve -- it is identical for every system in the catalog, so storing
it per row would exactly double the output.

Both writers follow the generator's convention: write `.parquet.tmp` and rename
on success, so a rank killed mid-write leaves no file rather than a truncated
one that looks complete. That is what makes `--skip-existing` trustworthy.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ..shardio import BufferedParquetWriter
from . import config as C


# ==========================================================================
# Which systems get a stored curve
# ==========================================================================
def source_id_ranks(gaia_source_ids, seed=None):
    """A uniform 64-bit rank per source id: `blake2s(f"{seed}:{id}")`.

    `pipeline/subsample.py`, verbatim -- including hashing the id through its
    decimal string, so the rank does not depend on whether the column arrived
    as int64, uint64 or object, and including the little-endian read, which is
    not a free choice: the other byte order gives a different sample.
    """
    seed = C.SUBSAMPLE_SEED if seed is None else int(seed)
    prefix = f"{int(seed)}:".encode("utf-8")
    ids = np.asarray(gaia_source_ids)
    return np.fromiter(
        (int.from_bytes(
            hashlib.blake2s(prefix + str(int(i)).encode("utf-8"), digest_size=8).digest(),
            "little")
         for i in ids),
        dtype=np.uint64, count=len(ids))


def in_paper_subsample(gaia_source_ids, seed=None, cutoff=None):
    """The shared 10,000-star down-selection, decided one source id at a time.

    `subsample_frame` takes the `SUBSAMPLE_SIZE` smallest ranks over the parent
    table, which a rank holding a single shard cannot evaluate. Comparing
    against the precomputed quantile `config.SUBSAMPLE_RANK_CUTOFF` gives the
    identical set -- not an approximation of it -- with no communication
    between ranks and no read of the merged truth table.
    """
    cutoff = C.SUBSAMPLE_RANK_CUTOFF if cutoff is None else int(cutoff)
    return source_id_ranks(gaia_source_ids, seed) <= np.uint64(cutoff)


# ==========================================================================
# The characterization table
# ==========================================================================
def truth_columns(population):
    """The truth columns carried into the characterization table.

    Everything the three paper figures need and nothing the generator alone
    cares about: the five seed columns, `source_id_dr2` and `population` are
    dropped (`population` becomes a partition directory, not a repeated string
    column). The per-companion families are appended for as many companions as
    the population injects.
    """
    columns = list(C.TRUTH_COLUMNS_SYSTEM)
    n = C.POPULATIONS[population]
    for k in range(1, n + 1):
        columns += [c.format(k=k) for c in C.TRUTH_COLUMNS_COMPANION]
    if n == 2:
        columns += list(C.TRUTH_COLUMNS_PAIR)
    return columns


def to_analysis_schema(frame):
    """Add the aliases `epochalypse_figures` reads, alongside the originals.

    The generator records companion masses in Jupiter masses and inclinations
    in degrees; the figure stack reads `Mp_k_msun` and `i_k_rad`. Both unit
    conversions happen here and nowhere else -- the same mapping
    `sharded_systems.to_analysis_schema` writes down, so a figure cell is
    identical whether it is fed a 10,000-system subset or this table.

    The originals are kept rather than renamed: five extra columns per
    companion is a rounding error against 70, and it means the table can be
    read back by anything that speaks either vocabulary.
    """
    added = {}
    for k in (1, 2):
        for source, target, factor in ((f"sma_{k}", f"a_{k}_au", 1.0),
                                       (f"ecc_{k}", f"e_{k}", 1.0),
                                       (f"mass_pl_{k}", f"Mp_{k}_msun", C.MJUP_IN_MSUN),
                                       (f"alpha_mas_{k}", f"alpha_{k}_mas", 1.0)):
            if source in frame.columns:
                added[target] = frame[source].to_numpy(float) * factor
        if f"inc_{k}" in frame.columns:
            added[f"i_{k}_rad"] = np.radians(frame[f"inc_{k}"].to_numpy(float))
    for name, values in added.items():
        frame[name] = values
    return frame


class CharacterizationWriter(BufferedParquetWriter):
    """Buffers characterization records and writes one parquet per work unit.

    Only `_table` is ours: the buffering, the atomic rename, and the context
    manager come from `shardio.BufferedParquetWriter`.
    """

    def __init__(self, path, population, shard, truths):
        super().__init__(path, C.CHARS_FLUSH_EVERY, C.PARQUET_COMPRESSION,
                         compression_level=C.PARQUET_COMPRESSION_LEVEL)
        self._shard = shard
        self._truths = truths[truth_columns(population)]

    @property
    def n_systems(self):
        return self.n_rows

    def add(self, index, record):
        record = dict(record)
        record["shard"] = self._shard
        record["shard_row"] = index
        super().add(record)

    def _table(self, rows):
        frame = pd.DataFrame.from_records(rows)
        joined = self._truths.iloc[frame["shard_row"].to_numpy()].reset_index(drop=True)
        frame = to_analysis_schema(pd.concat([joined, frame], axis=1))
        frame["gaia_source_id"] = frame["gaia_source_id"].astype("int64")
        return pa.Table.from_pandas(frame, preserve_index=False)


# ==========================================================================
# The raw curves
# ==========================================================================
class PowerWriter:
    """Buffers Delta-chi^2 curves and writes one parquet per work unit.

    One row per stored system: `gaia_source_id`, `shard_row`, and `power` as a
    fixed-size list. Fixed-size rather than variable-length because every curve
    is on the same grid -- parquet then stores the values as one flat column
    with no per-row offsets, and a reader can memory-map a slice of it.

    A `PowerWriter` with `mode="none"` is a no-op object, so the caller has no
    branch: it always constructs one and always calls `add`.

    Deliberately NOT built on `shardio.BufferedParquetWriter`: the no-op mode
    must not create the directory, and the fixed-size-list packing keeps two
    parallel buffers. Both would need flags on the base that no other writer
    wants.
    """

    def __init__(self, path, n_periods, mode=None, decimate=None, dtype=None):
        self.mode = C.POWER_MODE if mode is None else mode
        self.decimate = C.POWER_DECIMATE if decimate is None else int(decimate)
        self.dtype = np.dtype(C.POWER_DTYPE if dtype is None else dtype)
        self.n_stored = int(np.ceil(n_periods / self.decimate))
        self.path = Path(path)
        self._tmp = self.path.with_suffix(".parquet.tmp")
        self._writer = None
        self._rows, self._ids = [], []
        self.n_systems = 0
        if self.mode != "none":
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def wants(self, gaia_source_ids):
        """Which of these systems this writer will store, as a boolean array."""
        if self.mode == "none":
            return np.zeros(len(gaia_source_ids), bool)
        if self.mode == "all":
            return np.ones(len(gaia_source_ids), bool)
        if self.mode == "subsample":
            return in_paper_subsample(gaia_source_ids)
        raise ValueError(f"unknown POWER_MODE {self.mode!r}")

    def add(self, gaia_source_id, shard_row, power):
        if self.mode == "none" or power is None:
            return
        self._rows.append(np.asarray(power[::self.decimate], dtype=self.dtype))
        self._ids.append((int(gaia_source_id), int(shard_row)))
        self.n_systems += 1
        if len(self._rows) >= C.POWER_FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self._rows:
            return
        flat = pa.array(np.concatenate(self._rows))
        table = pa.table({
            "gaia_source_id": pa.array([i for i, _ in self._ids], pa.int64()),
            "shard_row": pa.array([r for _, r in self._ids], pa.int32()),
            "power": pa.FixedSizeListArray.from_arrays(flat, self.n_stored),
        })
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self._tmp, table.schema, compression=C.PARQUET_COMPRESSION,
                compression_level=C.PARQUET_COMPRESSION_LEVEL, use_dictionary=False)
        self._writer.write_table(table)
        self._rows, self._ids = [], []

    def close(self):
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._tmp.replace(self.path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def write_period_grid(periods, decimate=None, path=None):
    """Write the trial periods every stored curve is aligned to, once per run.

    Both the full grid and the stored (decimated) one, because the summary
    columns were measured on the full grid and the curves are sampled on the
    decimated one, and a reader needs to be able to tell which is which.
    """
    decimate = C.POWER_DECIMATE if decimate is None else int(decimate)
    path = C.period_grid_path() if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stored = np.asarray(periods, float)[::decimate]
    table = pa.table({"index": pa.array(np.arange(len(stored)), pa.int32()),
                      "period_yr": pa.array(stored, pa.float64())})
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression=C.PARQUET_COMPRESSION)
    tmp.replace(path)
    return path
