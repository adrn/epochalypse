"""Two output streams per work unit: the per-system table and the samples.

**The per-system table** is one row per system -- the diagnostics, three period
point estimates, and the injected truth columns joined on. It is small (~90
columns, ~2 GB for the whole catalog) and it is what most analysis reads: the
ESS column alone answers "did the library resolve this posterior", which is the
question that decides whether a system needs a second, MCMC pass.

**The samples** are `TOP_K` weighted draws per system. Stored one row per
system, with each parameter a fixed-size list of `TOP_K` float32, rather than
one row per sample: `gaia_source_id` and a sample index repeated 1024 times
would add ~210 GB of pure bookkeeping over the catalog, and parquet stores a
fixed-size list as one flat column with no per-row offsets, so a reader can
memory-map a slice. It also makes the join to the per-system table a row-for-row
one.

**The samples are weighted.** `weight` is normalized over the *whole* prior
library, so it sums to `weight_captured` rather than to 1, and any average over
these draws that ignores it is wrong. That is a harv sharp bit and it survives
into this table unchanged.

Both writers follow the generator's convention: write `.parquet.tmp` and rename
on success, so a rank killed mid-write leaves no file rather than a truncated
one that looks complete. That is what makes `--skip-existing` trustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from ..periodogram.writers import truth_columns
from ..shardio import BufferedParquetWriter
from . import config as C


class SystemWriter(BufferedParquetWriter):
    """Buffers per-system records and writes one parquet per work unit.

    Only `_table` is ours: the buffering, the atomic rename, and the context
    manager come from `shardio.BufferedParquetWriter`. The truth join is the
    same one `periodogram.writers.CharacterizationWriter` does, against the same
    column list, so the two analyses can be merged on `gaia_source_id` without
    either being the authority on what a truth column is called.
    """

    def __init__(self, path, population, shard, truths):
        super().__init__(
            path,
            C.SYSTEMS_FLUSH_EVERY,
            C.PARQUET_COMPRESSION,
            compression_level=C.PARQUET_COMPRESSION_LEVEL,
        )
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
        frame = pd.concat([joined, frame], axis=1)
        frame["gaia_source_id"] = frame["gaia_source_id"].astype("int64")
        return pa.Table.from_pandas(frame, preserve_index=False)


class SampleWriter(BufferedParquetWriter):
    """Buffers top-K sample blocks and writes one parquet per work unit.

    The schema is taken from the first system's parameter names rather than
    hard-coded, so switching `config.USE_THIELE_INNES` changes the columns
    without changing this file. Units are not in the schema -- they are recorded
    once in the run manifest, because they are identical for every row in the
    catalog and a per-column unit string would be repeated 17.2 M times.
    """

    def __init__(self, path, top_k=None, dtype=None):
        self.top_k = C.TOP_K if top_k is None else int(top_k)
        self.dtype = np.dtype(C.SAMPLE_DTYPE if dtype is None else dtype)
        super().__init__(
            path,
            C.SAMPLES_FLUSH_EVERY,
            C.PARQUET_COMPRESSION,
            compression_level=C.PARQUET_COMPRESSION_LEVEL,
            use_dictionary=False,
        )

    @property
    def n_systems(self):
        return self.n_rows

    def add(self, gaia_source_id, shard_row, columns):
        """`columns` maps parameter name -> a `(top_k,)` array of draws."""
        cast = {}
        for name, values in columns.items():
            values = np.asarray(values, dtype=self.dtype)
            if values.shape != (self.top_k,):
                msg = f"{name}: expected ({self.top_k},), got {values.shape}"
                raise ValueError(msg)
            cast[name] = values
        super().add((int(gaia_source_id), int(shard_row), cast))

    def _table(self, rows):
        table = {
            "gaia_source_id": pa.array([r[0] for r in rows], pa.int64()),
            "shard_row": pa.array([r[1] for r in rows], pa.int32()),
        }
        for name in rows[0][2]:
            flat = pa.array(np.concatenate([r[2][name] for r in rows]))
            table[name] = pa.FixedSizeListArray.from_arrays(flat, self.top_k)
        return pa.table(table)
