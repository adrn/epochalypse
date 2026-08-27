"""Reading the generated catalog's parquet shards, one shard at a time.

The analysis side of this repository has a bridge to the sharded catalog
already -- `src/sharded_systems.py` -- but it solves the opposite problem. It
*materializes* an arbitrary 10,000-system subset out of 320 shards, locating
each system through parquet row-group statistics because the systems it wants
are scattered across every file. That is exactly right when the analysis touches
0.2% of the catalog and touches each of those systems many times.

Here the analysis touches all of it, once. So the unit of work is the shard the
generator already wrote, read start to finish, and nothing is materialized:

    shard 00007 of 00320  ->  17,890 systems  ->  17,890 characterization rows

Two facts about `astrometry.ShardWriter`'s output make this cheap, and both are
checked by `ShardReader` rather than assumed:

1. **The truths shard and the epochs shard are in the same order.** `add()`
   appends to both buffers together, so truths row *i* is the *i*-th system in
   the epoch file. That is the whole index -- there is no id lookup anywhere in
   this module's hot path.

2. **A system's epochs never straddle a row group.** `flush()` writes whole
   buffered systems, so every row group holds an integer number of complete
   systems: the first 2,000 (one `FLUSH_EVERY` buffer) in row group 0, then one
   system per row group. Row groups are therefore a legitimate partition of the
   system list, which is what `--split` divides on.

The reader streams row groups rather than reading the file: a shard is ~100 MB
of epochs, so reading it whole would be fine, but streaming keeps a rank's
resident memory at one row group regardless of how the generator was configured.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C
from . import fitting

# The five epoch columns the fit needs. `gaia_source_id` is read only to verify
# the ordering assumption above, and only on the mixed row group.
EPOCH_COLUMNS = [
    "obs_time_tcb",
    "scan_pos_angle",
    "parallax_factor_al",
    "centroid_pos_al",
    "centroid_pos_error_al",
]


def discover_shards(population):
    """`(shard_numbers, n_shards)` for a population, from the file names.

    `n_shards` is the `_of_NNNNN` suffix the generator wrote, not the number of
    files found: a run whose shards are still being copied would otherwise be
    silently characterized in part. A mismatch is reported by the caller.
    """
    directory = C.shard_dir(population)
    files = sorted(directory.glob("epochs_rank*_of_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no epochs_rank*.parquet in {directory}")
    counts = {int(f.stem.split("_of_")[1]) for f in files}
    if len(counts) != 1:
        raise RuntimeError(f"{directory}: mixed shard counts {sorted(counts)}")
    n_shards = counts.pop()
    numbers = sorted(int(f.stem.split("_rank")[1].split("_of_")[0]) for f in files)
    return numbers, n_shards


def work_units(populations=None, n_parts=1):
    """Every `(population, shard, n_shards, part, n_parts)` the run must do.

    Ordered population-major, then by shard, so a rank's contiguous slice of
    this list is a contiguous region of one population's directory. Per-system
    cost varies by only ~15% across the catalog (the frequency loop dominates,
    not the epoch count), so I/O locality is worth more than load balancing.
    """
    populations = C.POPULATIONS if populations is None else tuple(populations)
    units = []
    for population in populations:
        numbers, n_shards = discover_shards(population)
        for shard in numbers:
            for part in range(n_parts):
                units.append((population, shard, n_shards, part, n_parts))
    return units


class ShardReader:
    """One (population, shard) pair, iterated system by system.

    `truths` is the shard's truth table, one row per system in shard order.
    `iter_systems(part, n_parts)` yields `(index, truth_row, t, psi, pf, y, yerr)`
    for that part's systems, reading only the row groups the part covers.
    """

    def __init__(self, population, shard, n_shards):
        self.population = population
        self.shard = shard
        self.n_shards = n_shards
        self.epochs_path = C.shard_epochs(population, shard, n_shards)
        self.truths_path = C.shard_truths(population, shard, n_shards)
        for path in (self.epochs_path, self.truths_path):
            if not Path(path).exists():
                raise FileNotFoundError(path)

        self.truths = pd.read_parquet(self.truths_path)
        self._handle = pq.ParquetFile(self.epochs_path)
        self._groups = self._group_spans()

        counted = sum(n for _, n in self._groups)
        if counted != len(self.truths):
            raise RuntimeError(
                f"{self.epochs_path.name}: row groups hold {counted} systems but the "
                f"truth shard has {len(self.truths)}; the pair is not from one writer"
            )

    # ------------------------------------------------------------------
    def _group_spans(self):
        """`[(row_group, n_systems)]`, using statistics wherever they suffice.

        A row group whose `gaia_source_id` min equals its max holds exactly one
        system -- true for ~89% of them -- and needs no read at all. The few
        mixed groups (one per shard, the writer's first flush) are read for
        their id column alone and their systems counted by consecutive runs.
        """
        metadata = self._handle.metadata
        names = [metadata.schema.column(i).name for i in range(metadata.num_columns)]
        col = names.index("gaia_source_id")

        spans = []
        for group in range(metadata.num_row_groups):
            stats = metadata.row_group(group).column(col).statistics
            if stats is not None and stats.min == stats.max:
                spans.append((group, 1))
                continue
            ids = (
                self._handle.read_row_group(group, columns=["gaia_source_id"])
                .column(0)
                .to_numpy(zero_copy_only=False)
            )
            spans.append((group, 1 + int((ids[1:] != ids[:-1]).sum())))
        return spans

    def _part_groups(self, part, n_parts):
        """The row groups of one part, and the system index its first one starts at.

        Parts are cut on *systems*, not on row groups, so that splitting a shard
        four ways gives four near-equal amounts of work even though row group 0
        holds 2,000 systems and the rest hold one each.
        """
        total = len(self.truths)
        lo = part * total // n_parts
        hi = (part + 1) * total // n_parts
        groups, start, index = [], None, 0
        for group, n in self._groups:
            if index < hi and index + n > lo:
                if start is None:
                    start = index
                groups.append(group)
            index += n
        return groups, (0 if start is None else start), lo, hi

    # ------------------------------------------------------------------
    def n_systems(self, part=0, n_parts=1):
        total = len(self.truths)
        return (part + 1) * total // n_parts - part * total // n_parts

    def iter_systems(self, part=0, n_parts=1):
        """Yield `(index, truth, t, psi, pf, y, yerr)` over this part's systems.

        `index` is the truth-table row index, i.e. the system's position in the
        shard, so a record can be traced back to its epochs with nothing but
        the shard number and this integer.
        """
        groups, index, lo, hi = self._part_groups(part, n_parts)
        for group in groups:
            block = self._handle.read_row_group(
                group, columns=["gaia_source_id"] + EPOCH_COLUMNS
            )
            ids = block.column(0).to_numpy(zero_copy_only=False)
            columns = [
                block.column(1 + i).to_numpy() for i in range(len(EPOCH_COLUMNS))
            ]
            # boundaries between consecutive systems inside this row group
            edges = np.flatnonzero(ids[1:] != ids[:-1]) + 1
            edges = np.concatenate([[0], edges, [len(ids)]])
            for a, b in zip(edges[:-1], edges[1:]):
                if lo <= index < hi:
                    order = np.argsort(columns[0][a:b])  # by obs_time_tcb
                    t, psi, pf, y, yerr = fitting.epoch_arrays(
                        *[c[a:b][order] for c in columns]
                    )
                    yield index, self.truths.iloc[index], t, psi, pf, y, yerr
                index += 1

    def close(self):
        self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
