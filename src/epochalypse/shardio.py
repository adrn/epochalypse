"""One buffered parquet writer, for the three stages that each wanted their own.

Every parallel stage writes the same way: buffer rows, flush a row group when
the buffer fills so peak memory stays bounded regardless of how much work lands
in a rank, write to `.tmp`, and rename on success. The rename is what makes
`--skip-existing` trustworthy -- a rank killed mid-write leaves no file rather
than a truncated one that looks complete.

Subclasses supply only `_table(rows)`: how a list of buffered items becomes an
Arrow table. `astrometry.ShardWriter` builds epochs from DataFrames, the
characterization writer joins truth columns, the power writer packs fixed-size
lists -- all three used to carry their own copy of everything above.
"""
from __future__ import annotations

from pathlib import Path


class BufferedParquetWriter:
    """Append row groups to `path`, atomically. Subclass and define `_table`."""

    def __init__(self, path, flush_every, compression, compression_level=None,
                 **writer_kwargs):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_suffix(".parquet.tmp")
        self._flush_every = int(flush_every)
        self._compression = compression
        self._writer_kwargs = dict(writer_kwargs)
        if compression_level is not None:
            self._writer_kwargs["compression_level"] = compression_level
        self._rows = []
        self._writer = None
        self.n_rows = 0

    def _table(self, rows):
        raise NotImplementedError

    def add(self, row):
        self._rows.append(row)
        self.n_rows += 1
        if len(self._rows) >= self._flush_every:
            self.flush()

    def flush(self):
        import pyarrow.parquet as pq

        if not self._rows:
            return
        table = self._table(self._rows)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self._tmp, table.schema,
                                            compression=self._compression,
                                            **self._writer_kwargs)
        self._writer.write_table(table)
        self._rows = []

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
