"""One work unit: search every system in one shard (or one part of one).

This is the whole of the compute. `scripts/characterize_mpi.py` is a loop over
`run_unit` and a `gather` at the end; there is no other work in this pipeline.
It lives in the package rather than in a script because the MPI driver and the
tests both call it -- importing it from a script is what used to require a
`sys.path` hack in the test suite.
"""

from __future__ import annotations

import time

from . import config as C
from .grid import frequency_segments, segment_periods
from .periodogram import characterize_system
from .shards import ShardReader
from .writers import CharacterizationWriter, PowerWriter


def run_unit(
    population,
    shard,
    n_shards,
    part=0,
    n_parts=1,
    *,
    segments=None,
    limit=None,
    skip_existing=False,
    power_mode=None,
    progress_every=2000,
    verbose=True,
):
    """Search every system in one work unit; write its two parquet files.

    Returns a summary dict. A system that raises is recorded and skipped rather
    than taken as fatal: one unusable star must not cost a rank its shard, and
    at 17 M systems a per-system exception that happens once in a million still
    happens seventeen times.
    """
    segments = frequency_segments() if segments is None else segments
    periods = segment_periods(segments)
    chars_path = C.chars_shard(population, shard, n_shards, part, n_parts)
    power_path = C.power_shard(population, shard, n_shards, part, n_parts)

    if skip_existing and chars_path.exists():
        if verbose:
            print(
                f"[{population} {shard:05d}.{part}] already done, skipping", flush=True
            )
        return {
            "population": population,
            "shard": shard,
            "part": part,
            "n_systems": 0,
            "n_failed": 0,
            "skipped": True,
            "seconds": 0.0,
        }

    started = time.time()
    failures = []
    with ShardReader(population, shard, n_shards) as reader:
        power = PowerWriter(power_path, len(periods), mode=power_mode)
        # `wants` is evaluated for the whole shard at once: the subsample mode
        # is a blake2s per id and doing it per system inside the loop would put
        # a hash in the hot path for no reason.
        wanted = power.wants(reader.truths["gaia_source_id"].to_numpy())
        n_unit = reader.n_systems(part, n_parts)

        with (
            CharacterizationWriter(
                chars_path, population, shard, reader.truths
            ) as chars,
            power,
        ):
            for count, (index, truth, t, psi, pf, y, yerr) in enumerate(
                reader.iter_systems(part, n_parts)
            ):
                if limit and count >= limit:
                    break
                try:
                    record, curve = characterize_system(
                        t,
                        psi,
                        pf,
                        y,
                        yerr,
                        truth=truth,
                        segments=segments,
                        want_power=bool(wanted[index]),
                    )
                except Exception as error:
                    failures.append(
                        {
                            "population": population,
                            "shard": shard,
                            "shard_row": index,
                            "gaia_source_id": truth["gaia_source_id"],
                            "reason": repr(error),
                        }
                    )
                    continue
                chars.add(index, record)
                power.add(truth["gaia_source_id"], index, curve)
                if verbose and progress_every and (count + 1) % progress_every == 0:
                    rate = (count + 1) / (time.time() - started)
                    print(
                        f"[{population} {shard:05d}.{part}] {count + 1:,}/{n_unit:,} "
                        f"({rate:.1f}/s)",
                        flush=True,
                    )

    if failures:
        import pandas as pd

        path = C.failed_dir() / f"{population}_shard{shard:05d}_part{part:02d}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(path, index=False)

    elapsed = time.time() - started
    summary = {
        "population": population,
        "shard": shard,
        "part": part,
        "n_systems": chars.n_systems,
        "n_power": power.n_systems,
        "n_failed": len(failures),
        "skipped": False,
        "seconds": elapsed,
    }
    if verbose:
        rate = chars.n_systems / elapsed if elapsed else 0.0
        print(
            f"[{population} {shard:05d}.{part}] {chars.n_systems:>7,} systems in "
            f"{elapsed / 60:6.1f} min ({rate:5.1f}/s), {power.n_systems:,} curves stored"
            + (f", {len(failures)} FAILED" if failures else ""),
            flush=True,
        )
    return summary
