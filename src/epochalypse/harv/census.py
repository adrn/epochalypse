"""Reading the per-system output, and the two definitions everything shares.

`recovered()` and `high_snr_mask()` live here rather than in the finish script
because the figures need exactly the same rules. They were duplicated once
already and the copy in the script scored recovery against `period_1` alone,
which marks a correct fit on a two-companion system's *second* orbit as a
failure. One definition, two callers.
"""

from __future__ import annotations

import numpy as np
import pyarrow.dataset as ds

from ..periodogram import config as PG
from . import config as C

# Binning shared by the text census and the figures, so the two cut the sample
# identically. The period edges span the *searched* range -- there is no point
# resolving decades the prior no longer covers -- so they follow PERIOD_MIN_YR /
# PERIOD_MAX_YR rather than being fixed numbers.
LOG_PERIOD_BINS = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0])
ECC_BINS = np.array([0.0, 0.3, 0.5, 0.7, 0.9, 1.0])

# Fixed rather than derived from the data, so runs with different recovery rates
# still line up column for column. Starts at the high-SNR floor.
SNR_BINS = np.array([PG.HIGH_SNR_MIN, 10.0, 20.0, 40.0, 80.0, np.inf])


def bin_index(values, bins):
    """`np.digitize` with the top edge folded into the last bin.

    `digitize` puts a value equal to the last edge *past* the final bin, so a
    plain `digitize(...) - 1` silently drops it. That is invisible for period
    bins but guaranteed for any binning built from `nanmax`, where the largest
    value always sits exactly on the edge -- and for an SNR scan that is the
    most interesting system in the sample.
    """
    return np.clip(np.digitize(np.asarray(values, float), bins) - 1, 0, len(bins) - 2)


def read_systems(population, columns=None):
    """One population's per-system rows, merged file if present else the shards.

    The merged parquet is what `--stages merge` writes; before that runs the
    shards are the only copy, and a diagnostic should not require a merge first.
    """
    merged = C.merged_systems(population)
    source = merged if merged.exists() else C.systems_dir(population)
    if not source.exists():
        raise FileNotFoundError(f"no harv output at {source}")
    return ds.dataset(source, format="parquet").to_table(columns=columns)


def detectability_path(population):
    """Where `project_snr_mpi.py` writes this population's projections."""
    return C.catalog_root() / "detectability" / population


def with_detectability(table, population):
    """Left-join `snr_detectable_k`, `snr_expected_k` and `retained_k` onto a
    per-system table, on `gaia_source_id`.

    Returns the table UNCHANGED when the projection stage has not been run, so
    every figure keeps working on output produced before it existed. Callers
    check for the column rather than assuming it.

    Joined at read time rather than written into the truth columns because the
    projection is a separate stage over an existing catalog. For a future
    catalog it belongs in `simulate_mpi.py`, where the reflex is already in hand
    -- see `scripts/project_snr_mpi.py`.

    An Arrow join rather than a pandas merge so `best_truth` keeps working on
    the result: it reads `f"{name}_{k}"` off a Table, and it is the one place
    that knows how to pick the companion the fit actually matched.
    """
    source = detectability_path(population)
    if not source.exists():
        return table  # the projection stage has not been run; callers fall back
    if "gaia_source_id" not in table.column_names:
        # Not a missing stage -- a caller that built its own column list and
        # left out the join key. Silent fallback here reads as "no projections
        # exist", which sends someone to rerun a stage that already ran.
        import warnings

        warnings.warn(
            f"detectability for {population} exists but the table has no "
            "gaia_source_id to join on; add it to the requested columns",
            RuntimeWarning,
            stacklevel=2,
        )
        return table
    n = C.POPULATIONS[population]
    columns = ["gaia_source_id"] + [
        f"{name}_{k}"
        for name in ("retained", "snr_detectable", "snr_expected")
        for k in range(1, n + 1)
    ]
    projected = ds.dataset(source, format="parquet").to_table(columns=columns)
    return table.join(projected, keys="gaia_source_id", join_type="left outer")


def has_detectability(table, population):
    """Did the join land? False means fall back to `snr_total` and say so."""
    n = C.POPULATIONS[population]
    return not n or f"snr_detectable_{n}" in table.column_names


def system_columns(population, extra=()):
    """The columns a diagnostic needs, for however many companions there are.

    `gaia_source_id` is always included: it is the key `with_detectability`
    joins on, and a caller who forgot it got an Arrow error about a missing
    field reference rather than anything about SNR. Deduplicated, so passing it
    through `extra` as well is harmless.
    """
    n = C.POPULATIONS[population]
    columns = [
        "gaia_source_id",
        "ess",
        "weight_captured",
        "logZ_int",
        "period_best_yr",
        *extra,
    ]
    for k in range(1, n + 1):
        columns += [f"period_{k}", f"ecc_{k}", f"snr_total_{k}"]
    return list(dict.fromkeys(columns))


def period_columns(population):
    """`period_1 .. period_n` for however many companions the population has."""
    return [f"period_{k}" for k in range(1, C.POPULATIONS[population] + 1)]


def high_snr_mask(table, population):
    """Every injected companion clears the SNR floor -- the generator's rule.

    `None` for the companion-free control, which has no high-SNR subset.
    """
    n = C.POPULATIONS[population]
    if n == 0:
        return None
    snr = np.column_stack(
        [np.asarray(table[f"snr_total_{k}"], float) for k in range(1, n + 1)]
    )
    return np.isfinite(snr).all(axis=1) & (snr >= PG.HIGH_SNR_MIN).all(axis=1)


def recovered(table, population):
    """|ln(P_best/P_true)| < ln(tol), against the BEST-matching companion.

    harv fits a *single* companion, so in a two-companion system it can
    legitimately lock onto either orbit. Scoring only against `period_1` marks a
    correct fit on companion 2 as a failure and understates `2_companion`.

    The tolerance is the periodogram stage's `PERIOD_RECOVER_TOL`, imported so
    the two analyses cannot recover to different bars.
    """
    n = C.POPULATIONS[population]
    if n == 0:
        return None
    best = np.asarray(table["period_best_yr"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        offsets = np.column_stack(
            [
                np.abs(np.log(best / np.asarray(table[f"period_{k}"], float)))
                for k in range(1, n + 1)
            ]
        )
    return np.nanmin(offsets, axis=1) < np.log(PG.PERIOD_RECOVER_TOL)


def railed(table):
    """Did the fit collapse to the "no orbit" solution at the prior floor?

    The Thiele-Innes amplitude prior scales as `(P/P0)^(2/3)`, so the shortest
    period in the prior is where an orbit is forced to zero amplitude and the
    model reduces to a five-parameter astrometric fit. A best sample sitting
    there is a **non-detection**, which is a different failure from finding the
    wrong period -- and on the first 300k-system run it was 65% of all misses,
    so a recovery percentage that mixes the two says very little.

    Relative to `PERIOD_MIN_YR`, never an absolute period, so it follows the
    prior when the bounds move.
    """
    return np.asarray(table["period_best_yr"], float) < (
        C.PERIOD_MIN_YR * C.RAIL_FACTOR
    )


def in_search_range(table, population):
    """Could this system have been recovered at all?

    `PERIOD_MIN_YR`/`PERIOD_MAX_YR` are narrower than the injected prior on
    purpose (see `config`), so a system injected outside them is unrecoverable
    by construction rather than by measurement. Counting those as failures
    understates the method and makes the narrowed prior look worse than it is,
    so every recovery number should be quoted over this subset with the
    out-of-range count stated beside it.

    Judged on the matched companion, the same one `recovered()` scores against.
    """
    if C.POPULATIONS[population] == 0:
        return None
    truth = best_truth(table, population, "period")
    return (truth >= C.PERIOD_MIN_YR) & (truth <= C.PERIOD_MAX_YR)


def best_truth(table, population, name):
    """The truth column of whichever companion `recovered()` matched.

    So a figure binning by "the eccentricity of the orbit that was found" agrees
    with the recovery flag rather than always using companion 1.
    """
    n = C.POPULATIONS[population]
    if n == 0:
        return None
    best = np.asarray(table["period_best_yr"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        offsets = np.column_stack(
            [
                np.abs(np.log(best / np.asarray(table[f"period_{k}"], float)))
                for k in range(1, n + 1)
            ]
        )
    which = np.nanargmin(np.nan_to_num(offsets, nan=np.inf), axis=1)
    values = np.column_stack(
        [np.asarray(table[f"{name}_{k}"], float) for k in range(1, n + 1)]
    )
    return values[np.arange(len(which)), which]
