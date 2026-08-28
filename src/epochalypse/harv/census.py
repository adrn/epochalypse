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


def system_columns(population, extra=()):
    """The columns a diagnostic needs, for however many companions there are."""
    n = C.POPULATIONS[population]
    columns = ["ess", "weight_captured", "logZ_int", "period_best_yr", *extra]
    for k in range(1, n + 1):
        columns += [f"period_{k}", f"ecc_{k}", f"snr_total_{k}"]
    return columns


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
