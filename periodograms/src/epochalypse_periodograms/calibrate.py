"""Detection thresholds, calibrated on the companion-free control.

The periodogram Delta-chi^2 is maximized over ~16,600 trial periods, so under
the null it is inflated by the look-elsewhere effect and is *not* chi^2_4
distributed. It is further inflated because the catalog's noise model injects
scatter at the sigma_UEVA scale but reports the smaller sigma_formal, and the
search uses fixed 1/sigma_formal^2 weights, so it over-fits that excess. Neither
effect has a closed form worth trusting at this grid size, so the thresholds are
measured: the quantile of `0_companion` that leaves a `TARGET_FP` false-positive
rate.

Two things make that calibration stronger here than in the 10,000-system
notebook it comes from. The control is the *same 5.7 M stars* as the two
companion populations -- the generator draws all three from the same parent
sample, so every star's parallax, transit count and per-epoch precision appears
in the null and in the signal. And the quantile is measured on 5.7 M systems
rather than 10,000, so the 99.5th percentile is set by ~29,000 systems in the
tail rather than by ~50: the threshold stops being the noisy quantity it is at
notebook scale.

The budget is split evenly between the two independent channels -- the
periodogram peak and the acceleration test -- so each gets `TARGET_FP / 2` and
the union comes out at `TARGET_FP`. The realized rate is measured after the
fact rather than assumed, because the channels are not quite independent.

The thresholds MUST be recalibrated for any change in the grid, the bounds, or
the noise model. The look-elsewhere inflation grows with the trial count, and
with `FIT_JITTER = True` the orbit channel moves by orders of magnitude.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from . import config as C


def thresholds_from_null(top_power, accel_delta_chi2, target_fp=None):
    """`(thr_orbit, thr_accel)` at a total false-positive rate of `target_fp`."""
    target_fp = C.TARGET_FP if target_fp is None else float(target_fp)
    q = 100.0 * (1.0 - target_fp / 2.0)
    return (float(np.nanpercentile(np.asarray(top_power, float), q)),
            float(np.nanpercentile(np.asarray(accel_delta_chi2, float), q)))


def calibrate(population="0_companion", target_fp=None, output_root=None):
    """Read the control population's characterization and derive the thresholds.

    Only two columns are read off disk. At 5.7 M rows that is ~90 MB rather than
    the ~2 GB the whole table would be, which is the difference between this
    running on a login node and not.
    """
    if output_root is not None:
        C.set_output_root(output_root)
    dataset = ds.dataset(C.chars_dir(population), format="parquet")
    table = dataset.to_table(columns=["top_power", "accel_delta_chi2"])
    top = table.column("top_power").to_numpy(zero_copy_only=False)
    accel = table.column("accel_delta_chi2").to_numpy(zero_copy_only=False)

    thr_orbit, thr_accel = thresholds_from_null(top, accel, target_fp)
    peak = top > thr_orbit
    acc = accel > thr_accel
    return {
        "population": population,
        "target_fp": C.TARGET_FP if target_fp is None else float(target_fp),
        "n_null_systems": int(len(top)),
        "thr_orbit": thr_orbit,
        "thr_accel": thr_accel,
        "realized_fp": float((peak | acc).mean()),
        "realized_fp_peak": float(peak.mean()),
        "realized_fp_accel": float(acc.mean()),
        "fit_jitter": C.FIT_JITTER,
    }


def write_calibration(calibration, path=None):
    path = C.calibration_path() if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration, indent=2) + "\n")
    return path


def read_calibration(path=None):
    path = C.calibration_path() if path is None else Path(path)
    return json.loads(Path(path).read_text())


def apply_calibration(frame, thr_orbit=None, thr_accel=None, baseline=None):
    """Add the null-calibrated detection flags every figure keys off, in place.

    Identical to `epochalypse_figures.apply_calibration`, and deliberately kept
    out of the per-system loop: it is four vectorized comparisons on columns
    that are already written, so it costs nothing to apply at read time and
    everything to bake in -- baking it in would mean rewriting 6 GB whenever
    `TARGET_FP` changed.

    `detected_cal` fires if either independent channel clears its threshold.
    `period_reliable_cal` is the stricter data-only flag: detected, unimodal,
    and best period well inside the mission baseline. The two channels are also
    recorded separately, because `peak_significant_cal` is what a "the period
    is localized" claim has to rest on -- `klass == "unimodal"` only says the
    competitive region is narrow, and it is evaluated against the classifier's
    own `DELTA_BIC_DETECT = 10`, far below `thr_orbit`. Without the peak flag a
    system can be called period-localized on the acceleration channel alone.
    """
    if thr_orbit is None or thr_accel is None:
        calibration = read_calibration()
        thr_orbit = calibration["thr_orbit"] if thr_orbit is None else thr_orbit
        thr_accel = calibration["thr_accel"] if thr_accel is None else thr_accel
    baseline = C.DR4_BASELINE_YEARS if baseline is None else baseline

    peak = frame["top_power"].to_numpy(float) > thr_orbit
    accel = frame["accel_delta_chi2"].to_numpy(float) > thr_accel
    frame["peak_significant_cal"] = peak
    frame["accel_significant_cal"] = accel
    frame["detected_cal"] = peak | accel
    frame["period_reliable_cal"] = (
        (peak | accel) & (frame["klass"].to_numpy(object) == "unimodal")
        & (frame["best_period"].to_numpy(float) < baseline))
    return frame


def select_high_snr(frame, snr_min=None):
    """Systems in which *every* injected companion clears SNR_tot >= `snr_min`.

    This is the whole of the "high-SNR population" -- there is no separate run.
    The generator draws all three populations from the unbiased prior and
    records SNR_tot rather than rejecting on it, so the floor is an analysis
    choice applied to rows that were already characterized. A companion whose
    SNR_tot is not finite never clears it.

    One caveat a post-hoc cut carries and rejection sampling did not: stars
    enter the selection weighted by their acceptance probability, so the
    high-SNR populations are drawn from a slightly nearer subset of the parent
    sample (median parallax 3.34 mas against 2.87 for the parent).
    """
    snr_min = C.HIGH_SNR_MIN if snr_min is None else float(snr_min)
    columns = [c for c in ("snr_total_1", "snr_total_2") if c in frame.columns]
    if not columns:
        raise KeyError("no snr_total_* columns to cut on")
    snr = frame[columns].to_numpy(float)
    return frame[np.isfinite(snr).all(axis=1) & (snr >= snr_min).all(axis=1)]


def census(population, thr_orbit=None, thr_accel=None, columns=None, output_root=None):
    """Class counts for one population, and for its high-SNR subset.

    Reads only the columns the classification needs, so this runs over 5.7 M
    systems in seconds and is the cheapest check that a run came out sane.
    """
    if output_root is not None:
        C.set_output_root(output_root)
    needed = ["top_power", "accel_delta_chi2", "klass", "best_period",
              "snr_total_1", "snr_total_2"]
    available = set(ds.dataset(C.chars_dir(population), format="parquet").schema.names)
    frame = ds.dataset(C.chars_dir(population), format="parquet").to_table(
        columns=[c for c in (columns or needed) if c in available]).to_pandas()
    apply_calibration(frame, thr_orbit, thr_accel)

    def counts(f):
        narrow = f["peak_significant_cal"] & (f["klass"] == "unimodal")
        return {"n": int(len(f)),
                "undet": int((~f["detected_cal"]).sum()),
                "narrow": int((f["detected_cal"] & narrow).sum()),
                "broad": int((f["detected_cal"] & ~narrow).sum())}

    out = {"population": population, "all": counts(frame)}
    if "snr_total_1" in frame.columns:
        out["high_snr"] = counts(select_high_snr(frame))
    return out


def load_characterization(population, columns=None, high_snr=False, output_root=None):
    """One population's characterization table as a DataFrame, flags applied.

    `columns` is worth passing: the full table is ~70 columns over 5.7 M rows,
    which is ~3 GB in pandas, and the three paper figures each need under a
    dozen of them. The columns the calibration needs are added automatically.

    With `high_snr=True` the SNR_tot >= `HIGH_SNR_MIN` cut is applied after the
    read -- the high-SNR population is a row selection of this same table, not
    a separate run.
    """
    if output_root is not None:
        C.set_output_root(output_root)
    dataset = ds.dataset(C.chars_dir(population), format="parquet")
    if columns is not None:
        needed = {"top_power", "accel_delta_chi2", "klass", "best_period"}
        if high_snr:
            needed |= {"snr_total_1", "snr_total_2"}
        columns = [c for c in dict.fromkeys(list(columns) + sorted(needed))
                   if c in set(dataset.schema.names)]
    frame = dataset.to_table(columns=columns).to_pandas()
    if high_snr:
        frame = select_high_snr(frame)
    return apply_calibration(frame.reset_index(drop=True))


def merged_path(population, high_snr=False):
    suffix = "_high_snr" if high_snr else ""
    return C.OUTPUT_ROOT / f"characterization_{population}{suffix}.parquet"


def merge(population, high_snr=False, columns=None, output_root=None):
    """Write one population's shards out as a single parquet file.

    Optional -- every reader in this package takes the sharded directory
    directly, and pyarrow reads a directory of parquet as one dataset. It is
    here because a single 6 GB file is easier to copy off a cluster than 320
    files, and because the high-SNR views are small enough (400 k and 17 k rows)
    to be worth materializing.
    """
    if output_root is not None:
        C.set_output_root(output_root)
    frame = load_characterization(population, columns=columns, high_snr=high_snr)
    path = merged_path(population, high_snr)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, index=False, compression=C.PARQUET_COMPRESSION)
    tmp.replace(path)
    return path, len(frame)
