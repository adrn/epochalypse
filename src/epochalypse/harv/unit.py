"""One work unit: fit every system in one shard (or one part of one).

This is the whole of the compute. `scripts/harv_mpi.py` is a loop over
`run_unit` and a `gather` at the end. It lives in the package rather than in a
script because the MPI driver and the tests both call it.

Per system: read its epochs, pad to a bucket, build the data-floored
parameterization, and evaluate the shared prior library against it, keeping the
`TOP_K` samples with the largest importance weight. Nothing here communicates
with any other system or any other rank.
"""

from __future__ import annotations

import time
import warnings

import jax
import numpy as np

from ..periodogram.shards import ShardReader
from ..planets import system_seed
from . import adapt
from . import config as C
from . import library as L
from .writers import SampleWriter, SystemWriter

# x64 is not cosmetic here either: harv's own docs (`sharp-bits.md`) are explicit
# that float32 makes the marginalized likelihoods -- and so the importance
# weights -- unstable. Must run before any jax array is made.
jax.config.update("jax_enable_x64", True)

# Two harv warnings that are correct and would fire 17.2 million times. The
# first is a statement about the prior, identical for every system: `parallax`
# is HalfNormal, so it is sampled rather than marginalized. The second says the
# posterior was not resolved by the library -- which is true for every strongly
# detected system, is exactly what the `ess` column reports, and is the thing
# this pipeline measures rather than an anomaly it should announce.
warnings.filterwarnings(
    "ignore", message="Non-Gaussian linear prior", category=UserWarning
)
warnings.filterwarnings(
    "ignore", message="Under-resolved rejection run", category=UserWarning
)


def fit_system(
    t,
    psi,
    pf,
    y,
    yerr,
    *,
    prior_samples,
    seed,
    prior=None,
    m_star_msun=None,
    top_k=None,
):
    """Fit one system. Returns `(record_without_ids, {param: (top_k,) array})`.

    The sampler is rebuilt per system because two of its pieces are: the
    parameterization's `a_floor` is `med(sigma_AL)/sqrt(N)` from this star's own
    uncertainties, and the prior's `sigma_a0` is the companion-mass ceiling
    scaled by this star's mass. Neither costs a JIT compile, and neither touches
    the *library*, which stays the catalog-wide one -- that is what makes 17.2 M
    posteriors comparable.

    Pass `prior` to override, which is what a sweep over a constant `sigma_a0`
    does; otherwise it is built here from `m_star_msun`.
    """
    from harv import RejectionSampler

    top_k = C.TOP_K if top_k is None else int(top_k)
    data, par, n_epochs = adapt.prepare(t, psi, pf, y, yerr)
    n_padded = len(data.time)

    if prior is None:
        prior = L.prior(m_star_msun=m_star_msun)
    model = L.model(par)
    sampler = RejectionSampler(prior, model, batch_size=C.BATCH_SIZE)
    samples = sampler.run_with_samples(data, prior_samples, top_k=top_k, seed=seed)

    # Padding adds a constant to every log-likelihood. It cancels in the weights
    # and the ESS; it does not cancel in these two, which are absolute.
    offset = adapt.pad_log_offset(n_epochs, n_padded)
    meta = samples.metadata

    columns = {
        name: np.asarray(q.value, dtype=np.float64)
        for name, q in {**samples.nonlinear, **samples.linear}.items()
    }
    columns["ln_likelihood"] = np.asarray(samples.ln_likelihood, dtype=np.float64)
    columns["ln_prior"] = np.asarray(samples.ln_prior, dtype=np.float64)
    # Read once, here: `samples.weight` is a derived property that needs the
    # metadata, and it is NOT stored -- see `library.SAMPLE_UNITS`. Every summary
    # below is computed on this float64 array, before the float32 storage cast.
    weight = np.asarray(samples.weight, dtype=np.float64)

    record = {
        "n_epochs": n_epochs,
        "n_padded": n_padded,
        "ess": float(meta["logZ_int_ess"]),
        "weight_captured": float(meta["weight_captured"]),
        "logZ_int": float(meta["logZ_int"]) - offset,
        "logZ_int_mcse": float(meta["logZ_int_mcse"]),
        "max_log_likelihood": float(meta["max_log_likelihood"]) - offset,
        "n_prior_samples": int(meta["n_prior_samples"]),
        "top_k": top_k,
        "seed": int(seed),
        "t_ref_yr": float(meta["t_ref"]),
    }
    record.update(_sample_summary(columns, weight))
    record.update(_fit_quality(model, data, columns, weight))
    return record, columns


# Every per-system summary derived from the draws, so a NaN row has the same
# columns as a good one and the parquet schema cannot depend on which system a
# writer happened to see first.
SUMMARY_COLUMNS = (
    "period_best_yr",
    "period_wmean_yr",
    "period_wstd_yr",
    "ecc_best",
    "a0_wmean_mas",
    "a0_wstd_mas",
    "parallax_wmean_mas",
    "weight_railed",
)


def _weighted(values, weight, total):
    """`(mean, std)` of `values` under `weight`, renormalized by `total`."""
    mean = float((weight * values).sum() / total)
    var = float((weight * (values - mean) ** 2).sum() / total)
    return mean, float(np.sqrt(max(var, 0.0)))


def _sample_summary(columns, weight):
    """Point estimates from the weighted draws, before the storage cast.

    `weight` sums to `weight_captured`, not to 1, so every average here
    renormalizes. An analysis that treats the stored draws as equal-weight is
    wrong, and this is the one place in the pipeline that already knows it.

    Amplitude and parallax are summarized by their weighted **mean**, not by the
    best draw's value. harv draws the analytically marginalized linear
    parameters from their conditional Gaussian rather than returning its mean,
    so any single row carries conditional-sampling scatter on top of the
    posterior width. Period and eccentricity are genuine prior draws, so the
    best-weight row is meaningful for those.

    `weight_railed` is the posterior mass sitting at the "no orbit" solution.
    It is the continuous form of `census.railed`, which reads the single best
    draw and so cannot tell a marginal detection from a decisive non-detection.
    """
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0.0:
        return dict.fromkeys(SUMMARY_COLUMNS, np.nan)

    period = columns["period"]
    best = int(np.argmax(weight))
    p_mean, p_std = _weighted(period, weight, total)
    a0 = L.semi_major_axis_mas(*(columns[f"ti_{c}"] for c in "ABFG"))
    a0_mean, a0_std = _weighted(a0, weight, total)
    plx_mean, _ = _weighted(columns["parallax"], weight, total)
    railed = period < C.PERIOD_MIN_YR * C.RAIL_FACTOR
    return {
        "period_best_yr": float(period[best]),
        "period_wmean_yr": p_mean,
        "period_wstd_yr": p_std,
        "ecc_best": float(columns["eccentricity"][best]),
        "a0_wmean_mas": a0_mean,
        "a0_wstd_mas": a0_std,
        "parallax_wmean_mas": plx_mean,
        "weight_railed": float(weight[railed].sum() / total),
    }


def _fit_quality(model, data, columns, weight):
    """`chi2` of the best draw and of the no-orbit fit it has to beat.

    Their difference is the detection statistic this stage otherwise never
    computes: `logZ_int` integrates over the whole prior *including* the null
    region, so it cannot say on its own whether the orbit term bought anything.
    Two nested least-squares fits over ~100 rows, against a ~25 s library sweep.

    BOTH sides are refit. The best draw's nonlinear parameters fix the orbit's
    shape, but its stored linear parameters are a conditional *draw*, not an
    optimum -- scoring those against a least-squares null is not a likelihood
    ratio and can come out negative. See `adapt.linear_solution`.

    Not an evidence ratio -- there is no Occam penalty in it. A railed system
    with a large `chi2_null - chi2_best` is one the amplitude prior rejected
    despite a real improvement in fit, which is precisely the failure mode
    `config.M_MAX_MJUP` exists to control.
    """
    if not np.isfinite(weight).any() or float(weight.sum()) <= 0.0:
        return {"chi2_best": np.nan, "chi2_null": np.nan}
    best = int(np.argmax(weight))
    design = L.design_matrix(
        model,
        data,
        columns["period"][best],
        columns["eccentricity"][best],
        columns["phase_peri"][best],
    )
    al = np.asarray(data.al_position.value, dtype=np.float64)
    err = np.asarray(data.al_position_err.value, dtype=np.float64)
    return {
        "chi2_best": adapt.linear_solution(design, al, err)[1],
        "chi2_null": adapt.linear_solution(design, al, err, 5)[1],
    }


def passes_snr(truth, population, min_snr):
    """Every injected companion clears `min_snr` -- the generator's own rule.

    The same "all companions, not any" test as `sources.select_high_snr` and
    `census.high_snr_mask`, applied one truth row at a time so a unit can skip a
    system before paying 25 s to fit it.

    A control system has no companion and so no SNR, and returns False rather
    than True: `--min-snr` means "spend the budget on systems with signal", and
    silently fitting the whole control population would be the opposite.
    """
    n = C.POPULATIONS[population]
    if not n:
        return False
    return all(
        np.isfinite(truth[f"snr_total_{k}"]) and truth[f"snr_total_{k}"] >= min_snr
        for k in range(1, n + 1)
    )


def unit_costs(units, min_snr=None):
    """Relative cost of every work unit, planned before any fitting starts.

    Time per system is close to linear in the PADDED epoch count -- that is what
    `EPOCH_BUCKETS` trades against compile count -- so a unit's cost is the sum
    of `bucket_for(n_transits)` over the systems it will actually fit. The truth
    tables already carry `n_transits_dr4`, so this needs no measurement and no
    trial run.

    Worth the trouble because the spread is large and does not average out:
    measured 29.5 to 79.1 minutes for units of the same 105 systems, with only
    ~2 units per rank. Feed the result to `mpi.balance`.

    Returns a list parallel to `units`. Reads one column from each truth table,
    so call it on one rank and `mpi.broadcast` the result.
    """
    import pyarrow.parquet as pq

    from ..periodogram import config as PG

    cache, costs = {}, []
    for population, shard, n_shards, part, n_parts in units:
        key = (population, shard)
        if key not in cache:
            n = C.POPULATIONS[population]
            columns = ["n_transits_dr4"] + (
                [f"snr_total_{k}" for k in range(1, n + 1)] if min_snr and n else []
            )
            table = pq.read_table(
                PG.shard_truths(population, shard, n_shards), columns=columns
            )
            transits = np.asarray(table["n_transits_dr4"], dtype=np.int64)
            keep = np.ones(len(transits), dtype=bool)
            if min_snr and n:
                snr = np.column_stack(
                    [
                        np.asarray(table[f"snr_total_{k}"], float)
                        for k in range(1, n + 1)
                    ]
                )
                keep = np.isfinite(snr).all(axis=1) & (snr >= min_snr).all(axis=1)
            cache[key] = (transits, keep)
        transits, keep = cache[key]

        # the same [lo, hi) split ShardReader.iter_systems walks, then the cap
        total = len(transits)
        lo = part * total // n_parts
        hi = (part + 1) * total // n_parts
        rows = np.arange(lo, hi)[keep[lo:hi]]
        limit = C.limit_per_shard(n_shards * n_parts)
        if limit:
            rows = rows[:limit]
        costs.append(float(sum(C.bucket_for(int(transits[r])) for r in rows)))
    return costs


def run_unit(
    population,
    shard,
    n_shards,
    part=0,
    n_parts=1,
    *,
    prior_samples=None,
    top_k=None,
    limit=None,
    min_snr=None,
    skip_existing=False,
    progress_every=50,
    verbose=True,
):
    """Fit every system in one work unit; write its two parquet files.

    Returns a summary dict. A system that raises is recorded and skipped rather
    than taken as fatal: one unusable star must not cost a rank its shard, and
    at 17 M systems a per-system exception that happens once in a million still
    happens seventeen times.
    """
    systems_path = C.systems_shard(population, shard, n_shards, part, n_parts)
    samples_path = C.samples_shard(population, shard, n_shards, part, n_parts)

    if skip_existing and systems_path.exists():
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

    # SUBSAMPLE is a budget per population; a unit is one of n_shards x n_parts
    # of them, so each takes that share. Explicit `limit` wins.
    if limit is None:
        limit = C.limit_per_shard(n_shards * n_parts)
    if prior_samples is None:
        prior_samples = L.draw()
    # Only when a constant is pinned (a sweep) is one prior good for every
    # system; otherwise it is built per system from the host mass.
    prior = L.prior() if C.SIGMA_A0_AU is not None else None

    started = time.time()
    failures = []
    with ShardReader(population, shard, n_shards) as reader:
        n_unit = reader.n_systems(part, n_parts)
        with (
            SystemWriter(systems_path, population, shard, reader.truths) as systems,
            SampleWriter(samples_path, top_k) as samples,
        ):
            # `limit` counts systems FITTED, not scanned: with --min-snr the two
            # differ by ~15x, and a cap on scanned systems would silently return
            # a fraction of what was asked for.
            count = 0
            for index, truth, t, psi, pf, y, yerr in reader.iter_systems(part, n_parts):
                if min_snr is not None and not passes_snr(truth, population, min_snr):
                    continue
                if limit and count >= limit:
                    break
                count += 1
                gaia_source_id = truth["gaia_source_id"]
                try:
                    record, columns = fit_system(
                        t,
                        psi,
                        pf,
                        y,
                        yerr,
                        prior=prior,
                        prior_samples=prior_samples,
                        seed=system_seed(C.SEED, population, gaia_source_id),
                        m_star_msun=truth["mass_st_msun"],
                        top_k=top_k,
                    )
                except Exception as error:
                    failures.append(
                        {
                            "population": population,
                            "shard": shard,
                            "shard_row": index,
                            "gaia_source_id": gaia_source_id,
                            "reason": repr(error),
                        }
                    )
                    continue
                systems.add(index, record)
                samples.add(gaia_source_id, index, columns)
                if verbose and progress_every and count % progress_every == 0:
                    rate = count / (time.time() - started)
                    print(
                        f"[{population} {shard:05d}.{part}] {count:,}/{n_unit:,} "
                        f"({rate:.2f}/s)",
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
        "n_systems": systems.n_systems,
        "n_failed": len(failures),
        "skipped": False,
        "seconds": elapsed,
    }
    if verbose:
        rate = systems.n_systems / elapsed if elapsed else 0.0
        print(
            f"[{population} {shard:05d}.{part}] {systems.n_systems:>7,} systems in "
            f"{elapsed / 60:6.1f} min ({rate:5.2f}/s)"
            + (f", {len(failures)} FAILED" if failures else ""),
            flush=True,
        )
    return summary
