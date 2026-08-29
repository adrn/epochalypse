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
    sampler = RejectionSampler(prior, L.model(par), batch_size=C.BATCH_SIZE)
    samples = sampler.run_with_samples(data, prior_samples, top_k=top_k, seed=seed)

    # Padding adds a constant to every log-likelihood. It cancels in the weights
    # and the ESS; it does not cancel in these two, which are absolute.
    offset = adapt.pad_log_offset(n_epochs, n_padded)
    meta = samples.metadata

    columns = {
        name: np.asarray(q.value, dtype=np.float64)
        for name, q in {**samples.nonlinear, **samples.linear}.items()
    }
    weight = np.asarray(samples.weight, dtype=np.float64)
    columns["weight"] = weight
    columns["ln_likelihood"] = np.asarray(samples.ln_likelihood, dtype=np.float64)
    columns["ln_prior"] = np.asarray(samples.ln_prior, dtype=np.float64)

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
    record.update(_period_summary(columns["period"], weight))
    return record, columns


def _period_summary(period, weight):
    """Three point estimates, on the float64 draws before the storage cast.

    `weight` sums to `weight_captured`, not to 1, so every average here
    renormalizes. An analysis that treats the stored draws as equal-weight is
    wrong, and this is the one place in the pipeline that already knows it.
    """
    total = float(weight.sum())
    if not np.isfinite(total) or total <= 0.0:
        return {
            "period_best_yr": np.nan,
            "period_wmean_yr": np.nan,
            "period_wstd_yr": np.nan,
        }
    mean = float((weight * period).sum() / total)
    var = float((weight * (period - mean) ** 2).sum() / total)
    return {
        "period_best_yr": float(period[int(np.argmax(weight))]),
        "period_wmean_yr": mean,
        "period_wstd_yr": float(np.sqrt(max(var, 0.0))),
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
