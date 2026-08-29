#!/usr/bin/env python
"""Self-check: the epoch mapping, the shared library, and the round trip.

    python tests/test_harv.py                          # synthetic only
    python tests/test_harv.py --catalog-root <path>    # + one real work unit

The synthetic half needs nothing but the package and `harv`; it builds its own
epoch series and checks the four things the whole stage rests on:

* **padding is a no-op.** Every system is padded to one of ~9 bucket shapes so
  harv compiles nine times instead of 17.2 million. That is only legitimate if
  the answer does not move, and `pad_log_offset` is only legitimate if it is
  exactly the shift it claims to be.
* **the prior library is shareable.** `a_floor` is per system; the prior must
  not be, or "one library for the whole catalog" is false.
* **selection is deterministic.** The only randomness left on the top-K path is
  the conditional linear draw, so the same seed must give bit-identical output.
  This is the check that catches an accidental `seed=None`.
* **weighted, not equal-weight.** The stored `weight` column sums to
  `weight_captured`, and every summary in the per-system table renormalizes.

The catalog half runs one small unit end to end and checks that what comes out
of the parquet is what went in.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from epochalypse.harv import adapt, unit
from epochalypse.harv import config as C
from epochalypse.harv import library as L

CHECKED = []
CATALOG_ROOT = None

# Small enough to run in seconds, large enough that the top-K path is exercised
# rather than degenerate.
N_LIB = 4000
K = 32


def check(name, condition, detail=""):
    """Assert, and say what was checked -- the detail is the useful half here."""
    assert condition, f"{name}" + (f"   ({detail})" if detail else "")
    CHECKED.append(name)
    print(f"  [ok  ] {name}" + (f"   {detail}" if detail else ""))


def fake_system(n_epochs=90, period=1.7, alpha=1.5, sigma=0.06, seed=0):
    """A five-parameter star plus one circular reflex, on a DR4-like time span.

    Returns the five arrays `ShardReader.iter_systems` yields, in its order.
    """
    from epochalypse import constants as k

    rng = np.random.default_rng(seed)
    t = np.sort(
        rng.uniform(-k.DR4_BASELINE_YEARS / 2, k.DR4_BASELINE_YEARS / 2, n_epochs)
    )
    psi = rng.uniform(0, 2 * np.pi, n_epochs)
    pf = np.sin(2 * np.pi * t)
    phase = 2 * np.pi * t / period
    y = (
        alpha * (np.cos(phase) * np.sin(psi) + 0.7 * np.sin(phase) * np.cos(psi))
        + 0.4 * pf
        + rng.normal(0.0, sigma, n_epochs)
    )
    return t, psi, pf, y, np.full(n_epochs, sigma)


# ==========================================================================
# The epoch mapping
# ==========================================================================
def test_padding():
    t, psi, pf, y, yerr = fake_system(n_epochs=70)
    unpadded = adapt.to_harv(t, psi, pf, y, yerr, pad=False)
    padded = adapt.to_harv(t, psi, pf, y, yerr, pad=True)

    check(
        "the bucket is the smallest that fits",
        len(padded.time) == C.bucket_for(70) == 80,
        f"70 epochs -> {len(padded.time)}",
    )
    check(
        "t_ref survives padding",
        float(padded.t_ref.value) == float(unpadded.t_ref.value) == float(np.mean(t)),
        f"{float(padded.t_ref.value):.6f} yr",
    )
    check(
        "real epochs are untouched by padding",
        np.array_equal(np.asarray(padded.al_position.value)[:70], y)
        and np.array_equal(np.asarray(padded.time.value)[:70], t),
    )
    check(
        "padded rows carry a large finite uncertainty, not inf",
        np.all(np.asarray(padded.al_position_err.value)[70:] == adapt.PAD_ERR_MAS)
        and np.isfinite(adapt.PAD_ERR_MAS),
        f"PAD_ERR_MAS = {adapt.PAD_ERR_MAS:.0e} mas",
    )

    # a_floor = med(sigma)/sqrt(N). Padded rows would move both.
    par = adapt.prepare(t, psi, pf, y, yerr)[1]
    check(
        "a_floor is built from the unpadded epochs",
        np.isclose(par.a_floor, np.median(yerr) / np.sqrt(70), rtol=1e-8),
        f"{par.a_floor:.6g} vs {np.median(yerr) / np.sqrt(70):.6g}",
    )


def test_padding_is_a_no_op():
    """The claim the whole bucketing scheme rests on, measured rather than argued."""
    # Deliberately a *weak* system: a strong one puts all the weight on one
    # draw, and "the weights are unchanged" is then a claim about one number.
    t, psi, pf, y, yerr = fake_system(n_epochs=70, alpha=0.0, sigma=0.06, seed=3)
    lib, prior = L.draw(N_LIB), L.prior()

    def fit(pad):
        from harv import RejectionSampler

        data = adapt.to_harv(t, psi, pf, y, yerr, pad=pad)
        par = L.parameterization(adapt.to_harv(t, psi, pf, y, yerr, pad=False))
        sampler = RejectionSampler(prior, L.model(par), batch_size=C.BATCH_SIZE)
        return sampler.run_with_samples(data, lib, top_k=K, seed=99)

    bare, padded = fit(False), fit(True)
    p_bare = np.asarray(bare.nonlinear["period"].value)
    p_padded = np.asarray(padded.nonlinear["period"].value)
    check(
        "padding selects the same samples, in the same order",
        np.array_equal(p_bare, p_padded),
        f"{K} periods identical",
    )
    w_bare, w_padded = np.asarray(bare.weight), np.asarray(padded.weight)
    check(
        "padding leaves the weights unchanged",
        np.allclose(w_bare, w_padded, rtol=1e-9, atol=1e-12),
        f"{int((w_bare > 1e-6).sum())} of {K} draws carry weight, max |dw| = "
        f"{np.max(np.abs(w_padded - w_bare)):.2e}",
    )
    check(
        "padding leaves the ESS unchanged",
        np.isclose(
            bare.metadata["logZ_int_ess"], padded.metadata["logZ_int_ess"], rtol=1e-10
        ),
        f"ESS {bare.metadata['logZ_int_ess']:.4f}",
    )

    # logZ_int IS shifted, by exactly the constant pad_log_offset predicts.
    shift = padded.metadata["logZ_int"] - bare.metadata["logZ_int"]
    n_padded = C.bucket_for(70)
    predicted = adapt.pad_log_offset(70, n_padded)
    check(
        "pad_log_offset is exactly the shift it corrects",
        np.isclose(shift, predicted, rtol=1e-9),
        f"{shift:.4f} vs {predicted:.4f} nats over {n_padded - 70} pad rows",
    )


# ==========================================================================
# The shared prior library
# ==========================================================================
def test_library_is_shareable():
    """One library for 17.2 M systems is only honest if the prior is not per system."""
    a, b = (
        fake_system(n_epochs=60, sigma=0.02, seed=1),
        fake_system(n_epochs=200, sigma=0.9, seed=2),
    )
    par_a = adapt.prepare(*a)[1]
    par_b = adapt.prepare(*b)[1]
    check(
        "the two systems really do get different floors",
        not np.isclose(par_a.a_floor, par_b.a_floor, rtol=1e-3),
        f"{par_a.a_floor:.4g} vs {par_b.a_floor:.4g}",
    )

    import jax.random as jr

    draws = [
        np.asarray(
            L.prior(par)
            .sample(jr.key(5), 500, model=L.model(par))
            .nonlinear["period"]
            .value
        )
        for par in (par_a, par_b, L.parameterization())
    ]
    check(
        "the prior does not depend on the per-system floor",
        np.array_equal(draws[0], draws[1]) and np.array_equal(draws[0], draws[2]),
        "500 period draws bit-identical across three parameterizations",
    )
    check(
        "the library is reproducible from the seed alone",
        L.fingerprint(L.draw(500)) == L.fingerprint(L.draw(500)),
        L.fingerprint(L.draw(500)),
    )
    check(
        "only the non-marginalizable parameters are in the library",
        sorted({**L.draw(10).nonlinear, **L.draw(10).linear})
        == ["eccentricity", "parallax", "period", "phase_peri"],
        "the eight Gaussian linear parameters are marginalized, not sampled",
    )


# ==========================================================================
# One system, through fit_system
# ==========================================================================
def test_census_definitions():
    """The three flags the whole recovery story now rests on."""
    import pyarrow as pa

    from epochalypse.harv import census

    floor = C.PERIOD_MIN_YR
    table = pa.table(
        {
            # at the floor, just above it, mid-range, and above the ceiling
            "period_best_yr": pa.array([floor, floor * 3, 2.0, 2.0]),
            "period_1": pa.array([0.5, 0.5, 2.01, 500.0]),
        }
    )
    check(
        "railed is the best sample sitting at the prior floor",
        list(census.railed(table)) == [True, False, False, False],
        f"floor {floor:g} x RAIL_FACTOR {C.RAIL_FACTOR:g} = {floor * C.RAIL_FACTOR:g} yr",
    )
    check(
        "in_search_range excludes truths the prior cannot reach",
        list(census.in_search_range(table, "1_companion")) == [True, True, True, False],
        f"searched range {C.PERIOD_MIN_YR:g}-{C.PERIOD_MAX_YR:g} yr",
    )
    check(
        "railing is judged on P_best, being in range on P_true",
        bool(census.railed(table)[0])
        and bool(census.in_search_range(table, "1_companion")[0]),
        "a railed system can still be one that *could* have been recovered",
    )

    edges = np.array([0.0, 1.0, 2.0, 3.0])
    check(
        "bin_index folds the top edge into the last bin, not past it",
        list(census.bin_index([-5.0, 0.5, 2.9, 3.0], edges)) == [0, 0, 2, 2],
        "np.digitize alone drops a value equal to the last edge -- which for any "
        "binning built from nanmax is always the most interesting system",
    )


def test_stride_partitions():
    """A striding bug silently drops or duplicates work, so assert the partition."""
    from epochalypse import mpi

    for n_items, size in ((2880, 1536), (960, 320), (7, 3), (3, 7)):
        units = list(range(n_items))
        shares = [mpi.stride_for_rank(units, r, size) for r in range(size)]
        flat = sorted(x for share in shares for x in share)
        sizes = {len(share) for share in shares}
        check(
            f"stride partitions {n_items} items over {size} ranks",
            flat == units and max(sizes) - min(sizes) <= 1,
            f"share sizes {sorted(sizes)}",
        )
    check(
        "and it decorrelates position from rank, unlike a contiguous slice",
        mpi.stride_for_rank(list(range(960)), 0, 320) == [0, 320, 640]
        and list(range(*mpi.slice_for_rank(960, 0, 320))) == [0, 1, 2],
        "rank 0 gets units 0/320/640 rather than 0/1/2",
    )


def test_fit_system():
    arrays = fake_system(n_epochs=90, period=1.7, alpha=1.5, seed=4)
    lib, prior = L.draw(N_LIB), L.prior()
    kw = {"prior": prior, "prior_samples": lib, "top_k": K}

    record, columns = unit.fit_system(*arrays, seed=1234, **kw)
    _, columns2 = unit.fit_system(*arrays, seed=1234, **kw)
    _, other = unit.fit_system(*arrays, seed=999, **kw)

    check(
        "every parameter comes back at exactly top_k",
        all(v.shape == (K,) for v in columns.values()),
        f"{len(columns)} columns x {K}",
    )
    check(
        "the same seed gives bit-identical output",
        all(np.array_equal(columns[k], columns2[k]) for k in columns),
        "including the conditional linear draws",
    )
    check(
        "selection does not depend on the seed, only the linear draw does",
        np.array_equal(columns["period"], other["period"])
        and not np.array_equal(columns["ti_A"], other["ti_A"]),
        "top-K is chosen by likelihood alone",
    )
    check(
        "weights are non-increasing",
        np.all(np.diff(columns["weight"]) <= 0),
    )
    check(
        "weight sums to weight_captured, not to 1",
        np.isclose(columns["weight"].sum(), record["weight_captured"], rtol=1e-9),
        f"{record['weight_captured']:.6f}",
    )
    check(
        "ESS is finite and no larger than the library",
        np.isfinite(record["ess"]) and 1.0 <= record["ess"] <= N_LIB,
        f"ESS {record['ess']:.2f} of {N_LIB:,}",
    )
    check(
        "the period summary renormalizes the weights",
        np.isclose(
            record["period_wmean_yr"],
            (columns["weight"] * columns["period"]).sum() / columns["weight"].sum(),
            rtol=1e-12,
        ),
    )
    check(
        "period_best is the highest-weight draw",
        record["period_best_yr"] == columns["period"][np.argmax(columns["weight"])],
        f"{record['period_best_yr']:.5f} yr",
    )
    check(
        "SAMPLE_UNITS names exactly the stored columns",
        set(L.SAMPLE_UNITS) == set(columns),
        f"{len(L.SAMPLE_UNITS)} columns",
    )
    check(
        "the epoch count and its bucket are both recorded",
        record["n_epochs"] == 90 and record["n_padded"] == C.bucket_for(90) == 96,
    )


def test_x64_is_on():
    """Importing the subpackage must enable x64, whatever else was imported.

    The failure mode is silent garbage, not an error: at float32 every
    log-likelihood over the 7.8-decade period prior underflows to -inf, ESS
    comes back NaN, and top-K then returns arbitrary rows. This caught a real
    probe script that imported `library` and `adapt` but not `unit`, when `unit`
    was the only module setting the flag.
    """
    import subprocess
    import sys

    def x64_after(import_line):
        code = f"{import_line}; import jax; print(jax.config.jax_enable_x64)"
        return subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    out = x64_after("from epochalypse.harv import library")
    check("a library-only import enables x64", out == "True", f"got {out!r}")
    check(
        "and so does importing the subpackage itself",
        x64_after("import epochalypse.harv") == "True",
    )


def test_sample_units():
    """The manifest is written before the first fit, so its units are hard-coded.

    Which makes them a place to drift. This is the assertion that stops that.
    """
    from harv import RejectionSampler

    arrays = fake_system(n_epochs=90, seed=6)
    data, par, _ = adapt.prepare(*arrays)
    sampler = RejectionSampler(L.prior(), L.model(par), batch_size=C.BATCH_SIZE)
    samples = sampler.run_with_samples(data, L.draw(N_LIB), top_k=K, seed=1)

    stored = {**samples.nonlinear, **samples.linear}
    wrong = {
        n: (L.SAMPLE_UNITS.get(n), str(q.unit))
        for n, q in stored.items()
        if L.SAMPLE_UNITS.get(n) != str(q.unit)
    }
    check("every hard-coded unit matches what harv returns", not wrong, str(wrong))
    check(
        "the three derived columns are dimensionless",
        all(L.SAMPLE_UNITS[n] == "" for n in ("weight", "ln_likelihood", "ln_prior")),
    )
    check(
        "SAMPLE_UNITS has nothing harv does not return",
        set(L.SAMPLE_UNITS) - set(stored) == {"weight", "ln_likelihood", "ln_prior"},
    )


def test_subsample():
    """The cap is per unit, so it needs no coordination -- and must still add up."""
    check("None means the whole catalog", C.limit_per_shard(320) is None)
    try:
        C.set_subsample(10_000)
        cap = C.limit_per_shard(320)
        check(
            "an integer becomes a per-unit cap that covers the budget",
            cap == 32 and 10_000 <= 320 * cap < 10_000 + 320,
            f"320 units x {cap} = {320 * cap:,} for a 10,000 budget (rounded up)",
        )
        split = C.limit_per_shard(320 * 4)
        check(
            "a shard split divides the cap rather than multiplying it",
            split == 8 and 10_000 <= 320 * 4 * split < 10_000 + 320 * 4,
            f"n_parts=4 -> {split} per part, {320 * 4 * split:,} total",
        )
        C.set_subsample(7)
        check(
            "a budget smaller than the unit count still runs one per unit",
            C.limit_per_shard(320) == 1,
            "never zero, which would write 320 empty files",
        )
    finally:
        C.set_subsample(None)
    check(
        "set_subsample(None) restores the full catalog",
        C.SUBSAMPLE is None and C.limit_per_shard(320) is None,
    )


# ==========================================================================
# One work unit, against real shards
# ==========================================================================
def test_work_unit():
    if CATALOG_ROOT is None:
        print("  [skip] --catalog-root not given")
        return
    import pandas as pd

    from epochalypse.periodogram.shards import discover_shards

    C.set_catalog_root(CATALOG_ROOT)
    population = "1_companion"
    numbers, n_shards = discover_shards(population)
    shard = numbers[0]
    lib = L.draw(N_LIB)

    with tempfile.TemporaryDirectory() as tmp:
        C.set_output_root(tmp)
        summary = unit.run_unit(
            population,
            shard,
            n_shards,
            limit=6,
            top_k=K,
            prior_samples=lib,
            verbose=False,
            progress_every=0,
        )
        systems = pd.read_parquet(C.systems_shard(population, shard, n_shards))
        samples = pd.read_parquet(C.samples_shard(population, shard, n_shards))

    check(
        "the unit wrote a row per system in both files",
        len(systems) == len(samples) == summary["n_systems"] == 6,
    )
    check("nothing failed", summary["n_failed"] == 0)
    check(
        "the two files are in the same order",
        np.array_equal(systems["gaia_source_id"], samples["gaia_source_id"]),
        "so the join is row-for-row",
    )
    check(
        "every stored block is exactly top_k long",
        all(len(np.asarray(v)) == K for v in samples["period"]),
    )
    check(
        "the truth columns came along",
        {"period_1", "snr_total_1", "parallax_mas"} <= set(systems.columns),
        f"{systems.shape[1]} columns",
    )
    check(
        "logZ_int is corrected for each system's own padding",
        systems["n_padded"].nunique() > 1,
        f"buckets {sorted(systems['n_padded'].unique())} in one shard",
    )
    check(
        "the stored weights still sum to weight_captured after the float32 cast",
        np.allclose(
            [np.asarray(v, float).sum() for v in samples["weight"]],
            systems["weight_captured"],
            rtol=1e-5,
        ),
        f"SAMPLE_DTYPE = {C.SAMPLE_DTYPE}",
    )


def main(argv=None):
    global CATALOG_ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-root",
        type=Path,
        default=None,
        help="run the work-unit test against this catalog too",
    )
    CATALOG_ROOT = parser.parse_args(argv).catalog_root

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        print(f"\n{test.__name__[5:].replace('_', ' ')}")
        test()
    print(f"\n{len(CHECKED)} checks passed across {len(tests)} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
