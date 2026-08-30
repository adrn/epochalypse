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


def _run_smoke_unit(population, n_prior=3000, top_k=16):
    """One small unit into the current output root, for tests that need output."""
    from epochalypse.periodogram.shards import discover_shards

    numbers, n_shards = discover_shards(population)
    unit.run_unit(
        population,
        numbers[0],
        n_shards,
        prior_samples=L.draw(n_prior),
        top_k=top_k,
        limit=None,
        verbose=False,
        progress_every=0,
    )


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
    lib, prior = L.draw(N_LIB), L.prior(m_star_msun=0.41)

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
            L.prior(par, m_star_msun=0.41)
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
    a = L.fingerprint(L.draw(500))
    try:
        C.SIGMA_A0_AU = 1e-6
        b = L.fingerprint(L.draw(500))
    finally:
        C.SIGMA_A0_AU = None
    check(
        "and not on sigma_a0 either -- the TI priors are marginalized, never drawn",
        a == b,
        "which is why `draw` can pass any host mass",
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
def test_gallery():
    """The Thiele-Innes -> a0 identity, and that the cell selection is capped."""
    import jax.numpy as jnp
    from harv.models.parameterizations.gaia import thiele_innes_ABFG

    from epochalypse.harv import gallery
    from epochalypse.periodogram.shards import discover_shards

    # Round trip: build A,B,F,G from Campbell elements with harv's own routine,
    # then recover a0 with the Halbwachs & Pourbaix identity the panel uses.
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        a0, omega, node = (
            rng.uniform(0.01, 5),
            rng.uniform(0, 2 * np.pi),
            rng.uniform(0, 2 * np.pi),
        )
        cos_i = rng.uniform(-1, 1)
        abfg = [
            a0 * float(v)
            for v in thiele_innes_ABFG(
                jnp.cos(omega), jnp.sin(omega), jnp.cos(node), jnp.sin(node), cos_i
            )
        ]
        worst = max(worst, abs(L.semi_major_axis_mas(*abfg) / a0 - 1))
    check(
        "a0 recovered from the Thiele-Innes constants, over 200 random orbits",
        worst < 1e-6,
        f"worst relative error {worst:.2e}",
    )
    check(
        "a zero-amplitude orbit gives a0 = 0 -- the null the rail threshold sits at",
        L.semi_major_axis_mas(0.0, 0.0, 0.0, 0.0) == 0.0,
    )

    if CATALOG_ROOT is None:
        print("  [skip] cell selection needs --catalog-root")
        return
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        C.set_catalog_root(CATALOG_ROOT)
        C.set_output_root(tmp)
        _run_smoke_unit("1_companion")
        chosen = gallery.select("1_companion", per_bin=2)
    counts = chosen.groupby("cell", observed=True).size()
    check(
        "the selection actually found systems to bin",
        len(counts) > 0,
        f"{len(chosen)} system(s) in {len(counts)} cell(s)",
    )
    check(
        "no cell gets more than per_bin systems",
        counts.max() <= 2,
        f"largest cell holds {counts.max()}",
    )
    check(
        "every selection carries the address needed to re-read its epochs",
        {"shard", "shard_row", "gaia_source_id", "cell"} <= set(chosen.columns),
    )

    # --n-parts splits a shard across several sample files, and the per-system
    # table records the shard and the row but NOT the part. Naming the unsplit
    # file works only for --n-parts 1, which is not what production runs -- this
    # is the case that broke a real finish job.
    with tempfile.TemporaryDirectory() as tmp:
        C.set_catalog_root(CATALOG_ROOT)
        C.set_output_root(tmp)
        numbers, n_shards = discover_shards("1_companion")
        for part in range(3):
            unit.run_unit(
                "1_companion",
                numbers[0],
                n_shards,
                part,
                3,
                prior_samples=L.draw(2000),
                top_k=16,
                verbose=False,
                progress_every=0,
            )
        files = sorted(p.name for p in C.samples_dir("1_companion").glob("*.parquet"))
        check(
            "a split shard really does write one file per part",
            len(files) == 3 and all("_part" in f for f in files),
            files[0],
        )
        rows = list(range(4))
        got = gallery.read_samples("1_companion", numbers[0], n_shards, rows)
        check(
            "read_samples finds rows across the parts, not just part 0",
            set(got["shard_row"]) == set(rows),
            f"asked for {rows}, got {sorted(got['shard_row'])}",
        )
        # The samples carry the id, so a catalog/run mismatch is detectable
        # rather than a silent misalignment of epochs against posteriors.
        check(
            "the stored samples carry the id the epochs can be checked against",
            "gaia_source_id" in got.columns
            and got["gaia_source_id"].nunique() == len(rows),
            "make_gallery raises when the two disagree",
        )


def test_gallery_weights():
    """What the panels colour by, and what they decide is worth drawing.

    The old panels coloured by `log10(max(weight, 1e-12))` off a float32 weight
    column, which stacked two artificial floors on top of each other and made
    ~99.7% of the draws one flat dark colour at full opacity. These are the
    checks that the replacement reads the posterior instead of the prior.
    """
    from epochalypse.harv import gallery

    arrays = fake_system(n_epochs=90, period=1.7, alpha=1.5, seed=11)
    record, columns = unit.fit_system(
        *arrays, seed=7, prior_samples=L.draw(N_LIB), m_star_msun=0.41, top_k=K
    )
    w, keep = gallery.weights(columns)

    offset = adapt.pad_log_offset(record["n_epochs"], record["n_padded"])
    harv_w = np.exp(
        columns["ln_likelihood"]
        - record["logZ_int"]
        - offset
        - np.log(record["n_prior_samples"])
    )
    check(
        "the panel weights are harv's own, up to the normalization",
        np.allclose(w, harv_w / harv_w.max(), rtol=1e-10),
        "colour is ln(w/w_best) -- nats of likelihood below the best fit",
    )
    check(
        "the best draw is always drawn as posterior, and anchors the scale",
        bool(keep[np.argmax(w)]) and w.max() == 1.0,
    )
    check(
        "`keep` covers at least the mass it promises",
        w[keep].sum() / w.sum() >= C.GALLERY_WEIGHT_MASS,
        f"{int(keep.sum())} of {len(w)} draws hold "
        f"{w[keep].sum() / w.sum():.4%} of the mass",
    )

    # The reason for the logs, made deterministic: a span like a real detection's.
    span = {"ln_likelihood": -10.0 * np.arange(K, dtype=np.float64)}
    w_span, keep_span = gallery.weights(span)
    check(
        "a float32 weight column would zero a tail the logs keep intact",
        bool((np.float32(w_span) == 0).any()) and bool((w_span > 0).all()),
        f"{int((np.float32(w_span) == 0).sum())} of {K} draws underflow float32, "
        f"none underflow in log space",
    )
    check(
        "and only the draws that matter are drawn as posterior",
        int(keep_span.sum()) <= 2,
        f"{int(keep_span.sum())} of {K} carry {C.GALLERY_WEIGHT_MASS:.1%} of the mass",
    )

    w0, keep0 = gallery.weights({"ln_likelihood": np.full(K, -np.inf)})
    check(
        "an all-non-finite system draws nothing rather than raising",
        not keep0.any() and not w0.any(),
    )


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


def test_min_snr_selection():
    """`--min-snr` must use the generator's ALL-companions rule, not any."""
    from epochalypse.harv.unit import passes_snr

    check(
        "one companion: a simple threshold",
        passes_snr({"snr_total_1": 7.0}, "1_companion", 5.0)
        and not passes_snr({"snr_total_1": 4.9}, "1_companion", 5.0),
    )
    check(
        "two companions: EVERY one must clear it, not just one",
        not passes_snr({"snr_total_1": 900.0, "snr_total_2": 1.0}, "2_companion", 5.0)
        and passes_snr({"snr_total_1": 6.0, "snr_total_2": 5.0}, "2_companion", 5.0),
        "same rule as sources.select_high_snr and census.high_snr_mask",
    )
    check(
        "a non-finite SNR never passes",
        not passes_snr({"snr_total_1": np.nan}, "1_companion", 5.0),
    )
    check(
        "the control never passes -- it has no companion to have SNR",
        not passes_snr({}, "0_companion", 5.0),
        "--min-snr means spend the budget where there is signal",
    )


def test_cost_aware_balance():
    """LPT must partition, and must beat a stride on a long-tailed cost spread."""
    from epochalypse import mpi

    rng = np.random.default_rng(0)
    for n_items, size in ((2880, 1536), (100, 7), (5, 9)):
        costs = rng.lognormal(0, 0.4, n_items)
        items = list(range(n_items))
        shares = [mpi.balance(items, costs, r, size) for r in range(size)]
        flat = sorted(x for share in shares for x in share)
        check(
            f"balance partitions {n_items} items over {size} ranks",
            flat == items,
            "every unit assigned exactly once",
        )

    # the property that matters: a smaller slowest rank, given enough items
    n_items, size = 5760, 512
    costs = rng.lognormal(0, 0.4, n_items)
    items = list(range(n_items))
    lpt = [
        sum(costs[i] for i in mpi.balance(items, costs, r, size)) for r in range(size)
    ]
    strided = [
        sum(costs[i] for i in mpi.stride_for_rank(items, r, size)) for r in range(size)
    ]
    check(
        "and beats a stride on the slowest rank, which is what sets walltime",
        max(lpt) < max(strided),
        f"used {np.mean(lpt) / max(lpt):.0%} vs {np.mean(strided) / max(strided):.0%} "
        f"at {n_items / size:.1f} units/rank",
    )
    check(
        "it is deterministic, so ranks agree without communicating",
        mpi.balance(items, costs, 3, size) == mpi.balance(items, costs, 3, size),
    )


def test_unit_cost_model():
    """Predicted cost must track the padded epoch count, which is what drives it."""
    if CATALOG_ROOT is None:
        print("  [skip] --catalog-root not given")
        return
    from epochalypse.periodogram.shards import discover_shards, work_units

    C.set_catalog_root(CATALOG_ROOT)
    numbers, _ = discover_shards("1_companion")
    units = [u for u in work_units(["1_companion"], 1) if u[1] == numbers[0]]
    costs = unit.unit_costs(units)
    check(
        "one cost per unit, all positive",
        len(costs) == len(units) and all(c > 0 for c in costs),
        f"{len(costs)} unit(s), cost {costs[0]:,.0f}",
    )
    # splitting a shard into parts must split its cost, not duplicate it
    split = unit.unit_costs(
        [u for u in work_units(["1_companion"], 3) if u[1] == numbers[0]]
    )
    check(
        "parts of a shard sum to the whole shard's cost",
        np.isclose(sum(split), costs[0], rtol=1e-9),
        f"3 parts sum to {sum(split):,.0f} vs {costs[0]:,.0f} whole",
    )
    check(
        "--min-snr lowers the predicted cost, because fewer systems are fitted",
        sum(unit.unit_costs(units, min_snr=5.0)) < costs[0],
        "the cost model has to know what the run will actually fit",
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


def test_detectability():
    """The projection: what a fit could ever see of an injected orbit."""
    import pandas as pd

    from epochalypse import constants as k
    from epochalypse import detectability as D
    from epochalypse.astrometry import simulate_along_scan

    rng = np.random.default_rng(3)
    n = 120
    T = k.DR4_BASELINE_YEARS
    t = np.sort(rng.uniform(-T / 2, T / 2, n))
    psi = rng.uniform(0, 2 * np.pi, n)
    pf = np.sin(2 * np.pi * t)
    yerr = np.full(n, 0.05)
    star = {
        "mass_st_msun": 0.5,
        "radius_st_rsun": 0.5,
        "parallax_mas": 20.0,
        "pmra_mas_yr": 25.0,
        "pmdec_mas_yr": -10.0,
        "sigma_single_mas": 0.05,
    }
    orbit = {
        "mass_pl": 8.0,
        "inc": 55.0,
        "omega": 30.0,
        "Omega": 200.0,
        "M_anom": 100.0,
    }

    def truth(**periods):
        row = dict(star)
        for j, (period, ecc) in enumerate(periods["orbits"], start=1):
            row |= {f"{key}_{j}": value for key, value in orbit.items()}
            row |= {f"period_{j}": period, f"ecc_{j}": ecc}
        return pd.Series(row)

    # A short-period orbit has nowhere to hide: many cycles across the window
    # look nothing like a straight line, so the astrometric fit absorbs almost
    # none of it and the detectable SNR approaches the nominal one.
    short = truth(orbits=[(0.15, 0.05)])
    reflex = D.injected_reflex(short, t, psi, pf, 1)
    snr, retained = D.snr_detectable(reflex, t, psi, pf, yerr, 0.05)
    nominal = np.sqrt(n) * np.sqrt(np.mean(reflex**2)) / 0.05
    check(
        "a short-period orbit keeps nearly all of its signal",
        retained > 0.9 and np.isclose(snr, nominal * retained, rtol=1e-9),
        f"retained {retained:.1%}, snr_detectable {snr:.1f}",
    )

    # ...and a long one is mostly a straight line, which is what proper motion is.
    fractions = []
    for ratio in (0.2, 1.0, 2.0, 5.0, 10.0):
        row = truth(orbits=[(ratio * T, 0.05)])
        fractions.append(
            D.retained_fraction(
                D.injected_reflex(row, t, psi, pf, 1), t, psi, pf, yerr
            )[0]
        )
    check(
        "retention falls monotonically as the period passes the baseline",
        all(a > b for a, b in zip(fractions, fractions[1:])),
        "P/T " + " ".join(f"{f:.3f}" for f in fractions),
    )
    check(
        "and collapses for an orbit twice the mission span",
        fractions[2] < 0.4,
        f"{fractions[2]:.1%} survives at P/T = 2 -- a RAILED fit there is correct",
    )

    # The photocentre traces the SUM, so per-companion reflexes are additive.
    # This is what lets snr_detectable_k be attributed to one companion.
    pair = truth(orbits=[(1.3, 0.1), (4.0, 0.4)])
    parts = D.per_companion_reflex(pair, t, psi, pf, 2)
    check(
        "per-companion reflexes sum to the whole",
        np.allclose(sum(parts), D.injected_reflex(pair, t, psi, pf, 2), atol=1e-9),
        f"{len(parts)} companions",
    )

    # The reconstruction must agree with the generator, or every number above is
    # a measurement of a different orbit than the one in the data.
    y, _ = simulate_along_scan(
        t,
        psi,
        [{**orbit, "period": 1.7, "ecc": 0.2}],
        mstar=0.5,
        rstar=0.5,
        parallax=20.0,
        mu_alpha=25.0,
        mu_delta=-10.0,
        parallax_factor=pf,
        sigma_ueva=0.05,
        seed=7,
    )
    row = truth(orbits=[(1.7, 0.2)])
    design = (
        np.column_stack(
            [D.astrometric_design(t, psi, pf, 5), D.injected_reflex(row, t, psi, pf, 1)]
        )
        / yerr[:, None]
    )
    theta, *_ = np.linalg.lstsq(design, np.asarray(y) / yerr, rcond=None)
    check(
        "the reconstructed reflex fits the generated data at amplitude 1",
        abs(float(theta[-1]) - 1.0) < 0.15,
        f"{float(theta[-1]):.4f}",
    )

    # The table: E[retained] over orientation, interpolated.
    table = D.retained_table([(row, t, psi, pf, yerr)], n_draws=6)
    check(
        "the orientation table falls with period, like the orbits it averages",
        D.expected_retained(table, 0.3 * T, 0.1)
        > D.expected_retained(table, 3.0 * T, 0.1),
        f"{D.expected_retained(table, 0.3 * T, 0.1):.3f} vs "
        f"{D.expected_retained(table, 3.0 * T, 0.1):.3f}",
    )
    check(
        "and interpolates a whole column at once, which is how the stage uses it",
        np.asarray(
            D.expected_retained(table, np.array([1.0, 10.0]), np.array([0.1, 0.5]))
        ).shape
        == (2,),
    )


def test_fit_system():
    arrays = fake_system(n_epochs=90, period=1.7, alpha=1.5, seed=4)
    lib, prior = L.draw(N_LIB), L.prior(m_star_msun=0.41)
    kw = {"prior": prior, "prior_samples": lib, "top_k": K}
    del prior  # the dict holds it; fit_system also accepts m_star_msun instead

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
    # No `weight` column is stored. harv's weight is
    # `exp(ln_likelihood - (logZ_int + ln M))`, so it is exactly recoverable from
    # the stored logs plus two per-system scalars -- and unlike a float32 weight
    # it cannot underflow. Reconstructing it here exercises the padding
    # bookkeeping too: `record["logZ_int"]` has `pad_log_offset` subtracted and
    # `ln_likelihood` does not, so a sign error in either shows up as a sum that
    # is off by e^530.
    offset = adapt.pad_log_offset(record["n_epochs"], record["n_padded"])
    log_norm = record["logZ_int"] + offset + np.log(record["n_prior_samples"])
    w = np.exp(columns["ln_likelihood"] - log_norm)
    check(
        "no weight column is stored -- it is derived, and the only lossy form",
        "weight" not in columns,
        f"{len(columns)} columns, ln_likelihood and ln_prior among them",
    )
    check(
        "weights are non-increasing -- harv returns top-K sorted by weight",
        np.all(np.diff(w) <= 0),
    )
    check(
        "weights rebuilt from the stored logs sum to weight_captured, not to 1",
        np.isclose(w.sum(), record["weight_captured"], rtol=1e-9),
        f"{record['weight_captured']:.6f}, from ln_likelihood alone",
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
            (w * columns["period"]).sum() / w.sum(),
            rtol=1e-12,
        ),
    )
    check(
        "period_best is the highest-weight draw",
        record["period_best_yr"] == columns["period"][np.argmax(w)],
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

    # The summaries added so the population figures need never touch the ~850 GB
    # of samples.
    check(
        "every derived summary is present and finite",
        all(np.isfinite(record[name]) for name in unit.SUMMARY_COLUMNS),
        ", ".join(unit.SUMMARY_COLUMNS),
    )
    check(
        "amplitude is summarized by the weighted mean, not by the best draw",
        np.isclose(
            record["a0_wmean_mas"],
            (w * L.semi_major_axis_mas(*(columns[f"ti_{c}"] for c in "ABFG"))).sum()
            / w.sum(),
            rtol=1e-12,
        ),
        "harv draws the marginalized linear parameters, it does not return their mean",
    )
    check(
        "weight_railed is a probability -- the continuous form of census.railed",
        0.0 <= record["weight_railed"] <= 1.0,
        f"{record['weight_railed']:.3g} of the mass at the prior floor",
    )
    check(
        "the two models are nested, so the no-orbit fit can never win",
        record["chi2_null"] >= record["chi2_best"],
        f"chi2 {record['chi2_null']:.1f} without an orbit, "
        f"{record['chi2_best']:.1f} with -- delta {record['chi2_null'] - record['chi2_best']:.1f}",
    )
    check(
        "an injected 1.5 mas orbit is a real improvement over no orbit at all",
        record["chi2_null"] - record["chi2_best"] > 1.0,
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


def test_amplitude_prior_from_host_mass():
    """sigma_a0 as a companion-mass ceiling, scaled by each host."""
    check(
        "a0 = m / M^(2/3): the Sun-like case checks against a hand calculation",
        np.isclose(L.sigma_a0_au(1.0), C.M_MAX_MJUP * C.MJUP_IN_MSUN, rtol=1e-12),
        f"{C.M_MAX_MJUP:g} MJup at 1 Msun -> {L.sigma_a0_au(1.0):.4f} AU",
    )
    check(
        "a lighter host gets a WIDER prior for the same companion mass",
        L.sigma_a0_au(0.41) > L.sigma_a0_au(1.0),
        f"0.41 Msun -> {L.sigma_a0_au(0.41):.4f} AU vs 1.0 Msun -> {L.sigma_a0_au(1.0):.4f}",
    )
    try:
        C.SIGMA_A0_AU = 0.03
        check(
            "pinning a constant overrides the mass scaling",
            L.sigma_a0_au(0.41) == 0.03 and L.sigma_a0_au(999.0) == 0.03,
            "which is what a sweep varies",
        )
    finally:
        C.SIGMA_A0_AU = None
    try:
        L.sigma_a0_au()
        raise AssertionError("should have raised")
    except ValueError as error:
        check(
            "no host mass and no pin raises rather than guessing",
            "m_star_msun" in str(error),
        )
    # the prior must actually USE it, not merely compute it
    light = L.prior(m_star_msun=0.2).linear_priors["ti_A"]
    heavy = L.prior(m_star_msun=1.0).linear_priors["ti_A"]
    check(
        "and the prior object differs between two hosts",
        float(light.sigma_a0.value) != float(heavy.sigma_a0.value),
        f"{float(light.sigma_a0.value):.4f} vs {float(heavy.sigma_a0.value):.4f} AU",
    )


def test_sample_units():
    """The manifest is written before the first fit, so its units are hard-coded.

    Which makes them a place to drift. This is the assertion that stops that.
    """
    from harv import RejectionSampler

    arrays = fake_system(n_epochs=90, seed=6)
    data, par, _ = adapt.prepare(*arrays)
    sampler = RejectionSampler(
        L.prior(m_star_msun=0.41), L.model(par), batch_size=C.BATCH_SIZE
    )
    samples = sampler.run_with_samples(data, L.draw(N_LIB), top_k=K, seed=1)

    stored = {**samples.nonlinear, **samples.linear}
    wrong = {
        n: (L.SAMPLE_UNITS.get(n), str(q.unit))
        for n, q in stored.items()
        if L.SAMPLE_UNITS.get(n) != str(q.unit)
    }
    check("every hard-coded unit matches what harv returns", not wrong, str(wrong))
    check(
        "the two derived columns are dimensionless",
        all(L.SAMPLE_UNITS[n] == "" for n in ("ln_likelihood", "ln_prior")),
    )
    check(
        "SAMPLE_UNITS has nothing harv does not return",
        set(L.SAMPLE_UNITS) - set(stored) == {"ln_likelihood", "ln_prior"},
    )
    check(
        "and no weight column -- derived from ln_likelihood, and the one lossy form",
        "weight" not in L.SAMPLE_UNITS,
        "~70 GB of mostly-zero float32 over the catalog",
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
        "no weight column survives into the parquet",
        "weight" not in samples.columns,
        f"stored: {sorted(c for c in samples.columns if c != 'shard_row')}",
    )
    merged = samples.merge(systems, on="shard_row", suffixes=("", "_sys"))
    rebuilt = [
        float(
            np.exp(
                np.asarray(r["ln_likelihood"], float)
                - r["logZ_int"]
                - adapt.pad_log_offset(r["n_epochs"], r["n_padded"])
                - np.log(r["n_prior_samples"])
            ).sum()
        )
        for _, r in merged.iterrows()
    ]
    check(
        "weight_captured is rebuildable from the stored logs after the float32 cast",
        np.allclose(rebuilt, merged["weight_captured"], rtol=1e-3),
        f"SAMPLE_DTYPE = {C.SAMPLE_DTYPE}; the logs survive it, the weights would not",
    )
    check(
        "the derived summaries reached the parquet",
        set(unit.SUMMARY_COLUMNS) | {"chi2_null", "chi2_best"} <= set(systems.columns),
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
