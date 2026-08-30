#!/usr/bin/env python
"""Dissect one system: why did its period land where it did?

    python scripts/harv_inspect.py 155720052270716800 1148009440706631808 \
        --catalog-root $OUT_ROOT --output-root $HARV_ROOT

The gallery shows you *that* a fit went somewhere unexpected. This says *why*,
by separating the three explanations that a single posterior cannot:

  1. THE DATA CANNOT TELL.  The profile chi-square is flat between the true
     period and the reported one, so both fit equally well and which one was
     reported is a coin flip the library made. Near one cycle across the
     mission span this is the default expectation, not a surprise: position and
     proper motion are free linear parameters, so a slow orbit is partly
     absorbable into them and the period ridge goes shallow.
  2. THE LIBRARY MISSED IT.  The data prefer the truth, but at
     N_PRIOR_SAMPLES over four decades x eccentricity x phase the effective
     grid is ~100 per nonlinear dimension, and nothing landed close enough.
     Refitting at 10x the library moves the answer.
  3. THE LIKELIHOOD REALLY PREFERS THE WRONG PERIOD.  The restricted-prior
     Bayes factor favours the reported period by many nats. Then no library
     size helps and the model or the noise is the story.

Three tests, in that order:

  profile   chi-square against trial period, refitting all nine linear
            parameters at each and minimizing over eccentricity and phase. This
            is the LIKELIHOOD alone -- no priors, no Occam penalty -- so it says
            what the data can and cannot distinguish. Delta chi-square between
            the truth and the reported period is the headline number.
  bayes     logZ_int over a narrow period window centred on the truth, against
            the same-width window centred on the reported period. Equal prior
            volume, so the difference IS the log Bayes factor, and unlike the
            profile it includes the marginalization penalty harv actually pays.
  refit     the same system at larger N_PRIOR_SAMPLES. If the answer moves to
            the truth, it was resolution.

`--dump` writes the epochs and the stored draws to one npz, for looking at
somewhere else.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from epochalypse import constants as k
from epochalypse.harv import adapt, census, gallery, unit
from epochalypse.harv import config as C
from epochalypse.harv import library as L
from epochalypse.periodogram.shards import ShardReader, discover_shards

TESTS = ("profile", "bayes", "refit")


def locate(gaia_source_id, populations=None):
    """`(population, shard, shard_row)` for one id, searched across populations."""
    for population in populations or list(C.POPULATIONS):
        try:
            table = census.read_systems(
                population, ["gaia_source_id", "shard", "shard_row"]
            )
        except FileNotFoundError:
            continue
        hit = np.flatnonzero(np.asarray(table["gaia_source_id"]) == gaia_source_id)
        if hit.size:
            row = int(hit[0])
            return (
                population,
                int(np.asarray(table["shard"])[row]),
                int(np.asarray(table["shard_row"])[row]),
            )
    msg = f"gaia_source_id {gaia_source_id} is in no harv output under {C.OUTPUT_ROOT}"
    raise SystemExit(msg)


def read_one(population, shard, shard_row):
    """`(truth_row, arrays)` for one system. Stops as soon as it is found."""
    _, n_shards = discover_shards(population)
    with ShardReader(population, shard, n_shards) as reader:
        for index, truth, *arrays in reader.iter_systems():
            if index == shard_row:
                return truth, tuple(arrays)
    msg = f"row {shard_row} not in {population} shard {shard}"
    raise SystemExit(msg)


def profile_chi2(model, data, periods, n_ecc, n_phase):
    """Best chi-square achievable at each trial period.

    All nine linear parameters are refit at every grid point -- the orbit's
    amplitude and orientation are free, as they are in the fit -- and the
    minimum is taken over eccentricity and phase. So this is the profile
    likelihood in period alone, with everything else given its best shot.

    Returns `(chi2, a0_mas, ecc, phase)`, each one value per trial period.
    """
    al = np.asarray(data.al_position.value, dtype=np.float64)
    err = np.asarray(data.al_position_err.value, dtype=np.float64)
    eccs = np.linspace(0.0, 0.95, n_ecc)
    phases = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)

    chi2 = np.full(len(periods), np.inf)
    a0 = np.zeros(len(periods))
    at_ecc = np.zeros(len(periods))
    at_phase = np.zeros(len(periods))
    for i, period in enumerate(periods):
        for ecc in eccs:
            for phase in phases:
                design = L.design_matrix(model, data, period, ecc, phase)
                theta, value = adapt.linear_solution(design, al, err)
                if value < chi2[i]:
                    chi2[i] = value
                    a0[i] = L.semi_major_axis_mas(*theta[5:9])
                    at_ecc[i], at_phase[i] = ecc, phase
    return chi2, a0, at_ecc, at_phase


def two_orbit_chi2(model, data, truth, n_comp, n_phase):
    """Best chi-square with BOTH injected orbits in the model.

    harv fits a SINGLE Keplerian. On a two-companion system the photocentre
    traces the sum of two orbits, so a one-orbit fit has to compromise: it
    generically lands on an amplitude larger than either companion's, at a
    period that is neither, and it spills what it cannot fit into the free
    linear parameters -- which is how an unmodelled companion biases the
    PARALLAX.

    This builds the design matrix the truth deserves: five astrometric columns
    plus four Thiele-Innes columns per injected orbit, every period and
    eccentricity pinned at its injected value, amplitudes and orientations free,
    minimized over the two phases. If this beats the best single-orbit profile
    by a large margin, the single-Keplerian model is the limitation and no
    amount of prior sampling will close the gap.

    Returns `(chi2, a0_per_orbit)`.
    """
    al = np.asarray(data.al_position.value, dtype=np.float64)
    err = np.asarray(data.al_position_err.value, dtype=np.float64)
    phases = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False)

    orbits = [
        (float(truth[f"period_{j}"]), float(truth[f"ecc_{j}"]))
        for j in range(1, n_comp + 1)
    ]
    best, best_a0 = np.inf, None
    for combination in np.ndindex(*([n_phase] * n_comp)):
        blocks = []
        for (period, ecc), index in zip(orbits, combination):
            design = L.design_matrix(model, data, period, ecc, phases[index])
            if not blocks:
                blocks.append(design[:, :5])  # astrometry, shared, orbit-independent
            blocks.append(design[:, 5:])
        stacked = np.concatenate(blocks, axis=1)
        theta, value = adapt.linear_solution(stacked, al, err)
        if value < best:
            best = value
            best_a0 = [
                L.semi_major_axis_mas(*theta[5 + 4 * j : 9 + 4 * j])
                for j in range(n_comp)
            ]
    return best, best_a0


def within(periods, chi2, threshold):
    """The contiguous period range around the minimum inside `threshold` nats."""
    best = int(np.argmin(chi2))
    ok = chi2 <= chi2[best] + threshold
    lo = hi = best
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(ok) - 1 and ok[hi + 1]:
        hi += 1
    return periods[lo], periods[hi]


def restricted_logz(arrays, m_star_msun, centre, half_dex, n_samples, seed, top_k):
    """`logZ_int` with the period prior confined to a window around `centre`.

    Two windows of the same width in log-period have the same prior volume, so
    the difference between their `logZ_int` is exactly the log Bayes factor
    between "the period is near here" and "the period is near there" -- with
    every other prior, the padding correction and the marginalization penalty
    identical on both sides.
    """
    saved = (C.PERIOD_MIN_YR, C.PERIOD_MAX_YR)
    try:
        C.PERIOD_MIN_YR = float(centre) * 10.0**-half_dex
        C.PERIOD_MAX_YR = float(centre) * 10.0**half_dex
        library = L.draw(n_samples)
        return unit.fit_system(
            *arrays,
            prior_samples=library,
            seed=seed,
            m_star_msun=m_star_msun,
            top_k=top_k,
        )
    finally:
        C.PERIOD_MIN_YR, C.PERIOD_MAX_YR = saved


def inspect(gaia_source_id, args):
    population, shard, shard_row = locate(gaia_source_id, args.populations)
    truth, arrays = read_one(population, shard, shard_row)
    t = arrays[0]
    span = float(t.max() - t.min())

    n_comp = C.POPULATIONS[population]
    stored = census.read_systems(
        population,
        census.system_columns(
            population,
            (
                "gaia_source_id",
                "shard_row",
                "n_epochs",
                "n_padded",
                "n_prior_samples",
                "period_wmean_yr",
                "period_wstd_yr",
            )
            + tuple(f"alpha_mas_{j}" for j in range(1, n_comp + 1)),
        ),
    )
    row = stored.filter(
        np.asarray(stored["gaia_source_id"]) == gaia_source_id
    ).to_pydict()
    row = {key: value[0] for key, value in row.items()}

    data, par, n_epochs = adapt.prepare(*arrays)
    model = L.model(par)
    al = np.asarray(data.al_position.value, dtype=np.float64)
    err = np.asarray(data.al_position_err.value, dtype=np.float64)
    design0 = L.design_matrix(model, data, 1.0, 0.0, 0.0)
    chi2_null = adapt.linear_solution(design0, al, err, 5)[1]
    n_real = int(adapt.real_rows(err).sum())

    p_best = float(row["period_best_yr"])
    print(
        f"\n{'=' * 78}\ngaia {gaia_source_id}   {population} shard {shard} row {shard_row}"
    )
    print("=" * 78)
    print(f"  host mass        {truth['mass_st_msun']:.3f} Msun")
    print(f"  parallax (true)  {truth['parallax_mas']:.3f} mas")
    print(
        f"  epochs           {n_epochs} real over {span:.2f} yr (padded to {n_padded_of(row)})"
    )
    print(f"  chi2/N, no orbit {chi2_null / n_real:.3f}")
    for j in range(1, n_comp + 1):
        p_true = float(truth[f"period_{j}"])
        print(
            f"  injected {j}: P = {p_true:.4f} yr ({span / p_true:.2f} cycles observed)"
            f"  e = {truth[f'ecc_{j}']:.3f}  alpha = {truth[f'alpha_mas_{j}']:.4f} mas"
            f"  SNR = {truth[f'snr_total_{j}']:.1f}"
        )
    print(
        f"  reported:   P = {p_best:.4f} yr ({span / p_best:.2f} cycles)"
        f"  wmean {row['period_wmean_yr']:.4f}  wstd {row['period_wstd_yr']:.4g}"
    )

    block = gallery.read_samples(
        population, shard, discover_shards(population)[1], [shard_row]
    )
    block = block.iloc[0]
    w, keep = gallery.weights(block)
    print(
        f"  ess {row['ess']:.2f}   weight_captured {row['weight_captured']:.4f}   "
        f"{int(keep.sum())} of {len(w)} draws carry "
        f"{C.GALLERY_WEIGHT_MASS:.1%} of the mass"
    )
    if int(keep.sum()) <= 3:
        print(
            "  ^ the posterior is NOT tight, it is UNRESOLVED. A spread computed from"
            "\n    two or three draws is a property of the library, not of the data."
        )

    if not n_comp:
        print("\n  control population: no injected companion, nothing to profile.")
        return
    p_true = float(
        census.best_truth(stored, population, "period")[
            int(
                np.flatnonzero(np.asarray(stored["gaia_source_id"]) == gaia_source_id)[
                    0
                ]
            )
        ]
    )

    if "profile" in args.tests:
        run_profile(
            model, data, p_true, p_best, chi2_null, n_real, args, gaia_source_id
        )
        if n_comp > 1:
            best, a0s = two_orbit_chi2(model, data, truth, n_comp, args.n_phase)
            print(f"\n--- both injected orbits in the model ({n_comp} companions) ---")
            print(
                f"  chi2 with both orbits   {best:10.2f}  ({best / n_real:.3f} per epoch)"
            )
            for j, a0 in enumerate(a0s, start=1):
                print(
                    f"    orbit {j}: fitted a0 = {a0:.4f} mas   "
                    f"injected alpha = {float(truth[f'alpha_mas_{j}']):.4f} mas   "
                    f"P = {float(truth[f'period_{j}']):.4f} yr"
                )
            print(
                "  harv fits ONE Keplerian. If this is much better than the best"
                "\n  single-orbit chi2 above, the model is the limitation: a one-orbit"
                "\n  fit to a two-orbit wobble lands on an inflated amplitude at a"
                "\n  period that is neither, and spills the rest into parallax and"
                "\n  proper motion. No library size fixes that."
            )
    if "bayes" in args.tests:
        run_bayes(arrays, truth, p_true, p_best, args)
    if "refit" in args.tests:
        run_refit(arrays, truth, p_true, p_best, args)

    if args.dump:
        path = Path(args.dump) / f"harv_system_{gaia_source_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            t=arrays[0],
            psi=arrays[1],
            pf=arrays[2],
            y=arrays[3],
            yerr=arrays[4],
            truth=np.array([(key, str(value)) for key, value in truth.items()]),
            **{name: np.asarray(block[name], float) for name in L.SAMPLE_UNITS},
        )
        print(f"\n  dumped -> {path}  ({path.stat().st_size / 1e3:.0f} kB)")


def n_padded_of(row):
    return int(row.get("n_padded", 0))


def run_profile(model, data, p_true, p_best, chi2_null, n_real, args, gaia_source_id):
    """The likelihood alone: can the data tell these two periods apart?"""
    print(
        f"\n--- profile chi-square ({args.n_period} periods x {args.n_ecc} e x "
        f"{args.n_phase} phase) ---"
    )
    grid = np.unique(
        np.concatenate(
            [
                np.logspace(
                    np.log10(C.PERIOD_MIN_YR), np.log10(C.PERIOD_MAX_YR), args.n_period
                ),
                # the two periods actually in question, exactly
                np.geomspace(p_true * 0.9, p_true * 1.1, 21),
                np.geomspace(p_best * 0.9, p_best * 1.1, 21),
                [p_true, p_best],
            ]
        )
    )
    started = time.time()
    chi2, a0, ecc, _ = profile_chi2(model, data, grid, args.n_ecc, args.n_phase)
    print(
        f"  {len(grid) * args.n_ecc * args.n_phase:,} fits in {time.time() - started:.0f} s"
    )

    at_true = float(chi2[np.argmin(np.abs(grid - p_true))])
    at_best = float(chi2[np.argmin(np.abs(grid - p_best))])
    floor = float(chi2.min())
    print(
        f"  chi2 no orbit        {chi2_null:10.2f}  ({chi2_null / n_real:.3f} per epoch)"
    )
    print(f"  chi2 at P_true       {at_true:10.2f}  ({at_true / n_real:.3f})")
    print(f"  chi2 at P_reported   {at_best:10.2f}  ({at_best / n_real:.3f})")
    print(
        f"  chi2 global minimum  {floor:10.2f}  at P = {grid[np.argmin(chi2)]:.4f} yr"
    )
    print(f"\n  delta chi2 (true - reported) = {at_true - at_best:+.2f}")
    if abs(at_true - at_best) < 4.0:
        print(
            "  -> THE DATA CANNOT TELL THEM APART. Both periods fit within noise, so"
            "\n     the reported one was chosen by whichever draw the library happened"
            "\n     to place closest. Not a bias -- a coin flip. Expect it to land on"
            "\n     either side of the truth across systems."
        )
    elif at_true > at_best:
        print(
            f"  -> the data genuinely prefer the reported period by "
            f"{at_true - at_best:.1f} in chi2."
            "\n     A larger library will not fix this; look at the model or the noise."
        )
    else:
        print(
            f"  -> the truth fits BETTER by {at_best - at_true:.1f} in chi2, so the"
            "\n     library simply never sampled near it. Resolution, not physics."
        )
    lo, hi = within(grid, chi2, 1.0)
    print(f"\n  periods within delta chi2 < 1 of the best: {lo:.3f} - {hi:.3f} yr")
    print("  ...which is the honest period uncertainty, versus the reported wstd.")

    print("\n  P [yr]     delta chi2 vs best    a0 [mas]   e")
    show = np.logspace(np.log10(C.PERIOD_MIN_YR), np.log10(C.PERIOD_MAX_YR), 40)
    for period in show:
        i = int(np.argmin(np.abs(grid - period)))
        bar = "#" * int(
            min(40, max(0, (chi2[i] - floor) / max(chi2_null - floor, 1) * 40))
        )
        print(
            f"  {grid[i]:8.3f}  {chi2[i] - floor:10.1f}  {a0[i]:10.4f}  {ecc[i]:4.2f}  {bar}"
        )

    if args.figure:
        save_profile(grid, chi2, a0, p_true, p_best, chi2_null, gaia_source_id, args)


def save_profile(grid, chi2, a0, p_true, p_best, chi2_null, gaia_source_id, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    floor = float(chi2.min())
    axes[0].plot(grid, chi2 - floor, "-", color="#050CDB", lw=1.4)
    axes[0].axhline(chi2_null - floor, color="0.5", ls="-", lw=1, label="no orbit")
    for value, color, label in (
        (p_true, "#C2185B", "injected"),
        (p_best, "#0288D1", "reported"),
    ):
        axes[0].axvline(value, color=color, lw=1.3, ls="--", label=label)
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="trial period [yr]",
        ylabel=r"$\Delta\chi^2$ above the best fit",
    )
    axes[0].legend(fontsize=8)
    axes[0].set_title("profile likelihood: what the data alone can say", fontsize=10)

    axes[1].plot(grid, a0, "-", color="#6A1B9A", lw=1.4, label=r"best-fit $a_0$")
    for value, color in ((p_true, "#C2185B"), (p_best, "#0288D1")):
        axes[1].axvline(value, color=color, lw=1.3, ls="--")
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="trial period [yr]",
        ylabel=r"$a_0$ at the profile optimum [mas]",
    )
    axes[1].legend(fontsize=8)
    axes[1].set_title("the amplitude the period buys", fontsize=10)
    fig.suptitle(f"gaia {gaia_source_id}", fontsize=11)
    path = Path(args.figure) / f"harv_profile_{gaia_source_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def run_bayes(arrays, truth, p_true, p_best, args):
    """The same question harv answers, restricted to one period window at a time."""
    print(f"\n--- restricted-prior Bayes factor (+/- {args.half_dex} dex windows) ---")
    out = {}
    for label, centre in (("P_true", p_true), ("P_reported", p_best)):
        record, _ = restricted_logz(
            arrays,
            truth["mass_st_msun"],
            centre,
            args.half_dex,
            args.n_window_samples,
            args.seed,
            args.top_k,
        )
        out[label] = record
        print(
            f"  {label:<12} centred {centre:8.4f} yr   logZ_int {record['logZ_int']:12.3f}"
            f"   ess {record['ess']:7.2f}   P_best {record['period_best_yr']:.4f}"
        )
    delta = out["P_true"]["logZ_int"] - out["P_reported"]["logZ_int"]
    print(f"\n  ln B (true vs reported) = {delta:+.2f} nats")
    if abs(delta) < 2.3:
        print(
            "  -> inconclusive (|ln B| < 2.3, i.e. within 10:1). The evidence does not"
            "\n     separate them, which is the same verdict as a flat profile."
        )
    elif delta > 0:
        print(
            "  -> the evidence FAVOURS the true period. The full-range run missed it"
            "\n     because it never drew close enough -- a resolution problem."
        )
    else:
        print(
            "  -> the evidence favours the reported period even with the prior confined"
            "\n     around the truth. That is a real preference, not a sampling miss."
        )


def run_refit(arrays, truth, p_true, p_best, args):
    """Does more library move the answer? The direct test of resolution."""
    print("\n--- refit at larger N_PRIOR_SAMPLES ---")
    print(
        f"  {'M':>12}  {'P_best':>10}  {'ess':>8}  {'wcap':>7}  {'|ln P/P_true|':>13}"
    )
    for n_samples in args.library_sizes:
        started = time.time()
        record, _ = unit.fit_system(
            *arrays,
            prior_samples=L.draw(int(n_samples)),
            seed=args.seed,
            m_star_msun=truth["mass_st_msun"],
            top_k=args.top_k,
        )
        offset = abs(np.log(record["period_best_yr"] / p_true))
        print(
            f"  {int(n_samples):>12,}  {record['period_best_yr']:>10.4f}  "
            f"{record['ess']:>8.2f}  {record['weight_captured']:>7.4f}  "
            f"{offset:>13.4f}   ({time.time() - started:.0f} s)"
        )
    print(
        f"  reference: reported {p_best:.4f}, injected {p_true:.4f}, "
        f"|ln| = {abs(np.log(p_best / p_true)):.4f}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("gaia_source_ids", type=int, nargs="+")
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--populations", nargs="+", choices=list(C.POPULATIONS))
    parser.add_argument("--tests", nargs="+", choices=TESTS, default=list(TESTS))
    parser.add_argument("--n-period", type=int, default=90)
    parser.add_argument("--n-ecc", type=int, default=6)
    parser.add_argument("--n-phase", type=int, default=24)
    parser.add_argument(
        "--half-dex", type=float, default=0.02, help="Bayes window half-width"
    )
    parser.add_argument("--n-window-samples", type=int, default=200_000)
    parser.add_argument(
        "--library-sizes",
        type=float,
        nargs="+",
        default=[1e6, 1e7],
        help="library sizes for the refit test",
    )
    parser.add_argument("--top-k", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=C.SEED)
    parser.add_argument("--figure", type=Path, help="directory for the profile PNG")
    parser.add_argument(
        "--dump", type=Path, help="directory for an npz of the raw data"
    )
    args = parser.parse_args(argv)

    C.set_catalog_root(args.catalog_root)
    C.set_output_root(args.output_root)
    print(
        f"catalog {args.catalog_root}\noutput  {args.output_root}\n"
        f"mission baseline {k.DR4_BASELINE_YEARS} yr, "
        f"period prior {C.PERIOD_MIN_YR:g}-{C.PERIOD_MAX_YR:g} yr"
    )
    for gaia_source_id in args.gaia_source_ids:
        inspect(gaia_source_id, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
