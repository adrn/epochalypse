#!/usr/bin/env python
"""Is `snr_total` the signal that is actually in the along-scan data?

    python scripts/check_snr.py --catalog-root $OUT_ROOT --ids 568042036585081856
    python scripts/check_snr.py --catalog-root $OUT_ROOT --sample 2000

Over the epoch shards: two least-squares solves and one projection per system.
No prior library and no posterior sampling, so thousands of systems in a minute.

THE NO-SIGNAL FLOOR IS NOT ONE. The generator injects noise at one scale and
reports it at another, on purpose:

    sigma_injected = sigma_UEVA,single   -- AL measurement + CALIBRATION terms
    sigma_reported = sigma_formal        -- attitude + AL, no calibration term

so a downstream fit weights by an uncertainty that does not describe the scatter
it is looking at. That is the point: equating them would give an artificially
self-consistent data set. The consequence for this script is that a system with
NO companion at all has

    chi2 / (N - k)  ->  r^2,      r = sigma_single / sigma_reported

and r is 1.28 at the median of the real catalog, not 1. Every chi-square below
is therefore divided by r^2 before anything is read off it. Getting this wrong
makes a healthy catalog look like it has 60% more signal than it should, which
is exactly the false alarm this script raised on its first run.

With that floor, a signal of along-scan rms `A` against the INJECTED noise over
`N` epochs leaves

    chi2 / (N - k)  ~  r^2 (1 + (A / sigma_injected)^2)

and the recorded metric claims `snr_total = sqrt(N) * alpha / sigma_single`,
which divides by the injected scale -- so `snr_total` is the honest measure of
detectability and needs no correction of its own.

Three numbers that must agree, and the disagreement names the culprit:

    snr_total       what the catalog recorded
    snr_detectable  what geometry says survives -- the injected orbit's own
                    design columns with the five astrometric columns projected
                    OUT of them, which is the only part of the signal any fit
                    can ever use. Pure linear algebra on the scan law: no
                    re-simulation, no phase convention, no fit.
    snr_measured    the excess scatter actually left after the five-parameter
                    fit, sqrt(N (chi2/dof - 1))

  * **detectable << total, and measured tracks detectable** -> the orbit is
    real and the five-parameter solution ate it. Expected for a period near or
    beyond the baseline, and it means `snr_eff`'s `1/(1 + (sma/a_crit)^3)`
    penalty is too weak, not that the data are wrong. A "RAILED" verdict on
    such a system is CORRECT, and so is the eye that says the residuals look
    flat.
  * **detectable ~ total, but measured ~ 0** -> the signal is not in the data at
    the recorded amplitude. A generator bug, and `alpha_mas` against the fitted
    amplitude localizes it.
  * **all three agree** -> `snr_total` is honest and the fit is what went wrong.

The two noise scales are also printed, because they are NOT the same variable.
`astrometry.simulate_along_scan` injects `sigma_ueva = sigma_single * jitter`
while the shard records `centroid_pos_error_al = sigma_reported * jitter`, and
`snr_single` divides by `sigma_single`. `sigma_ratio` below is that ratio per
system: it is EXPECTED to sit above 1 and sets the chi-square floor. Watch its
tail rather than its median -- a star whose injected noise is many times its
reported error will look wildly over-dispersed to any fit, and whether that is
intended is a question for the noise model, not for this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from epochalypse.periodogram import config as PG
from epochalypse.periodogram.shards import ShardReader, discover_shards


def astrometric_design(t, psi, pf, n_columns=5):
    """The along-scan astrometric basis, straight from the simulator's own model.

    `simulate_along_scan` writes
    `al = sin(psi) (ra0 + mu_a t) + cos(psi) (dec0 + mu_d t) + parallax * pf`,
    so these five columns span exactly the motion a Gaia five-parameter solution
    can absorb. Any basis spanning the same space gives the same residual, so
    there is nothing to get wrong here and no need for harv's design matrix.

    """
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)
    columns = [sin_psi, cos_psi, t * sin_psi, t * cos_psi, pf]
    return np.column_stack(columns[:n_columns])


def reduced_chi2(t, psi, pf, y, yerr, n_columns=5):
    """Reduced chi-square of the best `n_columns`-parameter astrometric fit."""
    design = astrometric_design(t, psi, pf, n_columns)
    a = design / yerr[:, None]
    b = y / yerr
    theta, *_ = np.linalg.lstsq(a, b, rcond=None)
    residual = (y - design @ theta) / yerr
    return float(np.sum(residual**2) / max(len(y) - n_columns, 1))


def retained_fraction(t, psi, pf, yerr, period, ecc):
    """How much of an orbit at this period survives the five-parameter fit.

    An orbit whose period is comparable to or longer than the mission span is
    partly a straight line plus a curve across the data, and position, proper
    motion and parallax are FREE. Whatever part of the orbit those five columns
    can reproduce is subtracted along with them and can never be detected --
    no matter how large `alpha` is.

    So project the orbit's four Thiele-Innes columns onto the orthogonal
    complement of the astrometric basis (inverse-variance weighted, since that
    is the metric the fit minimizes in) and return the fraction of amplitude
    left. This is geometry alone: it needs the scan law and the period, not the
    data, not the amplitude, and not the orbital phase.

    `snr_eff` tries to charge for exactly this with `1/(1 + (sma/a_crit)^3)`.
    Comparing the two is the point of the script.
    """
    from epochalypse.harv import adapt
    from epochalypse.harv import library as L

    data, par, _ = adapt.prepare(t, psi, pf, np.zeros_like(t), yerr)
    design = L.design_matrix(L.model(par), data, period, ecc, 0.0)
    real = adapt.real_rows(np.asarray(data.al_position_err.value, float))
    weight = 1.0 / yerr[:, None]

    astro = np.asarray(design)[real][:, :5] * weight
    orbit = np.asarray(design)[real][:, 5:] * weight
    # least squares of every orbit column against the astrometric basis at once
    fitted, *_ = np.linalg.lstsq(astro, orbit, rcond=None)
    left = orbit - astro @ fitted
    total = float(np.sum(orbit**2))
    return float(np.sqrt(np.sum(left**2) / total)) if total > 0 else 0.0


def audit(truth, arrays, n_companions):
    """Everything measurable about one system's signal, and what was claimed."""
    t, psi, pf, y, yerr = arrays
    n = len(t)
    row = {
        "gaia_source_id": int(truth["gaia_source_id"]),
        "n_epochs": n,
        "span_yr": float(t.max() - t.min()),
        "sigma_reported": float(np.median(yerr)),
        "sigma_single": float(truth["sigma_single_mas"]),
        "parallax_mas": float(truth["parallax_mas"]),
        "chi2_astro": reduced_chi2(t, psi, pf, y, yerr, 5),
    }
    row["sigma_ratio"] = row["sigma_single"] / row["sigma_reported"]
    # The floor: with no companion at all, chi-square lands on r^2, not on 1,
    # because the data carry sigma_injected while the fit weights by
    # sigma_reported. Dividing it out puts the excess in units of the INJECTED
    # noise, which is what snr_total is defined against.
    floor = row["sigma_ratio"] ** 2
    row["chi2_floor"] = floor
    row["snr_measured"] = np.sqrt(n) * float(
        np.sqrt(max(row["chi2_astro"] / floor - 1.0, 0.0))
    )

    if n_companions:
        row["alpha_mas"] = float(
            sum(truth[f"alpha_mas_{j}"] for j in range(1, n_companions + 1))
        )
        row["period_min_yr"] = float(
            min(truth[f"period_{j}"] for j in range(1, n_companions + 1))
        )
        row["snr_total"] = float(
            max(truth[f"snr_total_{j}"] for j in range(1, n_companions + 1))
        )
        row["snr_single"] = float(
            max(truth[f"snr_single_{j}"] for j in range(1, n_companions + 1))
        )
        shortest = int(
            np.argmin([truth[f"period_{j}"] for j in range(1, n_companions + 1)]) + 1
        )
        row["ecc_min"] = float(truth[f"ecc_{shortest}"])
        # what snr_total predicts the k=5 residual should look like. The 1/2 is
        # the projection of a 2-D orbit onto a rotating scan direction.
        row["chi2_predicted"] = floor * (1.0 + 0.5 * row["snr_total"] ** 2 / n)
        row["retained"] = retained_fraction(
            t, psi, pf, yerr, row["period_min_yr"], row["ecc_min"]
        )
        row["snr_detectable"] = row["snr_total"] * row["retained"]
    return row


def verdict(row):
    """Which of the three explanations this system's numbers support."""
    if "snr_total" not in row:
        return "control (no companion)"
    if row["chi2_predicted"] / row["chi2_floor"] - 1.0 < 0.25:
        return "no signal claimed"
    measured, detectable, total = (
        row["snr_measured"],
        row["snr_detectable"],
        row["snr_total"],
    )
    # sqrt(N (chi2-1)) has scatter of order sqrt(2N)/... below a few sigma, so a
    # single weak system cannot be classified. Read the binned table for those.
    if detectable < 5.0:
        return "too weak to judge individually"
    if measured > 0.5 * total:
        return "consistent -- snr_total is in the data"
    if detectable < 0.5 * total and measured > 0.4 * detectable:
        return "ABSORBED by the 5-param fit (snr_eff penalty too weak)"
    return "MISSING -- signal not in the data at the recorded amplitude"


def report_one(row):
    print(f"\n{'=' * 72}\ngaia {row['gaia_source_id']}\n{'=' * 72}")
    print(f"  epochs              {row['n_epochs']} over {row['span_yr']:.2f} yr")
    print(f"  parallax            {row['parallax_mas']:.3f} mas")
    print(
        f"  sigma reported      {row['sigma_reported']:.5f} mas   (the plotted error bar)"
    )
    print(
        f"  sigma_single        {row['sigma_single']:.5f} mas   "
        f"(what snr divides by; ratio {row['sigma_ratio']:.3f})"
    )
    if abs(row["sigma_ratio"] - 1.0) > 0.15:
        print(
            "  ^^ THESE DISAGREE. snr_single divides by sigma_single, the data carry"
            "\n     sigma_reported, and every SNR is off by exactly this factor."
        )
    if "snr_total" in row:
        print(
            f"\n  injected alpha      {row['alpha_mas']:.5f} mas (summed over companions)"
        )
        print(f"  shortest period     {row['period_min_yr']:.4f} yr")
        print(f"  alpha / sigma       {row['alpha_mas'] / row['sigma_reported']:.3f}")
        print(f"  recorded snr_single {row['snr_single']:.3f}")
        print(f"  recorded snr_total  {row['snr_total']:.2f}")
        print(f"\n  chi2/dof, NO-companion floor    {row['chi2_floor']:8.3f}  (= r^2)")
        print(f"  chi2/dof after the 5-param fit  {row['chi2_astro']:8.3f}")
        print(f"  chi2/dof PREDICTED from snr_tot {row['chi2_predicted']:8.3f}")
        print(
            f"\n  orbit amplitude surviving the astrometric fit: {row['retained']:.1%}"
        )
        print(f"\n  snr_total      (recorded)   {row['snr_total']:8.2f}")
        print(f"  snr_detectable (geometry)   {row['snr_detectable']:8.2f}")
        print(f"  snr_measured   (data)       {row['snr_measured']:8.2f}")
    print(f"\n  --> {verdict(row)}")


def iterate(population, n_companions, ids=None, sample=None, min_snr=None, rng=None):
    """Yield audited rows, either for specific ids or for a random sample."""
    numbers, n_shards = discover_shards(population)
    wanted = set(ids or ())
    order = list(numbers)
    if sample and rng is not None:
        rng.shuffle(order)
    seen = 0
    for shard in order:
        with ShardReader(population, shard, n_shards) as reader:
            for _index, truth, *arrays in reader.iter_systems():
                if wanted:
                    if int(truth["gaia_source_id"]) not in wanted:
                        continue
                elif min_snr is not None and n_companions:
                    snr = [truth[f"snr_total_{j}"] for j in range(1, n_companions + 1)]
                    if not all(np.isfinite(s) and s >= min_snr for s in snr):
                        continue
                yield audit(truth, tuple(arrays), n_companions)
                seen += 1
                if wanted:
                    wanted.discard(int(truth["gaia_source_id"]))
                    if not wanted:
                        return
                elif sample and seen >= sample:
                    return


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument(
        "--population",
        help="default 1_companion for --sample; every population for --ids",
    )
    parser.add_argument("--ids", type=int, nargs="+", help="specific gaia_source_ids")
    parser.add_argument("--sample", type=int, help="audit this many random systems")
    parser.add_argument("--min-snr", type=float, default=PG.HIGH_SNR_MIN)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--figure", type=Path, help="write the summary PNG here")
    args = parser.parse_args(argv)

    PG.set_catalog_root(args.catalog_root)
    rng = np.random.default_rng(args.seed)
    # With explicit ids, search every population: a source id does not say which
    # one it is in, and "no systems matched" because the default was wrong is a
    # confusing way to find that out.
    populations = (
        list(PG.POPULATIONS)
        if args.ids and args.population is None
        else [args.population or "1_companion"]
    )
    rows = []
    for population in populations:
        rows += list(
            iterate(
                population,
                int(population.split("_")[0]),
                ids=args.ids,
                sample=args.sample,
                min_snr=args.min_snr,
                rng=rng,
            )
        )
        if rows and args.ids and len(rows) == len(args.ids):
            break
    if not rows:
        raise SystemExit("no systems matched")

    if args.ids:
        for row in rows:
            report_one(row)
        return 0

    import pandas as pd

    frame = pd.DataFrame(rows)
    print(
        f"\n{args.population}: {len(frame):,} systems with snr_total >= {args.min_snr:g}"
    )
    ratio = frame["sigma_ratio"]
    print(
        f"\n  sigma_single / sigma_reported: median {ratio.median():.4f}, "
        f"range {ratio.min():.4f} - {ratio.max():.4f}"
    )
    print(
        f"  -> chi2 floor with NO companion is {ratio.median() ** 2:.3f}, not 1."
        "\n     Deliberate: the injected scale carries the calibration term and the"
        "\n     reported one does not. Divided out of everything below."
    )
    tail = float((ratio > 3.0).mean())
    if tail > 0.001:
        print(
            f"  !! {tail:.2%} of systems have sigma_injected > 3x sigma_reported"
            f" (max {ratio.max():.1f}x)."
            "\n     Those look badly over-dispersed to any fit. Worth checking that the"
            "\n     tail of the noise model is intended."
        )

    counts = frame.apply(verdict, axis=1).value_counts()
    print("\n  verdicts:")
    for label, count in counts.items():
        print(f"    {count:>7,}  {count / len(frame):>6.1%}  {label}")

    print("\n  measured vs claimed, binned by recorded snr_total:")
    print(
        f"    {'snr_total':>16}  {'n':>7}  {'retained':>9}  "
        f"{'detectable':>11}  {'measured':>9}  {'floor':>7}  {'chi2 k=5':>9}  "
        f"{'predicted':>9}"
    )
    edges = np.array([5, 10, 20, 40, 80, 160, np.inf])
    index = np.clip(np.digitize(frame["snr_total"], edges) - 1, 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        sel = index == b
        if not sel.sum():
            continue
        top = edges[b + 1]
        label = f"{edges[b]:g}-" + ("inf" if not np.isfinite(top) else f"{top:g}")
        print(
            f"    {label:>16}  {int(sel.sum()):>7,}  "
            f"{frame['retained'][sel].median():>8.1%}  "
            f"{frame['snr_detectable'][sel].median():>11.2f}  "
            f"{frame['snr_measured'][sel].median():>9.2f}  "
            f"{frame['chi2_floor'][sel].median():>7.3f}  "
            f"{frame['chi2_astro'][sel].median():>9.3f}  "
            f"{frame['chi2_predicted'][sel].median():>9.3f}"
        )
    print(
        "\n  'measured' tracking 'snr_total' means the recorded SNR is honest."
        "\n  'retained' well below 100% with 'measured' tracking 'detectable' instead"
        "\n  means the orbit is real and the five-parameter fit ate it -- snr_eff is"
        "\n  not charging enough for long periods, and a RAILED verdict is correct."
        "\n  'measured' near zero while 'detectable' is high means the signal is not"
        "\n  in the data at the recorded amplitude, which would be a generator bug."
    )

    if args.figure:
        save_figure(frame, args)
    return 0


def save_figure(frame, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")

    ax = axes[0]
    ax.scatter(
        frame["snr_total"],
        frame["snr_measured"],
        s=6,
        lw=0,
        alpha=0.35,
        color="#050CDB",
        label="measured in the data",
    )
    ax.scatter(
        frame["snr_total"],
        frame["snr_detectable"],
        s=6,
        lw=0,
        alpha=0.35,
        color="#6A1B9A",
        label="detectable (geometry)",
    )
    grid = np.array([frame["snr_total"].min(), frame["snr_total"].max()])
    ax.plot(grid, grid, "-", color="#C2185B", lw=1.5, label="1:1 (honest)")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel=r"recorded snr$_{\rm tot}$",
        ylabel="SNR",
    )
    ax.legend(fontsize=8)
    ax.set_title("does the recorded SNR exist in the data?", fontsize=10)

    ax = axes[1]
    ax.scatter(
        frame["period_min_yr"], frame["retained"], s=6, lw=0, alpha=0.4, color="#6A1B9A"
    )
    from epochalypse import constants as k

    ax.axvline(
        k.DR4_BASELINE_YEARS,
        color="#C2185B",
        ls="--",
        lw=1.3,
        label=f"{k.DR4_BASELINE_YEARS} yr baseline",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="injected period [yr]",
        ylabel="orbit amplitude surviving the 5-param fit",
    )
    ax.legend(fontsize=8)
    ax.set_title("what the astrometric solution eats", fontsize=10)

    ax = axes[2]
    ax.scatter(
        frame["snr_detectable"],
        frame["snr_measured"],
        s=6,
        lw=0,
        alpha=0.4,
        color="#050CDB",
    )
    lim = np.array(
        [max(frame["snr_detectable"].min(), 0.1), frame["snr_detectable"].max()]
    )
    ax.plot(lim, lim, "-", color="#C2185B", lw=1.5, label="1:1")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="detectable SNR (geometry)",
        ylabel="measured SNR (data)",
    )
    ax.legend(fontsize=8)
    ax.set_title("if this is 1:1, the generator is fine", fontsize=10)
    fig.suptitle(
        f"{args.population}: is the recorded SNR the signal in the data?", fontsize=11
    )
    path = Path(args.figure)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
