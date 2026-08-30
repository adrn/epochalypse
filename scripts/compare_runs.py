#!/usr/bin/env python
"""Put several harv runs side by side, so an arm can be judged against a baseline.

    python scripts/compare_runs.py \
        --roots $HARV_ROOT-reported $HARV_ROOT-injected $HARV_ROOT-jitter \
        --labels reported injected jitter \
        --figure $HARV_ROOT-compare/compare.png

Every run writes its own `--output-root` with its own manifest, census and
figures. That keeps arms from contaminating each other, and makes them
awkward to compare: the numbers that matter live in six separate text dumps and
the figures are six separate PNGs that have to be flipped between.

This reads the per-system tables directly -- no merge required -- and produces
one table and one figure per question:

  headline    n, ESS, railed, recovered, per run
  by period   recovery in each injected-period decade, per run
  by SNR      rail fraction against SNR, per run -- the curve that says whether
              the amplitude prior or the data is setting the threshold
  settings    what actually differed between the arms, read from the manifests
              rather than from the directory names

**The settings table is the one to read first.** An arm that differs in
`n_prior_samples` as well as in the thing being tested is not a comparison, and
this is where that shows up. `sweep_summary.py` learned the same lesson for the
sigma_a0 sweep; the difference here is that these arms differ in the LIKELIHOOD
(the weights on the data) rather than only in the prior, so their `logZ_int`
values are not comparable across arms even when everything else matches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from epochalypse.harv import census
from epochalypse.harv import config as C
from epochalypse.periodogram import config as PG

# The settings worth diffing across arms. Anything that changes what the fit
# saw, plus the two that make a comparison invalid if they drift.
SETTINGS = (
    "n_prior_samples",
    "top_k",
    "period_min_yr",
    "period_max_yr",
    "m_max_mjup",
    "sigma_a0_au",
    "jitter_sigma_mas",
    "eccentricity_prior",
    "fingerprint",
)


def load(root, population):
    """One run's per-system frame plus the census flags, and its manifest."""
    C.set_output_root(root)
    manifest = {}
    if C.manifest_path().exists():
        manifest = json.loads(C.manifest_path().read_text())
    extra = ("n_epochs",)
    table = census.read_systems(population, census.system_columns(population, extra))
    frame = table.to_pandas()
    frame["recovered"] = census.recovered(table, population)
    frame["railed"] = census.railed(table)
    frame["searchable"] = census.in_search_range(table, population)
    for name in ("period", "snr_total"):
        frame[f"{name}_best"] = census.best_truth(table, population, name)
    mask = census.high_snr_mask(table, population)
    if mask is not None:
        frame = frame[mask]
    return frame.reset_index(drop=True), manifest


def settings_table(runs):
    """What actually differed between the arms, from the manifests."""
    print("=== settings (from each run's manifest, not its directory name) ===")
    width = max(len(label) for label, _, _ in runs) + 2
    rows = []
    for name in SETTINGS:
        values = [str(m.get("library", {}).get(name, "-")) for _, _, m in runs]
        if len(set(values)) > 1 or name == "fingerprint":
            rows.append((name, values))
    if not rows:
        print("  every listed setting is identical across the arms.\n")
        return
    header = "".join(f"{label:>{width}}" for label, _, _ in runs)
    print(f"  {'setting':<22}{header}")
    for name, values in rows:
        print(f"  {name:<22}" + "".join(f"{v[: width - 2]:>{width}}" for v in values))
    differing = {n for n, _ in rows} - {"fingerprint"}
    if len(differing) > 1:
        print(
            f"\n  !! {len(differing)} settings differ: {sorted(differing)}."
            "\n     More than one moving part means this is not a controlled"
            "\n     comparison and any difference below is unattributable."
        )
    print()


def headline(runs, population):
    print(f"=== {population}, high-SNR, within the searched range ===")
    print(
        f"  {'run':<14}{'n':>9}{'ESS med':>10}{'railed':>9}{'recovered':>11}"
        f"{'wcap med':>10}"
    )
    for label, frame, _ in runs:
        inside = frame[frame["searchable"].astype(bool)]
        if inside.empty:
            continue
        print(
            f"  {label:<14}{len(inside):>9,}"
            f"{np.nanmedian(inside['ess']):>10.2f}"
            f"{inside['railed'].mean():>8.1%} "
            f"{inside['recovered'].mean():>10.1%} "
            f"{np.nanmedian(inside['weight_captured']):>10.4f}"
        )
    print()


def by_bin(runs, key, bins, value, title, fmt="{:.1%}"):
    """One row per bin, one column per run."""
    print(f"=== {title} ===")
    width = max(len(label) for label, _, _ in runs) + 2
    header = "".join(f"{label:>{width}}" for label, _, _ in runs)
    print(f"  {'bin':<20}{header}   n (first run)")
    reference = None
    for b in range(len(bins) - 1):
        cells, counts = [], []
        for _, frame, _ in runs:
            inside = frame[frame["searchable"].astype(bool)]
            index = census.bin_index(inside[key], bins)
            sel = index == b
            counts.append(int(sel.sum()))
            cells.append(
                fmt.format(float(inside[value].to_numpy()[sel].mean()))
                if sel.sum()
                else "-"
            )
        if not counts[0]:
            continue
        reference = reference or counts[0]
        label = f"{bins[b]:g} to {bins[b + 1]:g}"
        print(
            f"  {label:<20}"
            + "".join(f"{c:>{width}}" for c in cells)
            + f"   {counts[0]:>7,}"
        )
    print()


def overlay(runs, population, path):
    """Three curves per question, one line per run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")
    snr_bins = np.linspace(np.log10(PG.HIGH_SNR_MIN), 2.6, 10)
    for label, frame, _ in runs:
        inside = frame[frame["searchable"].astype(bool)]
        if inside.empty:
            continue
        inside = inside.assign(
            log_period=np.log10(inside["period_best"]),
            log_snr=np.log10(np.maximum(inside["snr_total_best"], 1e-3)),
        )
        for ax, key, bins, column in (
            (axes[0], "log_period", census.LOG_PERIOD_BINS, "recovered"),
            (axes[1], "log_snr", snr_bins, "railed"),
        ):
            index = census.bin_index(inside[key], bins)
            xs, ys = [], []
            for b in range(len(bins) - 1):
                sel = index == b
                if sel.sum() >= 5:
                    xs.append(0.5 * (bins[b] + bins[b + 1]))
                    ys.append(float(inside[column].to_numpy()[sel].mean()))
            ax.plot(xs, ys, "o-", lw=1.5, ms=4, label=label)
        ess = np.asarray(inside["ess"], float)
        ess = ess[np.isfinite(ess)]
        axes[2].hist(
            np.log10(np.maximum(ess, 1.0)),
            bins=50,
            histtype="step",
            lw=1.6,
            density=True,
            label=f"{label}  (median {np.median(ess):.1f})",
        )

    axes[0].set(
        xlabel=r"$\log_{10}$ injected period [yr]",
        ylabel="fraction recovered",
        ylim=(0, 1),
    )
    axes[0].set_title("recovery: higher is better", fontsize=10)
    axes[1].axvline(np.log10(PG.HIGH_SNR_MIN), color="k", ls="--", lw=1)
    axes[1].set(
        xlabel=r"$\log_{10}$ SNR$_{\rm tot}$", ylabel="railed fraction", ylim=(0, 1)
    )
    axes[1].set_title("non-detections: lower is better", fontsize=10)
    axes[2].axvline(np.log10(C.ESS_RESOLVED), color="k", ls="--", lw=1)
    axes[2].set(xlabel=r"$\log_{10}$ ESS", ylabel="density")
    axes[2].set_title(
        "did the library resolve it?\n(higher is better here, unlike recovery)",
        fontsize=9,
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.suptitle(f"{population}: arms compared on identical systems", fontsize=11)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  wrote {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", help="default: directory names")
    parser.add_argument("--population", default="1_companion")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)

    labels = args.labels or [root.name for root in args.roots]
    if len(labels) != len(args.roots):
        parser.error("--labels must match --roots")

    runs = []
    for label, root in zip(labels, args.roots):
        frame, manifest = load(root, args.population)
        runs.append((label, frame, manifest))
        print(f"{label:<14} {len(frame):>9,} systems   {root}")
    print()

    settings_table(runs)
    headline(runs, args.population)
    by_bin(
        runs,
        "period_best",
        10.0**census.LOG_PERIOD_BINS,
        "recovered",
        "recovery by injected period [yr]",
    )
    by_bin(
        runs,
        "snr_total_best",
        census.SNR_BINS[:-1].tolist() + [1e9],
        "railed",
        "rail fraction by SNR",
    )
    if args.figure:
        overlay(runs, args.population, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
