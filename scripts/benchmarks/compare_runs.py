#!/usr/bin/env python
"""Put several harv runs side by side, so an arm can be judged against a baseline.

    python scripts/benchmarks/compare_runs.py --roots $HARV_ROOT-err-* \
        --catalog-root $OUT_ROOT --figure $HARV_ROOT-compare/errors.png

Labels default to the directory names, which is safer than passing `--labels`
against a shell glob -- the two orderings have to agree and nothing checks that
they describe the same arms.

Put `--figure` where `--roots` cannot glob back onto it. A PNG written to
`$HARV_ROOT-err-compare.png` matches `$HARV_ROOT-err-*`, and the next
invocation treats it as an arm.

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
this is where that shows up.

**Read the rows against each other, never against the production run.** A sweep
runs at a smaller library size to stay cheap, which lowers recovery across every
arm; only the relative comparison at fixed M is meaningful, which is why a sweep
must always include the current setting as a control arm.

**`logZ_int` is not comparable across arms that changed the error model.** The
sigma_a0 arms differ only in the prior, so their evidences can be read against
each other. The error-model arms differ in the LIKELIHOOD -- the weights on the
data -- so theirs cannot, no matter how well everything else matches. Compare
them on recovery and rail fraction instead.

Every definition comes from `epochalypse.harv.census`, the same one the census
and the figures use, so this cannot drift from what they report.
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
    table = census.with_detectability(
        census.read_systems(population, census.system_columns(population, extra)),
        population,
    )
    frame = table.to_pandas()
    frame["recovered"] = census.recovered(table, population)
    frame["railed"] = census.railed(table)
    frame["searchable"] = census.in_search_range(table, population)
    names = ["period", "snr_total"]
    if census.has_detectability(table, population):
        names.append("snr_detectable")
    for name in names:
        frame[f"{name}_best"] = census.best_truth(table, population, name)
    mask = census.high_snr_mask(table, population)
    if mask is not None:
        frame = frame[mask]
    return frame.reset_index(drop=True), manifest


def settings_table(runs):
    """What actually differed between the arms, from the manifests."""
    print("=== settings (from each run's manifest, not its directory name) ===")
    width = max(len(label) for label, *_ in runs) + 2
    rows = []
    for name in SETTINGS:
        values = [str(run[2].get("library", {}).get(name, "-")) for run in runs]
        if len(set(values)) > 1 or name == "fingerprint":
            rows.append((name, values))
    if not rows:
        print("  every listed setting is identical across the arms.\n")
        return
    header = "".join(f"{label:>{width}}" for label, *_ in runs)
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


def jitter_posterior(root, population, n_systems=400, max_files=200):
    """What a jitter arm actually inferred, read off its stored samples.

    THE CONTROL ARM DEPENDS ON THIS. Learning an excess variance on data whose
    uncertainties are already correct should return jitter ~ 0. If it does not,
    the jitter is absorbing the LIBRARY's inadequacy -- an orbit the prior
    samples cannot match exactly is cheaper to explain as extra noise -- rather
    than the data's. That failure mode makes the jitter arm look good for the
    wrong reason, and it is invisible in recovery and rail rate alone.

    Reads files until it has `n_systems`, rather than a fixed number of files.
    A subsampled sweep writes one file per work unit and a unit may fit only a
    couple of systems, so "read two files" collected FOUR systems on a real
    sweep and quoted a median off them.
    """
    import pyarrow.parquet as pq

    C.set_output_root(root)
    files = sorted(C.samples_dir(population).glob("samples_*.parquet"))
    weighted = []
    for path in files[:max_files]:
        table = pq.read_table(path)
        if "jitter" not in table.column_names:
            return None
        for row in range(table.num_rows):
            jitter = np.asarray(table["jitter"][row].as_py(), float)
            lnw = np.asarray(table["ln_likelihood"][row].as_py(), float)
            if not np.isfinite(lnw).any():
                continue
            w = np.exp(lnw - np.nanmax(lnw))
            weighted.append(float((w * jitter).sum() / w.sum()))
        if len(weighted) >= n_systems:
            break
    return np.array(weighted) if weighted else None


def jitter_table(runs, population):
    """One line per arm that learned a jitter, or nothing if none did."""
    rows = []
    for label, _frame, manifest, root in runs:
        if manifest.get("library", {}).get("jitter_sigma_mas") is None:
            continue
        values = jitter_posterior(root, population)
        if values is None or not values.size:
            continue
        rows.append((label, values))
    if not rows:
        return
    print("=== learned jitter, from the stored samples ===")
    print(f"  {'run':<24}{'median':>10}{'16-84%':>20}{'n':>8}")
    for label, values in rows:
        lo, mid, hi = np.percentile(values, [16, 50, 84])
        flag = "   << too few to read" if values.size < 50 else ""
        print(f"  {label:<24}{mid:>10.4f}{lo:>11.4f} -{hi:>7.4f}{values.size:>8}{flag}")
    print(
        "\n  On an arm whose uncertainties are ALREADY correct (--error-mode"
        "\n  injected), a learned jitter should come back near zero. A large one"
        "\n  there means it is absorbing the library's inability to match the orbit"
        "\n  rather than any excess noise in the data -- which inflates the null's"
        "\n  likelihood and can make railing WORSE while looking like a better fit.\n"
    )


def headline(runs, population):
    print(f"=== {population}, high-SNR, within the searched range ===")
    print(
        f"  {'run':<14}{'n':>9}{'ESS med':>10}{'railed':>9}{'recovered':>11}"
        f"{'wcap med':>10}"
    )
    for label, frame, *_ in runs:
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
    width = max(len(label) for label, *_ in runs) + 2
    header = "".join(f"{label:>{width}}" for label, *_ in runs)
    print(f"  {'bin':<20}{header}   n (first run)")
    reference = None
    for b in range(len(bins) - 1):
        cells, counts = [], []
        for _label, frame, *_ in runs:
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
    column = (
        "snr_detectable_best"
        if "snr_detectable_best" in runs[0][1]
        else "snr_total_best"
    )
    for label, frame, *_ in runs:
        inside = frame[frame["searchable"].astype(bool)]
        if inside.empty:
            continue
        inside = inside.assign(
            log_period=np.log10(inside["period_best"]),
            log_snr=np.log10(np.maximum(inside[column], 1e-3)),
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
        xlabel=r"$\log_{10}$ SNR$_{\rm det}$"
        if "detect" in column
        else r"$\log_{10}$ SNR$_{\rm tot}$",
        ylabel="railed fraction",
        ylim=(0, 1),
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
    parser.add_argument(
        "--catalog-root",
        type=Path,
        help="needed to join the detectability columns, so the rail table bins "
        "on SNR_det rather than the recorded SNR. Without it the tables say "
        "which they used, but they say 'recorded'",
    )
    parser.add_argument("--population", default="1_companion")
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args(argv)

    # A shell glob picks up whatever matches, including this script's own
    # output: `--roots $HARV_ROOT-err-* --figure $HARV_ROOT-err-compare.png`
    # writes a PNG that the next invocation's glob then treats as a run root.
    # Say that, rather than dying six frames down in read_systems.
    roots = [root for root in args.roots if root.is_dir()]
    dropped = [root for root in args.roots if not root.is_dir()]
    if dropped:
        print(
            "skipping "
            + ", ".join(root.name for root in dropped)
            + " -- not a directory."
            + (
                "\n  A figure written inside the globbed path matches the glob;"
                "\n  put it somewhere the pattern does not reach."
                if any(root.suffix in (".png", ".pdf") for root in dropped)
                else ""
            )
            + "\n"
        )
    missing = [r for r in roots if not (r / "systems").exists()]
    if missing:
        parser.error(
            "no harv output under: "
            + ", ".join(str(r) for r in missing)
            + " (expected a systems/ directory in each)"
        )
    if not roots:
        parser.error("no run directories to compare")
    args.roots = roots

    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)

    labels = args.labels or [root.name for root in args.roots]
    if len(labels) != len(args.roots):
        parser.error(f"--labels has {len(labels)} entries for {len(roots)} roots")

    runs = []
    for label, root in zip(labels, args.roots):
        frame, manifest = load(root, args.population)
        runs.append((label, frame, manifest, root))
        print(f"{label:<14} {len(frame):>9,} systems   {root}")
    print()

    settings_table(runs)
    jitter_table(runs, args.population)
    headline(runs, args.population)
    by_bin(
        runs,
        "period_best",
        10.0**census.LOG_PERIOD_BINS,
        "recovered",
        "recovery by injected period [yr]",
    )
    # The detectable SNR when the projection stage has run: binning arms on
    # snr_total puts systems the astrometric fit ate into the same cell as ones
    # it kept, and a rail rate averaged over those two is not a rate.
    column = (
        "snr_detectable_best"
        if "snr_detectable_best" in runs[0][1]
        else "snr_total_best"
    )
    by_bin(
        runs,
        column,
        census.SNR_BINS[:-1].tolist() + [1e9],
        "railed",
        f"rail fraction by {'DETECTABLE' if 'detect' in column else 'recorded'} SNR",
    )
    if args.figure:
        overlay(runs, args.population, args.figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
