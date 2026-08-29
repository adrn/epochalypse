"""Four diagnostic figures, each answering a question this stage actually raised.

| figure | question |
| --- | --- |
| `harv_recovery_map` | where in (period, eccentricity) does the method work? |
| `harv_period_aliases` | when it fails, *where* does the period land? |
| `harv_library` | did the prior library resolve these posteriors? |
| `harv_detection` | at what signal strength does recovery turn on? |

They are diagnostics, not paper figures: PNG only, no TeX. The catalog figures
use `usetex=True`, which has already failed once on a machine without a TeX
install, and a missing font must not take down a finish job.

Read them in order. `harv_recovery_map` says whether a low recovery number is
the baseline's fault (bad everywhere except 0.1-10 yr) or the prior's (bad at
high eccentricity even in the sweet spot). `harv_period_aliases` then says
whether the misses are aliases -- clustered on the annual or 2x tracks, which
more prior samples fix -- or scattered, which means the data does not constrain
those systems and no library size helps. `harv_library` sizes the second pass.
`harv_detection` is the completeness curve.
"""

from __future__ import annotations

import matplotlib
import numpy as np

from ..periodogram import config as PG
from . import census
from . import config as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = (
    "recovery_map",
    "period_aliases",
    "library",
    "detection",
)

LOG_PERIOD_BINS = census.LOG_PERIOD_BINS
ECC_BINS = census.ECC_BINS
POP_COLORS = {
    "0_companion": "#777777",
    "1_companion": "#050CDB",
    "2_companion": "#C2185B",
}


def _apply_style():
    """Diagnostics typography. No TeX -- see the module docstring."""
    matplotlib.rc("font", family="sans-serif")
    matplotlib.rc("text", usetex=False)
    matplotlib.rc("axes", grid=True, axisbelow=True)
    matplotlib.rc("grid", alpha=0.25, linewidth=0.6)


def _save(fig, stem):
    C.figure_dir().mkdir(parents=True, exist_ok=True)
    path = C.figure_dir() / f"harv_{stem}.png"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")
    return path


def _frame(population, high_snr=True, in_range=False, extra=()):
    """One population as a DataFrame with the census flags and matched truths.

    `in_range` drops systems injected outside the searched period range. Any
    panel quoting *recovery* wants that, because those systems cannot be
    recovered by construction; panels showing ESS or evidence want the whole
    sample.
    """
    table = census.read_systems(population, census.system_columns(population, extra))
    frame = table.to_pandas()
    frame["recovered"] = census.recovered(table, population)
    frame["railed"] = census.railed(table)
    frame["searchable"] = census.in_search_range(table, population)
    for name in ("period", "ecc", "snr_total"):
        frame[f"{name}_best"] = census.best_truth(table, population, name)
    mask = census.high_snr_mask(table, population)
    if high_snr and mask is not None:
        frame = frame[mask]
    if in_range and frame["searchable"].notna().any():
        frame = frame[frame["searchable"].astype(bool)]
    return frame.reset_index(drop=True)


def _binned(frame, key, bins):
    """(centres, fraction recovered, n) over `bins`, skipping empty bins."""
    index = census.bin_index(frame[key], bins)
    out = []
    for b in range(len(bins) - 1):
        sel = index == b
        n = int(sel.sum())
        if n:
            out.append(
                (
                    0.5 * (bins[b] + bins[b + 1]),
                    float(frame["recovered"].to_numpy()[sel].mean()),
                    n,
                )
            )
    return (
        np.array([o[0] for o in out]),
        np.array([o[1] for o in out]),
        np.array([o[2] for o in out]),
    )


# ==========================================================================
def plot_recovery_map(population="1_companion"):
    """Recovery over (period, eccentricity), with both marginals.

    The grid is the point. A low headline recovery number has two unrelated
    causes and they compound: the 5.5-year baseline can only constrain roughly
    0.1-10 yr of a 7.8-decade prior, and the prior's eccentricity coverage may
    not match what was injected. Only the joint view separates them.
    """
    frame = _frame(population, in_range=True)
    frame["log_period"] = np.log10(frame["period_best"])

    fig, axes = plt.subplots(
        1, 3, figsize=(16, 4.6), gridspec_kw={"width_ratios": [1.5, 1, 1]}
    )
    fig.suptitle(
        f"{population}, high-SNR (SNR_tot >= {PG.HIGH_SNR_MIN:g}) within the searched "
        f"range {C.PERIOD_MIN_YR:g}-{C.PERIOD_MAX_YR:g} yr -- {len(frame):,} systems, "
        f"recovery = |ln(P_best/P_true)| < ln {PG.PERIOD_RECOVER_TOL:g}",
        fontsize=11,
    )

    grid = np.full((len(LOG_PERIOD_BINS) - 1, len(ECC_BINS) - 1), np.nan)
    counts = np.zeros_like(grid)
    pi = census.bin_index(frame["log_period"], LOG_PERIOD_BINS)
    ei = census.bin_index(frame["ecc_best"], ECC_BINS)
    rec = frame["recovered"].to_numpy()
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            sel = (pi == i) & (ei == j)
            if sel.sum():
                grid[i, j] = rec[sel].mean()
                counts[i, j] = sel.sum()

    ax = axes[0]
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        vmin=0,
        vmax=1,
        cmap="viridis",
        extent=(0, grid.shape[1], 0, grid.shape[0]),
    )
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if counts[i, j]:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{grid[i, j]:.0%}\n{int(counts[i, j])}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if grid[i, j] < 0.5 else "black",
                )
    ax.set_xticks(np.arange(grid.shape[1]) + 0.5)
    ax.set_xticklabels(
        [f"{ECC_BINS[j]:g}-{ECC_BINS[j + 1]:g}" for j in range(grid.shape[1])],
        fontsize=8,
    )
    ax.set_yticks(np.arange(grid.shape[0]) + 0.5)
    ax.set_yticklabels(
        [
            f"{LOG_PERIOD_BINS[i]:g} to {LOG_PERIOD_BINS[i + 1]:g}"
            for i in range(grid.shape[0])
        ],
        fontsize=8,
    )
    ax.set_xlabel("injected eccentricity")
    ax.set_ylabel(r"$\log_{10}$ injected period [yr]")
    ax.set_title("fraction recovered (count below)", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046)

    for ax, key, bins, xlabel in (
        (axes[1], "log_period", LOG_PERIOD_BINS, r"$\log_{10}$ injected period [yr]"),
        (axes[2], "ecc_best", ECC_BINS, "injected eccentricity"),
    ):
        centres, fraction, n = _binned(frame, key, bins)
        # binomial standard error, so a bin with 3 systems cannot be read as a trend
        err = np.sqrt(fraction * (1 - fraction) / np.maximum(n, 1))
        ax.errorbar(
            centres,
            fraction,
            yerr=err,
            marker="o",
            color=POP_COLORS[population],
            capsize=3,
            lw=1.5,
        )
        for x, y, m in zip(centres, fraction, n):
            ax.annotate(
                f"{m}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                color="0.35",
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("fraction recovered")
        ax.set_ylim(0, 1)
    axes[1].axvspan(np.log10(0.1), np.log10(10), color="0.85", zorder=0)
    axes[1].text(
        0.5,
        0.95,
        "0.1-10 yr:\nwhat a 5.5 yr,\n~80-epoch baseline\ncan constrain",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="0.35",
    )
    return _save(fig, "recovery_map")


def plot_period_aliases(population="1_companion"):
    """`P_best` against `P_true`, which is where the failure mode is visible.

    With ESS ~ 1 the reported period is a single prior draw, and an astrometric
    likelihood is multi-modal: the annual parallax term puts aliases at
    `1/P_best = 1/P_true +- 1/yr`, and the 2x / 0.5x harmonics compete too.
    Misses **on** those tracks are a resolution problem that more prior samples
    fix. Misses scattered everywhere, or piled at the prior edges, mean the data
    does not constrain those systems and no library size will help.
    """
    frame = _frame(population, in_range=True)
    p_true = np.asarray(frame["period_best"], float)
    p_best = np.asarray(frame["period_best_yr"], float)
    snr = np.asarray(frame["snr_total_best"], float)
    rec = frame["recovered"].to_numpy().astype(bool)

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.35, 1]}
    )
    ax = axes[0]
    grid = np.logspace(np.log10(C.PERIOD_MIN_YR), np.log10(C.PERIOD_MAX_YR), 400)
    ax.plot(grid, grid, "-", color="0.2", lw=1, label="correct")
    ax.fill_between(
        grid,
        grid / PG.PERIOD_RECOVER_TOL,
        grid * PG.PERIOD_RECOVER_TOL,
        color="0.2",
        alpha=0.18,
        lw=0,
        label=f"within {PG.PERIOD_RECOVER_TOL:g}x (counted recovered)",
    )
    for sign, style in ((+1, "--"), (-1, ":")):
        with np.errstate(divide="ignore", invalid="ignore"):
            alias = 1.0 / (1.0 / grid + sign * 1.0)  # 1/P +- 1 per year
        good = np.isfinite(alias) & (alias > 0)
        ax.plot(
            grid[good],
            alias[good],
            style,
            color="#C2185B",
            lw=1.2,
            label="annual alias" if sign > 0 else None,
        )
    ax.plot(
        grid, 2 * grid, "-.", color="#0288D1", lw=1.0, label=r"$2\times$ / $0.5\times$"
    )
    ax.plot(grid, 0.5 * grid, "-.", color="#0288D1", lw=1.0)

    order = np.argsort(snr)
    sc = ax.scatter(
        p_true[order],
        p_best[order],
        c=np.log10(np.maximum(snr[order], 1e-3)),
        s=11,
        cmap="plasma",
        alpha=0.8,
        lw=0,
    )
    fig.colorbar(sc, ax=ax, fraction=0.046, label=r"$\log_{10}$ SNR$_{\rm tot}$")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="injected period [yr]",
        ylabel="recovered period [yr]",
        xlim=(C.PERIOD_MIN_YR, C.PERIOD_MAX_YR),
        ylim=(C.PERIOD_MIN_YR, C.PERIOD_MAX_YR),
    )
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.set_title(f"{population}, high-SNR -- {rec.mean():.1%} recovered", fontsize=10)

    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.log10(p_best / p_true)
    ratio = ratio[np.isfinite(ratio) & ~rec]
    ax.hist(ratio, bins=np.linspace(-6, 6, 121), color="#C2185B", alpha=0.85)
    # the two harmonic marks sit only 0.6 dex apart on a 12-dex axis, so they
    # collide if both are centred -- push each away from the other
    for x, label, align in (
        (np.log10(2), r"$2\times$", "left"),
        (-np.log10(2), r"$0.5\times$", "right"),
    ):
        ax.axvline(x, color="#0288D1", ls="-.", lw=1)
        ax.text(
            x, ax.get_ylim()[1] * 0.95, label, fontsize=8, ha=align, color="#0288D1"
        )
    ax.axvspan(
        -np.log10(PG.PERIOD_RECOVER_TOL),
        np.log10(PG.PERIOD_RECOVER_TOL),
        color="0.2",
        alpha=0.18,
        lw=0,
    )
    ax.set(
        xlabel=r"$\log_{10}(P_{\rm best}/P_{\rm true})$ for the misses only",
        ylabel="systems",
    )
    ax.set_title(
        f"{len(ratio):,} misses: clustered = alias, flat = unconstrained", fontsize=10
    )
    return _save(fig, "period_aliases")


def plot_library(populations=None):
    """Did the library resolve these posteriors, and was `TOP_K` enough?

    Two diagnostics that answer different questions, plus the relation between
    ESS and recovery -- which runs *backwards* and is the panel most likely to
    be misread. A well-constrained period gives a sharp posterior, which a
    fixed-size library resolves with fewer effective samples. Low ESS where
    recovery is high is correct. ESS says "the library did not sample this
    posterior", never "the answer is bad".
    """
    populations = list(C.POPULATIONS) if populations is None else list(populations)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), layout="constrained")

    for population in populations:
        frame = _frame(population, high_snr=False)
        ess = np.asarray(frame["ess"], float)
        ess = ess[np.isfinite(ess)]
        axes[0].hist(
            np.log10(np.maximum(ess, 1.0)),
            bins=60,
            histtype="step",
            lw=1.8,
            color=POP_COLORS[population],
            density=True,
            label=f"{population}  (median {np.median(ess):.1f})",
        )
        wcap = np.asarray(frame["weight_captured"], float)
        axes[1].hist(
            wcap[np.isfinite(wcap)],
            bins=np.linspace(0, 1.0001, 60),
            histtype="step",
            lw=1.8,
            color=POP_COLORS[population],
            density=True,
            label=population,
        )

    axes[0].axvline(np.log10(C.ESS_RESOLVED), color="k", ls="--", lw=1)
    # right-aligned: the line often sits at the axis edge, and a left-aligned
    # label there is drawn outside the axes and clipped.
    axes[0].text(
        np.log10(C.ESS_RESOLVED),
        axes[0].get_ylim()[1] * 0.9,
        f"ESS = {C.ESS_RESOLVED:g}  \nbelow: a localization,  \nnot a posterior  ",
        fontsize=7.5,
        va="top",
        ha="right",
    )
    axes[0].set(xlabel=r"$\log_{10}$ ESS", ylabel="density")
    axes[0].set_title("did the library resolve the posterior?", fontsize=10)
    axes[0].legend(fontsize=7.5)

    axes[1].set(xlabel="weight_captured", ylabel="density")
    axes[1].set_title(f"was TOP_K = {C.TOP_K} enough?  (want 1.0)", fontsize=10)
    axes[1].legend(fontsize=7.5)

    # recovery against ESS -- the anti-correlation
    frame = _frame("1_companion", in_range=True)
    frame["log_ess"] = np.log10(np.maximum(np.asarray(frame["ess"], float), 1.0))
    bins = np.linspace(0, max(frame["log_ess"].max(), 1.0), 9)
    centres, fraction, n = _binned(frame, "log_ess", bins)
    err = np.sqrt(fraction * (1 - fraction) / np.maximum(n, 1))
    axes[2].errorbar(
        centres,
        fraction,
        yerr=err,
        marker="o",
        color=POP_COLORS["1_companion"],
        capsize=3,
        lw=1.5,
    )
    for x, y, m in zip(centres, fraction, n):
        axes[2].annotate(
            f"{m}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7,
            color="0.35",
        )
    axes[2].axvline(np.log10(C.ESS_RESOLVED), color="k", ls="--", lw=1)
    axes[2].set(xlabel=r"$\log_{10}$ ESS", ylabel="fraction recovered", ylim=(0, 1))
    axes[2].set_title(
        "1_companion high-SNR: recovery falls as ESS rises\n"
        "(sharp posterior = low ESS = well constrained)",
        fontsize=9,
    )
    return _save(fig, "library")


def plot_detection(population="1_companion"):
    """The null distribution, and the completeness curve against it.

    `0_companion` has no injected signal, so its `logZ_int` is the null: any
    detection statistic has to be read against it, exactly as the periodogram
    stage calibrates its thresholds on the same control. The right panel is the
    completeness curve -- recovery against injected SNR, which is what tells you
    where the method turns on rather than where you chose to cut.
    """
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), layout="constrained")

    for pop in C.POPULATIONS:
        frame = _frame(pop, high_snr=False)
        z = np.asarray(frame["logZ_int"], float)
        z = z[np.isfinite(z)]
        if not len(z):
            continue
        axes[0].hist(
            z,
            bins=80,
            histtype="step",
            lw=1.8,
            density=True,
            color=POP_COLORS[pop],
            label=f"{pop}  (median {np.median(z):.0f})",
        )
    axes[0].set(xlabel=r"$\log Z_{\rm int}$  (padding removed)", ylabel="density")
    axes[0].set_title("evidence: the control is the null distribution", fontsize=10)
    axes[0].legend(fontsize=7.5)

    frame = _frame(population, high_snr=False, in_range=True)
    snr = np.asarray(frame["snr_total_best"], float)
    frame["log_snr"] = np.log10(np.maximum(snr, 1e-3))
    bins = np.linspace(-2, np.nanmax(frame["log_snr"]), 12)
    centres, fraction, n = _binned(frame, "log_snr", bins)
    err = np.sqrt(fraction * (1 - fraction) / np.maximum(n, 1))
    axes[1].errorbar(
        centres,
        fraction,
        yerr=err,
        marker="o",
        capsize=3,
        lw=1.5,
        color=POP_COLORS[population],
    )
    for x, y, m in zip(centres, fraction, n):
        axes[1].annotate(
            f"{m:,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7,
            color="0.35",
        )
    axes[1].axvline(np.log10(PG.HIGH_SNR_MIN), color="k", ls="--", lw=1)
    axes[1].text(
        np.log10(PG.HIGH_SNR_MIN),
        0.95,
        f"  HIGH_SNR_MIN = {PG.HIGH_SNR_MIN:g}",
        fontsize=8,
        va="top",
    )
    axes[1].set(
        xlabel=r"$\log_{10}$ SNR$_{\rm tot}$ of the matched companion",
        ylabel="fraction recovered",
        ylim=(0, 1),
    )
    axes[1].set_title(f"{population}: completeness against injected SNR", fontsize=10)

    # Rail fraction against SNR -- the curve that says whether the AMPLITUDE
    # prior is setting the detection threshold rather than the data. sigma_a0
    # controls the Occam penalty on a real orbit relative to the no-orbit
    # solution, and that penalty grows with period because the prior width
    # scales as (P/P0)^(2/3). If railing falls off a cliff at some SNR and is
    # near zero above it, that cliff IS the threshold -- and it should sit at
    # HIGH_SNR_MIN, not well above it. A cliff at SNR ~ 7 against a cut at 5 is
    # what would justify changing sigma_a0.
    index = census.bin_index(frame["log_snr"], bins)
    xs, rail, ns = [], [], []
    for b in range(len(bins) - 1):
        sel = index == b
        if sel.sum():
            xs.append(0.5 * (bins[b] + bins[b + 1]))
            rail.append(float(frame["railed"].to_numpy()[sel].mean()))
            ns.append(int(sel.sum()))
    xs, rail, ns = np.array(xs), np.array(rail), np.array(ns)
    err = np.sqrt(rail * (1 - rail) / np.maximum(ns, 1))
    axes[2].errorbar(xs, rail, yerr=err, marker="o", capsize=3, lw=1.5, color="#C2185B")
    for x, y, m in zip(xs, rail, ns):
        axes[2].annotate(
            f"{m:,}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=7,
            color="0.35",
        )
    axes[2].axvline(np.log10(PG.HIGH_SNR_MIN), color="k", ls="--", lw=1)
    axes[2].text(
        np.log10(PG.HIGH_SNR_MIN),
        0.95,
        f"  HIGH_SNR_MIN = {PG.HIGH_SNR_MIN:g}",
        fontsize=8,
        va="top",
    )
    axes[2].set(
        xlabel=r"$\log_{10}$ SNR$_{\rm tot}$ of the matched companion",
        ylabel="railed fraction",
        ylim=(0, 1),
    )
    axes[2].set_title(
        "no-detection rate: a cliff above HIGH_SNR_MIN means\n"
        "the amplitude prior, not the data, sets the threshold",
        fontsize=9,
    )
    return _save(fig, "detection")


def make_figures(names=None):
    """Draw the requested diagnostics; unknown names raise rather than pass."""
    _apply_style()
    names = FIGURES if names is None else tuple(names)
    unknown = set(names) - set(FIGURES)
    if unknown:
        raise ValueError(f"unknown figure(s) {sorted(unknown)}; have {list(FIGURES)}")
    made = []
    for name in names:
        print(f"{name}:")
        made.append(globals()[f"plot_{name}"]())
    return made
