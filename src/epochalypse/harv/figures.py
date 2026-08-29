"""Four diagnostic figures, each answering a question this stage actually raised.

| figure | question |
| --- | --- |
| `harv_recovery_map` | where in (period, eccentricity) does the method work? |
| `harv_period_aliases` | when it fails, *where* does the period land? |
| `harv_library` | did the prior library resolve these posteriors? |
| `harv_detection` | at what signal strength does recovery turn on? |
| `harv_amplitude` | is the amplitude PRIOR setting the detection threshold? |
| `harv_precision` | is the reported period an estimate with an honest error bar? |

They are diagnostics, not paper figures: PNG only, no TeX. The catalog figures
use `usetex=True`, which has already failed once on a machine without a TeX
install, and a missing font must not take down a finish job.

Read them in order. `harv_recovery_map` says whether a low recovery number is
the baseline's fault (bad everywhere except 0.1-10 yr) or the prior's (bad at
high eccentricity even in the sweet spot). `harv_period_aliases` then says
whether the misses are aliases -- clustered on the annual or 2x tracks, which
more prior samples fix -- or scattered, which means the data does not constrain
those systems and no library size helps. `harv_library` sizes the second pass.
`harv_detection` is the completeness curve -- and `harv_amplitude` is what it
cannot answer, because a rail rate against SNR shows the symptom of a too-broad
amplitude prior without ever showing the prior. `harv_precision` is last and is
about believing the numbers rather than about how many of them are right.
"""

from __future__ import annotations

import matplotlib
import numpy as np

from ..periodogram import config as PG
from . import census
from . import config as C
from . import library as L

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = (
    "recovery_map",
    "period_aliases",
    "library",
    "detection",
    "amplitude",
    "precision",
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


TRUTHS = ("period", "ecc", "snr_total")


def _frame(population, high_snr=True, in_range=False, extra=(), truths=TRUTHS):
    """One population as a DataFrame with the census flags and matched truths.

    `in_range` drops systems injected outside the searched period range. Any
    panel quoting *recovery* wants that, because those systems cannot be
    recovered by construction; panels showing ESS or evidence want the whole
    sample.

    `truths` names the per-companion columns to resolve to the matched
    companion. Anything in it must also be requested through `extra` for each
    companion, since `census.system_columns` only knows about the standard three.
    """
    table = census.read_systems(population, census.system_columns(population, extra))
    frame = table.to_pandas()
    frame["recovered"] = census.recovered(table, population)
    frame["railed"] = census.railed(table)
    frame["searchable"] = census.in_search_range(table, population)
    for name in truths:
        frame[f"{name}_best"] = census.best_truth(table, population, name)
    mask = census.high_snr_mask(table, population)
    if high_snr and mask is not None:
        frame = frame[mask]
    if in_range and frame["searchable"].notna().any():
        frame = frame[frame["searchable"].astype(bool)]
    return frame.reset_index(drop=True)


def _companion_columns(population, *names):
    """`("alpha_mas", ...)` -> every companion's copy, for `_frame(extra=...)`."""
    n = C.POPULATIONS[population]
    return tuple(f"{name}_{k}" for name in names for k in range(1, n + 1))


def _binned(frame, key, bins, value="recovered"):
    """(centres, fraction, n) of a boolean column over `bins`, skipping empties."""
    index = census.bin_index(frame[key], bins)
    values = frame[value].to_numpy()
    out = []
    for b in range(len(bins) - 1):
        sel = index == b
        n = int(sel.sum())
        if n:
            out.append((0.5 * (bins[b] + bins[b + 1]), float(values[sel].mean()), n))
    return (
        np.array([o[0] for o in out]),
        np.array([o[1] for o in out]),
        np.array([o[2] for o in out]),
    )


def _fraction_curve(ax, centres, fraction, n, color, annotate=True):
    """A binned fraction with binomial errors and per-point counts.

    The counts are the point: a bin holding three systems must not read as a
    trend, and the error bar alone does not always make that obvious.
    """
    err = np.sqrt(fraction * (1 - fraction) / np.maximum(n, 1))
    ax.errorbar(centres, fraction, yerr=err, marker="o", color=color, capsize=3, lw=1.5)
    if annotate:
        for x, y, m in zip(centres, fraction, n):
            ax.annotate(
                f"{m:,}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 9),
                ha="center",
                fontsize=7,
                color="0.35",
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
        _fraction_curve(ax, centres, fraction, n, POP_COLORS[population])
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
    _fraction_curve(axes[2], centres, fraction, n, POP_COLORS["1_companion"])
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
    _fraction_curve(axes[1], centres, fraction, n, POP_COLORS[population])
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
    xs, rail, ns = _binned(frame, "log_snr", bins, value="railed")
    _fraction_curve(axes[2], xs, rail, ns, "#C2185B")
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


# ==========================================================================
def plot_amplitude(population="1_companion"):
    """The orbit-amplitude prior against the amplitudes that were injected.

    This is the figure for the thing that actually sets the detection
    threshold. `sigma_a0` fixes the Occam penalty a real orbit pays against the
    no-orbit solution, and because harv scales the width as
    `(P/P0)^(2/3) x parallax` that penalty grows with period -- so it falls on
    real orbits and barely at all on the null. Everywhere else in these
    diagnostics that shows up only as its *symptom*, the rail rate.

    The middle panel is the one that decides the argument. Railing plotted
    against `alpha / sigma_a` says the prior is choking detections; railing that
    tracks only SNR says the data is too weak. They are different problems with
    different fixes, and a rail-vs-SNR curve alone cannot tell them apart.
    """
    frame = _frame(
        population,
        in_range=True,
        extra=("mass_st_msun", "parallax_mas")
        + _companion_columns(population, "alpha_mas", "mass_pl"),
        truths=(*TRUTHS, "alpha_mas", "mass_pl"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    fig.suptitle(
        f"{population}, high-SNR within the searched range -- {len(frame):,} systems.  "
        rf"prior: $\sigma_a(P) = \sigma_{{a0}}(M_\star)\,(P/P_0)^{{2/3}}\,\varpi$, "
        rf"$\sigma_{{a0}} = {C.M_MAX_MJUP:g}\,M_{{\rm Jup}} / M_\star^{{2/3}}$",
        fontsize=11,
    )

    period = np.asarray(frame["period_best"], float)
    alpha = np.asarray(frame["alpha_mas_best"], float)
    # the prior width at each system's OWN injected period, in mas
    scale = L.sigma_a0_au(frame["mass_st_msun"].to_numpy()) * np.asarray(
        frame["parallax_mas"], float
    )
    sigma_a = scale * (period / C.P0_YR) ** (2.0 / 3.0)

    ax = axes[0]
    for label, sel, color in (
        ("recovered", frame["recovered"].to_numpy().astype(bool), "#050CDB"),
        ("railed", frame["railed"].to_numpy().astype(bool), "#C2185B"),
        (
            "wrong period",
            ~frame["recovered"].to_numpy().astype(bool)
            & ~frame["railed"].to_numpy().astype(bool),
            "0.6",
        ),
    ):
        if sel.any():
            ax.scatter(
                period[sel], alpha[sel], s=6, lw=0, alpha=0.5, color=color, label=label
            )
    grid = np.logspace(np.log10(C.PERIOD_MIN_YR), np.log10(C.PERIOD_MAX_YR), 200)
    lo, mid, hi = np.nanpercentile(scale, [16, 50, 84])
    shape = (grid / C.P0_YR) ** (2.0 / 3.0)
    ax.plot(grid, mid * shape, "-", color="#6A1B9A", lw=1.8, label=r"$\sigma_a(P)$")
    ax.fill_between(
        grid, lo * shape, hi * shape, color="#6A1B9A", alpha=0.18, lw=0, label="16-84%"
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="injected period [yr]",
        ylabel=r"injected $\alpha$ [mas]",
        xlim=(C.PERIOD_MIN_YR, C.PERIOD_MAX_YR),
    )
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9)
    ax.set_title("injected amplitude against the prior on it", fontsize=10)

    # The middle panel. If railing turns off where alpha crosses sigma_a, the
    # prior is the threshold.
    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        frame["log_ratio"] = np.log10(alpha / sigma_a)
    good = frame[np.isfinite(frame["log_ratio"])]
    bins = np.linspace(-3.5, 1.5, 16)
    for value, color, label in (
        ("railed", "#C2185B", "railed (no detection)"),
        ("recovered", "#050CDB", "recovered"),
    ):
        centres, fraction, n = _binned(good, "log_ratio", bins, value=value)
        _fraction_curve(ax, centres, fraction, n, color, annotate=value == "railed")
        ax.plot([], [], "o-", color=color, label=label)
    ax.axvline(0.0, color="k", ls="--", lw=1)
    ax.text(
        0.0,
        0.97,
        r"  $\alpha = \sigma_a$",
        fontsize=8,
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.set(
        xlabel=r"$\log_{10}(\alpha_{\rm inj} / \sigma_a(P_{\rm true}))$",
        ylabel="fraction",
        ylim=(0, 1),
    )
    ax.legend(fontsize=7.5)
    ax.set_title(
        "railing against how many prior sigmas the truth sits at\n"
        "(a cliff here, not at a fixed SNR, means the PRIOR is the threshold)",
        fontsize=9,
    )

    # Panel 3, in the units the prior is actually expressed in.
    ax = axes[2]
    mass = np.asarray(frame["mass_pl_best"], float)
    edges = np.logspace(
        np.log10(max(np.nanmin(mass), 1e-3)), np.log10(np.nanmax(mass)), 40
    )
    for label, sel, color in (
        ("recovered", frame["recovered"].to_numpy().astype(bool), "#050CDB"),
        ("railed", frame["railed"].to_numpy().astype(bool), "#C2185B"),
    ):
        if sel.any():
            ax.hist(
                mass[sel],
                bins=edges,
                histtype="step",
                lw=1.8,
                color=color,
                label=f"{label}  ({int(sel.sum()):,})",
            )
    ax.axvline(C.M_MAX_MJUP, color="#6A1B9A", ls="--", lw=1.4)
    ax.text(
        C.M_MAX_MJUP,
        0.97,
        f"  M_MAX = {C.M_MAX_MJUP:g}",
        fontsize=8,
        va="top",
        transform=ax.get_xaxis_transform(),
    )
    ax.set(
        xscale="log",
        xlabel=r"injected companion mass [$M_{\rm Jup}$]",
        ylabel="systems",
    )
    ax.legend(fontsize=7.5)
    ax.set_title("who the prior expects, and who it loses", fontsize=10)
    return _save(fig, "amplitude")


# ==========================================================================
def plot_precision(population="1_companion"):
    """Is the reported period an estimate with an honest uncertainty on it?

    `period_wmean_yr` and `period_wstd_yr` are computed for every system and
    were never plotted. The first panel is the test that matters: if the stored
    spread were a real uncertainty, the pull would be a unit normal. At
    `ess ~ 1` it cannot be -- a handful of prior draws localize the peak but do
    not sample its width -- and the size of the failure is worth knowing before
    anyone quotes these as error bars.

    The third panel separates epoch count from SNR, which are correlated in the
    catalog. Cost is linear in the padded epoch count, so if recovery does not
    improve with epochs at fixed SNR, the largest buckets are buying nothing.
    """
    frame = _frame(
        population,
        in_range=True,
        extra=("n_transits_dr4", "period_wmean_yr", "period_wstd_yr"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout="constrained")
    fig.suptitle(
        f"{population}, high-SNR within the searched range -- {len(frame):,} systems",
        fontsize=11,
    )

    p_true = np.asarray(frame["period_best"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pull = (np.asarray(frame["period_wmean_yr"], float) - p_true) / np.asarray(
            frame["period_wstd_yr"], float
        )
    rec = frame["recovered"].to_numpy().astype(bool)

    ax = axes[0]
    edges = np.linspace(-10, 10, 81)
    for label, sel, color in (
        ("all in range", np.ones_like(rec), "0.45"),
        ("recovered", rec, "#050CDB"),
    ):
        values = pull[sel & np.isfinite(pull)]
        if not values.size:
            continue
        # A ROBUST scale, not np.std. When the library resolves a posterior with
        # one draw the stored spread collapses to zero, the pull divides by it,
        # and a standard deviation over the result is ~1e150 -- a number about
        # the outliers and not about the calibration.
        off = float(np.mean(np.abs(values) > 10))
        ax.hist(
            values,
            bins=edges,
            histtype="step",
            lw=1.8,
            density=True,
            color=color,
            label=f"{label}: med $|$pull$|$ {np.median(np.abs(values)):.3g}, "
            f"{off:.0%} past $\\pm$10",
        )
    x = np.linspace(-10, 10, 400)
    ax.plot(
        x,
        np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi),
        "--",
        color="k",
        lw=1.2,
        label="unit normal (calibrated)",
    )
    # The sharpest form of the finding: a spread of exactly zero is not a small
    # uncertainty, it is no uncertainty at all.
    dead = float(np.mean(np.asarray(frame["period_wstd_yr"], float) == 0.0))
    ax.text(
        0.02,
        0.72,
        f"{dead:.0%} report $P_{{\\rm wstd}} = 0$\nexactly: one draw holds\nall the weight",
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        color="#C2185B",
    )
    ax.set(xlabel=r"$(P_{\rm wmean} - P_{\rm true}) / P_{\rm wstd}$", ylabel="density")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("is the reported spread an uncertainty?", fontsize=10)

    ax = axes[1]
    snr = np.asarray(frame["snr_total_best"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = np.abs(np.asarray(frame["period_best_yr"], float) / p_true - 1.0)
    ok = rec & np.isfinite(error) & (error > 0)
    bins = np.linspace(np.log10(PG.HIGH_SNR_MIN), np.log10(np.nanmax(snr)), 10)
    index = census.bin_index(np.log10(np.maximum(snr, 1e-3)), bins)
    centres, med, lo, hi = [], [], [], []
    for b in range(len(bins) - 1):
        sel = (index == b) & ok
        if sel.sum() >= 5:
            q = np.nanpercentile(error[sel], [16, 50, 84])
            centres.append(0.5 * (bins[b] + bins[b + 1]))
            lo.append(q[0])
            med.append(q[1])
            hi.append(q[2])
    if centres:
        ax.plot(centres, med, "o-", color="#050CDB", lw=1.5)
        ax.fill_between(
            centres, lo, hi, color="#050CDB", alpha=0.2, lw=0, label="16-84%"
        )
        ax.legend(fontsize=7.5)
    ax.set(
        yscale="log",
        xlabel=r"$\log_{10}$ SNR$_{\rm tot}$",
        ylabel=r"$|P_{\rm best}/P_{\rm true} - 1|$",
    )
    ax.set_title("fractional period error, recovered only", fontsize=10)

    # Epoch count at FIXED SNR, so the two are not confounded.
    ax = axes[2]
    edges = np.array([40, 70, 90, 110, 140, 300])
    for b in range(len(census.SNR_BINS) - 1):
        band = (snr >= census.SNR_BINS[b]) & (snr < census.SNR_BINS[b + 1])
        if band.sum() < 20:
            continue
        centres, fraction, _ = _binned(frame[band], "n_transits_dr4", edges)
        if not len(centres):
            continue
        top = census.SNR_BINS[b + 1]
        ax.plot(
            centres,
            fraction,
            "o-",
            lw=1.4,
            label=f"SNR {census.SNR_BINS[b]:g}-"
            + ("inf" if not np.isfinite(top) else f"{top:g}")
            + f"  ({int(band.sum()):,})",
        )
    ax.set(xlabel="DR4 transits", ylabel="fraction recovered", ylim=(0, 1))
    ax.legend(fontsize=7)
    ax.set_title("epochs at fixed SNR", fontsize=10)
    return _save(fig, "precision")


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
