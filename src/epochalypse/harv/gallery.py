"""Per-system diagnostics: the data, the model, and the posterior samples.

The population figures say *how often* the fit works. They cannot say *what
happens* when it does not, and for that you have to look at individual systems.
This picks a representative handful -- `config.GALLERY_PER_BIN` from each cell
of a 2-D grid in (SNR, injected period) -- and draws six panels each.

**Start with the 0.79-1.26 yr cells.** A one-year orbit is degenerate with
parallax, because parallax is a free linear parameter in the model with a
deliberately broad prior, so the same along-scan signal can be attributed to
either. If that degeneracy is real, those systems' posteriors are **bimodal**:
one mode with a companion and a small parallax, one with no companion and an
inflated parallax. Panels (c) and (d) are drawn to show exactly that.

**The samples are weighted, and almost none of them count.** `TOP_K` keeps 1024
draws by *rank*, not by merit, so at `ess ~ 1-8` a handful carry the mass and the
rest are prior draws that happened to rank highest. Drawing them all alike is why
these panels used to be a picture of the prior -- see `weights()`.

The model is reconstructed with harv's own `_base_design_matrix` via
`library.design_matrix`, not a reimplementation of the Thiele-Innes projection --
so `AL = X @ theta` for the nine linear parameters stored beside each sample, and
the orbit is the last four columns. A reimplementation would be a second thing to
keep correct.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pyarrow.dataset as ds

from ..periodogram.shards import ShardReader, discover_shards
from . import adapt, census
from . import config as C
from . import figures as _figures
from . import library as L

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORBIT = "#C2185B"
PARALLAX = "#0288D1"


def weights(block, mass=None):
    """Importance weights and the posterior/prior split, **from the logs**.

    harv's weight is `exp(ln_likelihood - (logZ_int + ln M))`, normalized over
    the whole prior library. `ln_prior` is not in it -- the library is drawn
    *from* the prior, so a prior draw's importance weight is the likelihood
    alone, and adding the prior back would double-count.

    Recomputing it from the stored `ln_likelihood` rather than reading a stored
    `weight` column is what makes these panels honest. A strong detection spans
    ~1e-130 in weight: at `SAMPLE_DTYPE = float32` that carries no information
    below ~1e-38 and stores as exactly 0.0 below ~1e-45, so a stored weight
    column has a hard floor that has nothing to do with the posterior. The logs
    never underflow. (It also works unchanged on output from before the weight
    column was dropped, because `ln_likelihood` was always stored.)

    Returns `(w, keep)`: `w` normalized so the best draw is 1.0 -- an absolute
    normalization would be meaningless here anyway, since the denominator runs
    over a library these 1024 rows are a tail of -- and `keep` the draws inside
    `config.GALLERY_WEIGHT_MASS` of the total mass. The rest are **prior draws,
    not bad fits**, and belong in the background.

    Sorted here rather than trusting harv's descending top-K order: two lines,
    and it takes a cross-package assumption out of every panel. `test_harv.py`
    asserts the ordering separately, which is where a harv change should surface.
    """
    lnw = np.asarray(block["ln_likelihood"], dtype=np.float64)
    finite = np.isfinite(lnw)
    w = np.zeros_like(lnw)
    if not finite.any():
        return w, finite
    w[finite] = np.exp(lnw[finite] - lnw[finite].max())

    mass = C.GALLERY_WEIGHT_MASS if mass is None else float(mass)
    order = np.argsort(w)[::-1]
    cumulative = np.empty_like(w)
    cumulative[order] = np.cumsum(w[order])
    # `cumulative - w` is the mass ahead of each draw, so the draw that crosses
    # the threshold is kept rather than being the first one dropped.
    return w, finite & ((cumulative - w) < mass * w.sum())


def _weighted_scatter(ax, x, y, w, keep):
    """Draw weighted samples so the posterior is visible on top of the prior.

    Three encodings of the same number, because one is not enough at this
    dynamic range: color is `ln(w/w_best)` -- nats of likelihood below the best
    fit, which is interpretable in a way that `log10 weight` normalized over an
    unseen 10^6-sample library is not -- and size and opacity both track the
    weight, so a draw carrying 1% of the mass cannot look like one carrying 90%.
    Heavy draws are plotted last so they land on top.
    """
    if (~keep).any():
        ax.scatter(
            x[~keep],
            y[~keep],
            s=3,
            color="0.82",
            lw=0,
            zorder=1,
            label=f"{int((~keep).sum())} prior draws "
            f"(<{1 - C.GALLERY_WEIGHT_MASS:.1%} of the mass)",
        )
    order = np.argsort(w[keep])
    rel = w[keep][order]
    with np.errstate(divide="ignore"):
        shade = np.log(rel)
    return ax.scatter(
        x[keep][order],
        y[keep][order],
        c=shade,
        s=8 + 62 * rel,
        alpha=0.35 + 0.65 * rel,
        vmin=min(float(np.min(shade)), -1.0),
        vmax=0.0,
        cmap="magma_r",
        lw=0,
        zorder=2,
    )


def _cells(frame):
    """Assign each system to a (SNR, log-period) cell; returns the labels."""
    log_bins = np.asarray(C.GALLERY_LOG_PERIOD_BINS)
    si = census.bin_index(frame["snr_total_best"], census.SNR_BINS)
    pi = census.bin_index(np.log10(frame["period_best"]), log_bins)
    labels = []
    for s, p in zip(si, pi):
        hi = census.SNR_BINS[s + 1]
        labels.append(
            f"snr{census.SNR_BINS[s]:g}-{'inf' if not np.isfinite(hi) else f'{hi:g}'}"
            f"_logP{log_bins[p]:+.1f}to{log_bins[p + 1]:+.1f}"
        )
    return np.array(labels)


def select(population="1_companion", per_bin=None, high_snr=True):
    """`per_bin` systems from every populated (SNR, period) cell.

    **Stratified by outcome within each cell**, round-robin across recovered /
    railed / wrong-period, falling back to whatever is present when a category is
    empty. Taking the first eight in `gaia_source_id` order instead means a cell
    at 90% recovery shows eight successes and teaches nothing -- and the failures
    are the reason the gallery exists.

    Deterministic within each category, so the same catalog always yields the
    same gallery and a figure can be compared across runs.
    """
    per_bin = C.GALLERY_PER_BIN if per_bin is None else int(per_bin)
    extra = (
        "shard",
        "shard_row",
        "gaia_source_id",
        "parallax_mas",
        "mass_st_msun",
        "n_epochs",
        "n_padded",
    )
    n = C.POPULATIONS[population]
    columns = census.system_columns(population, extra) + [
        f"alpha_mas_{k}" for k in range(1, n + 1)
    ]
    table = census.read_systems(population, columns)
    frame = table.to_pandas()
    frame["recovered"] = census.recovered(table, population)
    frame["railed"] = census.railed(table)
    for name in ("period", "ecc", "snr_total", "alpha_mas"):
        frame[f"{name}_best"] = census.best_truth(table, population, name)
    mask = census.high_snr_mask(table, population)
    if high_snr and mask is not None:
        frame = frame[mask]
    frame = frame[
        census.in_search_range(table, population)[mask] if high_snr else slice(None)
    ]
    frame = frame.copy()
    frame["cell"] = _cells(frame)
    frame["outcome"] = np.where(
        frame["recovered"],
        "recovered",
        np.where(frame["railed"], "railed", "wrong period"),
    )
    frame = frame.sort_values("gaia_source_id")
    # rank within (cell, outcome), then order by rank first: that interleaves the
    # categories, so `head(per_bin)` deals one of each before a second of any.
    frame["_rank"] = frame.groupby(["cell", "outcome"], observed=True).cumcount()
    return (
        frame.sort_values(["cell", "_rank", "outcome"])
        .groupby("cell", observed=True)
        .head(per_bin)
        .drop(columns="_rank")
        .reset_index(drop=True)
    )


def plot_system(row, block, arrays, out_dir):
    """Six panels for one system: the data, the fit, and where the posterior went."""
    t, psi, pf, y, yerr = arrays
    data, par, n_epochs = adapt.prepare(t, psi, pf, y, yerr)
    model = L.model(par)

    w, keep = weights(block)
    period = np.asarray(block["period"], float)
    parallax = np.asarray(block["parallax"], float)
    a0 = L.semi_major_axis_mas(*(np.asarray(block[f"ti_{c}"], float) for c in "ABFG"))
    best = int(np.argmax(w))

    design = L.design_matrix(
        model,
        data,
        period[best],
        block["eccentricity"][best],
        block["phase_peri"][best],
    )
    theta = np.array([float(np.asarray(block[name])[best]) for name in L.LINEAR_ORDER])
    al = np.asarray(data.al_position.value, float)
    err = np.asarray(data.al_position_err.value, float)
    real = adapt.real_rows(err)
    t_yr = np.asarray(data.time.value, float) - float(data.t_ref.value)
    astrometric = design[:, :5] @ theta[:5]  # the five-parameter solution
    orbit = design[:, 5:] @ theta[5:]  # the companion's own contribution

    # The fit a companion has to beat: the best no-orbit solution. It absorbs
    # part of a real orbit into proper motion and parallax, which is why its
    # residuals in (a) are not simply the orbit curve.
    #
    # Delta chi^2 compares the two models at THEIR OWN maxima, so both sides are
    # refit -- `theta` above is a conditional draw, not an optimum, and scoring
    # it against a least-squares null is not a likelihood ratio. `chi2_draw` is
    # what the sampler actually reported and is the one panel (b) plots.
    theta_null, chi2_null = adapt.linear_solution(design, al, err, 5)
    _, chi2_fit = adapt.linear_solution(design, al, err)
    null_resid = al - design[:, :5] @ theta_null
    chi2_draw = adapt.chi2(al - design @ theta, err)
    n_real = int(real.sum())

    sigma_a0 = L.sigma_a0_au(row["mass_st_msun"])

    fig, axes = plt.subplots(2, 3, figsize=(19.5, 9.0), layout="constrained")
    p_true, snr = row["period_best"], row["snr_total_best"]
    fig.suptitle(
        f"gaia {int(row['gaia_source_id'])}   SNR$_{{tot}}$={snr:.1f}   "
        f"$P_{{true}}$={p_true:.4f} yr   $e$={row['ecc_best']:.2f}   "
        f"$\\alpha$={row['alpha_mas_best']:.3f} mas   "
        + (
            "RECOVERED"
            if row["recovered"]
            else ("RAILED" if row["railed"] else "MISSED")
        ),
        fontsize=12,
    )

    # (a) the data, with the 5-parameter astrometric solution removed. Raw AL is
    # ~250 mas rms and the orbit ~1 mas, so nothing is visible until it is.
    ax = axes[0, 0]
    order = np.argsort(t_yr[real])
    ax.plot(
        t_yr[real][order],
        null_resid[real][order],
        "o",
        ms=3,
        mfc="none",
        mec="0.7",
        mew=0.7,
        zorder=1,
        label="residual with NO orbit",
    )
    ax.errorbar(
        t_yr[real],
        (al - astrometric)[real],
        yerr=err[real],
        fmt="o",
        ms=3,
        lw=0.8,
        color="0.25",
        zorder=2,
        label="data $-$ astrometric solution",
    )
    ax.plot(
        t_yr[real][order],
        orbit[real][order],
        "-",
        lw=1.2,
        color=ORBIT,
        zorder=3,
        label="best-weight orbit term",
    )
    ax.set(xlabel="time $-$ $t_{ref}$ [yr]", ylabel="along-scan [mas]")
    ax.legend(fontsize=7.5)
    ax.set_title("the signal the fit is working with", fontsize=10)

    # (b) model against data. NOT a phase fold: along-scan is a 1-D projection
    # at a scan angle that changes epoch to epoch, so the same orbital phase
    # lands at different AL values and a fold never closes up even for a perfect
    # fit. A 1:1 comparison is projection-independent and says directly whether
    # the model explains the signal.
    ax = axes[0, 1]
    observed = (al - astrometric)[real]
    predicted = orbit[real]
    ax.errorbar(
        predicted, observed, yerr=err[real], fmt="o", ms=3, lw=0.8, color="0.25"
    )
    span = np.array(
        [min(predicted.min(), observed.min()), max(predicted.max(), observed.max())]
    )
    ax.plot(span, span, "-", lw=1.2, color=ORBIT, label="1:1")
    ax.set(xlabel="model orbit term [mas]", ylabel="data $-$ astrometric [mas]")
    ax.legend(fontsize=7.5)
    # Delta chi^2 is the fit improvement the orbit bought, BEFORE any Occam
    # penalty. A railed system with a large one is a detection the amplitude
    # prior rejected, which is a different diagnosis from a weak signal.
    ax.set_title(
        rf"best-weight draw: $\chi^2/N$ = {chi2_draw / n_real:.2f}"
        "\n"
        rf"refit at that period: $\Delta\chi^2$ = {chi2_null - chi2_fit:.1f} "
        rf"over {n_real} epochs",
        fontsize=10,
    )

    # (c) the period posterior itself. The scatter panels show where the draws
    # ARE; this shows how much they are worth, which is the only way multimodality
    # is visible when 1000+ of the 1024 draws carry no mass at all.
    ax = axes[0, 2]
    edges = np.linspace(np.log10(C.PERIOD_MIN_YR), np.log10(C.PERIOD_MAX_YR), 60)
    density, _ = np.histogram(np.log10(period), bins=edges, weights=w)
    if density.max() > 0:
        density = density / density.max()
    ax.stairs(density, edges, fill=True, color=ORBIT, alpha=0.75)
    ax.axvline(np.log10(p_true), color=ORBIT, lw=1.4, label="injected period")
    ax.axvline(0.0, color=PARALLAX, ls="--", lw=1.2, label="1 yr (parallax period)")
    ax.axvline(
        np.log10(C.PERIOD_MIN_YR * C.RAIL_FACTOR),
        color="k",
        ls="--",
        lw=1,
        label="rail threshold",
    )
    ax.set(
        xlabel=r"$\log_{10}$ period [yr]",
        ylabel="posterior density (peak = 1)",
        xlim=(edges[0], edges[-1]),
    )
    ax.legend(fontsize=7)
    ax.set_title("weighted period posterior: two peaks = degenerate", fontsize=10)

    # (d) THE degeneracy panel. A one-year orbit and the parallax term explain
    # the same signal, so a bimodal posterior here is the diagnosis.
    ax = axes[1, 0]
    scat = _weighted_scatter(ax, period, parallax, w, keep)
    fig.colorbar(scat, ax=ax, label=r"$\ln(w / w_{\rm best})$")
    ax.axvline(p_true, color=ORBIT, lw=1.2, label="injected period")
    ax.axvline(1.0, color=PARALLAX, ls="--", lw=1.2, label="1 yr")
    ax.axhline(row["parallax_mas"], color=ORBIT, ls=":", lw=1.2, label="true parallax")
    ax.set(
        xscale="log",
        xlabel="period [yr]",
        ylabel="parallax [mas]",
        xlim=(C.PERIOD_MIN_YR, C.PERIOD_MAX_YR),
    )
    ax.legend(fontsize=7)
    ax.set_title("period vs parallax: two clumps = the 1 yr degeneracy", fontsize=10)

    # (e) amplitude vs period, against the prior that shapes it. The no-orbit
    # mode lives at a0 -> 0 and the shortest period, which is what "railed"
    # means -- and the prior curve is what pushes it there.
    ax = axes[1, 1]
    scat = _weighted_scatter(ax, period, np.maximum(a0, 1e-6), w, keep)
    fig.colorbar(scat, ax=ax, label=r"$\ln(w / w_{\rm best})$")
    grid = np.logspace(np.log10(C.PERIOD_MIN_YR), np.log10(C.PERIOD_MAX_YR), 200)
    ax.plot(
        grid,
        sigma_a0 * (grid / C.P0_YR) ** (2.0 / 3.0) * row["parallax_mas"],
        "-",
        color="#6A1B9A",
        lw=1.4,
        zorder=3,
        label=rf"prior scale $\sigma_a(P)$, {C.M_MAX_MJUP:g} $M_{{\rm Jup}}$",
    )
    ax.axvline(p_true, color=ORBIT, lw=1.2)
    ax.axhline(
        row["alpha_mas_best"], color=ORBIT, ls=":", lw=1.2, label=r"injected $\alpha$"
    )
    ax.axvline(
        C.PERIOD_MIN_YR * C.RAIL_FACTOR,
        color="k",
        ls="--",
        lw=1,
        label="rail threshold",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="period [yr]",
        ylabel=r"$a_0$ from Thiele-Innes [mas]",
        xlim=(C.PERIOD_MIN_YR, C.PERIOD_MAX_YR),
    )
    ax.legend(fontsize=7)
    ax.set_title(
        r"amplitude vs period: $a_0 \to 0$ at the floor is the null", fontsize=10
    )

    # (f) the numbers, so the panels above do not have to carry them in titles.
    ax = axes[1, 2]
    ax.axis("off")
    ax.grid(False)
    n_keep = int(keep.sum())
    lines = [
        ("ESS", f"{row['ess']:.1f}  of {C.N_PRIOR_SAMPLES:,} library draws"),
        ("weight_captured", f"{row['weight_captured']:.4f}"),
        ("draws carrying the mass", f"{n_keep} of {len(w)} stored"),
        ("", ""),
        ("chi2/N, best draw", f"{chi2_draw / n_real:.3f}"),
        ("chi2/N, refit orbit", f"{chi2_fit / n_real:.3f}"),
        ("chi2/N, no orbit", f"{chi2_null / n_real:.3f}"),
        ("delta chi2 (refit)", f"{chi2_null - chi2_fit:.1f}"),
        ("logZ_int", f"{row['logZ_int']:.1f}"),
        ("", ""),
        ("epochs", f"{n_epochs} real, padded to {int(row['n_padded'])}"),
        ("host mass", f"{row['mass_st_msun']:.3f} Msun"),
        ("parallax", f"{row['parallax_mas']:.2f} mas (true)"),
        ("sigma_a0", f"{sigma_a0:.4f} AU"),
        ("injected alpha", f"{row['alpha_mas_best']:.4f} mas"),
        ("period, best draw", f"{period[best]:.5f} yr"),
        ("period, true", f"{p_true:.5f} yr"),
    ]
    ax.text(
        0.0,
        1.0,
        "\n".join(f"{k:<24}{v}" if k else "" for k, v in lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8.5,
    )
    ax.set_title("what the fit reported", fontsize=10, loc="left")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"gaia_{int(row['gaia_source_id'])}.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path


def read_samples(population, shard, n_shards, shard_rows):
    """The stored draws for specific systems of one shard.

    Globs the shard's sample files rather than naming one, because `--n-parts`
    splits a shard across several -- `samples_shard00001_of_00320_part00_of_03`
    and so on -- and the per-system table records the shard and the row but not
    which part they landed in. Naming the unsplit file works only for a run that
    used `--n-parts 1`, which is not what production does.

    The `shard_row` filter is pushed into the parquet reader, so a row group
    whose statistics cannot contain a wanted row is never read: one system's
    block of `TOP_K` draws is ~60 kB, but a whole part file is ~370 MB.
    """
    import pandas as pd

    pattern = f"samples_shard{shard:05d}_of_{n_shards:05d}*.parquet"
    files = sorted(C.samples_dir(population).glob(pattern))
    if not files:
        msg = f"no sample files matching {pattern} in {C.samples_dir(population)}"
        raise FileNotFoundError(msg)
    table = ds.dataset(files, format="parquet").to_table(
        filter=ds.field("shard_row").isin([int(r) for r in shard_rows])
    )
    if not table.num_rows:
        msg = (
            f"{population} shard {shard}: none of rows {sorted(shard_rows)} found "
            f"across {len(files)} sample file(s) -- are the systems and samples "
            "from the same run?"
        )
        raise ValueError(msg)
    return pd.DataFrame(table.to_pandas())


def make_gallery(population="1_companion", per_bin=None, verbose=True):
    """Draw the gallery, one subdirectory per (SNR, period) cell."""

    _figures._apply_style()
    chosen = select(population, per_bin)
    if chosen.empty:
        print(f"  no systems selected for {population}")
        return []
    root = C.figure_dir() / "gallery" / population
    _, n_shards = discover_shards(population)
    written = []
    # One reader per shard, not per system: a shard read is the expensive part.
    for shard, group in chosen.groupby("shard"):
        wanted = dict(zip(group["shard_row"], group.index))
        samples = read_samples(population, int(shard), n_shards, list(wanted))
        blocks = {int(r): samples.iloc[i] for i, r in enumerate(samples["shard_row"])}
        with ShardReader(population, int(shard), n_shards) as reader:
            for index, truth, t, psi, pf, y, yerr in reader.iter_systems():
                if index not in wanted:
                    continue
                row = chosen.loc[wanted[index]]
                # The samples are addressed by (shard, shard_row) and the epochs
                # by position in the shard, so pointing --catalog-root at a
                # catalog the run did not use lines up the wrong star's epochs
                # against the right star's posterior -- silently, and every panel
                # then looks like a fit that failed. The ids are in both tables;
                # check them rather than trusting the two roots to match.
                if int(truth["gaia_source_id"]) != int(blocks[index]["gaia_source_id"]):
                    msg = (
                        f"{population} shard {shard} row {index}: the catalog has "
                        f"gaia {int(truth['gaia_source_id'])} but the samples have "
                        f"gaia {int(blocks[index]['gaia_source_id'])}. "
                        "--catalog-root is not the catalog this run was fitted on."
                    )
                    raise ValueError(msg)
                written.append(
                    plot_system(
                        row, blocks[index], (t, psi, pf, y, yerr), root / row["cell"]
                    )
                )
    if verbose:
        cells = chosen["cell"].nunique()
        counts = chosen["outcome"].value_counts().to_dict()
        print(f"  {len(written)} figures across {cells} (SNR, period) cells -> {root}")
        print(f"  outcomes: {counts}")
    return written
