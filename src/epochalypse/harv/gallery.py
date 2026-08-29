"""Per-system diagnostics: the data, the model, and the posterior samples.

The population figures say *how often* the fit works. They cannot say *what
happens* when it does not, and for that you have to look at individual systems.
This picks a representative handful -- `config.GALLERY_PER_BIN` from each cell
of a 2-D grid in (SNR, injected period) -- and draws four panels each.

**Start with the 0.79-1.26 yr cells.** A one-year orbit is degenerate with
parallax, because parallax is a free linear parameter in the model with a
deliberately broad prior, so the same along-scan signal can be attributed to
either. If that degeneracy is real, those systems' posteriors are **bimodal**:
one mode with a companion and a small parallax, one with no companion and an
inflated parallax. Panel (c) is drawn to show exactly that.

The model is reconstructed with harv's own `_base_design_matrix`, not a
reimplementation of the Thiele-Innes projection -- so `AL = X @ theta` for the
nine linear parameters stored beside each sample, and the orbit is the last four
columns. A reimplementation would be a second thing to keep correct.
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

# The nine linear parameters, in the column order `_base_design_matrix` returns.
LINEAR = ("ra0", "dec0", "pmra", "pmdec", "parallax", "ti_A", "ti_B", "ti_F", "ti_G")


def semi_major_axis_mas(ti_a, ti_b, ti_f, ti_g):
    """`a0` from the Thiele-Innes constants (Halbwachs & Pourbaix identity).

    `u = (A^2+B^2+F^2+G^2)/2`, `v = AG - BF`, `a0 = sqrt(u + sqrt(u^2 - v^2))`.
    The same identity harv uses in its Jacobian correction, so the amplitude
    plotted here is the one the prior and the correction act on.
    """
    u = 0.5 * (ti_a**2 + ti_b**2 + ti_f**2 + ti_g**2)
    v = ti_a * ti_g - ti_b * ti_f
    return np.sqrt(u + np.sqrt(np.maximum(u * u - v * v, 0.0)))


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

    Deterministic -- systems are taken in `gaia_source_id` order -- so the same
    catalog always yields the same gallery and a figure can be compared across
    runs.
    """
    per_bin = C.GALLERY_PER_BIN if per_bin is None else int(per_bin)
    extra = ("shard", "shard_row", "gaia_source_id", "parallax_mas", "n_epochs")
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
    return (
        frame.sort_values("gaia_source_id")
        .groupby("cell", observed=True)
        .head(per_bin)
        .reset_index(drop=True)
    )


def _model_al(data, model, block, index):
    """`(design matrix, theta)` for one stored draw, via harv's own model."""
    from unxt import Quantity as Q

    nl = {
        # period keeps its unit (it is a time); the other two must be bare, as
        # `_solve_kepler` strips only the mean anomaly before the Kepler solve
        "period": Q(float(np.asarray(block["period"])[index]), "yr"),
        "eccentricity": float(np.asarray(block["eccentricity"])[index]),
        "phase_peri": float(np.asarray(block["phase_peri"])[index]),
    }
    theta = np.array([float(np.asarray(block[name])[index]) for name in LINEAR])
    return np.asarray(model._base_design_matrix(nl, data)), theta


def plot_system(row, block, arrays, out_dir):
    """Four panels for one system: the data, and where the posterior went."""
    t, psi, pf, y, yerr = arrays
    data, par, n_epochs = adapt.prepare(t, psi, pf, y, yerr)
    model = L.model(par)

    weight = np.asarray(block["weight"], float)
    period = np.asarray(block["period"], float)
    parallax = np.asarray(block["parallax"], float)
    a0 = semi_major_axis_mas(*(np.asarray(block[f"ti_{c}"], float) for c in "ABFG"))
    best = int(np.argmax(weight))

    X, theta = _model_al(data, model, block, best)
    al = np.asarray(data.al_position.value)
    err = np.asarray(data.al_position_err.value)
    real = err < adapt.PAD_ERR_MAS / 2  # drop the padded rows
    t_yr = np.asarray(data.time.value) - float(data.t_ref.value)
    astrometric = X[:, :5] @ theta[:5]  # the five-parameter solution
    orbit = X[:, 5:] @ theta[5:]  # the companion's own contribution

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    p_true, snr = row["period_best"], row["snr_total_best"]
    fig.suptitle(
        f"gaia {int(row['gaia_source_id'])}   SNR$_{{tot}}$={snr:.1f}   "
        f"$P_{{true}}$={p_true:.4f} yr   $e$={row['ecc_best']:.2f}   "
        f"$\\alpha$={row['alpha_mas_best']:.3f} mas   {n_epochs} epochs   "
        f"ESS={row['ess']:.1f}   "
        + (
            "RECOVERED"
            if row["recovered"]
            else ("RAILED" if row["railed"] else "MISSED")
        ),
        fontsize=11,
    )

    # (a) the data, with the 5-parameter astrometric solution removed. Raw AL is
    # ~250 mas rms and the orbit ~1 mas, so nothing is visible until it is.
    ax = axes[0, 0]
    ax.errorbar(
        t_yr[real],
        (al - astrometric)[real],
        yerr=err[real],
        fmt="o",
        ms=3,
        lw=0.8,
        color="0.25",
        label="data $-$ astrometric solution",
    )
    order = np.argsort(t_yr[real])
    ax.plot(
        t_yr[real][order],
        orbit[real][order],
        "-",
        lw=1.2,
        color="#C2185B",
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
    ax.plot(span, span, "-", lw=1.2, color="#C2185B", label="1:1")
    chi2 = np.mean(((observed - predicted) / err[real]) ** 2)
    ax.set(xlabel="model orbit term [mas]", ylabel="data $-$ astrometric [mas]")
    ax.legend(fontsize=7.5)
    ax.set_title(
        rf"best-weight draw: $\chi^2/N$ = {chi2:.1f}"
        + ("" if chi2 < 3 else "   (library too coarse to fit well)"),
        fontsize=10,
    )

    # (c) THE degeneracy panel. A one-year orbit and the parallax term explain
    # the same signal, so a bimodal posterior here is the diagnosis.
    ax = axes[1, 0]
    scat = ax.scatter(
        period,
        parallax,
        c=np.log10(np.maximum(weight, 1e-12)),
        s=14,
        cmap="viridis",
        lw=0,
    )
    fig.colorbar(scat, ax=ax, label=r"$\log_{10}$ weight")
    ax.axvline(p_true, color="#C2185B", lw=1.2, label="injected period")
    ax.axvline(1.0, color="#0288D1", ls="--", lw=1.2, label="1 yr (parallax period)")
    ax.axhline(
        row["parallax_mas"], color="#C2185B", ls=":", lw=1.2, label="true parallax"
    )
    ax.set(
        xscale="log",
        xlabel="period [yr]",
        ylabel="parallax [mas]",
        xlim=(C.PERIOD_MIN_YR, C.PERIOD_MAX_YR),
    )
    ax.legend(fontsize=7)
    ax.set_title("period vs parallax: two clumps = the 1 yr degeneracy", fontsize=10)

    # (d) amplitude vs period. The no-orbit mode lives at a0 -> 0 and the
    # shortest period, which is what "railed" means.
    ax = axes[1, 1]
    scat = ax.scatter(
        period,
        np.maximum(a0, 1e-6),
        c=np.log10(np.maximum(weight, 1e-12)),
        s=14,
        cmap="viridis",
        lw=0,
    )
    fig.colorbar(scat, ax=ax, label=r"$\log_{10}$ weight")
    ax.axvline(p_true, color="#C2185B", lw=1.2)
    ax.axhline(
        row["alpha_mas_best"],
        color="#C2185B",
        ls=":",
        lw=1.2,
        label=r"injected $\alpha$",
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
            for index, _truth, t, psi, pf, y, yerr in reader.iter_systems():
                if index not in wanted:
                    continue
                row = chosen.loc[wanted[index]]
                written.append(
                    plot_system(
                        row, blocks[index], (t, psi, pf, y, yerr), root / row["cell"]
                    )
                )
    if verbose:
        cells = chosen["cell"].nunique()
        print(f"  {len(written)} figures across {cells} (SNR, period) cells -> {root}")
    return written
