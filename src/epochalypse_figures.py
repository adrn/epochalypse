"""Shared classification + joint-figure machinery for the characterization notebooks.

Imported by both ``characterize_populations.ipynb`` (the baseline: circular single periodogram) and
``characterize_populations_robust.ipynb`` (the robustness suite: eccentric refinement + the
iterative two-planet periodogram), so the two notebooks classify and draw identically and only the
inputs differ. Nothing here reads or writes state on its own -- the caller supplies the frames, the
detection column and the classifier.

The four/five classes describe what the periodogram established about the PERIOD, in the vocabulary
of ``epochalypse_fitting.classify_periodogram``'s own ``klass``:

    undetected  below the null-calibrated detection threshold
    unimodal    one narrow peak -- "true"/"wrong" is whether that peak sits at the injected period
    broad       a competitive region too wide to localize (often railed to a period-grid edge, i.e.
                a lower limit rather than a mode). With ``split_multi`` the class is split on
                whether that REGION brackets the truth, which is the only question a non-localized
                constraint can answer -- its argmax is not trustworthy enough to test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import epochalypse_fitting as ef

MJUP = ef.MJUP_TO_MSUN
FIG_DIR = Path("outputs/figures")

# One marker area for every figure that goes in the paper -- the maps here and LW25 Fig. 1 as it is
# rebuilt in compare_lw25.ipynb, which imports these. Triangles are drawn slightly larger than
# circles for equal visual weight at the same nominal area.
MARKER_SIZE = 23
MARKER_SIZE_TRI = 29

# ...but "the same area in points" is not "the same size on the page". Every one of these figures is
# scaled to the column width when it is placed, so a marker's apparent size goes as (area) /
# (figure width)^2: at a fixed s = 23 the markers in a 24-inch-wide figure print at 46% the diameter
# of the same markers in the 13-inch LW25 rebuild. marker_size() rescales to that reference width so
# the three paper figures match once printed.
MARKER_REF_WIDTH_IN = 13.0


def marker_size(fig_width_in, tri=False):
    """Marker area that prints like ``MARKER_SIZE`` does on a ``MARKER_REF_WIDTH_IN``-wide figure."""
    base = MARKER_SIZE_TRI if tri else MARKER_SIZE
    return base * (float(fig_width_in) / MARKER_REF_WIDTH_IN) ** 2


# Drawn over the class palette (indigo, light purple, grey), so black is a poor choice: it reads as
# just another dark point. One warm accent, used for everything that annotates the map rather than
# being data -- the SNR_tot loci and the rings/letters marking the gallery's examples.
ACCENT = "k"

# ---------------------------------------------------------------------------------------------
# palette + labels (module-level so both notebooks agree; override from a notebook if auditioning,
# e.g. ``cf.CLASS_COLORS["multi"] = "#009E73"``)
# ---------------------------------------------------------------------------------------------
# Tol colorblind-friendly; undetected kept neutral grey as the "no data" class.
CLASS_COLORS = {"undet": "0.8", "ok": "#332288", "biased": "#DDCC77",
                "multi": "#44AA99", "uncon": "#882255",
                # 3-class scheme (see SCHEME_3 below)
                "narrow": "#332288", "broad": "#796BBA"}

CLASS_LABELS = {"undet": "undetected",
                "ok": "unimodal: narrow peak at the true period",
                "biased": "unimodal: narrow peak at the wrong period",
                "multi": "broad: competitive region contains the true period",
                "uncon": "broad: competitive region excludes the true period"}

# ---------------------------------------------------------------------------------------------
# class schemes
# ---------------------------------------------------------------------------------------------
# "5class"/"4class" are the truth-validated schemes: the peak-shape metric crossed with a cut
# against the injected period, which only a simulated benchmark can evaluate.
#
# "3class" keeps the peak-shape metric and drops the truth cut entirely, so it is computable on real
# DR4 data as well as on this catalog. It is the honest description of what a periodogram alone
# establishes -- whether the fit improves significantly over a single star, and if so whether the
# period is localized -- and it deliberately avoids the word "detection", which implies a vetted
# orbit rather than a thresholded statistic. The truth-based schemes above then become the
# *validation* of this one: within the narrow class, how often is the peak actually at the truth.
SCHEME_5, SCHEME_4, SCHEME_3 = "5class", "4class", "3class"

# Wording of the 3-class scheme, collected here so a paper can be re-worded in one place.
#
# The scheme is a two-step decision, and BOTH steps have to be legible in the labels:
#
#   (1) does any channel exceed the null-calibrated threshold?   no -> "undet"
#   (2) if so, is the competitive region narrow?                 yes -> "narrow", no -> "broad"
#
# Naming classes 2 and 3 for step (2) alone ("period localized" / "not localized") would be
# ambiguous, because the null class has periodogram peaks too -- 35% of it is `klass == "unimodal"`
# in 1_companion_detectable, i.e. a perfectly narrow peak that simply never cleared the threshold.
# The leading "significant" is what carries step (1) onto the classes that passed it.
LABELS_3 = {   # long form, for captions and tables
    "undet": ("no significant peak or acceleration: neither channel exceeds the threshold "
              "calibrated on the companion-free control at a 1\\% false-positive rate"),
    "narrow": ("significant peak, period localized: the competitive region spans less than "
               "0.05 dex in period"),
    "broad": ("significant peak or acceleration, period not localized: the competitive region is "
              "wide, or the significance comes from the acceleration channel alone")}
SHORT_3 = {"undet": "no significant peak or acceleration",
           "narrow": "significant peak, period localized",
           "broad": "significant peak or acceleration, period not localized"}


def _scheme(split_multi=True, scheme=None):
    """Resolve the class scheme: an explicit ``scheme`` wins, else the legacy ``split_multi`` bool."""
    if scheme is not None:
        return scheme
    return SCHEME_5 if split_multi else SCHEME_4

PERIOD_TOL = 1.25   # "true period found" if a peak/best period is within this factor of truth
WIDTH_THR = 0.05    # dex; "narrow" threshold for the double periodogram's own width metric

CIRC_COLS = ["peak1_period", "peak2_period", "best_period"]          # circular stage-1 estimates
ECC_COLS = ["best_period_ecc", "peak1_period", "peak2_period"]       # stage-2 eccentric-refined


def short_labels(split_multi=True, scheme=None):
    """Compact legend labels."""
    if _scheme(split_multi, scheme) == SCHEME_3:
        return dict(SHORT_3)
    return {"undet": "undetected",
            "ok": "unimodal: true period",
            "biased": "unimodal: wrong period",
            "multi": "broad: contains truth",
            "uncon": "broad: misses truth" if split_multi else "broad: period not localized"}


def draw_order(split_multi=True, scheme=None):
    """Scatter/stack order, back to front (the null class first = background)."""
    sch = _scheme(split_multi, scheme)
    if sch == SCHEME_3:
        return ["undet", "broad", "narrow"]
    return (["undet", "uncon", "multi", "biased", "ok"] if sch == SCHEME_5 else
            ["undet", "uncon", "biased", "ok"])


def legend_order(split_multi=True, scheme=None):
    """Legend order -- groups the two families, unlike the drawing order."""
    sch = _scheme(split_multi, scheme)
    if sch == SCHEME_3:
        return ["undet", "narrow", "broad"]
    return (["undet", "ok", "biased", "multi", "uncon"] if sch == SCHEME_5 else
            ["undet", "ok", "biased", "uncon"])


# ---------------------------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------------------------
def apply_calibration(df, thr_orbit, thr_accel, baseline=ef.DR4_BASELINE_YEARS):
    """Add the null-calibrated detection flags every figure keys off, in place.

    ``detected_cal`` fires if either independent channel clears its threshold (the periodogram peak
    or the acceleration test); thresholds come from the companion-free control at a fixed
    false-positive rate. ``period_reliable_cal`` is the stricter data-only flag: detected, unimodal,
    and best period well inside the mission baseline.

    The two channels are also recorded separately as ``peak_significant_cal`` and
    ``accel_significant_cal``. The peak flag is what a "the period is localized" claim has to rest
    on: ``klass == "unimodal"`` only says the competitive region is narrow, and it is evaluated
    against ``classify_periodogram``'s own internal ``delta_bic_detect = 10``, which is far below
    the null-calibrated ``thr_orbit``. Without the peak flag a system can be called period-localized
    on the strength of the acceleration channel alone.
    """
    peak = df["top_power"].to_numpy(float) > thr_orbit
    accel = df["accel_delta_chi2"].to_numpy(float) > thr_accel
    det = peak | accel
    df["peak_significant_cal"] = peak
    df["accel_significant_cal"] = accel
    df["detected_cal"] = det
    df["period_reliable_cal"] = det & (df["klass"] == "unimodal") & \
        (df["best_period"].to_numpy(float) < baseline)
    return df


def recovered_against(df, k, found_cols=CIRC_COLS, tol=PERIOD_TOL):
    """Is the true period_k among the periodogram's period estimates, within a factor ``tol``?"""
    Pt = df[f"period_{k}"].to_numpy(float)
    ok = np.zeros(len(df), bool)
    lt = np.log(tol)
    for col in found_cols:
        if col in df:
            with np.errstate(invalid="ignore", divide="ignore"):
                ok |= np.abs(np.log(df[col].to_numpy(float) / Pt)) < lt
    return ok


def is_primary(df, k):
    """A single periodogram addresses at most the dominant (higher snr_total) companion.

    The other companion is left grey -- not because it is undetectable but because a one-planet
    periodogram never tested for it (that is what the double periodogram in the robust notebook is
    for). snr_total is used here only to RANK the two companions, never as a detection threshold.
    For one-companion populations every companion is primary.
    """
    other = 2 if k == 1 else 1
    oc = f"snr_total_{other}"
    if oc not in df.columns or not df[oc].notna().any():
        return np.ones(len(df), bool)
    stk = np.nan_to_num(df[f"snr_total_{k}"].to_numpy(float), nan=-np.inf)
    sto = np.nan_to_num(df[oc].to_numpy(float), nan=-np.inf)
    return stk >= sto


def classes(df, k, det_col, split_multi=True, found_cols=CIRC_COLS, tol=PERIOD_TOL, scheme=None,
            require_peak=None, peak_col="peak_significant_cal"):
    """Class key per row for companion ``k`` of a single-periodogram frame.

    Significance is data-driven: ``det_col`` is the null-calibrated flag (periodogram OR
    acceleration channel, thresholded on the companion-free control) -- no arbitrary SNR cut.

    With ``scheme="3class"`` no truth column is read at all: the row is only asked whether the
    statistic clears the null-calibrated threshold and, if so, whether the competitive region is
    narrow. Use it for statements that must also hold on real data.

    ``require_peak`` additionally demands that the PERIODOGRAM channel clear its own threshold
    before a row can be called period-localized, so that class cannot be entered on the strength of
    the acceleration channel alone. It defaults to True for the 3-class scheme and to False for the
    legacy 4/5-class schemes, whose published numbers predate the flag; pass it explicitly to
    override either way.
    """
    sch = _scheme(split_multi, scheme)
    if require_peak is None:
        require_peak = sch == SCHEME_3

    active = df[det_col].to_numpy(bool) & is_primary(df, k)
    narrow = df["klass"].to_numpy(object) == "unimodal"
    if require_peak:
        if peak_col not in df.columns:
            raise ValueError(
                f"require_peak needs the '{peak_col}' column; re-run cf.apply_calibration(df, ...) "
                "on this frame (a bundle pickled before this flag existed will not have it).")
        narrow = narrow & df[peak_col].to_numpy(bool)

    if sch == SCHEME_3:
        out = np.full(len(df), "undet", object)
        out[active & narrow] = "narrow"
        out[active & ~narrow] = "broad"
        return out

    found = recovered_against(df, k, found_cols, tol)
    out = np.full(len(df), "undet", object)
    out[active & ~narrow] = "uncon"
    if sch == SCHEME_5:
        # A broad peak has no trustworthy argmax, so a point-estimate test is the wrong question for
        # it. Ask instead whether the competitive REGION brackets the truth: a wide but correct
        # period constraint is imprecision, not error.
        ib = f"period_{k}_in_bound"
        inside = (df[ib].fillna(0).to_numpy(bool) if ib in df.columns else np.zeros(len(df), bool))
        out[active & ~narrow & inside] = "multi"
    out[active & narrow & found] = "ok"
    out[active & narrow & ~found] = "biased"
    return out


def classes_double(df, k, thr1, thr2, split_multi=True, width_thr=WIDTH_THR, tol=PERIOD_TOL):
    """Class key per row for companion ``k`` of a DOUBLE-periodogram frame.

    The stronger recovered peak P1 is assigned to whichever true companion it is nearer to in log
    period; the other companion gets P2. Each peak is then tested against its own null-calibrated
    threshold (the second-planet threshold is calibrated on 1-companion systems, where the stage-1
    circular fit leaves residual harmonics).

    Note the ``split_multi`` test differs from :func:`classes`: the double periodogram does not
    record a competitive region (no ``period_k_in_bound``), so the broad class is split on whether
    either reported period lands on the truth. Teal is therefore not strictly comparable between a
    single-periodogram figure and a double-periodogram one.
    """
    P1, P2 = df["P1_pdg"].to_numpy(float), df["P2_pdg"].to_numpy(float)
    p1t, p2t = df["period_1"].to_numpy(float), df["period_2"].to_numpy(float)
    pw1, pw2 = df["top_power1"].to_numpy(float), df["top_power2"].to_numpy(float)
    w1, w2 = df["width1_dex"].to_numpy(float), df["width2_dex"].to_numpy(float)
    e1, e2 = df["edge1"].to_numpy(bool), df["edge2"].to_numpy(bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        p1_to_c1 = np.abs(np.log(P1 / p1t)) <= np.abs(np.log(P1 / p2t))
    if k == 1:
        Pk, powk, wk, ek = (np.where(p1_to_c1, a, b) for a, b in
                            [(P1, P2), (pw1, pw2), (w1, w2), (e1, e2)])
        thk, ptrue = np.where(p1_to_c1, thr1, thr2), p1t
    else:
        Pk, powk, wk, ek = (np.where(p1_to_c1, a, b) for a, b in
                            [(P2, P1), (pw2, pw1), (w2, w1), (e2, e1)])
        thk, ptrue = np.where(p1_to_c1, thr2, thr1), p2t
    detected = powk > thk
    with np.errstate(invalid="ignore", divide="ignore"):
        found = np.abs(np.log(Pk / ptrue)) < np.log(tol)
        other = np.where(Pk == P1, P2, P1)
        found_any = found | (np.abs(np.log(other / ptrue)) < np.log(tol))
    narrow = (wk < width_thr) & (~ek.astype(bool))
    out = np.full(len(df), "undet", object)
    out[detected & ~narrow] = "uncon"
    if split_multi:
        out[detected & ~narrow & found_any] = "multi"
    out[detected & narrow & found] = "ok"
    out[detected & narrow & ~found] = "biased"
    return out


# ---------------------------------------------------------------------------------------------
# axes + panel layout
# ---------------------------------------------------------------------------------------------
# Period axis runs from below the shortest injected period (5 d = 0.014 yr at A_MIN_AU = 0.1 AU)
# to above the longest; widen if the sim_planets.ipynb semi-major axis prior moves again.
XLIM, YLIM = (0.008, 3000.0), (2e-4, 140.0)

def _axis(col, label, scale, lim, nbins=34, line=None, factor=1.0):
    bins = (np.logspace(np.log10(lim[0]), np.log10(lim[1]), nbins) if scale == "log"
            else np.linspace(lim[0], lim[1], nbins))
    return dict(col=col, label=label, scale=scale, lim=lim, bins=bins, line=line, factor=factor)

# `line` is drawn as a reference line on that axis (the DR4 baseline on the period axis).
AX_P = _axis("period_{k}", "orbital period [yr]", "log", XLIM, line=ef.DR4_BASELINE_YEARS)
AX_M = _axis("Mp_{k}_msun", r"companion mass [$M_{\rm Jup}$]", "log", YLIM, factor=1.0 / MJUP)
AX_A = _axis("alpha_{k}_mas", r"reflex semi-major axis $\alpha$ [mas]", "log", (1e-4, 5e3))
AX_E = _axis("e_{k}", "eccentricity", "linear", (0.0, 1.0))

# The reflex amplitude penalised by the long-period suppression: for P >> T the orbit is sampled
# over a short arc and only part of the signal is accessible within the baseline. Plotting against
# this instead of alpha asks whether it, rather than alpha, is what sets detection.
#
# Two forms, written to DIFFERENT columns so a figure can never carry the wrong axis label:
#   "bounded"     alpha / (1 + (P/T)^2)  -- the suppression the catalog's SNR_eff actually uses
#                                           (Eq. snreff); tends to alpha for P << T.
#   "asymptotic"  alpha * (T/P)^2        -- its P >> T limit. Agrees at long period, but AMPLIFIES
#                                           at short period (a factor 336 at P = 0.3 yr).
_AEFF_SUFFIX = {"bounded": "bnd", "asymptotic": "asym"}

def alpha_eff(df, baseline=ef.DR4_BASELINE_YEARS, kind="bounded"):
    """Add ``alpha_eff_<kind>_{k}_mas`` to a frame, in place, and return it."""
    suf = _AEFF_SUFFIX[kind]
    for k in (1, 2):
        acol, pcol = f"alpha_{k}_mas", f"period_{k}"
        if acol not in df.columns or pcol not in df.columns:
            continue
        a = df[acol].to_numpy(float)
        P = df[pcol].to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            df[f"alpha_eff_{suf}_{k}_mas"] = (a / (1.0 + (P / baseline) ** 2) if kind == "bounded"
                                              else a * (baseline / P) ** 2)
    return df

AX_AEFF = _axis("alpha_eff_bnd_{k}_mas",
                r"$\alpha\,/\,\left[1+(P/T_{\rm baseline})^{2}\right]$ [mas]", "log", (1e-6, 1e4))
AX_AEFF_ASYM = _axis("alpha_eff_asym_{k}_mas",
                     r"$\alpha\,(T_{\rm baseline}/P)^{2}$ [mas]", "log", (1e-6, 1e4))

# ---------------------------------------------------------------------------------------------
# constant-SNR_tot loci in the mass-period plane
# ---------------------------------------------------------------------------------------------
# The catalog's own detectability proxy (sim_planets.ipynb, gen_orbits_csv), reproduced here so the
# curves drawn on a figure are the SAME quantity the populations were selected on:
#
#     alpha      = [Mp / (M* + Mp)] a ϖ                        [mas]
#     snr_single = alpha / sigma_AL
#     snr_eff    = snr_single / (1 + (a/a_crit)^3),  a_crit = (T^2 M*)^(1/3)
#     snr_total  = sqrt(N_fov) snr_eff
#
# a_crit is built from M* alone while a follows Kepler with the TOTAL mass, so the suppression is
# 1 + (P/T)^2 (M*+Mp)/M* -- not quite (P/T)^2. The difference reaches 14% for an 80 M_Jup companion
# and is kept, because it is what the catalog actually applied. Rebuilt from a characterization
# frame this reproduces its ``snr_total_k`` column to 1e-6.
#
# With a ~ P^(2/3), a locus of constant snr_total is Mp ~ P^(-2/3) [1 + (P/T)^2]: it FALLS as the
# orbit widens inside the baseline (a longer lever arm at fixed mass) and turns up as P^(4/3) past
# it, so it is a V with its minimum at P = T/sqrt(2), not a single power law.
def star_terms(df, k=1):
    """Per-row (M*, parallax, sigma_AL, N_fov) recovered from a characterization frame.

    The frame carries no stellar columns, but it carries enough to rebuild them exactly:
    ``M*+Mp = a^3/P^2`` by Kepler, and the parallax follows from the injected ``alpha_k_mas``.
    """
    P = df[f"period_{k}"].to_numpy(float)
    Mp = df[f"Mp_{k}_msun"].to_numpy(float)
    a = df[f"a_{k}_au"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mtot = a ** 3 / P ** 2
        parallax = df[f"alpha_{k}_mas"].to_numpy(float) * mtot / (Mp * a)
    return mtot - Mp, parallax, df["sigma_single_mas"].to_numpy(float), \
        df["n_transits_dr4"].to_numpy(float)


def snr_locus(df, snr, periods, k=1, quantile=0.5, baseline=ef.DR4_BASELINE_YEARS, iters=8):
    """Companion mass [Msun] whose ``snr_total`` equals ``snr``, per trial period.

    Evaluated on every star in ``df`` and reduced by ``quantile``, so the curve is the locus for a
    typical star of that population rather than for invented parameters; pass a tuple of quantiles
    to get a band. Returns an array shaped ``(len(periods),)`` (or ``(nq, len(periods))``).
    """
    Ms, par, sig, nfov = (v[:, None] for v in star_terms(df, k))
    p = np.asarray(periods, float)[None, :]
    ok = np.isfinite(Ms) & np.isfinite(par) & (Ms > 0) & (par > 0) & np.isfinite(sig)
    Ms, par, sig, nfov = (np.where(ok, v, np.nan) for v in (Ms, par, sig, nfov))
    mp = np.zeros_like(Ms * p)                     # Mp enters through a, q and the suppression
    for _ in range(iters):
        mtot = Ms + mp
        a = (p ** 2 * mtot) ** (1.0 / 3.0)
        supp = 1.0 + (p / baseline) ** 2 * mtot / Ms
        mp = np.clip(snr * sig * supp / (np.sqrt(nfov) * par * a) * mtot, 0.0, 0.5 * Ms)
    return np.nanquantile(mp, quantile, axis=0)


def snr_total(df, k=1, baseline=ef.DR4_BASELINE_YEARS):
    """The catalog's ``snr_total`` recomputed for arbitrary (period, mass) -- or checked against
    the frame's own column, which it reproduces to ~1e-6."""
    Ms, par, sig, nfov = star_terms(df, k)
    P = df[f"period_{k}"].to_numpy(float)
    Mp = df[f"Mp_{k}_msun"].to_numpy(float)
    a = (P ** 2 * (Ms + Mp)) ** (1.0 / 3.0)
    alpha = Mp / (Ms + Mp) * a * par
    return np.sqrt(nfov) * (alpha / sig) / (1.0 + (P / baseline) ** 2 * (Ms + Mp) / Ms)


def snr_at(df, period, mass, k=1, quantile=0.5, baseline=ef.DR4_BASELINE_YEARS):
    """``snr_total`` of a hypothetical companion (``period`` [yr], ``mass`` [Msun]) on the median
    star of ``df`` -- the inverse of :func:`snr_locus`, for reading a level off a measured boundary."""
    Ms, par, sig, nfov = (v[:, None] for v in star_terms(df, k))
    p = np.atleast_1d(np.asarray(period, float))[None, :]
    m = np.atleast_1d(np.asarray(mass, float))[None, :]
    mtot = Ms + m
    a = (p ** 2 * mtot) ** (1.0 / 3.0)
    alpha = m / mtot * a * par
    s = np.sqrt(nfov) * (alpha / sig) / (1.0 + (p / baseline) ** 2 * mtot / Ms)
    out = np.nanquantile(s, quantile, axis=0)
    return out if np.ndim(period) or np.ndim(mass) else float(out)


def class_boundary(df, flag, k=1, frac=0.5, pbins=None, mbins=None, nmin=25):
    """Where a class boundary actually lies: the mass at which ``frac`` of companions carry ``flag``.

    In each period bin the flagged fraction is measured in mass bins and the crossing of ``frac``
    is interpolated in log mass. Returns ``(period_centres, mass_at_crossing [Msun])`` with NaN
    wherever the fraction never reaches ``frac`` -- which is itself the result for localization
    past a few times the baseline, where no mass suffices.
    """
    P = df[f"period_{k}"].to_numpy(float)
    M = df[f"Mp_{k}_msun"].to_numpy(float)
    flag = np.asarray(flag, bool)
    pbins = np.logspace(-1.3, 3.0, 12) if pbins is None else np.asarray(pbins, float)
    mbins = np.logspace(np.log10(0.03 * MJUP), np.log10(80 * MJUP), 27) if mbins is None \
        else np.asarray(mbins, float)
    pc, mc_out = [], []
    for lo, hi in zip(pbins[:-1], pbins[1:]):
        sel = (P >= lo) & (P < hi)
        pc.append(np.sqrt(lo * hi))
        f, mc = [], []
        for mlo, mhi in zip(mbins[:-1], mbins[1:]):
            cell = sel & (M >= mlo) & (M < mhi)
            if cell.sum() >= nmin:
                f.append(flag[cell].mean())
                mc.append(np.sqrt(mlo * mhi))
        f, mc = np.asarray(f), np.asarray(mc)
        if len(f) < 4 or f.max() < frac:
            mc_out.append(np.nan)
            continue
        j = int(np.argmax(f >= frac))
        if j == 0:
            mc_out.append(mc[0])
            continue
        w = (frac - f[j - 1]) / (f[j] - f[j - 1])
        mc_out.append(10 ** (np.log10(mc[j - 1]) + w * (np.log10(mc[j]) - np.log10(mc[j - 1]))))
    return np.asarray(pc), np.asarray(mc_out)


# (population, n_companions, panel title) -- the 2x2 grid, random on top, high-SNR below.
SPECS = [("1_companion_agnostic", 1, "one companion: random"),
         ("2_companion_agnostic", 2, "two companions: random"),
         ("1_companion_detectable", 1, "one companion: high-SNR"),
         ("2_companion_detectable", 2, "two companions: high-SNR")]

SPECS_SINGLE = [s for s in SPECS if s[1] == 1]

# population -> the title the 2x2 map gives it, so any other figure can name it the same way
PANEL_TITLES = {spec[0]: spec[2] for spec in SPECS}


def companions(df, npl, ax_x, ax_y, classify):
    """Pool (x, y, class, companion index) over every companion in one panel."""
    X, Y, C, K = [], [], [], []
    for k in ([1, 2] if npl == 2 else [1]):
        cx, cy = ax_x["col"].format(k=k), ax_y["col"].format(k=k)
        d = df.dropna(subset=[cx, cy, f"period_{k}"])
        X.append(d[cx].to_numpy(float) * ax_x["factor"])
        Y.append(d[cy].to_numpy(float) * ax_y["factor"])
        C.append(classify(d, k))
        K.append(np.full(len(d), k))
    return np.concatenate(X), np.concatenate(Y), np.concatenate(C), np.concatenate(K)


def _label_along(ax, x, y, text, along=0.2, fontsize=13, color="0.1"):
    """Write ``text`` on the curve ``(x, y)``, along its screen slope.

    ``along`` is the fraction of the way from where the curve first enters the axes to its minimum,
    i.e. a position DOWN the rising left branch -- the one place on a V where neighbouring levels
    are far apart. Targeting an x instead would stack every label against the top edge, since that
    is where each branch enters. The rotation is measured in DISPLAY coordinates, so the label stays
    parallel to the curve whatever the aspect ratio or the axis scaling.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    ylo, yhi = ax.get_ylim()
    ok = np.isfinite(y) & (y > ylo * 2.0) & (y < yhi / 2.0)     # room for the text either side
    if ok.sum() < 3:
        return
    idx = np.flatnonzero(ok)
    i0, imin = idx[0], int(np.nanargmin(np.where(ok, y, np.inf)))
    j = int(round(i0 + along * (imin - i0))) if imin > i0 else idx[len(idx) // 2]
    j = int(np.clip(j, max(idx[0], 1), min(idx[-1], len(x) - 2)))
    (x0, y0), (x1, y1) = ax.transData.transform([(x[j - 1], y[j - 1]), (x[j + 1], y[j + 1])])
    angle = np.degrees(np.arctan2(y1 - y0, x1 - x0))
    ax.text(x[j], y[j], text, rotation=angle, rotation_mode="anchor", ha="center", va="bottom",
            fontsize=fontsize, color=color, zorder=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=color, lw=0.8, alpha=0.92))


def draw_snr_lines(ax, df, levels, ax_x=None, ax_y=None, color=ACCENT, lw=2.4, alpha=0.9,
                   along=None, fontsize=13):
    """Overlay constant-``snr_total`` loci on a period-vs-mass axes, labelled along each curve.

    Silently does nothing unless ``levels`` is given and the axes really are period vs. companion
    mass, where the curve is a locus of something. Call it AFTER the axes limits are set: the labels
    are rotated to each curve's slope on screen, which needs the data-to-display transform.

    Each curve is the mass at which HALF the frame's stars reach that SNR_tot (see :func:`snr_locus`);
    a point cloud reaches below it because a population's edge is set by its most favourable stars,
    which is a spread in the stars rather than in the scaling.
    """
    ax_x, ax_y = (AX_P if ax_x is None else ax_x), (AX_M if ax_y is None else ax_y)
    if not levels or not (ax_x["col"].startswith("period") and ax_y["col"].startswith("Mp_")):
        return
    levels = np.atleast_1d(levels)
    # Default: staggered down the rising branch, so neighbouring levels -- which run nearly
    # parallel there -- do not pile up in the top-left corner. ``along = 1`` instead puts every
    # label at its curve's minimum, where the curves are furthest apart vertically and horizontal;
    # that reads better on a tall panel whose top-left is already busy.
    if along is None:
        along = 0.18 + 0.22 * np.arange(len(levels))
    along = np.broadcast_to(np.atleast_1d(along), levels.shape)
    pgrid = np.logspace(np.log10(ax_x["lim"][0]), np.log10(ax_x["lim"][1]), 240)
    for s, al in zip(levels, along):
        mloc = snr_locus(df, float(s), pgrid) * ax_y["factor"]
        ax.plot(pgrid, mloc, color=color, lw=lw, ls="-", alpha=alpha, zorder=8)
        _label_along(ax, pgrid, mloc, rf"$\mathrm{{SNR}}_{{\rm tot}} = {s:g}$", along=float(al),
                     fontsize=fontsize, color=color)


def joint(specs, ax_x, ax_y, stem, *, frame, classify, include_undet=False, split_multi=True,
          scheme=None, name_tag="", figdir=None, figsize=(17, 12.5), save=True, snr_lines=None):
    """Grid of joint plots: main scatter + stacked class histograms below (x) and right (y).

    ``frame(spec)`` returns the DataFrame for a panel and ``classify(df, k, spec)`` its class keys,
    so the caller decides which frame and which classifier each panel uses (the robust notebook
    sends the two-companion panels through the double periodogram, for instance).

    ``include_undet`` stacks the null class into the marginals; when False each panel's empty corner
    instead reports the per-COMPANION null fraction. The null class always remains as the grey
    scatter background. ``scheme`` selects the class scheme (see :func:`_scheme`); note the caller's
    ``classify`` must produce keys from the same scheme.

    ``snr_lines`` draws constant-``snr_total`` loci (:func:`snr_locus`) over a mass-period panel,
    each computed on that panel's own stars. Ignored on any other pair of axes, where the curve
    would not be a locus of anything.
    """
    order = draw_order(split_multi, scheme)
    legend = legend_order(split_multi, scheme)
    short = short_labels(split_multi, scheme)
    stack = order if include_undet else [k for k in order if k != "undet"]
    nrow = int(np.ceil(len(specs) / 2))
    ncol_panels = 2 if len(specs) > 1 else 1
    fig = plt.figure(figsize=(figsize[0], figsize[1] * nrow / 2))
    outer = GridSpec(nrow, ncol_panels, figure=fig, hspace=0.26, wspace=0.16,
                     left=0.06, right=0.985, top=0.90, bottom=0.06)
    for idx, spec in enumerate(specs):
        pop, npl, title = spec
        r, c = divmod(idx, ncol_panels)
        inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[r, c], width_ratios=[4, 1],
                                        height_ratios=[4, 1], hspace=0.04, wspace=0.04)
        axm = fig.add_subplot(inner[0, 0])
        axr = fig.add_subplot(inner[0, 1], sharey=axm)
        axb = fig.add_subplot(inner[1, 0], sharex=axm)
        X, Y, C, K = companions(frame(spec), npl, ax_x, ax_y,
                                lambda d, k: classify(d, k, spec))
        for key in order:
            m = C == key
            for kk, mk, sz in [(1, "s", marker_size(figsize[0])),
                               (2, "^", marker_size(figsize[0], tri=True))]:
                mm = m & (K == kk)
                axm.scatter(X[mm], Y[mm], marker=mk, s=sz, c=CLASS_COLORS[key],
                            alpha=0.45 if key == "undet" else 0.7, linewidths=0, rasterized=True)
        axm.set(xscale=ax_x["scale"], yscale=ax_y["scale"], xlim=ax_x["lim"], ylim=ax_y["lim"])
        draw_snr_lines(axm, frame(spec), snr_lines, ax_x, ax_y)
        if ax_x["line"] is not None:
            axm.axvline(ax_x["line"], color="k", ls="--", lw=1.6)
            axb.axvline(ax_x["line"], color="k", ls="--", lw=1.6)
        if ax_y["line"] is not None:
            axm.axhline(ax_y["line"], color="k", ls="--", lw=1.6)
            axr.axhline(ax_y["line"], color="k", ls="--", lw=1.6)
        axm.set_title(title, fontsize=19, pad=8)
        axm.tick_params(labelbottom=False, labelsize=12)
        axm.set_ylabel(ax_y["label"], fontsize=15)
        axm.grid(True, which="both", alpha=0.12)
        axb.hist([X[C == k] for k in stack], bins=ax_x["bins"], stacked=True,
                 color=[CLASS_COLORS[k] for k in stack], edgecolor="none")
        axb.set(xscale=ax_x["scale"], xlim=ax_x["lim"])
        axb.set_xlabel(ax_x["label"], fontsize=15)
        axb.tick_params(labelsize=11)
        axb.set_ylabel("N", fontsize=12)
        axr.hist([Y[C == k] for k in stack], bins=ax_y["bins"], stacked=True,
                 orientation="horizontal", color=[CLASS_COLORS[k] for k in stack], edgecolor="none")
        axr.set(yscale=ax_y["scale"], ylim=ax_y["lim"])
        axr.tick_params(labelleft=False, labelsize=10)
        if not include_undet:
            axc = fig.add_subplot(inner[1, 1]); axc.axis("off")
            axc.text(0.5, 0.60, f"{100 * np.mean(C == 'undet'):.0f}\\%", ha="center", va="center",
                     fontsize=17, fontweight="bold", color="0.35")
            axc.text(0.5, 0.18, f"{short['undet']}\n(of companions)", ha="center", va="center",
                     fontsize=9.5, color="0.35")
    # Two legends rather than one wrapped block, so the reading order is fixed: what the SYMBOLS
    # mean on the top row, what the COLOURS mean (the classes) underneath.
    marks = [Line2D([0], [0], marker="s", color="0.35", lw=0, ms=9, label="inner"),
             Line2D([0], [0], marker="^", color="0.35", lw=0, ms=10, label="outer")]
    if ax_x["line"] is not None or ax_y["line"] is not None:
        marks.append(Line2D([0], [0], color="k", ls="--", lw=1.6,
                            label=r"\textit{Gaia} DR4 baseline"))
    if snr_lines and ax_x["col"].startswith("period") and ax_y["col"].startswith("Mp_"):
        marks.append(Line2D([0], [0], color=ACCENT, lw=2.4, alpha=0.9,
                            label=r"constant $\mathrm{SNR}_{\rm tot}$ (median over stars)"))
    classes_h = [Line2D([0], [0], marker="s", color=CLASS_COLORS[k], lw=0, ms=12, label=short[k])
                 for k in legend]
    fig.legend(handles=marks, loc="upper center", ncol=len(marks), frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, 0.995))
    fig.legend(handles=classes_h, loc="upper center", ncol=len(classes_h), frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, 0.964))
    if save:
        # resolved at call time, so setting cf.FIG_DIR from a notebook actually takes effect
        figdir = Path(FIG_DIR if figdir is None else figdir)
        figdir.mkdir(parents=True, exist_ok=True)
        ut = "_wundet" if include_undet else "_noundet"
        fig.savefig(figdir / f"{stem}{name_tag}{ut}.pdf")
        fig.savefig(figdir / f"{stem}{name_tag}{ut}.png", dpi=130)
    return fig


def census(specs, *, frame, classify, split_multi=True, scheme=None):
    """Per-population class fractions [%], as a plain dict-of-dicts (for printing/tables)."""
    out = {}
    for spec in specs:
        pop, npl, _ = spec
        _, _, C, _ = companions(frame(spec), npl, AX_P, AX_M, lambda d, k: classify(d, k, spec))
        out[pop] = {k: 100.0 * float(np.mean(C == k)) for k in draw_order(split_multi, scheme)}
    return out

# ---------------------------------------------------------------------------------------------
# example periodograms: three systems per class, placed on the population map
# ---------------------------------------------------------------------------------------------
# Where to look for each example: {class key: [(period [yr], companion mass [M_Jup]), ...]}. The
# nearest system OF THAT CLASS in the log-log plane is taken, so the anchors only have to say
# roughly where in the map each row should come from. These span the injected ranges in both axes
# and put each class next to the regime it owns: localized periods inside the baseline, wide
# constraints just outside it, nothing at all at low mass or very long period.
# A point may carry a third element, a dict of column -> required value, when position and class do
# not pin down the case worth showing. The broad class is split almost evenly between systems whose
# periodogram peak is itself significant and systems carried by the acceleration channel alone;
# nearest-in-plane alone happens to draw mostly the latter, which over-states how common they are.
# Two of the three broad panels are therefore pinned to a significant peak, and the acceleration-only
# case is left at the longest period, where it is the typical outcome.
EXAMPLE_ANCHORS = {
    # All three localized examples sit inside the mission baseline, and that is not a choice: past
    # it the long-period tail of the periodogram always rises to within ~20% of the peak (in
    # Delta-chi^2 it is still thousands below, so the peak stays narrow by the width criterion, but
    # on a log axis the curve no longer LOOKS like an isolated peak). Inside the baseline the peak
    # stands about a decade clear of the tail, which is what these panels should show.
    # The short-period anchor is kept under 13 M_Jup (planet, not brown dwarf) and away from the
    # very shortest periods, where the peak is a sub-pixel spike on a five-decade log axis: at
    # ~0.5 yr the peak still stands a decade above the surrounding forest and reads as a peak.
    "narrow": [(0.55, 7.0), (1.4, 5.0), (2.1, 19.0)],
    "broad":  [(7.0, 10.0, {"peak_significant_cal": True}),
               (25.0, 30.0, {"peak_significant_cal": True}), (400.0, 60.0)],
    "undet":  [(0.1, 0.01), (3.0, 0.5), (500.0, 20.0)],
}


# Compact row labels for the gallery: the full wording is already in the map's legend, and a rotated
# 50-character label does not fit beside a panel.
ROW_LABELS = {"undet": "not significant", "narrow": "period localized",
              "broad": "period not localized", "ok": "unimodal: true period",
              "biased": "unimodal: wrong period", "multi": "broad: contains truth",
              "uncon": "broad: misses truth"}


def _flat_anchors(anchors):
    """``EXAMPLE_ANCHORS`` (dict of class -> points) or a flat ``[(P, M, class[, require])]``.

    Returns ``(rows, flat)``: the (class, count) of each gallery row, and every anchor normalised to
    ``(period, mass, class or None, require dict)``.
    """
    def norm(pt, key):
        p, m, extra = (list(pt) + [None])[:3]
        return (p, m, key, extra if isinstance(extra, dict) else None)

    if isinstance(anchors, dict):
        rows = [(key, len(pts)) for key, pts in anchors.items()]
        return rows, [norm(pt, key) for key, pts in anchors.items() for pt in pts]
    flat = [norm(tuple(a)[:2] + tuple(a)[3:], a[2] if len(a) > 2 else None) for a in anchors]
    return [(None, len(flat))], flat


def select_examples(df, anchors=EXAMPLE_ANCHORS, *, classify, k=1):
    """Pick one system per anchor: nearest in log(period), log(mass), inside the anchor's class.

    ``classify(df, k)`` returns the class key per row (the same callable the maps use), so the
    examples can never disagree with the population figure they annotate. Returns a DataFrame with
    the picked rows plus ``cls`` and a single-letter ``tag`` column, in anchor order.
    """
    _, flat = _flat_anchors(anchors)
    d = df.dropna(subset=[f"period_{k}", f"Mp_{k}_msun"]).copy()
    d["cls"] = classify(d, k)
    lp = np.log10(d[f"period_{k}"].to_numpy(float))
    lm = np.log10(d[f"Mp_{k}_msun"].to_numpy(float) / MJUP)
    picked, taken = [], set()
    for i, (p_a, m_a, want, require) in enumerate(flat):
        cost = (lp - np.log10(p_a)) ** 2 + (lm - np.log10(m_a)) ** 2
        if want is not None:
            cost = np.where(d["cls"].to_numpy(object) == want, cost, np.inf)
        for col, val in (require or {}).items():
            cost = np.where(d[col].to_numpy() == val, cost, np.inf)
        for j in taken:
            cost[j] = np.inf
        if not np.isfinite(cost).any():
            raise ValueError(f"no system left for anchor {(p_a, m_a, want, require)}")
        j = int(np.argmin(cost))
        taken.add(j)
        row = d.iloc[j].copy()
        row["tag"] = chr(ord("A") + i)
        picked.append(row)
    return pd.DataFrame(picked).reset_index(drop=True)


def _line_color(cls):
    """Class colour for lines and headers: the null class's scatter grey is too light to read."""
    return "0.45" if cls == "undet" else CLASS_COLORS[cls]


def _param_title(truth, row, k=1):
    """The injected parameters of one example, as a two-line panel title."""
    g = lambda v, n=2: f"{float(v):.{n}g}"
    return (rf"$M_{{\rm c}} = {g(row[f'Mp_{k}_msun'] / MJUP)}\,M_{{\rm Jup}}$, "
            rf"$M_\star = {g(truth['mass_st_msun'])}\,M_\odot$" "\n"
            rf"$P = {g(row[f'period_{k}'], 3)}$ yr, "
            rf"$d = {g(1000.0 / truth['parallax_mas'], 3)}$ pc" "\n"
            rf"$e = {g(row[f'e_{k}'])}$, "
            rf"$i = {g(np.degrees(truth[f'i_{k}_rad']), 3)}^\circ$")


def _draw_periodogram(ax, periods, power, row, color, thr_orbit, baseline, k=1, fs=1.0):
    ax.plot(periods, np.clip(power, 1e-3, None), lw=0.9 * fs, color=color, zorder=3)
    ax.axvline(float(row[f"period_{k}"]), color="#2e6f95", ls="-", lw=1.4 * fs, zorder=2)
    ax.axvline(baseline, color="k", ls="--", lw=1.2 * fs, zorder=2)
    if thr_orbit is not None:
        ax.axhline(thr_orbit, color="#CC6677", ls=":", lw=1.4 * fs, zorder=2)
    top = max(float(np.nanmax(power)), 1.0 if thr_orbit is None else float(thr_orbit))
    pos = power[np.isfinite(power) & (power > 0)]
    bot = min(1.0, 0.8 * float(np.nanmin(pos))) if pos.size else 1.0   # the null cases sit low
    ax.set(xscale="log", yscale="log", xlim=(periods[0], periods[-1]),
           ylim=(max(bot, 1e-2), 6.0 * top))
    ax.tick_params(labelsize=10 * fs, pad=2)
    ax.grid(True, which="major", alpha=0.12)


def examples(df, population, *, classify, periodogram, anchors=EXAMPLE_ANCHORS, k=1,
             thr_orbit=None, scheme=None, split_multi=True, baseline=ef.DR4_BASELINE_YEARS,
             ax_x=AX_P, ax_y=AX_M, stem="characterizability_examples", name_tag="",
             figdir=None, figsize=(18.0, 10.5), save=True, data_root=None, verbose=True,
             snr_lines=None):
    """The population map beside the periodograms of individual systems drawn from it.

    Left: every companion of ``population`` in the plane of ``ax_y`` vs. ``ax_x``, coloured by the
    same classification as the maps, with the examples ringed and lettered. Right: one periodogram
    per example, one row per class, each titled with the system's injected parameters. Nothing is
    refitted or re-derived here -- the curve is the period search itself, and the vertical line is
    the period that was injected.

    ``periodogram(t, psi, pf, y, yerr) -> (periods, power)`` keeps the figure agnostic about which
    period search made the classification; the caller passes the one its notebook is about.
    ``snr_lines`` overlays the same constant-SNR_tot loci as :func:`joint` (see :func:`draw_snr_lines`).
    """
    short = short_labels(split_multi, scheme)
    # This figure is much wider than the LW25 rebuild, so it is scaled down harder when placed in a
    # paper. Every size below -- markers, type, line widths -- is multiplied by the same factor, so
    # the three paper figures print at matching sizes rather than matching point values.
    fs = float(figsize[0]) / MARKER_REF_WIDTH_IN
    rows, _ = _flat_anchors(anchors)
    picks = select_examples(df, anchors, classify=classify, k=k)
    h5 = ef.systems_h5_path(population, ef.DATA_ROOT if data_root is None else data_root)
    truths = ef.load_truths(h5)

    nrow, ncol = len(rows), max(n for _, n in rows)
    fig = plt.figure(figsize=figsize)
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.5], wspace=0.17,
                     left=0.058, right=0.995, top=0.875, bottom=0.075)

    # ---- left: the population map, with the examples marked ----------------------------------
    axm = fig.add_subplot(outer[0, 0])
    cx, cy = ax_x["col"].format(k=k), ax_y["col"].format(k=k)
    d = df.dropna(subset=[cx, cy, f"period_{k}"])
    C = classify(d, k)
    X = d[cx].to_numpy(float) * ax_x["factor"]
    Y = d[cy].to_numpy(float) * ax_y["factor"]
    for key in draw_order(split_multi, scheme):
        m = C == key
        axm.scatter(X[m], Y[m], marker="s", s=marker_size(figsize[0]), c=CLASS_COLORS[key],
                    linewidths=0, alpha=0.4 if key == "undet" else 0.75, rasterized=True)
    if ax_x["line"] is not None:
        axm.axvline(ax_x["line"], color="k", ls="--", lw=1.6 * fs)
    axm.set(xscale=ax_x["scale"], yscale=ax_y["scale"], xlim=ax_x["lim"], ylim=ax_y["lim"])
    # the same loci as the Step 4 map, labelled the same way: along each curve, at its slope
    draw_snr_lines(axm, df, snr_lines, ax_x, ax_y, lw=2.4 * fs, fontsize=13 * fs)
    axm.set_xlabel(ax_x["label"], fontsize=15 * fs)
    axm.set_ylabel(ax_y["label"], fontsize=15 * fs)
    axm.tick_params(labelsize=12 * fs)
    axm.grid(True, which="both", alpha=0.12)
    axm.set_title(PANEL_TITLES.get(population, population.replace("_", " ")), fontsize=17 * fs, pad=8)
    for _, row in picks.iterrows():
        px = float(row[cx]) * ax_x["factor"]
        py = float(row[cy]) * ax_y["factor"]
        # the badge IS the marker: a ring plus an offset letter read as two separate circles
        axm.annotate(row["tag"], (px, py), ha="center", va="center", fontsize=15 * fs,
                     fontweight="bold", color=ACCENT, zorder=7,
                     bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec=ACCENT, lw=1.4 * fs,
                               alpha=0.95))
    handles = [Line2D([0], [0], marker="s", color=CLASS_COLORS[key], lw=0, ms=11 * fs,
                      label=short[key]) for key in legend_order(split_multi, scheme)]
    handles.append(Line2D([0], [0], color="k", ls="--", lw=1.6 * fs,
                          label=r"\textit{Gaia} DR4 baseline"))
    if snr_lines:
        handles.append(Line2D([0], [0], color=ACCENT, lw=2.4 * fs, alpha=0.9,
                              label=r"constant $\mathrm{SNR}_{\rm tot}$ (median over stars)"))
    axm.legend(handles=handles, loc="lower left", frameon=False, fontsize=10.5 * fs,
               handletextpad=0.5, borderaxespad=0.3)

    # ---- right: the periodograms, one row per class -------------------------------------------
    grid = GridSpecFromSubplotSpec(nrow, ncol, subplot_spec=outer[0, 1], hspace=0.85, wspace=0.34)
    i = 0
    for r, (key, npts) in enumerate(rows):
        axes_r = []
        for c in range(npts):
            row = picks.iloc[i]
            ri = int(row["row_index"])
            truth = truths.iloc[ri]
            t, psi, pf, y, yerr = ef.epoch_arrays(ef.load_epochs(h5, ri))
            periods, power = periodogram(t, psi, pf, y, yerr)
            color = _line_color(row["cls"])
            ax = fig.add_subplot(grid[r, c])
            _draw_periodogram(ax, periods, power, row, color, thr_orbit, baseline, k, fs=fs)
            ax.set_title(_param_title(truth, row, k), fontsize=10.5 * fs, linespacing=1.5, pad=5)
            ax.text(0.045, 0.90, row["tag"], transform=ax.transAxes, fontsize=14 * fs,
                    fontweight="bold", va="top", ha="left", color=ACCENT, zorder=9,
                    bbox=dict(boxstyle="circle,pad=0.20", fc="white", ec=ACCENT, lw=1.2 * fs,
                              alpha=0.95))
            # a curve that never crosses the threshold can still be in a significant class: the
            # acceleration channel is independent of the periodogram and carries ~40% of "broad"
            if bool(row.get("accel_significant_cal", False)) and \
                    not bool(row.get("peak_significant_cal", True)):
                ax.text(0.97, 0.05, "significant via acceleration", ha="right",
                        va="bottom", transform=ax.transAxes, fontsize=9.5 * fs, color=color)
            ax.set_xlabel("period [yr]", fontsize=12 * fs, labelpad=2)
            if c == 0:
                ax.set_ylabel(r"$\Delta\chi^2$", fontsize=13 * fs)
            axes_r.append(ax)
            if verbose:
                print(f"  {row['tag']}  row {ri:6d}  P = {row[f'period_{k}']:9.3f} yr  "
                      f"M = {row[f'Mp_{k}_msun'] / MJUP:7.3f} MJup  {row['cls']}")
            i += 1
        if key is not None:                      # class of the row, down its left edge
            box = axes_r[0].get_position()
            fig.text(box.x0 - 0.060, box.y0 + 0.5 * box.height, ROW_LABELS.get(key, short[key]),
                     rotation=90, va="center", ha="center", fontsize=13 * fs, fontweight="bold",
                     color=_line_color(key))

    keys = [Line2D([0], [0], color="#2e6f95", lw=1.8 * fs, label="injected period"),
            Line2D([0], [0], color="k", ls="--", lw=1.4 * fs,
                   label=r"\textit{Gaia} DR4 baseline")]
    if thr_orbit is not None:
        keys.append(Line2D([0], [0], color="#CC6677", ls=":", lw=1.6 * fs,
                           label="null-calibrated detection threshold"))
    right = outer[0, 1].get_position(fig)
    fig.legend(handles=keys, loc="upper center",
               bbox_to_anchor=(0.5 * (right.x0 + right.x1), 0.995),
               ncol=len(keys), frameon=False, fontsize=11.0 * fs, handletextpad=0.5,
               columnspacing=1.6)

    if save:
        figdir = Path(FIG_DIR if figdir is None else figdir)
        figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / f"{stem}{name_tag}.pdf")
        fig.savefig(figdir / f"{stem}{name_tag}.png", dpi=130)
    return fig, picks
