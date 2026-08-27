"""Every choice the characterization depends on -- one screen, no indirection.

Paths, grid bounds, thresholds, and column lists are plain module constants.
Read them as `config.P_MIN`; there is nothing to construct and nothing to pass
down. The catalog and output paths are functions because `--catalog-root` and
`--output-root` can move them; everything else is a value.

Physical constants come from `epochalypse_periodograms.constants` (astropy),
never typed in here.
"""

from __future__ import annotations

from pathlib import Path

# Re-exported so every stage reads one authority as `config.X`. Assignments
# rather than a bare import, because an unused-import pass would otherwise
# strip the ones only other modules use -- which is how the simulator lost
# DAYS_PER_YEAR once already.
from . import constants as _k

DAYS_PER_YEAR = _k.DAYS_PER_YEAR
DR4_BASELINE_YEARS = _k.DR4_BASELINE_YEARS
GAIA_EPOCH_TCB_JD = _k.GAIA_EPOCH_TCB_JD
MJUP_IN_MSUN = _k.MJUP_IN_MSUN
RSUN_IN_AU = _k.RSUN_IN_AU

# The package root: src/epochalypse_periodograms/config.py -> ../../ .
ROOT = Path(__file__).resolve().parents[2]

# ==========================================================================
# Inputs -- the generated catalog, not produced here
# ==========================================================================
# The directory holding `data/simulated_astrometry/<population>/` and the
# merged `data/injected_solutions_<population>.parquet`.
#
# The default is `<repo>/outputs`, which is where the generator's
# `config.OUTPUT_ROOT` puts them -- so with this package sitting at
# `<repo>/periodograms/`, a checkout that has already generated a catalog needs
# no `--catalog-root` at all. Point it elsewhere for a catalog delivered as a
# directory, e.g. `--catalog-root ../flatiron_server_run_500pc/epochalypse`.
CATALOG_ROOT = ROOT.parent / "outputs"


def set_catalog_root(path) -> None:
    """Point every input path at a different catalog run (`--catalog-root`)."""
    global CATALOG_ROOT
    CATALOG_ROOT = Path(path).resolve()


def catalog_data_dir():
    return CATALOG_ROOT / "data"


def shard_dir(population):
    return catalog_data_dir() / "simulated_astrometry" / population


def shard_epochs(population, shard, n_shards):
    return shard_dir(population) / f"epochs_rank{shard:05d}_of_{n_shards:05d}.parquet"


def shard_truths(population, shard, n_shards):
    return shard_dir(population) / f"truths_rank{shard:05d}_of_{n_shards:05d}.parquet"


# ==========================================================================
# Outputs
# ==========================================================================
OUTPUT_ROOT = ROOT / "outputs"


def set_output_root(path) -> None:
    """Point every output path somewhere else (`--output-root`, smoke tests).

    Called once per process before any work; each MPI rank parses its own argv,
    so there is no shared state to keep in step.
    """
    global OUTPUT_ROOT
    OUTPUT_ROOT = Path(path).resolve()


def chars_dir(population):
    return OUTPUT_ROOT / "characterization" / population


def power_dir(population):
    return OUTPUT_ROOT / "periodograms" / population


def _piece(shard, n_shards, part, n_parts):
    """`shard00007_of_00320` , or `..._part1_of_4` when a shard is split."""
    stem = f"shard{shard:05d}_of_{n_shards:05d}"
    return stem if n_parts == 1 else f"{stem}_part{part:02d}_of_{n_parts:02d}"


def chars_shard(population, shard, n_shards, part=0, n_parts=1):
    return chars_dir(population) / f"chars_{_piece(shard, n_shards, part, n_parts)}.parquet"


def power_shard(population, shard, n_shards, part=0, n_parts=1):
    return power_dir(population) / f"power_{_piece(shard, n_shards, part, n_parts)}.parquet"


def period_grid_path():
    """The trial periods every stored power array is sampled on -- written once.

    The grid depends only on P_MIN / P_MAX / N_SEGMENTS, never on the data, so
    it is global to a run and storing it per system would multiply the
    periodogram output by two. Every `power` array is aligned to this file.
    """
    return OUTPUT_ROOT / "periodograms" / "period_grid.parquet"


def manifest_path():
    return OUTPUT_ROOT / "manifest.json"


def calibration_path():
    return OUTPUT_ROOT / "calibration.json"


def failed_dir():
    return OUTPUT_ROOT / "failed"


# ==========================================================================
# Populations
# ==========================================================================
# The three *generated* populations, and the only three that are searched. The
# high-SNR samples are not separate runs: `1_companion_high_snr` is the subset
# of `1_companion`'s rows with SNR_tot >= HIGH_SNR_MIN on every companion, and
# every one of those rows was already characterized here. Selecting after the
# fact is what makes 5.7 M x 3 the whole job rather than 5.7 M x 5.
POPULATIONS = ("0_companion", "1_companion", "2_companion")

# Injected companions per population, so the truth join knows how many
# `*_k` column families to look for.
N_COMPANIONS = {"0_companion": 0, "1_companion": 1, "2_companion": 2}

HIGH_SNR_MIN = 5.0  # SNR_tot floor, applied to EVERY injected companion

# ==========================================================================
# Search grid
# ==========================================================================
# Bounds are `run_periodograms.P_MIN` / `P_MAX`, unchanged, and the reasoning
# there still applies: the grid must bracket every injected truth at BOTH ends,
# or a system rails at an edge and "is the truth inside the competitive
# region?" becomes unanswerable-by-construction rather than measured. The lower
# bound is set by the innermost orbit the generator can inject -- not the flat
# 0.001 AU semi-major axis floor but the per-star Roche-lobe screen, whose
# tightest orbit over stars.csv has a period of 8.4e-5 yr (44 minutes), so
# 5e-5 yr brackets the prior with 1.7x margin.
P_MIN, P_MAX = 5.0e-5, 3300.0

# kepmodel steps the trial *frequency* by a fixed amount, so it evaluates only
# on uniform frequency grids. One such grid per log-period segment lets the
# step be refreshed as the trial period grows; the union of their sample points
# is the search grid, and every sample is a native kepmodel evaluation (nothing
# is interpolated). N_SEGMENTS is the only cost knob -- more segments means
# less overshoot at the fine end and a total closer to a genuinely log-uniform
# grid. At 32 segments each spans 0.24 dex and the search costs ~35% more
# evaluations than the baseline grid has trial periods.
N_SEGMENTS = 32

# Each segment's step is set so its coarsest log10-period spacing is at most
# this, which is `np.diff(np.log10(run_periodograms.PERIODS)).max()`: the
# search is at least as finely sampled in log-period as the in-house grid
# *everywhere*. Held at ~688 trials per e-fold across every revision of the
# bounds, so `width_dex` stays comparable with earlier runs.
BASELINE_N_PERIODS = 12383
TARGET_DLOG = None  # None -> derived from P_MIN, P_MAX, BASELINE_N_PERIODS

# ==========================================================================
# Noise model
# ==========================================================================
# False: fixed 1/sigma_formal^2 weights, a like-for-like swap of the period
# search alone, and what the paper figures use. True: fit kepmodel's
# excess-noise term on the companion-free model before the search.
#
# False is also `kepmodel`'s own default: `AstroModel(..., excess_noise=
# term.Jitter(0))` puts the term in the covariance but leaves it out of
# `fit_param`, so it is held at zero unless a caller adds it. The Gaia/OHP
# tutorial adds it, in a step of its own before the periodogram
# (`model.fit_param += ['cov.excess_noise.sig']; model.fit()`) -- so "the
# documented default" points both ways depending on whether the library or the
# tutorial is being read.
#
# True is a genuinely different noise model, not a refinement. Fitted with no
# orbit in the model, the term absorbs the companion's signal along with the
# excess scatter, and chi2_base (hence the whole Delta-chi^2 scale) collapses
# by one to two orders of magnitude on strongly-signalled systems -- detections
# are suppressed along with false positives. The argument the other way is
# real: the catalog injects scatter at sigma_UEVA and reports sigma_formal, so
# there is excess the fixed weights genuinely do not capture. Thresholds are
# recalibrated on the matched control either way, so both runs are internally
# self-consistent, but their `top_power` columns are on different scales and
# must never be compared numerically. Running both is a straight 2x: the
# frequency loop dominates and nothing is shared between them.
FIT_JITTER = False

# ==========================================================================
# Classification
# ==========================================================================
# `classify_periodogram`'s own internal thresholds. These are NOT the detection
# thresholds -- those are calibrated on the companion-free control, in
# `calibrate.py`, and are far above DELTA_BIC_DETECT.
DELTA_BIC_DETECT = 10.0        # BIC improvement before a peak counts at all
DELTA_POWER_UNIMODAL = 10.0    # Delta-chi^2 within which a second peak competes
MIN_SEPARATION_DEX = 0.1       # peaks closer than this in log P are merged
WIDTH_DELTA = 4.0              # Delta-chi^2 defining the competitive region
WIDTH_CONSTRAINED_DEX = 0.05   # competitive width below which a period is "localized"
EDGE_FRAC = 0.02               # argmax within this fraction of the log range = railed
N_ORBIT_PARAMS = 4             # circular Thiele-Innes columns
PERIOD_RECOVER_TOL = 1.2       # |ln(P_best / P_true)| < ln(tol) counts as recovered

# Null false-positive rate the detection thresholds are calibrated to, split
# evenly between the two independent channels (periodogram peak, acceleration).
TARGET_FP = 0.01

# ==========================================================================
# Output layout
# ==========================================================================
PARQUET_COMPRESSION = "zstd"
PARQUET_COMPRESSION_LEVEL = 3
CHARS_FLUSH_EVERY = 5000   # systems buffered before a characterization row-group flush
POWER_FLUSH_EVERY = 500    # ~27 MB of float32 power per row group

# What to keep of the raw Delta-chi^2 curves. A curve is 53 kB per system after
# zstd, so "all" is ~915 GB over the full catalog -- ten times the size of the
# catalog it was computed from, which is why it is not the default. "subsample"
# keeps the paper's 10,000 stars per population, 1.6 GB, and any other curve can
# be regenerated from its epochs in 0.34 s (see `periodogram_source.py`).
# POWER_DECIMATE and POWER_DTYPE shrink "all" without changing which systems are
# stored, if it is ever wanted; see PERIODOGRAMS.md.
POWER_MODE = "subsample"    # "all" | "subsample" | "none"
POWER_DECIMATE = 1          # keep every Nth grid point (1 = the full grid)
POWER_DTYPE = "float32"     # "float32" | "float16"

# The paper's shared down-selection, for POWER_MODE == "subsample": the rule in
# `pipeline/subsample.py`, keyed on gaia_source_id, which is what makes the same
# 10,000 stars appear in every figure drawn from any table in the catalog.
SUBSAMPLE_SIZE = 10_000
SUBSAMPLE_SEED = 20260824

# `subsample_frame` selects the SIZE smallest blake2s ranks, which a rank
# holding one shard cannot evaluate -- it sees 0.3% of the parent and has no way
# to know where the quantile falls. So the quantile is written down instead: the
# 10,000th smallest rank over the 500 pc parent sample's 5,724,586 source ids at
# seed 20260824. Comparing against it reproduces `subsample_frame` EXACTLY,
# system by system, with no communication between ranks. Re-derive it with
# `python scripts/finish.py --stages subsample-cutoff` if the parent sample,
# the seed, or the size ever changes; `tests/test_periodograms.py` checks it
# against the shipped `subsample_10000_seed20260824.parquet`.
SUBSAMPLE_PARENT_SIZE = 5_724_586
SUBSAMPLE_RANK_CUTOFF = 32256280498693495   # inclusive upper bound on the rank

# ==========================================================================
# Columns carried through from the truth tables
# ==========================================================================
# Everything the three paper figures need, and nothing that only the generator
# cares about (the five seed columns, source_id_dr2, population). Per-companion
# families are appended for k = 1..N_COMPANIONS[population].
TRUTH_COLUMNS_SYSTEM = (
    "system_id", "gaia_source_id", "n_transits_dr4",
    "parallax_mas", "pmra_mas_yr", "pmdec_mas_yr",
    "mass_st_msun", "radius_st_rsun", "sigma_single_mas", "n_planets",
)
TRUTH_COLUMNS_COMPANION = (
    "sma_{k}", "ecc_{k}", "mass_pl_{k}", "inc_{k}", "Omega_{k}", "omega_{k}",
    "M_anom_{k}", "period_{k}", "alpha_mas_{k}", "snr_single_{k}",
    "snr_eff_{k}", "snr_total_{k}",
)
TRUTH_COLUMNS_PAIR = ("coplanar", "P_ratio", "near_2_1", "near_3_2", "near_resonance")
