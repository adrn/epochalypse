"""Every choice the harv stage depends on -- one screen, no indirection.

Read them as `config.N_PRIOR_SAMPLES`. Paths are functions because
`--output-root` can move them; everything else is a value.

The prior is deliberately ONE prior for the whole catalog: the same library,
the same bounds, every system. That is what makes 17.2 M posteriors comparable,
and it is why the numbers below are constants rather than per-system choices.
"""

from __future__ import annotations

from pathlib import Path

from .. import config as _cat
from .. import constants as _k
from ..periodogram import config as _pg

# Re-exported so every stage reads one authority as `config.X`.
POPULATIONS = _cat.POPULATIONS
PARQUET_COMPRESSION = _cat.PARQUET_COMPRESSION
MJUP_IN_MSUN = _k.MJUP_IN_MSUN

# ==========================================================================
# The prior library
# ==========================================================================
# Drawn once, cached, and reused for every system. Cost is linear in this and
# it is the only real knob: measured 2.5 us per sample per system on a real
# 108-epoch system (x64, single-threaded), so 10^6 is ~2.5 s/system and
# ~12,000 core-hours over the 17.2 M runs -- 12 h on 1000 cores.
#
# It also sets how well a posterior can be resolved. Effective sample size is
# roughly N_PRIOR_SAMPLES x (posterior volume / prior volume), so a strongly
# detected system lands near ESS ~ 20 here and an undetected one near ~10^3.
# Raising this is the lever if the ESS distribution from a first run is too low.
N_PRIOR_SAMPLES = 1_000_000

# Samples kept per system, by importance weight. Fixed, so the output table is
# uniform: exactly this many rows for every system, always.
TOP_K = 1024

# Prior samples evaluated per JIT'd batch. This is the setting that decides
# whether the stage is compute-bound or memory-bound, so it is worth the space.
#
# The (batch, n_epochs, n_linear) design matrix is the largest intermediate, and
# it is streamed once per batch -- 100 times per system at M=10^6. At float64,
# N=320 and 9 linear columns:
#
#   batch = 10^4  ->  230 MB per rank    batch = 10^3  ->  23 MB per rank
#
# Measured single-threaded on a laptop core, 10^4 is marginally the faster of
# the two (2.53 vs 2.66 us/sample) -- and that measurement is misleading,
# because one core had the whole cache and the full memory bandwidth to itself.
# On a 64-rank Rome node, 64 x 230 MB streamed 100 times per system saturates
# DRAM: the first production attempt ran at >=36 s/system against a 6 s/system
# budget and completed no unit in two hours. 23 MB per rank fits per-rank cache,
# which is what matters when the node is full.
#
# So: 10^3 for a full node. Raise it toward 10^4 only for a run with few ranks
# per node, and re-measure if you do. harv's default is 100_000, which is GPU
# advice -- at N=320 that intermediate is 2.3 GB.
BATCH_SIZE = 1_000

# Master seed. Each system's seed is derived from its gaia_source_id, the way
# `planets.system_seed` does, so a rerun of any subset reproduces exactly.
SEED = 20260827

# ==========================================================================
# Parameterization
# ==========================================================================
# Thiele-Innes, not Campbell. It marginalizes the four orientation parameters
# (arg_peri, lon_asc_node, cos_i, semi_major_axis) as linear constants, leaving
# only period, eccentricity, and phase_peri sampled -- 3 nonlinear dimensions
# instead of 6.
#
# Measured on a real SNR=360 system at M=2e5: period recovered to 0.14% versus
# 2.63% for the Campbell parameterization, a 19x accuracy gain, with an
# uninformative parallax prior in both cases. It does NOT improve ESS -- see
# HARV.md; nothing does, for a strong detection.
#
# Built per system via `ThieleInnesGaiaAstrometry.from_data(data)`, which is the
# construction harv recommends: it sets `a_floor = med(sigma_AL)/sqrt(N)` and
# enables the Jacobian correction. Without that correction a flat prior on the
# Thiele-Innes constants is NOT the Campbell prior, and harv warns the marginal
# likelihood can be dominated by spurious long-period solutions where the orbit
# is absorbed into proper motion.
USE_THIELE_INNES = True

# ==========================================================================
# Prior bounds
# ==========================================================================
# Period bounds. These are DELIBERATELY NARROWER than the periodogram's
# `P_MIN`/`P_MAX` (5e-5 to 3300 yr) and than the injected prior. The two stages
# are answering different questions: the periodogram searches the full injected
# range because a peak anywhere is worth reporting, while this stage spends a
# fixed sampling budget and every decade it covers costs density in the decades
# that can actually be constrained.
#
# DR4 is 5.5 yr with ~80 transits, so roughly 0.1-10 yr is constrainable at all
# -- below that the signal aliases, above it the orbit is absorbed into proper
# motion. 4.0 decades instead of 7.8 was measured worth about one order of
# magnitude of N_PRIOR_SAMPLES for period accuracy (0.01% error at M=1e6 versus
# 1.00% over the full range; scripts/benchmarks/RESULTS.md).
#
# It also moves the "no orbit" solution. The Thiele-Innes amplitude prior scales
# as (P/P0)^(2/3), so the shortest period in the prior is where an orbit is
# forced to zero amplitude and the model collapses to a five-parameter
# astrometric fit. At 5e-5 yr that null is cheap; at 0.01 yr sigma_a is 34x
# larger, so the null carries a far bigger Occam penalty of its own. On the
# first 300k-system run that null took 65% of all recovery failures.
#
# THE COST: a system whose injected period falls outside this window cannot be
# recovered, by construction rather than by measurement. `census.in_search_range`
# exists so that is reported separately instead of being counted as a failure --
# see HARV.md.
PERIOD_MIN_YR = 0.01  # 3.7 d; below this ~80 epochs over 5.5 yr alias
PERIOD_MAX_YR = 100.0  # ~18x the mission baseline

# The remaining scales are set from the catalog itself (500 pc sample):
#   parallax    5.9 - 74 mas, median 14
#   v_tan       6 - 119 km/s, median 20
#   sma         0.012 - 71 AU
# Each is a scale parameter, not a hard bound, so generous is the right side to
# err on -- a HalfNormal or Normal with these sigmas covers the catalog with
# room to spare.
# Deliberately uninformative -- NOT centered on the catalog's parallax or proper
# motion. Centering on them is measurably better (period error 2.63% -> 0.2%),
# but the catalog's values are the ones the epochs were simulated from, so a
# tight prior on them hands the sampler the truth and flatters the recovery
# test. Real DR3 astrometry for an astrometric binary is biased by the very
# orbit being fitted. The Thiele-Innes parameterization recovers that accuracy
# (0.14%) without using any truth value, which is why it is the choice above.
SIGMA_PARALLAX_MAS = 100.0  # HalfNormal; catalog maxes at 74
SIGMA_POS_MAS = 100.0  # reference-position offset; the catalog injects 0
SIGMA_VTAN_KMS = 100.0  # tangential velocity; catalog maxes at 119
P0_YR = 1.0

# ==========================================================================
# The orbit-amplitude prior, as a companion mass
# ==========================================================================
# This one is not a scale to tune -- it sets the DETECTION THRESHOLD. It is the
# width of the Gaussian prior on the orbit's astrometric amplitude, so it fixes
# the Occam penalty a real orbit pays against the no-orbit solution, and because
# harv scales that width as (P/P0)^(2/3) x parallax the penalty grows with
# period and so falls on real orbits and barely at all on the null.
#
# So it is expressed as the thing it physically is: the largest companion the
# prior expects. At P0, for a companion of mass m around a star of mass M,
#
#     a0 = a * m/(M+m)   with   a = (M P0^2)^(1/3)     ->   ~ m / M^(2/3)
#
# which is `sigma_a0_au()` in `library`. harv's own scaling exists to keep the
# prior "approximately constant in companion mass at fixed primary mass"; the
# M^(2/3) factor completes it, so one number covers every star.
#
# PER SYSTEM, AND FREE. sigma_a0 touches only the analytically marginalized
# Thiele-Innes priors, which are never drawn into the shared library -- it does
# not change the library fingerprint, and measured over six systems with
# different values it triggers no extra JIT compile.
#
# WHY 13: the deuterium-burning limit, the conventional planet/brown-dwarf
# boundary. It is a Gaussian scale, not a cutoff -- the catalog injects up to
# 51 M_Jup and 13 is only the 83rd percentile, so the most massive 17% sit
# beyond 1 sigma and pay a mild Occam cost. The sweep says that trade is worth
# taking (`scripts/benchmarks/RESULTS.md`).
#
# The sweep is EVIDENCE for this value, not its source: reading the argmax off a
# recovery curve measured against injected answers is fitting to the truth, the
# same objection that keeps the parallax prior uninformative. At the catalog's
# median host mass of 0.41 Msun, 13 M_Jup gives sigma_a0 = 0.022 AU -- which
# lands between the sweep's two best arms (0.01 and 0.03) without having been
# chosen from them.
M_MAX_MJUP = 13.0

# Pin a constant sigma_a0 in AU, disabling the mass scaling above. `None` for
# the per-system value. Only for sweeps, which vary a single constant across
# arms -- `scripts/benchmarks/sweep_sigma_a0.sh`.
SIGMA_A0_AU = None

# ==========================================================================
# The eccentricity prior
# ==========================================================================
# harv's default is Kipping (2013), Beta(0.867, 3.03) -- an astrophysical prior
# for real planets, mean e ~ 0.22. This catalog injects Uniform(0, 0.99), so
# Kipping under-samples the half of the truth it most needs to cover:
#
#              P(e>0.7)  P(e>0.9)
#   injected      0.293     0.091
#   Kipping       0.021     0.001     <- 14x and 91x under
#   this prior    0.225     0.048     <- 1.3x and 1.9x
#
# Measured consequence at M=1e6 on the high-SNR 1_companion subset: period
# recovery falls monotonically with the injected eccentricity, 45.3% at e<0.3
# down to 16.8% at e>0.9. A recovery test run under Kipping is measuring prior
# mismatch as much as method performance.
#
# A broad hump centred at 0.5 rather than Uniform(0, 0.99): it covers the
# injected range well while staying a proper informative prior. It still
# down-weights e > 0.9 by ~2x relative to the injection, which is where recovery
# is worst -- switch to Uniform if that residual matters.
ECC_LOC = 0.5
ECC_SCALE = 0.3

# ==========================================================================
# Jitter (excess variance)
# ==========================================================================
# The generator injects noise at `sigma_UEVA,single` (AL + calibration) and
# REPORTS `sigma_formal` (attitude + AL, no calibration term). That is
# deliberate -- equating them would give an artificially self-consistent data
# set -- but it means every fit weights by an uncertainty smaller than the
# scatter it is looking at. Measured over 3,000 high-SNR systems of the real
# catalog: median ratio 1.276, range 0.67 to 11.5.
#
# The consequence is not a small bias. Weight is `exp(-dchi2/2)` and chi-square
# scales as `1/sigma^2`, so under-reported errors sharpen the likelihood
# contrast between library draws by `r^2 = 1.63` in the exponent. That is a
# direct contributor to `ess ~ 1` and to the confidently-wrong periods the
# gallery shows: the sampler is not merely wrong, it is overconfident by
# construction.
#
# `harv.Jitter` adds `jitter^2` in quadrature to the diagonal. Set this to a
# scale in mas to switch it on; `None` leaves it off.
#
# THE COST, AND IT IS NOT SMALL. `Jitter` declares a NONLINEAR parameter, so the
# shared library goes from three sampled nonlinear dimensions to four. At fixed
# `N_PRIOR_SAMPLES` the effective resolution per dimension falls from
# `M^(1/3) ~ 100` to `M^(1/4) ~ 32`, and library size was measured to saturate
# near 10^6 for three. Adding a dimension is close to dividing the library, so
# this has to be measured against a matched-M baseline before it goes on by
# default -- `scripts/benchmarks/sweep_sigma_a0.sh` is the pattern.
#
# THE DESIGN TENSION. `sigma_reported` varies star to star across the catalog,
# but the library is ONE library for every system -- that is what makes 17.2 M
# posteriors comparable. So the jitter prior here is ABSOLUTE, in mas, and has
# to be broad enough to cover the whole catalog's noise range, which wastes
# resolution on every individual star. A per-star jitter prior would fix that
# and would break the shared library. `sigma_a0` escapes the same tension only
# because it shapes analytically marginalized priors and is never drawn.
JITTER_SIGMA_MAS = None


# ==========================================================================
# Epoch padding
# ==========================================================================
# harv JITs per epoch count and the catalog spans 44-298, so every distinct
# count would be a fresh compile -- 17.2 M of them. Pad each system up to the
# next bucket and set the padded uncertainties to infinity, which is exact
# because the covariance is diagonal: an infinite-variance epoch contributes
# nothing to the likelihood.
#
# ~8 buckets keeps compiles down while wasting little work; the cost of padding
# is linear in the padding, so the buckets are tighter where systems are dense.
EPOCH_BUCKETS = (64, 80, 96, 112, 128, 160, 200, 256, 320)


def bucket_for(n_epochs: int) -> int:
    """The smallest bucket that fits `n_epochs`."""
    for size in EPOCH_BUCKETS:
        if n_epochs <= size:
            return size
    msg = (
        f"{n_epochs} epochs exceeds the largest bucket {EPOCH_BUCKETS[-1]}; "
        "add a larger one to EPOCH_BUCKETS"
    )
    raise ValueError(msg)


# ==========================================================================
# Reporting thresholds
# ==========================================================================
# The two questions the per-system table exists to answer, and the lines that
# split them. Neither is a detection threshold -- detection is the periodogram
# stage's job, calibrated against the companion-free control.
#
# `ess` -- did the library resolve this posterior? Below ~10 effective samples
# the returned draws are a localization of the best-fitting orbit, not a
# posterior, and the uncertainty on them is not measured. Those systems are the
# candidates for a second, MCMC pass; the census counts them so the size of that
# pass is known before it is planned.
ESS_RESOLVED = 10.0

# `weight_captured` -- was TOP_K big enough? It is the posterior mass the stored
# samples carry. Near 1.0 means nothing was truncated. Below this, TOP_K threw
# real mass away and should be raised.
WEIGHT_CAPTURED_MIN = 0.9

# `railed` -- did the fit collapse to the "no orbit" solution? The amplitude
# prior scales as (P/P0)^(2/3), so the shortest period in the prior forces the
# orbit to zero amplitude. A best sample sitting there is a NON-DETECTION, which
# is a different failure from finding the wrong period, and mixing the two makes
# a recovery percentage uninterpretable. Multiplies PERIOD_MIN_YR rather than
# being an absolute period, so it follows the prior if the bounds move.
RAIL_FACTOR = 1.5

# ==========================================================================
# Output
# ==========================================================================
PARQUET_COMPRESSION_LEVEL = 3
SYSTEMS_FLUSH_EVERY = 5000  # systems buffered before a per-system row-group flush
SAMPLES_FLUSH_EVERY = 200  # ~12 MB of float32 samples per row group

# Samples are stored as float32: 12 parameters x 1024 samples x 17.2 M systems
# is ~850 GB at float32 and 1.7 TB at float64, and the third digit of a
# posterior draw is not a measurement. Every *summary* in the per-system table
# is computed on the float64 values before this cast, so the cast is a storage
# choice and not a numerical one.
SAMPLE_DTYPE = "float32"

# ==========================================================================
# Per-system diagnostic gallery
# ==========================================================================
# Individual systems, chosen by a 2-D grid in (SNR, injected period), so the
# gallery spans the regimes rather than whichever systems happen to be first.
# There is no point plotting 100,000 of them; a handful per cell is enough to
# see what a regime looks like, and the cells are what make it representative.
#
# The period edges deliberately isolate 0.79-1.26 yr. That cell is the one to
# look at first: a one-year orbit is degenerate with parallax, which is a free
# linear parameter in the model, so its posterior should be visibly BIMODAL --
# one mode attributing the signal to a companion, one to parallax.
GALLERY_PER_BIN = 8
GALLERY_LOG_PERIOD_BINS = (-2.0, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 2.0)

# How much posterior mass a gallery panel draws as *posterior*. With ess ~ 1-8
# only a handful of the TOP_K draws carry any weight at all; the rest are prior
# draws that happened to rank highest, and drawing them like solutions is what
# made the old panels ~99.7% a picture of the prior. Everything outside this
# fraction is drawn as faint grey coverage instead.
GALLERY_WEIGHT_MASS = 0.999

# ==========================================================================
# Subsampling
# ==========================================================================
# `None` runs the full catalog. An integer runs approximately that many systems
# per population, which is what makes a dev run, a timing run, or a first look
# at the ESS distribution affordable -- the full run is ~12,000 core-hours.
#
# It is converted to a per-shard cap rather than a global one, so no rank has to
# know what any other rank is doing and a subsampled run is still just "each
# rank does its own shards". The systems taken are the first of each shard,
# which is a contiguous block of the source list per shard and therefore ~320
# scattered patches of sky rather than one. Companion draws are seeded per
# source id and independent of position, so the *orbits* are an unbiased
# sample; the sky coverage is not, which matters only for a scan-law-dependent
# question. For those, run the full catalog.
SUBSAMPLE = None


def limit_per_shard(n_shards):
    """Systems per shard under `SUBSAMPLE`, or `None` for the full sample."""
    if SUBSAMPLE is None:
        return None
    return max(1, -(-int(SUBSAMPLE) // int(n_shards)))


def set_subsample(n) -> None:
    """Cap systems per population (`--subsample`); `None` for all of them."""
    global SUBSAMPLE
    SUBSAMPLE = None if n is None else int(n)


# ==========================================================================
# Paths
# ==========================================================================
OUTPUT_ROOT = _cat.OUTPUT_ROOT / "harv"


def catalog_root():
    """The catalog being read. Owned by `periodogram.config`, not duplicated.

    The epochs come through `periodogram.shards.ShardReader`, so that module's
    root *is* the root -- a second copy here would be a second way to point the
    two stages at different catalogs while both claimed to be configured.
    """
    return _pg.CATALOG_ROOT


def set_catalog_root(path) -> None:
    """Point the epoch shards at a catalog delivered as a directory."""
    _pg.set_catalog_root(path)


def set_output_root(path) -> None:
    """Point every output path somewhere else (`--output-root`)."""
    global OUTPUT_ROOT
    OUTPUT_ROOT = Path(path).resolve()


def manifest_path():
    """The run manifest: the prior library's settings, seed, and fingerprint."""
    return OUTPUT_ROOT / "manifest.json"


def samples_dir(population):
    return OUTPUT_ROOT / "samples" / population


def systems_dir(population):
    return OUTPUT_ROOT / "systems" / population


def samples_shard(population, shard, n_shards, part=0, n_parts=1):
    return (
        samples_dir(population)
        / f"samples_{_pg._piece(shard, n_shards, part, n_parts)}.parquet"
    )


def systems_shard(population, shard, n_shards, part=0, n_parts=1):
    return (
        systems_dir(population)
        / f"systems_{_pg._piece(shard, n_shards, part, n_parts)}.parquet"
    )


def merged_systems(population):
    """One population's per-system rows, merged. The samples are never merged.

    The per-system table is ~2 GB for the whole catalog, so one file per
    population is convenient. The samples are ~850 GB and stay sharded --
    read them with `pyarrow.dataset` over `samples_dir(population)`.
    """
    return OUTPUT_ROOT / f"harv_systems_{population}.parquet"


def figure_dir():
    return OUTPUT_ROOT / "figures"


def failed_dir():
    return OUTPUT_ROOT / "failed"
