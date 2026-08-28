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
from ..periodogram import config as _pg

# Re-exported so every stage reads one authority as `config.X`.
POPULATIONS = _cat.POPULATIONS
PARQUET_COMPRESSION = _cat.PARQUET_COMPRESSION

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
# Period bounds bracket the injected prior at BOTH ends, the same requirement
# the periodogram grid has and for the same reason: a truth outside the prior
# is unrecoverable by construction rather than by measurement. Imported so the
# two analyses cannot drift apart.
PERIOD_MIN_YR = _pg.P_MIN  # 5e-5 yr, the Roche-limited innermost orbit
PERIOD_MAX_YR = _pg.P_MAX  # 3300 yr

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
SIGMA_A0_AU = 1.0  # semi-major axis at P0, scaling as (P/P0)^(2/3)
P0_YR = 1.0

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


def failed_dir():
    return OUTPUT_ROOT / "failed"
