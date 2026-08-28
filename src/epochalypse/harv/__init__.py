"""Posterior inference over the catalog with harv's rejection sampler.

The generation half writes epochs; the periodogram half characterizes them in
the frequency domain; this half samples orbit posteriors for every system
against one shared prior library.

    config     every prior bound, library size, and path this stage decides
    adapt      epochalypse epoch rows -> harv `GaiaAstrometryData`, with padding
    library    the one prior library, drawn in process from a fixed seed
    unit       one work unit: every system in one shard
    writers    the parquet output, on `epochalypse.shardio`

Read the samples with their `weight` column. They are importance-weighted, not
equal-weight draws -- see HARV.md.
"""

import jax as _jax

# x64 belongs here, not in one module. harv's `sharp-bits.md` is explicit that
# float32 makes the marginalized likelihoods unstable, and the failure is
# SILENT GARBAGE rather than an error: over this stage's 7.8-decade period prior
# every log-likelihood underflows to -inf, `logZ_int_ess` comes back NaN, and
# top-K then returns arbitrary rows. A script that imports `library` and `adapt`
# but not `unit` used to get exactly that, because `unit` was the only module
# setting it. Must run before any jax array is made.
_jax.config.update("jax_enable_x64", True)
