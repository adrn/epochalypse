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
