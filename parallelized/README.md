# epochalypse — parallel pipeline

A parallelizable rebuild of the epochalypse catalog generator, aimed at ~4 million
stars instead of ~16 thousand.

The physics is unchanged: the same priors, the same Roche-lobe and Hill-stability
screens, the same jaxoplanet reflex model and DR3-calibrated noise model, the same
astropy-derived constants. What changed is *how work is divided* and *how the
high-SNR sample is defined*.

## The three changes

**1. Per-source determinism.** Every random stream is seeded from the Gaia source
id:

```
seed = blake2s(master_seed : population : gaia_source_id)
```

not from a row index in a list. A star's companions and noise realization depend
only on that star, so any subset can be generated in any order, on any number of
machines, and re-running one source reproduces it exactly. This is what makes the
pipeline shardable at all — index-based seeding cannot survive being split.

**2. No detectability rejection.** All three populations are drawn from the
unbiased prior. `SNR_tot` is computed and stored per companion but never used to
accept or reject. The high-SNR sample is selected *afterwards* as the top 1% by
recorded `SNR_tot` (`HIGH_SNR_FRACTION` in the config). At 4M stars, rejection
sampling to a fixed threshold is expensive and bakes the threshold into the data;
a quantile selection is instant and leaves it an analysis choice.

**3. Sharded output.** One parquet pair per (population, shard) rather than one
CSV per system — 12 million small files is not a workable layout.

## Populations

| population | kind | how |
| --- | --- | --- |
| `0_companion` | generated | noise-only control |
| `1_companion` | generated | one companion, unbiased prior |
| `2_companion` | generated | two companions, unbiased prior |
| `1_companion_detectable` | **derived** | top 1% of `1_companion` by `SNR_tot` |
| `2_companion_detectable` | **derived** | top 1% of `2_companion` by `SNR_tot` |

Derived populations are never simulated — they are a quantile over a column, so
re-selecting at a different fraction costs seconds and needs no regeneration.
The figures are unchanged from the serial pipeline and still compare random
against high-SNR; they simply read the derived view.

## Workflow

```bash
# 1. parent stellar sample + the per-source lookup indices (once)
python catalog_generation/generate_catalog.py --stages stars index

# 2. simulate. One shard is the unit of parallel work.
python catalog_generation/run_shard.py --shard 0 --n-shards 512          # one shard
python catalog_generation/run_shard.py --shards 0-31 --n-shards 512 --workers 8
python catalog_generation/run_shard.py --n-shards 512 --print-commands   # for SLURM etc.

# 3. merge shard truth tables, select the high-SNR views, draw figures
python catalog_generation/generate_catalog.py --stages merge select figures

# everything in one process (small runs only)
python catalog_generation/generate_catalog.py --workers 8
```

### One source at a time

`simulate_source.py` is the atom the whole pipeline is built from, and is useful
on its own for inspecting or debugging a system:

```bash
python catalog_generation/simulate_source.py --gaia-id 5484066448309985152 \
    --population 0_companion 1_companion 2_companion
python catalog_generation/simulate_source.py --gaia-id 5484066448309985152 \
    --population 2_companion --out-dir /tmp/one_source
python catalog_generation/simulate_source.py --gaia-id 5484066448309985152 --which-shard 512
```

A source simulated standalone is byte-identical to the same source inside a shard
(verified: max epoch and truth difference exactly 0).

## Layout

```
epochalypse_parallel/
├── catalog_generation/
│   ├── generate_catalog.py     configuration + stage driver (stars/index/simulate/merge/select/figures)
│   ├── simulate_source.py      ONE Gaia id + population -> epochs + truth
│   ├── run_shard.py            one shard of sources; the unit of parallel work
│   ├── build_indices.py        per-source lookup indices (run once after `stars`)
│   ├── make_figures.py         figures alone, from an existing catalog
│   └── pipeline/
│       ├── config.py           schema, incl. generated vs derived populations
│       ├── stars.py            parent stellar sample
│       ├── sources.py          per-source lookup: SourceCatalog, ScanLawStore, shard_of
│       ├── planets.py          per-source companion draw (Roche + Hill screens)
│       ├── astrometry.py       per-source epoch simulation + ShardWriter
│       └── figures.py          unchanged figures + the high-SNR selection adapter
├── src/                        shared constants and analysis modules
├── data/                       static inputs (copies)
└── outputs/                    generated: data/ (shards, indices, truth tables), figures/
```

## Why the lookup layer exists

At 4M stars the scan law is ~400M rows. `pipeline/sources.py` builds two small
indices once (`gaia_source_id -> row`, `gaia_source_id -> (offset, length)`) and
every worker memory-maps the Arrow table behind them, so a process touches only
the pages for the sources it actually simulates instead of loading tens of GB.

The scan-law index requires each source to occupy one contiguous block; that is
checked at build time and reported with the offending ids rather than silently
returning the wrong epochs.

## Scaling

Measured on this machine: **~27 systems/s with 4 workers** (3 populations,
including JAX compilation in short shards). Extrapolated linearly that is ~120
core-hours-equivalent for 3 × 4M systems at 4 workers — i.e. a few hours on 64
cores, and the work is embarrassingly parallel, so it scales with whatever you
throw at it. Per-shard cost is dominated by the epoch simulation, not the draws.

Sizing note: shard balance over 512 shards for the current 16k sample is
min 17 / median 32 / max 50 sources; at 4M it would be ~7,800 per shard.

## Status

Verified: per-source determinism (standalone ≡ in-shard, exactly 0 difference),
shard partition purity and balance, 8-shard parallel run across 4 workers, merge,
top-1% selection, and all six figures.

Not yet done: the analysis side (`src/run_periodograms.py` etc.) still assumes the
serial catalog's file layout and population names, so the characterization step
needs porting to the parquet shards before it can run against this catalog.
