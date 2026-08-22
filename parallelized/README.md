# epochalypse — parallel pipeline

Generates the simulated Gaia DR4 epoch-astrometry catalog for ~4 million stars.
Self-contained: everything it reads and writes is under this directory.

The physics is the serial pipeline's: the same priors, the same Roche-lobe and
Hill-stability screens, the same jaxoplanet reflex model and DR3-calibrated
noise model, the same astropy-derived constants. What changed is how the work is
divided, and how the high-SNR sample is defined.

**Per-source determinism.** Every random stream is seeded from the Gaia source
id, `blake2s(master_seed : population : gaia_source_id)`, not from a row index.
A star's companions and noise realization depend only on that star, so any
subset can be generated in any order, on any number of ranks, and re-running one
source reproduces it exactly. Index-based seeding cannot survive being split.

**No detectability rejection.** All three populations are drawn from the
unbiased prior. `SNR_tot` is recorded per companion but never used to accept or
reject; the high-SNR sample is selected afterwards as the top
`HIGH_SNR_FRACTION` by recorded `SNR_tot`. At 4M stars, rejection sampling to a
fixed threshold is expensive and bakes the threshold into the data.

## Running it

```bash
# 1. parent stellar sample + per-source lookup indices (once, ~3 s)
python catalog_generation/generate_catalog.py --stages stars index

# 2. simulate (the expensive part -- OMP_NUM_THREADS=1 in the job script)
mpirun -n 1024 python catalog_generation/run_mpi.py

# 3. merge the shards, select the high-SNR views, draw the figures
python catalog_generation/generate_catalog.py --stages merge select figures
```

Locally, `run_mpi.py` falls back to a single rank when mpi4py is absent, which
is how you test it:

```bash
python catalog_generation/run_mpi.py --limit 200      # one process, no MPI
mpirun -n 8 python catalog_generation/run_mpi.py      # 8 local processes
python catalog_generation/simulate_source.py 5484066448309985152   # one star, printed
python test_pipeline.py                               # self-check, no data needed
```

Each rank writes `.parquet.tmp` and renames on success, so a rank killed
mid-write leaves no file rather than a truncated one that looks complete. That
makes `--skip-existing` trustworthy: rerunning a job only redoes the ranks that
died.

## Populations

| population | kind | how |
| --- | --- | --- |
| `0_companion` | simulated | noise-only control |
| `1_companion` | simulated | one companion, unbiased prior |
| `2_companion` | simulated | two companions, unbiased prior |
| `*_high_snr` | selected | top 1% of the above by `SNR_tot` |

The high-SNR views are a quantile over a column, so re-selecting at a different
fraction costs seconds and needs no regeneration.

## Layout

```
parallelized/
├── catalog_generation/
│   ├── generate_catalog.py     stages: stars, index, merge, select, figures
│   ├── run_mpi.py              the simulation; MPI ranks, the cluster entry point
│   ├── simulate_source.py      print one star, for inspection
│   └── pipeline/
│       ├── config.py           every prior, path, seed, and figure choice
│       ├── stars.py            parent stellar sample
│       ├── sources.py          per-source lookup: SourceCatalog, ScanLawStore
│       ├── planets.py          per-source companion draw (Roche + Hill screens)
│       ├── astrometry.py       per-source epoch simulation + ShardWriter
│       └── figures.py          the catalog figures
├── src/                        shared constants; the analysis modules
├── data/                       static inputs
├── outputs/                    generated: data/ (shards, indices, truth tables), figures/
└── test_pipeline.py            self-check for the seeding, priors, and screens
```

## Why the lookup layer exists

At 4M stars the scan law is ~400M rows. `pipeline/sources.py` builds two small
indices once (`gaia_source_id -> row`, `gaia_source_id -> (offset, length)`) and
every rank memory-maps the Arrow table behind them, so a process touches only
the pages for the sources it actually simulates instead of loading tens of GB.
The scan-law index requires each source to occupy one contiguous block; that is
checked at build time and reported with the offending ids.

## Scaling: the compute is trivial, the warm-up is not

Measured per source on one core: a warm call is **3.5 ms**, the first call at a
new epoch count is **414 ms**, and the sample contains **212 distinct epoch
counts** (44–298 transits). JAX retraces per array shape, so a fresh process
pays ~0.9 s of startup plus **~88 s of compilation** before reaching steady
state.

For 4M stars × 3 populations = 12M systems that is ~12 core-hours of actual
compute against ~89 s of warm-up per rank. Give each rank tens of thousands of
sources; a rank with a few hundred is almost entirely compilation, which also
means short test runs are useless for timing. Past ~1,000 ranks the job is
mostly compilation and is a minutes-long job either way, so there is little
reason to push to the full core count.

If you ever do want 4,000+ ranks, the fix is to pad epoch arrays into ~8
bucketed shapes, collapsing 212 compilations into 8. That changes the shape
passed to `jr.normal`, so it needs a check that truncating a longer draw
reproduces the current noise realization before being relied on.

## Status

The simulator is unchanged from the serial pipeline given identical inputs and
seeds, and the same source produces identical output standalone and inside an
MPI rank. The random populations agree with the serial pipeline's on every
distribution (KS p-values 0.08–0.9 across sma, mass, eccentricity, inclination,
period, alpha, SNR) — they are different realizations by construction, since
the seeding scheme changed.

Not yet done: the analysis side (`src/run_periodograms.py`) still assumes the
serial catalog's file layout and population names, so the characterization step
needs porting to the parquet shards before it can run against this catalog.
