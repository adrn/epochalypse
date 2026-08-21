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

**3. Sharded output.** One parquet pair per (population, rank or shard) rather
than one CSV per system — 12 million small files is not a workable layout.
Writes are atomic (`.tmp` then rename), so `--skip-existing` can treat a present
file as a finished one.

Sources are addressable by integer task id as well as by Gaia source id, so a
scheduler can hand a worker a number without knowing anything about Gaia. The
mapping is frozen and fingerprinted at index-build time.

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

## Running it

There are three drivers, all thin wrappers over the same `simulate_source()`
function, so they produce identical output for the same star (verified below).
Pick by where you are running:

| | driver | when |
| --- | --- | --- |
| cluster, many nodes | `run_mpi.py` | production at 4M scale |
| job array, no MPI | `simulate_source.py --id-start` | same cost, no mpi4py |
| laptop | `run_shard.py --workers N` | local runs and testing |
| one system | `simulate_source.py --id N` | debugging, inspection |

```bash
# 1. parent stellar sample + per-source lookup indices (once, ~3 s)
python catalog_generation/generate_catalog.py --stages stars index

# 2a. simulate on a cluster
mpirun -n 1024 python catalog_generation/run_mpi.py

# 2b. ...or in a job array, 240 tasks of 50k sources
python catalog_generation/simulate_source.py --id-start $((SLURM_ARRAY_TASK_ID * 50000)) \
    --id-count 50000 --population one-companion --write

# 2c. ...or on a laptop
python catalog_generation/run_shard.py --shards 0-31 --n-shards 32 --workers 8

# 3. merge, select the high-SNR views, draw figures
python catalog_generation/generate_catalog.py --stages merge select figures
```

Set `OMP_NUM_THREADS=1` in any job script: with hundreds of ranks per node the
per-rank BLAS threads would otherwise oversubscribe the cores.

### One source at a time

A source is addressable two ways. Integer **task ids** are what a scheduler
wants — they run `0..N-1` and mean nothing beyond "the Nth star":

```bash
python catalog_generation/simulate_source.py --id=0 --population one-companion
python catalog_generation/simulate_source.py --id=1 --population two-companion
```

or by Gaia DR3 source id, for a specific star:

```bash
python catalog_generation/simulate_source.py --gaia-id 5484066448309985152 \
    --population no-companion one-companion two-companion
python catalog_generation/simulate_source.py --gaia-id 5484066448309985152 \
    --population 2_companion --out-dir /tmp/one_source
```

`no-companion` / `one-companion` / `two-companion` alias to `0_companion` /
`1_companion` / `2_companion`.

The integer mapping is the source list built by `build_indices.py`, frozen at
build time and fingerprinted in `index_manifest.json`. Every run verifies the
fingerprint, so a stale `stars.csv` on one node cannot silently change which
star `--id 7` means. Note the asymmetry that makes this safe: the *simulation*
is seeded from the Gaia source id, never the task id, so rebuilding the list
would change which worker computes what, never what the data is.

### How MPI is used

`mpirun -n 1024 python run_mpi.py` launches 1024 copies of the same script. Each
copy asks MPI which rank it is and how many there are, takes the corresponding
**contiguous** slice of the source list, and simulates it. Ranks never
communicate except for one `gather` at the end to print a summary, and no two
ranks write the same file — so nothing here needs MPI-IO or parallel HDF5. MPI
is a launcher, not a message bus.

Contiguous slices rather than round-robin because at 4M stars the scan law is
tens of GB and memory-mapped: a rank reading a contiguous block of sources
streams a contiguous region of the file, while round-robin would scatter reads
across the whole thing. Per-source cost varies little, so I/O locality is worth
more than load balancing.

mpi4py is optional — without it the script runs as a single rank, which is what
makes it usable on a laptop and for debugging:

```bash
python catalog_generation/run_mpi.py --limit 200      # one process, no MPI
mpirun -n 8 python catalog_generation/run_mpi.py      # 8 local processes
python catalog_generation/run_mpi.py --dry-run        # print each rank's slice
```

### Restarts

Each rank writes to `.parquet.tmp` and renames on success, so a rank killed
mid-write leaves no file rather than a truncated one that looks complete. That
makes `--skip-existing` trustworthy: rerunning a job only redoes the ranks that
died.

```bash
mpirun -n 1024 python catalog_generation/run_mpi.py --skip-existing
```

## Layout

```
epochalypse_parallel/
├── catalog_generation/
│   ├── generate_catalog.py     configuration + stage driver (stars/index/simulate/merge/select/figures)
│   ├── simulate_source.py      one source (--id N or --gaia-id) or a block (--id-start)
│   ├── run_mpi.py              MPI ranks; the cluster entry point
│   ├── run_shard.py            hash-partitioned shards over a local process pool
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

## Scaling: the compute is trivial, the warm-up is not

Measured per source on one core:

| | |
| --- | --- |
| warm call (this epoch count already compiled) | **3.5 ms** |
| first call at a new epoch count | **414 ms** — 118× |
| distinct epoch counts in the sample | **212** (44–298 transits) |
| process start + JAX init | **0.9 s** |

JAX retraces per array shape and every star has a different number of transits,
so a fresh process pays ~0.9 s of startup **plus ~88 s of compilation** before
reaching steady state. Warming up is visible: 181 → 63 → 24 → 16 ms/source over
the first 1,200 sources.

For 4M stars × 3 populations = 12M systems:

* **actual compute: ~12 core-hours** (12M × 3.5 ms)
* **one process per source: ~4,300 core-hours** of startup and compilation

That is the whole design argument. A single-source invocation is the right
*interface* but the wrong *granularity*: give each rank tens of thousands of
sources so the warm-up is amortized. A rank with a few hundred sources is almost
entirely compilation, which also means short test runs are useless for timing.

Rank sizing, at total = ranks × 89 s warm-up + 42,000 core-s of work:

| ranks | wall clock | efficiency |
| --- | --- | --- |
| 200 | ~5 min | 70% |
| 1,000 | ~2 min | 32% |
| 4,000 | ~1.7 min | 11% |

Past ~1,000 ranks the job is mostly compilation. It is a minutes-long job either
way, so there is little reason to push to the full core count. (These are
M-series laptop cores; cluster cores may be 2–3× slower, and writing ~50 GB of
parquet is a real cost not measured here.)

If you ever do want to run at 4,000+ ranks, the fix is to pad epoch arrays into
~8 bucketed shapes, collapsing 212 compilations into 8 (~3 s instead of ~88 s).
That changes the shape passed to `jr.normal`, so it needs a check that truncating
a longer draw reproduces the current noise realization before being relied on.

## Status

Verified, all with exactly zero difference:

* the same source run standalone, inside a shard, inside an MPI rank, and via
  `--id-start` all produce identical epochs and truth rows;
* the simulator is bit-identical to the serial pipeline given identical inputs
  and seeds;
* the random populations agree with the serial pipeline's on every distribution
  (KS p-values 0.08–0.9 across sma, mass, eccentricity, inclination, period,
  alpha, SNR) — they are different realizations by construction, since the
  seeding scheme changed, so this is a statistical rather than value-by-value
  comparison;
* `--skip-existing` redoes nothing on a rerun and leaves no `.tmp` files;
* merge, top-1% selection, and all six figures.

Not yet done: the analysis side (`src/run_periodograms.py` etc.) still assumes the
serial catalog's file layout and population names, so the characterization step
needs porting to the parquet shards before it can run against this catalog.
