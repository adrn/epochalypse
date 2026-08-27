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
`HIGH_SNR_MIN` on every injected companion. At 4M stars, rejection sampling to a
fixed threshold is expensive and bakes the threshold into the data; applying the
floor afterwards leaves it an analysis choice, and characterization applies the
same rule to the same rows.

## Setup

Dependencies are in `pyproject.toml`, pinned by `uv.lock` — commit the lock, it
is what records the versions a catalog was generated with.

```bash
uv sync
uv run python tests/test_pipeline.py   # self-check; needs only the committed inputs
```

`mpi4py` is deliberately not a default dependency: it has to be built against
the MPI that `mpirun`/`srun` actually use, and a wheel built against a different
one fails across nodes. On a cluster:

```bash
module load <site openmpi/mpich>
uv sync
MPICC=$(which mpicc) uv pip install --no-binary=mpi4py mpi4py
```

Without `mpi4py`, `simulate_mpi.py` runs as a single rank, which is what makes it
usable on a laptop. `uv sync --extra analysis` adds scipy and h5py for
`epochalypse.fitting`, which does not run against this catalog yet.

### Inputs

Two kinds, with different lifecycles.

**The delivered dataset** — the parent sample and the DR4 scan law, ~12 GB — is
relocatable with `--data-root`, because it changes from run to run and does not
belong in a home directory. The layout under that root mirrors `data/`, so
moving it is a straight copy:

```bash
--data-root $DATA_ROOT      # e.g. <scratch>/project-data/epochalypse
```

| under `<data-root>/g23h_epochalypse_stars/` | what |
| --- | --- |
| `G23H_within_500pc.arrow` (1.2 GB) | parent sample, one row per DR2 cross-match |
| `scanlaw_dr4_within_500pc_hpx64_transit_loss10.arrow` (10.7 GB) | DR4 scan law, one row per FoV transit |

`G23H_NAME` and `SCANLAW_NAME` in `src/epochalypse/config.py` are the filenames;
point `G23H_NAME` at the committed `G23H_sample_subset.arrow` to run the whole
pipeline against the 16k smoke sample.

**Reference data** is committed, versioned with the code, and never configured —
`data/pecaut_mamajek.txt` (the P&M13 sequence) and `data/gost_fov_counts_dr4.fits`
(the GOST healpix map, sky figure only). A fresh clone always has them, so the
tests need no dataset.

Outputs are relocatable the same way, with `--output-root`; the periodogram half
adds `--catalog-root` for reading a catalog someone else generated. All five
`mpi/*.sh` scripts set the ceph roots at the top.

The scan law **must be grouped by `gaia_source_id`**, one contiguous block per
source, or the offset index is meaningless. `build_indices` verifies this and
raises naming the offending ids rather than returning the wrong epochs.

## Running it

```bash
# 1. parent stellar sample + per-source lookup indices (once)
python scripts/generate_catalog.py --stages stars index

# 2. simulate -- the expensive part, and the only one that needs a cluster
mpirun python scripts/simulate_mpi.py

# 3. merge the shards, select the high-SNR views, draw the figures
python scripts/generate_catalog.py --stages merge select figures
```

Locally, `simulate_mpi.py` falls back to a single rank when mpi4py is absent, which
is how you test it:

```bash
python scripts/simulate_mpi.py --limit 200         # one process, no MPI
mpirun -n 8 python scripts/simulate_mpi.py         # 8 local processes
python scripts/simulate_source.py 5484066448309985152   # one star, printed
```

Both drivers take `--output-root` to write somewhere other than `outputs/`, and
`--populations` to restrict which of the three are touched.

Each rank writes `.parquet.tmp` and renames on success, so a rank killed
mid-write leaves no file rather than a truncated one that looks complete. That
makes `--skip-existing` trustworthy: rerunning a job only redoes the ranks that
died.

## On a Slurm cluster

The submit scripts live in `mpi/` and are the source of truth — `1-prep.sh`,
`2-sim.sh`, `3-finish.sh` for the catalog, then `4-periodograms.sh` and
`5-periodogram-finish.sh` for the characterization. Each sets its roots at the
top:

```bash
DATA_ROOT=$CEPH/project-data/epochalypse       # the delivered inputs, ~12 GB
OUT_ROOT=$CEPH/project-outputs/epochalypse     # the catalog, ~50 GB
```

`mkdir -p logs` first, then chain them:

```bash
prep=$(sbatch --parsable mpi/1-prep.sh)
sim=$(sbatch --parsable --dependency=afterok:$prep mpi/2-sim.sh)
sbatch --dependency=afterok:$sim mpi/3-finish.sh
```

Three things about those scripts that are easy to get wrong:

**No `-n` on `mpirun`, and no `mpi4py.futures`.** Both MPI stages are SPMD, not
worker pools: every rank runs the same code, asks `COMM_WORLD` which rank it is,
takes its own contiguous slice, and writes its own files. Ranks talk once, in a
`gather` at the end. The allocation already fixes the rank count, so
`--ntasks-per-node` is the knob that matters — see below, because it is memory
that binds, not cores.

**`OMP_NUM_THREADS=1`.** With tens of ranks per node the per-rank BLAS thread
pools oversubscribe the cores, and that is invisible until the job is slow.
`JAX_PLATFORMS=cpu` also skips a pointless GPU probe on the simulation side.

**Activate `.venv` and call `python`, not `uv run`.** Under `mpirun` every rank
would otherwise re-check the environment at once. Run `uv sync` once before
submitting.

`--skip-existing` on both MPI stages means a requeue after a node failure only
redoes the ranks that died, so it costs nothing to leave on.

### Per-rank memory sets `--ntasks-per-node`

Every rank memory-maps the scan law, so that 10.7 GB is shared page cache rather
than per-rank. Three things are *not* shared, measured on the 16k sample and
scaled to 4M sources:

| per rank | at 4M sources |
| --- | --- |
| `stars.csv` DataFrame, the 13 columns the simulator reads | 0.6 GB |
| `ScanLawStore._span_of` (id → offset, length) | 0.8 GB |
| `SourceCatalog._row_of` (id → row) | 0.5 GB |
| **total** | **~1.8 GB** |

So 64 ranks/node needs ~116 GB and 128 needs ~232 GB. Divide your node's RAM by
1.8 GB and leave headroom. The rank count is memory-bound rather than
core-bound, which is why `SourceCatalog.COLUMNS` exists: holding all 117 columns
of `stars.csv` instead of those 13 costs 4.2 GB per rank rather than 0.6, and
128 ranks/node would not fit.

One piece of headroom left: `build_indices` converts the scan law's
`gaia_source_id` column to Python strings and factorizes it. At ~400M transits
that is why the prep job asks for a few hundred GB; keeping the ids as int64
would mostly remove it.

## Populations

| population | kind | how |
| --- | --- | --- |
| `0_companion` | simulated | noise-only control |
| `1_companion` | simulated | one companion, unbiased prior |
| `2_companion` | simulated | two companions, unbiased prior |
| `*_high_snr` | selected | every companion clears `SNR_tot >= 5` |

The high-SNR views are a threshold on a recorded column, so re-selecting at a
different floor costs seconds and needs no regeneration.

## Layout

```
epochalypse/
├── src/epochalypse/            the library
│   ├── config.py               every prior, path, seed, and figure choice
│   ├── constants.py            physical + mission constants, from astropy
│   ├── stars.py                parent stellar sample
│   ├── sources.py              per-source lookup + the high-SNR selection
│   ├── planets.py              per-source companion draw (Roche + Hill screens)
│   ├── astrometry.py           per-source epoch simulation + ShardWriter
│   ├── figures.py              the catalog figures
│   ├── mpi.py                  the MPI plumbing both parallel stages share
│   ├── shardio.py              the buffered parquet writer they share
│   └── periodogram/            the characterization half (see PERIODOGRAMS.md)
├── scripts/                    the entry points
│   ├── generate_catalog.py     stages: stars, index, merge, select, figures
│   ├── simulate_mpi.py         the simulation; MPI ranks, the cluster entry point
│   ├── simulate_source.py      print one star, for inspection
│   ├── characterize_mpi.py     the periodogram search; MPI ranks
│   ├── characterize_finish.py  stages: calibrate, census, merge
│   └── periodogram_source.py   print one system's periodogram, for inspection
├── pyproject.toml              dependencies; uv.lock pins them
├── data/                       static inputs (see Setup)
├── outputs/                    generated: data/ (shards, indices, truth tables), figures/
└── tests/                      test_pipeline.py, test_periodograms.py
```

## Why the lookup layer exists

At 4M stars the scan law is ~400M rows. `epochalypse/sources.py` builds two small
indices once (`gaia_source_id -> row`, `gaia_source_id -> (offset, length)`) and
every rank memory-maps the Arrow table behind them, so a process touches only
the pages for the sources it actually simulates instead of loading tens of GB.

## Scaling: the compute is trivial, the warm-up is not

Measured per source on one core: a warm call is **3.5 ms**, the first call at a
new epoch count is **414 ms**, and the sample contains **212 distinct epoch
counts** (44–298 transits). JAX retraces per array shape, so a fresh process
pays ~0.9 s of startup plus **~88 s of compilation** before reaching steady
state.

For 4M stars × 3 populations = 12M systems that is ~12 core-hours of actual
compute against ~89 s of warm-up per rank. Give each rank tens of thousands of
sources; a rank with a few hundred is almost entirely compilation, which also
means short test runs are useless for timing. At 256 ranks the whole simulation
is a handful of minutes of compute — writing ~50 GB of parquet to Ceph is the
part that is not measured here, which is why the walltime above is generous.

If you ever do want 4,000+ ranks, the fix is to pad epoch arrays into ~8
bucketed shapes, collapsing 212 compilations into 8. That changes the shape
passed to `jr.normal`, so it needs a check that truncating a longer draw
reproduces the current noise realization before being relied on.

## Status

The epoch simulation and noise model are unchanged from the serial pipeline:
given the same star and the same seeds, `0_companion` epochs are bit-identical.

**The companion draws are a different realization.** Two changes moved them, and
neither changed a prior: the seeding scheme went from row index to source id,
and the draw loop went from a 256-wide batch to one proposal at a time. `sma` is
untouched, and old-vs-new agrees distribution-by-distribution (two-sample KS
p = 0.2 on mass; KS p = 0.08–0.9 against the serial pipeline across sma, mass,
eccentricity, inclination, period, alpha, SNR). Any previously generated
1- or 2-companion catalog must be regenerated rather than mixed with new output.

The characterization half now runs against the parquet shards -- see
`PERIODOGRAMS.md`. The old serial analysis module was deleted with it; two of
its capabilities have no successor in the kepmodel path and are noted there.
