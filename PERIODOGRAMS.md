# epochalypse — periodograms

Runs the `kepmodel` astrometric periodogram over **every system in the DR4
catalog**: 5,724,586 stars × 3 populations = 17,173,758 period searches, on the
cluster the catalog was generated on. Self-contained: everything it reads is the
generator's parquet shards, everything it writes is under `outputs/`.

The statistic is `src/run_kepmodel_periodograms.py`'s, unchanged. What changed is
scale. That notebook characterizes 10,000 systems per population out of a
materialized subset; this characterizes all of them, off the shards, with no
subset in between.

**The high-SNR populations are not separate runs.** The generator draws all
three populations from the unbiased prior and records `SNR_tot` rather than
rejecting on it, so "one companion: high-SNR" is the subset of `1_companion`'s
rows with `SNR_tot >= 5` on every companion — rows this run has already
characterized. Selecting after the fact is what makes 5.7 M × 3 the whole job
rather than 5.7 M × 5, and it means re-cutting at a different floor costs a
boolean mask rather than 2,000 core-hours.

**One work unit is one shard.** The generator wrote 320 shards per population;
this reads them back one at a time, start to finish, and writes one
characterization file per shard. `src/sharded_systems.py` solves the opposite
problem — pulling 10,000 scattered systems out of 320 files through row-group
statistics — and is the right tool when the analysis touches 0.2% of the
catalog. Here it touches all of it, once, so nothing is materialized.

**The period grid is stored once, not once per system.** It depends on `P_MIN`,
`P_MAX` and `N_SEGMENTS` and on nothing about the star, so all 17 M curves are
sampled on the same 16,641 trial periods. `outputs/periodograms/period_grid.parquet`
holds them; storing a period axis beside every curve would exactly double the
largest output this pipeline produces. `tests/test_periodograms.py` asserts the
identity that rests on — that `kepmodel` visits exactly `segment_periods(segments)`.

## Setup

One environment for the whole repo -- see `PIPELINE.md` for `uv sync` and the
cluster `mpi4py` build. `kepmodel` (the period search) and its `spleaf` noise
terms come in with it.

```bash
uv run python tests/test_periodograms.py                          # synthetic, no catalog
uv run python tests/test_periodograms.py --catalog-root <catalog>  # + one real work unit
```

## Running it

```bash
# 1. the search -- the expensive part, and the only one that needs a cluster
mpirun python scripts/characterize_mpi.py --skip-existing

# 2. thresholds, class counts, and (optionally) merged tables
python scripts/characterize_finish.py --stages calibrate census merge
```

Locally:

```bash
python epochalypse.periodogram.unit.run_unit 1_companion 7                    # one shard, ~1.8 h
python epochalypse.periodogram.unit.run_unit 1_companion 7 --limit 300        # ~2 min
python scripts/characterize_mpi.py --max-units 2 --limit 20                    # one process, no MPI
mpirun -n 8 python scripts/characterize_mpi.py --max-units 8 --limit 20        # 8 local processes
python scripts/periodogram_source.py 1_companion 5484066448309985152 --plot one.pdf
```

**Pass `--max-units` for a laptop smoke test, not just `--limit`.** `--limit`
caps systems per unit but still walks all 960 units, so `--limit 20` alone is
still a few hours on one process. `--max-units 2 --limit 20` is fifteen seconds.

Both drivers take `--catalog-root` and `--output-root`; `characterize_mpi.py` also takes
`--populations` and `--power`.

Each unit writes `.parquet.tmp` and renames on success, so a rank killed
mid-write leaves no file rather than a truncated one that looks complete. That
is what makes `--skip-existing` trustworthy: rerunning a job only redoes the
units that died, and it costs nothing to leave on.

## On a Slurm cluster

`characterize_mpi.py` is SPMD, not a worker pool: every rank runs the same code, asks
`COMM_WORLD` which rank it is, takes its own contiguous slice of the work-unit
list, and writes its own files. Ranks talk once, at a `Barrier` after rank 0
writes the manifest, and once more in a `gather` at the end to print a summary.
So there is **no** `-m mpi4py.futures`, no `--mpi` flag, and no `-n` on
`mpirun` — the allocation already fixes the rank count.

`mpi/1-run.sh` and `mpi/2-finish.sh` are the two job scripts; chain them:

```bash
mkdir -p logs
run=$(sbatch --parsable mpi/1-run.sh)
sbatch --dependency=afterok:$run mpi/2-finish.sh
```

Both follow the generator's `mpi/*.sh`: `cd` to the checkout,
`source .venv/bin/activate`, and set their roots at the top. Both pass
`--catalog-root` and `--output-root`, because at ~915 GB the curves belong on
scratch rather than in the repo tree (the defaults are `<repo>/outputs` for the
shards,
`<repo>/outputs/periodograms` for the results) are the generator's own layout.
Run `uv sync` once before submitting; the scripts activate `.venv` and call
`python` rather than using `uv run`, since under `mpirun` every rank would
otherwise re-check the environment at once.

### Rank count

960 work units (320 shards × 3 populations) caps the useful rank count at 960
without `--n-parts`. Beyond that, `--n-parts N` cuts each shard into N
contiguous pieces — cut on *systems*, not row groups, so four parts really is
four near-equal quarters even though the writer's first row group holds 2,000
systems and the rest hold one each. Every part that touches the first row group
re-reads it, so keep N small.

Unlike the generator, **memory is not the binding constraint**. A rank holds one
row group of epochs and one shard's truth table, ~0.3 GB, so 128 ranks/node fits
on a rome node with room to spare. Cores are the constraint, which is why
`--ntasks-per-node` should be as high as the partition allows.

There is also **no JAX warm-up**. `kepmodel` is NumPy and S+LEAF, so a rank
reaches steady state in under a second, short test runs give honest timings, and
a rank with only a few hundred systems is not pathological the way it was during
generation.

## Cost

Measured, not estimated: **340 ms per system** on one Apple M-series core
(16,641 trial periods, `FIT_JITTER = False`), and **357 ms** including shard
reads and parquet writes. The cost is nearly flat in the epoch count — 320 ms at
48 epochs, 375 ms at 169 — because the 16,600-iteration frequency loop
dominates, not the linear algebra. A rome core is roughly 1.5–2× slower on
scalar Python, so budget **0.5–0.7 s/system** there.

| | systems | core-hours (M-series) | core-hours (rome, est.) |
| --- | --- | --- | --- |
| one shard | 17,890 | 1.8 | 2.5–3.5 |
| one population | 5,724,586 | 570 | 800–1,100 |
| **all three** | **17,173,758** | **1,700** | **2,400–3,300** |

| ranks | nodes × tasks | wall (rome, est.) |
| --- | --- | --- |
| 320 | 10 × 32 | 7.5–10 h |
| 640 | 10 × 64 | 4–5 h |
| 960 | 15 × 64 | 2.5–3.5 h |

`mpi/1-run.sh` asks for 320 ranks and 12 h, which is the conservative corner of
that table. The compute is embarrassingly parallel and the I/O is one sequential
read and one sequential write per unit, so the scaling should hold until the
filesystem notices — at 960 ranks writing `--power all` that is ~2.6 GB/s of
parquet, which is worth checking against your Ceph allocation before submitting.

## Storage

Two outputs, and they differ by three orders of magnitude.

**The characterization table** is one row per system: the periodogram summary,
both detection channels, the truth-based recovery flags, and the injected truth
columns joined on. 50 columns for `1_companion`, 74 for `2_companion`.
Measured at **343 bytes per system** after zstd (excluding the parquet footer,
which amortizes to ~5 B/row at 17,890 rows per shard):

| | rows | size |
| --- | --- | --- |
| `0_companion` | 5,724,586 | ~1.4 GB |
| `1_companion` | 5,724,586 | ~2.0 GB |
| `2_companion` | 5,724,586 | ~2.9 GB |
| **total** | **17,173,758** | **~6.3 GB** |

It is written for every system, always, and it is what the three paper figures
read.

**The raw periodograms** are one Δχ² curve per system: 16,641 float32 = 66.6 kB
raw, **53.3 kB after zstd** (measured on real curves — they are noisy, so
compression buys only ~20%).

| `--power` | what is stored | all three populations |
| --- | --- | --- |
| `all` | every system (the default) | ~915 GB |
| `none` | nothing | 0 |

**`all` is the default, and 915 GB needs somewhere to go.** That is roughly ten
times the catalog it was computed from, so point `--output-root` at ceph rather
than the repo tree:

```
--output-root $OUT_ROOT/periodograms
```

Storing every curve is deliberate. Which curves are interesting is an analysis
question — stacking them, refitting peaks, testing a different width metric —
and subsampling at write time answers it once, permanently, for whoever comes
next. A curve not written back can only be recovered by recomputing it, and
recomputing the catalog is ~2,400 core-hours. Subsample in analysis instead,
where the choice is cheap and reversible. (For a single system,
`scripts/periodogram_source.py` regenerates a curve in 0.34 s regardless.)

`config.POWER_DTYPE` halves the total without changing *which* systems are
kept:

| | float32 | float16 |
| --- | --- | --- |
| full grid | 915 GB | ~460 GB |

Decimation is safer than it looks: **every summary statistic the classification
depends on — `width_dex`, `top_power`, `best_period`, `period_k_in_bound` — is
computed on the full-resolution float64 curve before anything is thrown away.**
The stored curve is for plotting and re-analysis, not for the measurement, so
decimating by 4 costs a figure nothing at 16,641 points across 7.8 decades.
float16 is the more aggressive choice: it holds ~3 decimal digits, so a Δχ² of
1,000 is stored to ±0.5, against a `WIDTH_DELTA` of 4. Fine on a log axis, not
something to re-measure a width from.

## Output layout

```
outputs/
├── manifest.json                   the grid, the noise model, the classifier
│                                   thresholds, the code version -- written before
│                                   any results, so the output is never uninterpretable
├── calibration.json                thr_orbit / thr_accel and the realized null rate
├── characterization/
│   └── <population>/
│       └── chars_shard00007_of_00320.parquet     one row per system
├── periodograms/
│   ├── period_grid.parquet         the 16,641 trial periods, once per run
│   └── <population>/
│       └── power_shard00007_of_00320.parquet     one Delta-chi^2 curve per system
├── characterization_<population>[_high_snr].parquet    optional, `finish.py --stages merge`
└── failed/                         per-system exceptions, if any
```

`characterization/<population>/` is a pyarrow dataset — point
`ds.dataset(path, format="parquet")` at the directory and it reads as one table.
`calibrate.load_characterization` does that, applies the calibrated flags, and
takes a `columns=` list, which is worth passing: the full table is ~3 GB in
pandas and each paper figure needs under a dozen columns.

### The characterization columns

| | |
| --- | --- |
| `shard`, `shard_row` | the system's address — with these two, `ShardReader` finds its epochs |
| `gaia_source_id`, `system_id`, `n_epochs` | identity |
| `klass` | `undetected` / `broad` / `multimodal` / `unimodal` |
| `top_power`, `best_period`, `width_dex`, `best_at_edge`, `n_competitive` | the periodogram summary |
| `peak1_period`, `peak1_power`, `peak2_period`, `peak2_power` | the two tallest peaks |
| `delta_bic_best`, `chi2_5par` | the peak against the five-parameter model |
| `accel_delta_chi2`, `accel_delta_bic` | the second, independent detection channel |
| `detected`, `period_reliable` | data-only flags on the classifier's internal threshold |
| `kepmodel_fap`, `kepmodel_excess_noise_mas` | `kepmodel`'s own bookkeeping |
| `period_k_recovered`, `period_k_in_bound` | truth: point estimate, and whether the competitive region brackets the truth |
| the truth row | `period_k`, `sma_k`, `mass_pl_k`, `ecc_k`, `inc_k`, `alpha_mas_k`, `snr_total_k`, `snr_single_k`, `snr_eff_k`, `Omega_k`, `omega_k`, `M_anom_k`, `n_transits_dr4`, `parallax_mas`, `pmra_mas_yr`, `pmdec_mas_yr`, `mass_st_msun`, `radius_st_rsun`, `sigma_single_mas`, `n_planets`, `coplanar`, `P_ratio`, `near_resonance` |
| analysis aliases | `a_k_au`, `e_k`, `Mp_k_msun`, `i_k_rad`, `alpha_k_mas` |

The aliases exist because `epochalypse_figures` reads `Mp_k_msun` and `i_k_rad`
while the generator writes Jupiter masses and degrees. Both unit conversions
happen in `writers.to_analysis_schema` and nowhere else — the same mapping
`sharded_systems.to_analysis_schema` writes down, so a figure cell is identical
whether it is fed a 10,000-system subset or this table. The originals are kept
alongside rather than renamed.

The null-calibrated flags — `detected_cal`, `period_reliable_cal`,
`peak_significant_cal`, `accel_significant_cal` — are **not** stored. They are
four vectorized comparisons against `calibration.json`, applied at read time by
`calibrate.apply_calibration`, and baking them in would mean rewriting the whole
table whenever `TARGET_FP` changed.

## The noise model (`FIT_JITTER`)

Defaulted to `False`: fixed `1/σ_formal²` weights, a like-for-like swap of the
period search alone, and what the existing figures use.

"What does kepmodel default to?" points both ways, so it is worth being precise.
The **library** default is not to fit it — `AstroModel(..., excess_noise=
term.Jitter(0))` puts the term in the covariance but leaves it out of
`fit_param`, so it stays pinned at zero unless a caller adds it (verified
against the installed package: a fresh model's `fit_param` contains only the
linear terms). The **Gaia/OHP tutorial** does fit it, in a step of its own
before the periodogram:

```python
model.fit_param += ['cov.excess_noise.sig']
model.fit()
```

So `False` matches the library and `True` matches the tutorial. They are not a
default-and-refinement pair; they are different noise models:

- **For `True`:** the catalog's noise model injects scatter at the σ_UEVA scale
  but reports the smaller σ_formal, so there is genuine excess the fixed weights
  do not capture, and the periodogram over-fits it. The jitter term absorbs it
  directly instead of leaving it to the empirical null calibration.
- **Against `True`:** the term is fitted with *no orbit in the model*, so on a
  strongly-signalled system it absorbs the companion along with the excess
  scatter. `chi2_base` — and with it the whole Δχ² scale — collapses by one to
  two orders of magnitude, and detections are suppressed along with false
  positives.

Thresholds are recalibrated on the matched control either way, so both runs are
internally self-consistent. Their `top_power` columns are on different scales
and must never be compared numerically. Running both is a straight 2×: the
frequency loop dominates and nothing is shared between them.

## Calibration

The periodogram Δχ² is maximized over 16,641 trial periods, so under the null it
is inflated by the look-elsewhere effect and is not χ²₄-distributed. It is
further inflated because the catalog's noise model injects scatter at the
σ_UEVA scale but reports the smaller σ_formal, and the search uses fixed
1/σ_formal² weights, so it over-fits that excess. Neither has a closed form worth
trusting at this grid size, so the thresholds are measured on `0_companion` at a
1% false-positive rate, split evenly between the two independent channels.

Two things make that stronger here than in the notebook it comes from. The
control is the *same 5.7 M stars* as the two companion populations, so it is a
matched null — every star's parallax, transit count and per-epoch precision
appears in both. And the quantile is measured on 5.7 M systems rather than
10,000, so the 99.5th percentile is set by ~29,000 systems in the tail rather
than ~50. The threshold stops being the noisy quantity it is at notebook scale.

The thresholds MUST be recalibrated for any change in the grid, the bounds, or
the noise model, and `top_power` columns from a `FIT_JITTER = True` run are on a
different scale entirely and must not be compared numerically with these.

## Drawing the three figures

Everything the population maps need is in the characterization table:

```python
from epochalypse.periodogram.calibrate import load_characterization

# "one companion: random" -- the whole 5.7 M, flags already applied
df = load_characterization("1_companion",
                           columns=["period_1", "Mp_1_msun", "a_1_au", "e_1",
                                    "mass_st_msun", "parallax_mas", "snr_total_1",
                                    "alpha_1_mas", "sigma_single_mas", "n_transits_dr4"])

# "one companion: high-SNR" -- the same table, one boolean mask
hi = load_characterization("1_companion", columns=[...], high_snr=True)
```

Then the existing figure stack takes over unchanged — `epochalypse_figures.joint`,
`cf.SCHEME_3`, `cf.apply_calibration` — because the column names are the ones it
already reads. The one thing to reconsider at 5.7 M rows is the scatter: a
rasterized 5.7 M-point panel is a large PDF and mostly overplotted ink, so the
drawn panel probably wants a subsample of a few thousand, with the full table
behind every number in the caption. That is an analysis-side choice now -- the
write path stores every system, so pick the rule where the figure is made.

For the example-periodogram figure, `epochalypse_figures.examples` takes a
`periodogram(t, psi, pf, y, yerr) -> (periods, power)` callable. Recompute them -- nine curves, three seconds:

```python
from epochalypse.periodogram.grid import frequency_segments
from epochalypse.periodogram.periodogram import kepmodel_power

periods, power, _ = kepmodel_power(t, psi, pf, y, yerr,
                                   segments=frequency_segments())
```

(A random-access reader over the stored `power` shards was written but never
called, so it went with the merge. `power_shard()` parquet plus the shared
`period_grid.parquet` is what a reader would open.)

## Status

Verified against the delivered 500 pc catalog: `tests/test_periodograms.py
--catalog-root <catalog>` runs one work unit end to end and checks 33
invariants, including that a 4-way split partitions a shard exactly, that
`(shard, shard_row)` addresses the right star, and that a stored curve's peak
and argmax reproduce the `top_power` and `best_period` written beside it.

**Two capabilities were lost when the old serial analysis module was deleted**,
and neither has a successor in the kepmodel path:

* **No conditional second-period search.** `double_periodogram` used to run a
  two-pass CLEAN: find P1, then re-scan P2 with an orbit fixed at P1 in the
  design, masking P2 within `exclude_dex` of P1 as degenerate, giving a second
  power spectrum that could be calibrated for a second-planet false-positive
  rate. `characterize_system` instead reports the two tallest peaks of a
  *single* 1-planet periodogram as `peak1` / `peak2`. For two-companion systems
  the second-tallest peak is frequently an alias or harmonic of P1 rather than
  the second companion, so `period_2_recovered` should be read with that in
  mind. kepmodel supports multi-Keplerian fits, so rebuilding this is wiring
  rather than new maths.
* **No eccentricity from the periodogram.** `refine_orbit` used to emit
  `best_period_ecc` and `e_ecc`; there is no eccentric refinement now.

Both are recoverable from git (`src/epochalypse/fitting.py`, deleted).

Not done here: the figures themselves. `epochalypse_figures` still expects the
`_agnostic` / `_detectable` population keys and a `row_index` column, so
whichever notebook draws these three panels needs a small key mapping —
`1_companion` → "one companion: random", `1_companion` + `high_snr=True` → "one
companion: high-SNR" — the way
`characterize_populations_kepmodel_500pc.ipynb` already overrides `cf.SPECS`.
