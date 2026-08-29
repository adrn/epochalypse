# epochalypse — harv posteriors

Posterior inference on every simulated system: harv's importance sampler against
a single, catalog-wide prior library, keeping the 1,024 highest-weight samples
per system.

Where the periodogram stage answers *is there a signal, and at what period*,
this one answers *what are the orbital parameters, and how well are they
determined*. It runs on all three populations, control included, so the answer
to the second question has a null distribution to be read against.

**The output samples are weighted.** `weight` is normalized over the whole prior
library, not over the 1,024 stored rows, so it sums to `weight_captured` rather
than to 1. Any average over these draws that ignores it is wrong. This is a harv
sharp bit and it survives into the parquet unchanged.

## What one run of the sampler does

The prior library is drawn once and reused for every system. For one system,
harv evaluates the linear-marginalized log-likelihood of all `N_PRIOR_SAMPLES`
draws, and this stage keeps the `TOP_K` with the largest importance weight,
their weights, and two diagnostics.

Rejection sampling would return however many draws survive — around 1,000 for a
system with no signal and **one** for a well-detected one (harv's own Gaia BH3
study: 10⁷ prior samples in, "Accepted 1 posterior samples" out). Top-K by
weight instead returns 1,024 always. Three things follow: the output table is
uniform, the sample count stops silently encoding a diagnostic, and — because
`jax.lax.top_k` has a static output shape — the recompile-per-acceptance-count
problem goes away.

What it does **not** do is create information the library lacks. A detected
system with `ess = 1.5` still has an unresolved posterior; you now read that off
a column instead of inferring it from an array length.

## Two diagnostics, because they answer different questions

| column | question | reading |
| --- | --- | --- |
| `ess` | did the library resolve this posterior? | `< 10` (`ESS_RESOLVED`) means the returned draws localize the best orbit but do not measure its uncertainty |
| `weight_captured` | was `TOP_K` big enough? | `~1.0` means nothing was truncated; `0.1` means 90% of the posterior mass was thrown away |

A system can pass one and fail the other, which is why both are stored. In
practice `weight_captured` is ~1.0 nearly everywhere and `ess` is the binding
constraint.

**Low ESS on a strong detection is arithmetic, not a bug.** At SNR ≈ 360 the
Δχ² between the best prior draw and the second-best runs into the hundreds, so
`exp(−Δχ²/2)` annihilates every other sample regardless of how the prior is
written. Measured: neither tightening the parallax prior across four orders of
magnitude nor reparameterizing moved the ESS off 1.0. The levers that do work
are raising `N_PRIOR_SAMPLES` (ESS is linear in it), a periodogram-informed
importance proposal, or an MCMC second pass seeded from these samples. The
`ess` column is what sizes that pass.

## Setup

Nothing beyond `uv sync` — harv is a pinned git dependency in `pyproject.toml`.

```bash
uv run python tests/test_harv.py                          # synthetic, no catalog
uv run python tests/test_harv.py --catalog-root <path>    # + one real work unit
```

## Running it

```bash
# 1. the fits -- the expensive part, and the only one that needs a cluster
mpirun python scripts/harv_mpi.py --catalog-root $OUT_ROOT --output-root $HARV_ROOT

# 2. the census, and the merged per-system tables
python scripts/harv_finish.py --stages census merge --output-root $HARV_ROOT
```

Locally, `harv_mpi.py` falls back to a single rank when mpi4py is absent:

```bash
python scripts/harv_mpi.py --subsample 200 --max-units 3   # one process
mpirun -n 8 python scripts/harv_mpi.py --subsample 2000    # 8 local processes
```

## Subsampling

`config.SUBSAMPLE` (or `--subsample N`) fits approximately `N` systems per
population instead of all 5.7 M. `None` is the full catalog.

It is converted to a **per-unit** cap rather than a global one, so no rank needs
to know what any other rank is doing and a subsampled run is still just "each
rank does its own units". With 320 shards and `--n-parts 3` a budget of 20,000
becomes 21 systems per unit.

The systems taken are the first of each unit, which is a contiguous block of the
source list — so a subsample is ~960 scattered patches of sky, not one. Companion
draws are seeded per source id and independent of position, so the *orbits* are
an unbiased sample; the sky coverage is not. For a question that depends on the
scan law, run the full catalog.

**Run a subsample first.** The cost model below is a laptop extrapolation, and
the full run is a five-figure core-hour commitment.

## Cost

> **Superseded.** The single-core numbers below were measured on a laptop and
> are wrong by 8–10× for this kernel — see `scripts/benchmarks/RESULTS.md` for
> the measured replacement (genoa, 96 ranks/node, M=10⁵, ~24,000 core-h). The
> laptop table is kept only because the *ratios within it* still explain why
> `BATCH_SIZE` and the epoch bucket matter. Do not size a job from it.

**A full node costs 1.34× per rank, not the 5–10× this section used to claim.**

Measured single-threaded, x64, on one M-series core:

| `N_PRIOR_SAMPLES` | batch | N | warm | µs/sample | per system |
| --- | --- | --- | --- | --- | --- |
| 10⁴ | 1,000 | 108 | 0.035 s | 3.50 | 0.035 s |
| 10⁵ | 10,000 | 108 | 0.263 s | 2.63 | 0.26 s |
| 10⁶ | 10,000 | 108 | 2.528 s | 2.53 | 2.53 s |
| 10⁶ | 10,000 | 320 (padded) | 8.32 s | 0.83 | **8.3 s** |
| 10⁶ | 1,000 | 320 (padded) | 8.61 s | 0.86 | 8.6 s |

**Cost scales with the padded epoch count, not the real one.** The 2.53 s row is
a system near the *low* end of the catalog; at the largest bucket it is 8.3 s.
The catalog mean therefore depends entirely on the DR4 transit distribution, and
the two proxies available locally disagree by 2.5× — scaling the committed
sample's DR3 counts gives a median bucket of 80 (mean ~2.5 s/system, ~12,000
core-h), while the twelve systems in the smoke catalog have a median of 220
(mean ~6.3 s/system, ~30,000 core-h). Twelve sources from one sky region is not
a distribution, so neither number is trustworthy.

### What a full node costs — the failure that measured it

A 32-node × 64-rank job at `BATCH_SIZE = 10⁴` ran for two hours and completed no
unit. `SAMPLES_FLUSH_EVERY = 200` puts the first output file at 200 systems, so
the absence of any `.tmp` file after 120 minutes proves the rate was **≥36
s/system** against a 6 s/system budget — at least 6× worse, and enough that a
5,963-system unit needs ≥60 h against a 24 h walltime. The job would have burned
~49,000 core-hours for zero output.

The cause is memory bandwidth, and it is why `BATCH_SIZE` is now 10³. The
`(batch, n_epochs, n_linear)` intermediate is streamed once per batch, 100 times
per system at M=10⁶:

| batch | intermediate at N=320 | per node at 64 ranks |
| --- | --- | --- |
| 10⁴ | 230 MB | 14.7 GB streamed 100×/system |
| 10³ | 23 MB | fits per-rank cache |

Single-threaded, 10³ is 3% *slower* (8.61 vs 8.32 s) — which is exactly why the
laptop benchmark chose wrong. One core with the whole cache and full bandwidth to
itself is not the condition the job runs in.

### So: measure before you commit

```bash
mpirun python scripts/harv_mpi.py --subsample 20000 \
    --catalog-root $OUT_ROOT --output-root $HARV_ROOT --n-parts 3
```

That completes units, writes files, and prints a per-system rate from rank 0
every 50 systems. Size nodes, `--n-parts`, and `N_PRIOR_SAMPLES` from *that*
number. Nothing in this section is a substitute for it.

If the measured rate is still too slow, `N_PRIOR_SAMPLES` is the direct
multiplier: 10⁵ is 10× cheaper for 10× less ESS. Given that ESS is ~1 for a
strong detection at either setting, that trade is worth considering for a first
pass — the point estimate is what survives, and it is good.

JIT is ~0.5–3 s per distinct bucket shape, nine of them, so a few seconds per
rank. Never the problem.

## Epoch padding

harv JITs per epoch count and the catalog spans 44–298, so without padding every
distinct count is a fresh compile — 17.2 million of them. Each system is padded
up to one of nine `EPOCH_BUCKETS`, which needs three things to be right, none
obvious:

1. **Padded rows carry a large finite uncertainty, not `inf`.** Infinity zeroes
   the χ² contribution as intended, but the Gaussian normalization's `log σ`
   term diverges with it and `logZ` comes back `-inf`. `PAD_ERR_MAS = 1e6`
   leaves the χ² contribution at ~10⁻¹² of a real epoch's.
2. **`t_ref` is passed explicitly**, from the real epochs. harv derives it as
   `mean(time)` otherwise, so padding would drag the model's time origin.
3. **The parameterization is built from the unpadded data**, because
   `from_data` sets `a_floor = med(σ_AL)/√N` and padded rows move both.

Padding adds the same constant to every prior sample's log-likelihood, so it
cancels exactly in the weights, the top-K selection, and the ESS. It does *not*
cancel in `logZ_int` and `max_log_likelihood`, which are absolute — those have
`adapt.pad_log_offset` subtracted at write time, without which two systems in
different buckets would differ by up to ~4,000 nats for no physical reason.
`tests/test_harv.py` asserts all of it: same samples, same weights, same ESS,
and a `logZ` shift equal to the prediction.

## The prior

One prior for the whole catalog — same library, same bounds, every system. That
is what makes 17.2 M posteriors comparable, and it is why `harv/config.py` holds
values rather than per-system choices.

**Thiele-Innes, not Campbell.** It marginalizes the four orientation parameters
as linear constants, leaving `period`, `eccentricity` and `phase_peri` sampled —
three nonlinear dimensions instead of six. Measured on a real SNR≈360 system:
period recovered to **0.14%** versus 2.63% for Campbell, a 19× gain. It does not
help ESS — it makes it slightly *worse*, because marginalizing the orientation
analytically finds the best orientation at every trial period and so sharpens
the marginal likelihood. Accuracy and ESS pull in opposite directions here.

It is built per system through `ThieleInnesGaiaAstrometry.from_data`, which sets
`a_floor` and enables the Jacobian correction. Without that correction a flat
prior on the Thiele-Innes constants is **not** the Campbell prior, and harv warns
that the marginal likelihood can then be dominated by spurious long-period
solutions where the orbit is absorbed into proper motion.

**The scale priors are deliberately uninformative** — `sigma_parallax = 100 mas`
against a catalog that maxes at 74, and not centered on the catalog's own
parallax and proper motion. Centering on them is measurably better (period error
2.63% → 0.2%), and it is also cheating: the catalog's values are the ones the
epochs were simulated from, so a prior centered there hands the sampler the
truth. Real DR3 astrometry for an astrometric binary is biased by the very orbit
being fitted. Thiele-Innes recovers that accuracy without using any truth value,
which is why it is the parameterization and this is the prior.

**The period prior is deliberately narrower than the injected one** — 0.01 to
100 yr, 4.0 decades, against the periodogram's full 5×10⁻⁵ to 3300 yr. DR4 is
5.5 yr with ~80 transits, so only roughly 0.1–10 yr is constrainable at all, and
every decade the prior covers costs sampling density in the decades that can be.
Measured worth about one order of magnitude of `N_PRIOR_SAMPLES` for period
accuracy. It also moves the no-orbit solution from 5×10⁻⁵ to 0.01 yr, where
`σ_a` is 34× larger and the null therefore carries a much bigger Occam penalty
of its own.

The cost is that a system injected outside the window cannot be recovered by
construction. `census.in_search_range` exists so that is reported separately
rather than counted as a failure, and **every recovery number is quoted over the
searched range only.**

### One library, and how that is checked

The library is drawn in process from `config.SEED` on every rank. There is no
cache file: with Thiele-Innes it is four columns — `period`, `eccentricity`,
`phase_peri`, `parallax` — which is 32 MB at 10⁶ and 0.3 s to draw. The other
eight linear parameters are Gaussian, so harv marginalizes them analytically and
draws them conditionally per system; they are not sampled from the prior at all.
`parallax` is `HalfNormal`, which is not Gaussian, so it survives as an
explicitly sampled column.

A file would add a stage, a barrier, and a thousand ranks reading one ceph path
at once, for what a fixed seed already gives. What replaces it is a check: every
rank reports a `blake2s` fingerprint of its drawn arrays, and the final gather
prints whether they agree. Two ranks measuring against two different priors is
the one failure that would silently invalidate every comparison in the output,
so it is asserted rather than assumed.

`tests/test_harv.py` also asserts that the prior does not depend on the
per-system `a_floor` — the property that makes one shared library legitimate.

## On a Slurm cluster

`scripts/mpi/6-harv.sh` and `7-harv-finish.sh`, sourcing `scripts/mpi/env.sh`
like every other stage:

```bash
export HARV_ROOT=$OUT_ROOT/harv     # ~850 GB of samples, ~2 GB of per-system rows
```

`--catalog-root $OUT_ROOT`: the catalog being fitted is what the generator
wrote. It is a separate flag only so you can point it at a catalog someone else
generated and delivered as a directory.

### `--n-parts` is the walltime knob

A **unit** is one shard of one population — 17,889 systems — and it is what one
rank processes end to end. That sets a floor on walltime: you cannot finish
faster than one unit, however many cores you have. At `s` seconds per system a
unit is `17,889 s / 3600` hours, so at 6 s/system it is ~30 h and at 36 s/system
it is ~180 h. **Neither fits a 24 h allocation, which is what `--n-parts` is
for.** It cuts each shard into contiguous chunks of systems — `--n-parts 3`
gives 2,880 units of 5,963 systems — and parts are cut on systems, not row
groups, so three parts really is three near-equal thirds.

Pick `--n-parts` so one unit fits the walltime with margin, using the rate from
the subsample run:

    n_parts  >=  17,889 x s_per_system / (0.7 x walltime_seconds)

Then ask for enough ranks that no rank gets more than one unit: 960 × `n_parts`
of them. If you cannot, `--skip-existing` plus a dependent follow-up job is the
fallback — but a rank that gets two units needs twice the walltime, and that is
how a job ends up finishing 2 of 3 units and discarding the third at 79%.

### Per-rank memory

**1.85 GB**, measured with `getrusage` at production settings (M=10⁶,
`TOP_K=1024`, the largest 320-epoch bucket):

| | peak RSS |
| --- | --- |
| imports | 0.22 GB |
| + prior library at M=10⁶ | 0.85 GB |
| + fitting, 320 bucket | **1.85 GB** |

Note the library costs ~0.6 GB of process footprint even though its arrays are
32 MB — JAX's allocator holds the sampling intermediates. It is not the peak
driver, since fitting goes higher regardless.

So 128 ranks per node is **237 GB of a 1 TB node** — capacity is not the
constraint, and it never was. Memory *bandwidth* is, which `BATCH_SIZE` controls
and the Cost section covers. Set ranks per node from the throughput the
subsample run measures, not from RAM.

`--skip-existing` means a requeue after a node failure only redoes the units that
died, so it costs nothing to leave on. Each rank writes `.parquet.tmp` and
renames on success, which is what makes that trustworthy.

## Output layout

```
$HARV_ROOT/
├── manifest.json                    the library, its fingerprint, the units, the padding note
├── systems/<population>/            one row per system   (~2 GB total)
├── samples/<population>/            TOP_K draws per system (~850 GB total)
├── harv_systems_<population>.parquet   written by `--stages merge`
└── failed/                          per-system exceptions, if any
```

**The per-system table** carries the diagnostics, three period point estimates,
and the injected truth columns joined on — the same column list the periodogram
stage uses, so the two can be merged on `gaia_source_id` without either being
the authority.

| column | meaning |
| --- | --- |
| `ess`, `weight_captured` | the two diagnostics above |
| `logZ_int`, `logZ_int_mcse` | the evidence integral, padding removed |
| `max_log_likelihood` | padding removed |
| `n_epochs`, `n_padded` | the real epoch count and its bucket |
| `period_best_yr` | the highest-weight draw — the point estimate, good to ~0.1% on a strong detection |
| `period_wmean_yr`, `period_wstd_yr` | weight-renormalized mean and spread |
| `seed`, `n_prior_samples`, `top_k`, `t_ref_yr` | what this row was produced under |

**The samples** are stored one row per system, each parameter a fixed-size list
of `TOP_K` float32. Not one row per sample: `gaia_source_id` and a sample index
repeated 1,024 times would add ~210 GB of pure bookkeeping, and parquet stores a
fixed-size list as one flat column with no per-row offsets, so a reader can
memory-map a slice. It also makes the join to the per-system table row-for-row.

float32 because 12 parameters × 1,024 × 17.2 M is ~850 GB at float32 and 1.7 TB
at float64, and the third digit of a posterior draw is not a measurement. Every
summary in the per-system table is computed on the float64 values *before* the
cast, so it is a storage choice and not a numerical one.

Units are in `manifest.json`, not in the schema: they are identical for all
17.2 M rows. They are hard-coded in `library.SAMPLE_UNITS` because the manifest
is written before the first system is fitted, and asserted against a real fit in
the test suite so they cannot drift.

**The samples are never merged.** Read them with `pyarrow.dataset` over
`config.samples_dir(population)`, which memory-maps row groups instead of
materializing 850 GB.

## Reading the diagnostics

`harv_finish.py --stages census recovery figures` produces a text census, a
binned recovery breakdown, and four PNGs in `$HARV_ROOT/figures/`. **Read them
in this order** — a single recovery percentage is close to uninterpretable on its
own, because three unrelated things limit it and they compound.

| figure | question it answers |
| --- | --- |
| `harv_recovery_map` | where in (period, eccentricity) does the method work? |
| `harv_period_aliases` | when it fails, *where* does the period land? |
| `harv_library` | did the library resolve these posteriors? |
| `harv_detection` | at what signal strength does recovery turn on? |

**1. `harv_recovery_map`** — recovery over a period × eccentricity grid, with
both marginals and per-bin counts and binomial errors. The period profile is the
shape of the **recoverable window**: DR4 is 5.5 yr with ~80 transits, so only
roughly 0.1–10 yr is constrainable, while the injected prior spans 7.8 decades.
Most of that prior is unrecoverable *by construction*, and no library size fixes
it. The eccentricity profile is **prior coverage**, which is fixable — see
`config.ECC_LOC`. If recovery is high in the sweet spot and flat in
eccentricity, what remains is library resolution.

**2. `harv_period_aliases`** — `P_best` against `P_true`, with the ±tolerance
band and the alias tracks drawn on: the annual parallax term puts aliases at
`1/P_best = 1/P_true ± 1 yr⁻¹`, and the 2× / 0.5× harmonics compete too. With
`ess ≈ 1` the reported period is a *single* prior draw, so misses **on** those
tracks are a resolution problem that more samples fix, while misses scattered
flat mean the data does not constrain those systems.

**Check the railed fraction before reading this figure.** On the first
production run, aliasing was the *minority* failure: 65% of misses were the
sampler collapsing to the no-orbit solution at the prior floor, not picking a
wrong period. `--stages recovery` now splits the two, and `harv_detection`'s
third panel plots the rail rate against SNR.

**3. `harv_library`** — the ESS and `weight_captured` distributions, and
recovery against ESS. That third panel runs **backwards** and is the one most
likely to be misread: a well-constrained period gives a sharp posterior, which a
fixed-size library resolves with *fewer* effective samples. Low ESS where
recovery is high is correct. ESS says "the library did not sample this
posterior", never "the answer is wrong".

**4. `harv_detection`** — `logZ_int` per population, with `0_companion` as the
null distribution, and the completeness curve against injected SNR. The control
is the same one the periodogram stage calibrates its thresholds on.

### What a healthy run looks like

- `weight_captured` piled at 1.0, `wcap low` at 0% — `TOP_K` is ample.
- `0_companion` with the **highest** median ESS. No signal means the broadest
  posterior; if the control is not highest, something is wrong with the
  likelihood.
- Recovery peaking in the 0.1–10 yr band and falling off both sides.
- `logZ_int` for the companion populations shifted above the control.

### Measured, at M = 10⁶ on 20,160 systems per population

Recovery was 32.2% on the high-SNR `1_companion` subset. Binning it showed the
shape of the recoverable window (0% below 0.01 yr, 53.7% at 1–3.2 yr, 7.7% above
10 yr) and a monotonic decline with eccentricity. Neither is a bug: at that M the
library is under-resolved by ~2,000 nats even for a strongly detected system,
and the fit converges to the optimum by M = 10⁷. The two levers are
`N_PRIOR_SAMPLES` and the *width* of the period prior — narrowing it from 7.8 to
4.0 decades is worth roughly one order of magnitude of M for period accuracy, at
no cost. See `scripts/benchmarks/RESULTS.md` for the numbers.

## Reading the output

```python
import numpy as np, pandas as pd

sys = pd.read_parquet("harv_systems_1_companion.parquet")
resolved = sys[sys.ess >= 10]                      # posteriors, not localizations
escalate = sys[(sys.ess < 10) & (sys.snr_total_1 >= 5)]   # the MCMC second pass

smp = pd.read_parquet("samples/1_companion/samples_shard00007_of_00320.parquet")
row = smp.iloc[0]
period, w = np.asarray(row["period"]), np.asarray(row["weight"])
mean = (w * period).sum() / w.sum()   # renormalize: w sums to weight_captured
```

## Status

The pipeline runs end to end and `tests/test_harv.py` covers the four things it
rests on — padding is a no-op, the library is shareable, selection is
deterministic given the seed, and the stored weights survive the float32 cast.

Two numbers still want a real measurement rather than an extrapolation: the
per-system rate on a Rome core, and the JAX compile-cache size per rank. Both
come out of one `--subsample` run, and both change the allocation in
`6-harv.sh`. Run it before committing the full job.

The MCMC second pass does not exist. `ess` is what would size it.
