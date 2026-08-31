# What SNR means in this catalog

`snr_total` is not a detectability measure. This document is what it is, what it
is not, how far off it is, and what to use instead.

It matters beyond bookkeeping: every harv recovery figure was binned on it, so
the low-recovery cells at long period were partly an artefact of the x-axis.

## The mechanism, before any number

A Gaia five-parameter astrometric solution fits position, proper motion and
parallax. Those five are **free parameters** in every downstream fit, so
whatever part of a companion's signal they can reproduce is subtracted along
with them and can never be detected — no matter how large `alpha` is.

An orbit whose period is comparable to the mission span is mostly a straight
line plus a gentle curve across the data. **A straight line is exactly what
proper motion is.** So the longer the period, the more of the orbit the
astrometric solution eats, and past about twice the baseline almost all of it
goes.

`planets.draw_companions` already charges for this:

```python
a_crit   = (BASELINE_YEARS**2 * mstar) ** (1/3)     # sma whose period is T
snr_eff  = snr_single / (1 + (sma / a_crit)**3)      # == 1 + (P/T)^2
snr_total = sqrt(n_transits) * snr_eff
```

The scaling is right. The size of the charge is not.

## The three quantities

| quantity | definition | knows orientation? | answers |
| --- | --- | --- | --- |
| `snr_total` | `sqrt(N)·alpha/sigma_single / (1 + (P/T)²)` | no | **plausibly** observable |
| `snr_detectable` | `sqrt(N)·rms(reflex ⊥ astrometric basis)/sigma_single` | **yes**, exactly | observable **for this system** |
| `snr_expected` | `snr_detectable` marginalized over `(i, omega, Omega, M)` | no | observable **in expectation** |

`snr_detectable` is the right axis for a **method** question — given a signal
this strong is genuinely present, does the fit find it? It is the **wrong**
input to an occurrence-rate correction, because it conditions on the true
inclination and phase, which no real survey knows. That is what `snr_expected`
is for.

All three divide by `sigma_single`, the scale the noise was actually injected
at, so they are directly comparable. (The catalog *reports* `sigma_formal`,
which is smaller on purpose — see `HARV.md`, "Reading the weights".)

## How far off `snr_total` is

Measured over 3,000 high-SNR `1_companion` systems of the real catalog:

| | |
| --- | --- |
| median `retained` | **57–71%** of the orbit survives the astrometric fit |
| systems with `retained < 25%` | **18.6%** of the nominally high-SNR sample |
| worst case measured | gaia 1337286106820620928 — `retained` **5.4%**, `snr_total` 21.5 against a detectable **1.85**, an 11.6× overstatement |

So `HIGH_SNR_MIN = 5` admits systems whose detectable SNR is under 1, and the
typical system is overstated by ~1.7×.

### The period scaling is right; the normalization is not

`check_snr.py --calibrate`, exact projection against the formula, both
referenced to `alpha`:

| P/T | `snr_eff` formula | exact geometry | ratio |
| --- | --- | --- | --- |
| 0.2 | 0.962 | 0.543 | **1.77** |
| 0.5 | 0.800 | 0.521 | 1.54 |
| 1.0 | 0.500 | 0.457 | 1.10 |
| 2.0 | 0.200 | 0.176 | 1.14 |
| 5.0 | 0.038 | 0.037 | 1.05 |
| 10.0 | 0.010 | 0.007 | 1.40 |

**Do not change the exponent.** `(P/T)²` agrees with the exact projection to
10–30% across two decades. What the formula omits is the **along-scan
projection**: a 2-D orbit of semi-axis `alpha`, averaged over orientation and a
rotating scan direction, delivers ~0.53·`alpha` of along-scan rms, not `alpha`.
That is period-independent and biases the whole catalog by ~1.8×.

### And no better formula exists

`snr_eff` has no eccentricity term. Adding one would not fix it, because the
**spread**, not the median, is what defeats a closed form — at `P/T = 2`:

| e | geometry (median) | 16–84% spread |
| --- | --- | --- |
| 0.0 | 0.181 | 1.9× |
| 0.3 | 0.169 | 2.3× |
| 0.6 | 0.120 | 4.0× |
| 0.8 | 0.093 | **6.4×** |

The variable that dominates at long period is **where periastron falls relative
to the observing window**, which no function of `(P, a)` can see. The fix is an
exact per-system projection, not a better heuristic.

## The catalog itself is correct

This investigation began as a suspected generator bug. It is not one.

- Fitting the **exact known injected reflex** as a free amplitude alongside the
  astrometric basis returns 1 within its own uncertainty for **every one of
  3,000 systems**. (An earlier version flagged ~80 "bugs" by judging that
  amplitude against a fixed window instead of against `sigma_amp` — a weakly
  retained orbit constrains its own amplitude weakly.)
- Predicted chi-square tracks measured to **2–6%** across 1.5 decades of SNR,
  once the no-signal floor is set to `r² = (sigma_single/sigma_reported)²` and
  not to 1.
- `check_snr.py --self-test` reconstructs the reflex by calling
  `simulate_along_scan` twice and differencing, so a convention mismatch cannot
  hide: it recovers amplitude 0.998.

## What changed in the code

**Selection did not change.** `sources.select_high_snr`, `census.high_snr_mask`
and `harv_mpi --min-snr` all still cut on `snr_total`. A cheap a-priori proxy is
the right thing to decide where to spend compute, being generous is safer than
being restrictive, and changing it would change *which systems exist* in the
analysis — a much larger decision than an axis.

**Binning did.** Every harv figure now bins on `snr_detectable`, falling back to
`snr_total` when `project_snr_mpi.py` has not run and **saying so in the axis
label**. `harv_detection` carries both completeness curves — one against
`snr_detectable` (method) and one against `snr_expected` (survey) — and the gap
between them is this whole finding, drawn once.

`figures.SNR_LABELS` is the authority on how each is named; no figure may label
one of them generically as "SNR".

## Reproducing every number here

```bash
# the formula against exact geometry, and the eccentricity spread; no catalog
python scripts/diagnostics/check_snr.py --calibrate

# the catalog audit: retained, detectable, measured, per SNR bin
python scripts/diagnostics/check_snr.py --catalog-root $OUT_ROOT --sample 3000

# one system, in full
python scripts/diagnostics/check_snr.py --catalog-root $OUT_ROOT --ids <source_id>

# is E[retained] a property of the orbit alone?  (decides the table below)
python scripts/diagnostics/check_snr.py --catalog-root $OUT_ROOT --across-stars

# write snr_detectable / snr_expected for the whole catalog
mpirun python scripts/project_snr_mpi.py --catalog-root $OUT_ROOT
```

**`snr_expected` is tabulated, not drawn per system.** Measured on the real
catalog, 200 stars × 60 orientations drawn across shards (epoch counts 39–206),
the star-to-star spread of `E[retained]` is **3.6–8.7%** across the
`(P/T, e)` grid. `E[retained]` is therefore a
property of the *orbit*, one `(log10 P/T, e)` table serves the catalog, and
`snr_expected` costs ~nothing instead of ~1,900 core-hours.

Three cautions on that measurement, all learned the hard way.

It must use **common random numbers** — one orientation set reused for every
star — or sampling noise lands in the between-star spread and reads as a real
effect. At `n_trials = 12` that artefact alone was 49%, halving to 25% at 40
with nothing real changing.

**Read `between` in absolute terms, not against `within`.** Because the shared
Monte Carlo error has already cancelled out of the between-star comparison,
`between` sits *below* `within` in several cells — that is the design working.
`within` only shows that each star's own median is resolved; it is not a
threshold to clear. A few percent is negligible against the factor of 2–6
orientation spread the quantity summarizes.

**Sample stars across shards, not from one.** A shard is a contiguous region of
sky, and both things retention depends on — transit count and scan-angle
distribution — vary with ecliptic latitude. The first version drew from one
patch, covering 68–87 epochs of a catalog range of 44–298, and reported a
between-star spread of 1.4–4.6%. Sampling across shards widened the coverage to
39–206 and the spread to **3.6–8.7%** — so the narrow sample was understating
it by about a factor of two. Still a few percent, so the table stands, but the
residual is real: treat `snr_expected` as carrying a ~9% systematic, which is
small against the factor of 2–6 orientation spread it summarizes and should be
stated rather than ignored.

### Eccentricity is not monotonic, and changes sign with period

From the same run:

| P/T | E[ret] at e = 0.1 | E[ret] at e = 0.6 |
| --- | --- | --- |
| 2.0 | 0.291 | **0.324** |
| 5.0 | 0.054 | **0.024** |

At `P/T = 2` eccentricity *helps*: the fast periastron passage is the one part
of a partly-covered orbit the astrometric basis cannot mimic, while a circular
orbit over the same window is a smoother curve and more absorbable. By
`P/T = 5` the orbit spends nearly all the window near apastron, barely moving,
and eccentricity hurts instead.

This is a further argument that no closed form will do. A correction term in `e`
would need to change sign with period, on top of the factor-of-6 orientation
spread it still could not capture.

**Out of scope:** the periodogram stage cuts on `HIGH_SNR_MIN` too and would
benefit from the same treatment. Nothing there has been changed.

---

# Note for the lead author

*Self-contained; no familiarity with the analysis code assumed.*

**The problem.** `snr_total` as recorded in the catalog answers "could this
companion plausibly be seen?", not "can it be seen?". The gap is not small.
Because a Gaia five-parameter solution fits position, proper motion and parallax
as free parameters, it absorbs whatever part of an orbit resembles a straight
line plus a slow curve — which, for a period comparable to the mission baseline,
is most of it. Measured by exact projection over 3,000 high-SNR systems, the
median orbit keeps ~60% of its amplitude, **18.6% keep under 25%**, and the worst
case we looked at keeps **5.4%**: a recorded `snr_total` of 21.5 is a detectable
1.85. A system that the catalog calls a solid detection can be, in the data,
indistinguishable from noise.

`snr_eff`'s `1/(1 + (P/T)²)` penalty has the **right period scaling** — it
matches exact projection to 10–30% from `P/T = 0.2` to 10 — so nothing needs
re-deriving. What it omits is the along-scan projection: a 2-D orbit of
semi-major axis `alpha`, averaged over orientation and a rotating scan
direction, delivers about 0.53·`alpha` of along-scan rms. That is a
period-independent **~1.8× optimism affecting the whole catalog**, so a cut at
`SNR_tot >= 5` is really a cut near 3.

**Why a better formula will not fix it.** `snr_eff` has no eccentricity term,
and adding one does not help, because the problem is the spread rather than the
mean: at `P/T = 2` the true retained fraction varies by a factor of 1.9 at
`e = 0` and **6.4 at `e = 0.8`**, depending on where periastron happens to fall
relative to the observing window. No function of period and semi-major axis can
see that. It has to be projected per system, which costs one small
least-squares each and needs no re-simulation.

**The suggestion.** Report *two* SNRs, because they answer questions the paper
asks separately:

- **`SNR_det`** — the exact projection for a given system. Correct for
  characterization and for any statement about method performance ("given a
  signal this strong, we recover the period X% of the time").
- **`SNR_exp`** — the same quantity marginalized over inclination, node,
  argument of periastron and phase. **This is the one occurrence rates need.**
  `SNR_det` conditions on the true orientation, which no real survey knows, so a
  selection function built on it is not applicable to real targets. `SNR_exp`
  asks "for a planet with this period, eccentricity and mass around this star,
  how detectable is it *on average over the geometries we cannot observe*" —
  which is exactly what a completeness correction integrates.

We would keep `snr_total` as the catalog's selection proxy and simply document
its bias. Nothing about the simulated data changes.

**A refinement worth considering.** For occurrence rates the honest object is
arguably not an SNR at all but a **detection probability**: the fraction of
orientations for which a system would have been found. We have the machinery for
that already — it is the same projection evaluated over a set of random
orientations — and its width is real information. At `e = 0.8` a system can be
comfortably detectable or entirely invisible depending on geometry, and a single
expected SNR reports neither.

**Cost.** `SNR_det` for the full 17.2 M-system catalog is ~95 core-hours, a small
fraction of what the orbit fitting already costs, and needs no re-simulation
since it reads the existing epochs. `SNR_exp` is nearly free on top: we measured
that its orientation average depends on the orbit rather than on the star
(star-to-star spread 0.8–6.5%), so one interpolation table serves the catalog
instead of per-system sampling. Both are implemented and tested.

**The catalog is not in question.** We checked hard, because this first looked
like a generator bug. Fitting the exact known injected signal as a free
amplitude returns 1 within its uncertainty for all 3,000 systems tested, and
predicted scatter matches measured to 2–6% across 1.5 decades of SNR. The
simulation is doing what it says; only the summary statistic was optimistic.

**This is a suggestion, not a decision.** Happy to write it up either way, or to
drop it if the occurrence-rate section does not need the distinction — the
numbers above are the whole of the case for it.
