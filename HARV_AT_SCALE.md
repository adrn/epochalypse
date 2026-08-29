# Running harv at scale — takeaways from a 17-million-system catalog

Notes for a harv case study, from building and running a population pipeline on
the epochalypse simulated Gaia DR4 catalog: 5.7 M stars × 3 populations, 44–298
along-scan epochs each, one shared prior library, one `run_with_samples` call
per system. Everything below is measured on that run unless flagged as an
estimate.

The headline: **almost every number we guessed was wrong, and the ones that
mattered were wrong by an order of magnitude.** The guidance worth giving is
less "use these settings" than "measure these five things before you commit an
allocation."

---

## 1. Do not benchmark this on a laptop

An M4 Max performance core is **6.3× a genoa (Zen 4) core** on harv's kernel,
measured at matched epoch counts. Not 1.5×, not 2×. Clock and core width
plausibly account for 2–3.5× of that and the rest is unexplained — we ruled out
XLA CPU threading (setting `--xla_cpu_multi_thread_eigen=false` and
`OMP_NUM_THREADS=1` changes the laptop time by 2%) and confirmed identical
jax/jaxlib versions on both sides.

Every cost estimate we made before running on the cluster was derived from a
laptop and was wrong by ~8×. Two production jobs were sized from those numbers;
the second burned ~4,000 core-hours and produced no output at all.

**Guidance:** benchmark on the target hardware, at the target library size, with
the target ranks-per-node. A 20-minute job answers what a week of reasoning
cannot.

## 2. Cost is linear in the *padded* epoch count, and that changes everything

harv JITs per epoch count. A catalog spanning 44–298 epochs would compile 17.2 M
times, so we pad each system up to one of nine bucket sizes. That works, and the
padding is provably neutral — same samples, same weights, same ESS — provided:

- padded rows get a **large finite** uncertainty, not `inf`. Infinity zeroes the
  χ² contribution as intended, but the Gaussian normalization's `log σ` term
  diverges and `logZ` returns `-inf`.
- `t_ref` is passed **explicitly**, computed from the real epochs. harv derives
  it as `mean(time)` otherwise, so padding drags the model's time origin.
- the parameterization is built from the **unpadded** data, because
  `ThieleInnesGaiaAstrometry.from_data` sets `a_floor = med(σ_AL)/√N` and padded
  rows move both the median and N.

The consequence people will miss: **cost now scales with the bucket, not with
the system.** Our first benchmark used a 108-epoch system and reported
2.5 s/system; the catalog's mean bucket made the real figure 25 s. Padding is
right, but it makes any single-system timing unrepresentative unless you know
where that system sits in the bucket distribution.

## 3. `batch_size`: benchmark it under contention, not alone

harv's docs suggest `batch_size = n_prior_samples`; that is GPU advice. On CPU
the `(batch, n_epochs, n_linear)` intermediate is streamed once per batch, so
the working set is what matters:

| batch | intermediate at N=320 | single core | 64 ranks/node |
| --- | --- | --- | --- |
| 10⁴ | 230 MB | fastest by 3% | ≥6× slower than budget |
| 10³ | 23 MB | 3% slower | fits per-rank cache |

Measured single-threaded, 10⁴ wins. Measured on a full node, 10³ is **21%
faster** — and the node running 10⁴ was so bandwidth-starved it completed no
work unit in two hours. **A single-core benchmark gets the sign wrong.**

## 4. ESS is not a quality metric, and it runs backwards

For a strongly detected system, ESS ≈ 1 is arithmetic, not failure: the Δχ²
between the best prior draw and the second-best runs into the hundreds, so
`exp(−Δχ²/2)` annihilates everything else regardless of how the prior is
written. We confirmed that neither tightening the parallax prior across four
orders of magnitude nor reparameterizing moves it.

Worse for intuition: **recovery falls as ESS rises.** A well-constrained period
gives a sharp posterior, which a fixed-size library resolves with *fewer*
effective samples. Our best-recovering period bin has median ESS 1.7; the worst
has 7.7.

**Guidance:** report `ess` (did the library sample this posterior) and
`weight_captured` (was K big enough) as separate diagnostics, and do not let
either be read as "is this answer good". In our run `weight_captured` was 1.0
essentially everywhere, so truncation to K=1024 was never the binding
constraint; ESS was.

## 5. The prior on the linear amplitude sets the detection threshold

This is the result we would most want in a case study, because it is invisible
until you look for it and it silently governs what the pipeline can detect.

`sigma_a0` is the width of the Gaussian prior on the orbit's astrometric
amplitude. Because harv scales it as `(P/P₀)^(2/3) × ϖ`, the Occam penalty it
imposes grows with period — so it falls on real orbits and barely at all on the
no-orbit solution. Set it too wide and the evidence prefers "no companion" for
genuinely detectable systems.

We had it ~4,900× too wide. The symptom was not a warning but a **collapse to
the shortest period in the prior**, where the amplitude is forced to zero and
the model reduces to a five-parameter astrometric fit. That accounted for **65%
of all recovery failures**, and we initially misdiagnosed it as aliasing.

Rail fraction against injected SNR, high-SNR systems, before the fix:

| SNR | railed | recovered |
| --- | --- | --- |
| 5–10 | 50.2% | 5.3% |
| 10–20 | 20.4% | 34.5% |
| 20–40 | 2.4% | 68.9% |
| >80 | 0.0% | 86% |

The effective detection threshold sat near SNR 15–20 while our selection cut was
at 5. A sweep over `sigma_a0` at fixed library size moved overall recovery
36.1% → 44.9%.

**Guidance:** express this prior as the physical quantity it is — the largest
companion you expect. At the reference period a companion of mass *m* around a
star of mass *M* displaces the photocentre by `a₀ ≈ m / M^(2/3)`, so one
companion-mass ceiling gives the right scale for every star. It is free to
compute per system, because `sigma_a0` shapes only the analytically marginalized
priors: it never enters the shared library and triggers no extra JIT compile.

And do **not** pick it by maximizing recovery against known truth — that is
fitting to the answer. Use the sweep as evidence that a physically-motivated
value is in the right regime, not as the source of the number. Ours agreed:
13 M_Jup at the catalog's median host mass gives 0.022 AU, between the sweep's
two best arms.

**`log_uniform_in_a=True` is not the lever.** It sets `m = 4` in
`−m·ln(a₀ + a_floor)`, which rewards *small* a₀ more strongly and makes the
collapse worse.

## 6. Report failures in categories, not as one number

A single "recovery rate" conflates three unrelated things, and we spent a week
chasing the wrong one:

- **no detection** (collapsed to the prior floor) — a prior problem
- **outside the searched range** — impossible by construction, and not a failure
- **wrong period** — the only real failure

Splitting them turned an opaque 31.6% into a diagnosis. Bin recovery by injected
period *and* by SNR: the period profile shows the recoverable window (for a
5.5-year baseline, roughly 0.1–10 yr), and the SNR profile shows where the
prior's threshold sits.

## 7. Narrow the period prior to what the data can constrain

The injected prior spanned 7.8 decades; a 5.5-year, ~80-epoch baseline can
constrain about two. Every decade the prior covers costs sampling density in the
decades that matter. Narrowing 7.8 → 4.0 decades was worth **about one order of
magnitude of `n_prior_samples`** for period accuracy, and only 0.6% of high-SNR
systems fell outside the new range.

## 8. Library size saturates around 10⁶

| M | recovered (fixed prior) | |
| --- | --- | --- |
| 10⁵ | 28.8% | |
| 10⁶ | 35.5% | +6.7 |
| 10⁷ | 36.1% | **+0.6** |

Ten times the samples bought 0.6 points. Past the knee, the limit is the prior
and the data, not the library. Spend the budget on the prior and on more
sources.

Two caveats for a case study: this is with the amplitude prior *unfixed*, and
ESS does keep scaling linearly with M — so if you need posterior *widths* rather
than point estimates, the trade is different.

## 9. MPI: cost-aware assignment, and units small enough for it to matter

Per-unit cost varies 2.7× because it is linear in the padded epoch count. We
tried two assignment schemes and measured:

| | allocation used |
| --- | --- |
| contiguous slices | 57% |
| strided (decorrelated) | 52% |
| cost-aware (LPT), 1.9 units/rank | ~77% (simulated) |
| cost-aware, 3.8 units/rank | ~95% (simulated) |

The lesson is not "use LPT". It is that **striding fixes correlation, not
variance** — with ~2 units per rank there is nothing to average, so the slowest
rank is set by whoever drew the most expensive unit. And **when one unit costs
more than a rank's fair share, no scheduling can help**; that rank cannot shed
it. Unit size and assignment have to be fixed together.

Cost is predictable before any fitting — sum the padded epoch counts, which the
input catalog already carries — so this needs no trial run.

## 10. Operational notes worth a sidebar

- **Set `jax_enable_x64` at package import, not in one module.** At float32 the
  marginalized log-likelihoods over a wide period prior all underflow to `-inf`,
  ESS returns NaN, and top-K then returns arbitrary rows. It is silent garbage,
  not an error. A diagnostic script of ours imported two modules but not the one
  setting the flag and produced a convincing false alarm.
- **Suppress the per-call warnings.** harv's non-Gaussian-prior notice and its
  under-resolution warning are both correct and both fire once per system.
- **harv imports matplotlib**, which on a cluster with node-local cache
  directories means every rank rebuilds the font list. Point `MPLCONFIGDIR` at a
  pre-built shared cache.
- **Print progress inside a work unit.** Ours printed only on unit completion,
  and a unit was hours long; a failing 32-node job was undiagnosable from its
  log for two hours.
- **Report peak RSS.** Our memory model under-predicted by 1.5× at M=10⁶, which
  matters at M=10⁷ where the library dominates.
- **A weighted-sample output needs its weights used.** `top_k` returns
  importance weights normalized over the *whole* library, so they sum to
  `weight_captured`, not to 1. Every average over them must renormalize; this is
  the failure mode most likely to reach a published number silently.

---

## The five things to measure before committing an allocation

1. Warm seconds per system, **on the target CPU, at the target library size,
   with the node full**.
2. Peak RSS per rank at the target library size → ranks per node.
3. `batch_size`, **under contention**.
4. The rail fraction against SNR → is the amplitude prior setting your detection
   threshold?
5. Units per rank ≥ 4, with cost-aware assignment.

Each is one short job. Together they would have saved us two failed production
runs and about a week.
