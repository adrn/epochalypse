# harv benchmark results — 2026-08-28

Measured on Rusty with `scripts/benchmarks/`, against the 500 pc catalog at
`$OUT_ROOT`, `1_companion` shard 0, first 5 systems, `TOP_K = 1024`.

Every earlier cost estimate for this stage came from a laptop and **every one of
them was wrong by roughly an order of magnitude**. Two production jobs were
sized from them; the second burned ~4,000 core-hours and produced no output at
all. What follows is the measured replacement. Read "Corrections" before
trusting any number in `HARV.md` written before this date.

## Recommended configuration

```bash
#SBATCH -N 34
#SBATCH --ntasks-per-node=96
#SBATCH -C genoa
#SBATCH -t 12:00:00

mpirun python scripts/harv_mpi.py --skip-existing --n-parts 3 \
    --n-prior-samples 100000 --catalog-root $OUT_ROOT --output-root $HARV_ROOT
```

**~24,000 core-hours, ~8 h walltime**, one work unit per rank. The case for
each choice is below. M = 10⁶ over the whole catalog is ~117,000 core-hours and
150 genoa nodes — not a reasonable first pass. Run M = 10⁵ everywhere, then
escalate from the `ess` column, which is exactly what it is for.

## Experiment 1 — which CPU

One exclusive node each, identical systems, one rank.

| architecture | warm s/system | vs rome | cores/node | BW per core (approx) |
| --- | --- | --- | --- | --- |
| **genoa** | **16.59** | **1.59×** | 96 | ~9.6 GB/s |
| icelake | 22.50 | 1.17× | 64 | ~6.4 GB/s |
| rome | 26.41 | 1.00× | 128 | ~3.2 GB/s |

Genoa wins, and the ordering follows memory bandwidth per core, which is
consistent with a bandwidth-sensitive kernel. **The margin was predicted at
2–4× and is 1.59×** — real, worth taking, not transformative.

Rome is the worst available choice for this stage on both axes at once: the most
cores sharing the least bandwidth, and AVX2 only. The first two production
attempts both ran on rome.

## Experiment 2 — what packing a node costs

Same node, same systems, varying `--ntasks-per-node`.

| ranks/node | s/system/rank | sys/s/node | % of linear |
| --- | --- | --- | --- |
| 1 | 16.54 | 0.060 | 100% |
| 16 | 18.74 | 0.854 | 88% |
| 32 | 20.07 | 1.594 | 82% |
| 48 | 21.91 | 2.191 | 75% |
| **96** | **22.21** | **4.322** | **74%** |

**Fill the node.** A full node costs only 1.34× per rank and gives the best
throughput per node. Contention was hypothesised at 5–10× and reducing ranks
per node was recommended on that basis; both were wrong. 74% of linear at 96
ranks is good scaling for this kernel.

`bench-genoa` (16.59) and `bench-genoa-r1` (16.54) agree, so the measurement is
reproducible to ~0.3%.

## Experiment 3 — settings

| setting | warm s/system | vs baseline |
| --- | --- | --- |
| baseline: M=10⁶, batch=10³ | 16.59 | — |
| `--batch-size 10000` | 20.14 | **1.21× slower** |
| `--n-prior-samples 100000` | 3.37 | 4.9× faster |

**`BATCH_SIZE = 1000` is correct and by a wider margin than a laptop can see.**
The laptop measured B=10³ as 3% *slower* than B=10⁴; on genoa it is 21% faster.
The `(batch, n_epochs, n_linear)` intermediate is 230 MB at B=10⁴ and 23 MB at
B=10³, and only the smaller one fits per-rank cache.

**M = 10⁵ is 4.9× cheaper, not 10×** — per-system overhead (library flatten,
top-K gather, conditional linear draws, Python) does not scale with M. ESS does
scale with M, so this is a real trade, but for a strongly detected system ESS is
~1 at either setting and the point estimate — the thing that is actually good,
0.14% on period — is unaffected.

### The resulting allocations

| | s/sys at 96 ranks | catalog core-h | `--n-parts` | allocation |
| --- | --- | --- | --- | --- |
| M=10⁶ | 22.21 | ~117,000 | 15 | 150 genoa nodes × 7.4 h |
| **M=10⁵** | ~4.5 | **~24,000** | **3** | **34 genoa nodes × 7.5 h** |

Core-hours include a ×93/84 correction for the epoch-bucket mismatch described
under Open questions. `--n-parts` is set so one unit fits ~65% of a 12 h
walltime; see `HARV.md` for the formula. The M=10⁵ contention factor is assumed
equal to the measured M=10⁶ one (1.34×) — the working set per batch is
identical, so this should hold, but it is not measured.

## Corrections to earlier estimates

The single most useful result here is that **this kernel cannot be benchmarked
on a laptop.** Scaled to a common epoch bucket, an M-series performance core is
**7.6–10× faster than a genoa core** on this code:

| | laptop, bucket 320 | genoa, bucket 84 | genoa scaled to 320 | ratio |
| --- | --- | --- | --- | --- |
| M=10⁶ | 8.32 s | 16.54 s | ~63 s | 7.6× |
| M=10⁵ | 1.27 s | 3.37 s | ~12.8 s | 10.1× |

**The cause is not known.** The obvious candidate was XLA CPU threading — the
production scripts set `--xla_cpu_multi_thread_eigen=false
intra_op_parallelism_threads=1` and the laptop runs did not — but setting those
flags locally changes the time by 0.8% (1.27 → 1.28 s), so that is ruled out.
See Open questions.

What this invalidated:

| claim | where it came from | reality |
| --- | --- | --- |
| 2.53 µs/sample, 12,062 core-h at M=10⁶ | laptop, N=108 | ~117,000 core-h |
| "budget ~2× for Rome cores" | carried over from the periodogram stage, where it measured 2.4× | ~3× for rome on *this* kernel, and the laptop baseline was itself ~8× off |
| `BATCH_SIZE ≈ 10⁴` optimal | laptop, single-threaded | 10³ is 21% faster on genoa |
| contention 5–10×, use fewer ranks/node | inferred from a failed job | 1.34× at 96 ranks, fill the node |
| genoa 2–4× faster than rome | architecture reasoning | 1.59× |

The periodogram stage's "2× for Rome" *did* hold (measured 2.4×). Carrying it to
a JAX/XLA kernel with a completely different instruction mix was the mistake.

## Open questions

1. **Why is a genoa core 8× slower than an M-series core here?** Worth 10
   minutes, because if any of it is environmental the whole budget moves.
   Compare versions first — the laptop has jax/jaxlib 0.11.1:

   ```bash
   python -c "import jax, jaxlib; print(jax.__version__, jaxlib.__version__)"
   ```

   If the cluster is on 0.8.1 (harv's pin, which this project asked to have
   relaxed), that is three minor versions of XLA codegen and the first thing to
   test. Next candidate is whether the jaxlib wheel's XLA is emitting AVX-512
   for Zen 4 at all.

2. **The catalog's epoch-bucket distribution is unmeasured.** Cost is close to
   linear in the *padded* epoch count (bucket 80 → 16.5 s, bucket 96 → 19.0 s;
   96/80 = 1.20 against a 1.15 time ratio). The benchmark timed a mean bucket of
   84, and shard 0's systems are all N=76–82 — one region of sky. `PIPELINE.md`
   records 44–298 transits catalog-wide, and scaling the committed 16k sample's
   DR3 transit counts suggests a mean bucket near 93, which is the ×93/84 used
   above. A real census would replace that estimate:

   ```bash
   python - <<'PY'
   import glob, os, numpy as np, pandas as pd
   from epochalypse.harv import config as C
   d = os.environ["OUT_ROOT"] + "/data/simulated_astrometry/1_companion"
   f = sorted(glob.glob(d + "/truths_rank*.parquet"))[:8]
   n = pd.concat([pd.read_parquet(x, columns=["n_transits_dr4"]) for x in f])
   n = n["n_transits_dr4"].to_numpy()
   b = np.array([C.bucket_for(int(v)) for v in n])
   print(f"{len(n):,} systems  transits {n.min()}-{n.max()}  mean bucket {b.mean():.0f}")
   PY
   ```

3. **The science has never run at production settings.** Every correctness test
   used M ≤ 50,000 and `TOP_K` ≤ 64. The first real run must be followed by
   `harv_finish.py --stages census`, checking that `weight_captured` ≈ 1.0, that
   `0_companion` has the *highest* median ESS (no signal ⇒ broadest posterior),
   and that period recovery is clearly better for the high-SNR subset.

4. **`--n-parts 3` at M=10⁵ leaves a unit at 7.5 h.** Fine for a 12 h wall, but
   a requeue after a node failure redoes a whole unit. Raising it costs nothing
   except more ranks to stay at one unit each.

## Reproducing

```bash
zsh scripts/benchmarks/submit_all.sh            # all three experiments, 10 jobs
zsh scripts/benchmarks/submit_all.sh summary    # the table
```

Every rank fits the same systems in shard order, which is what makes the
architecture rows comparable and makes per-rank time in the rank scan a direct
contention factor. `--exclusive` is mandatory: a contention measurement on a
shared node measures the other job too.

The suite is ~10 jobs × 1 node × ≤30 min. It is cheap enough that it should be
re-run whenever jax, jaxlib, or harv moves — the `BATCH_SIZE` result alone shows
how far a stale number can travel.
