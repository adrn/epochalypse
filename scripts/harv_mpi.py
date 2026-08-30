#!/usr/bin/env python
"""Run harv across MPI ranks -- the only posterior-inference entry point.

`mpirun python scripts/harv_mpi.py` launches one copy of this script per slot in
the allocation. Each copy asks MPI two questions -- which rank am I, and how
many of us are there -- takes the corresponding contiguous slice of the
work-unit list, and fits it. Ranks never communicate except for one `gather` at
the end, and no two ranks write the same file, so nothing here needs MPI-IO or
parallel HDF5. MPI is being used purely as a launcher.

Same SPMD shape as `simulate_mpi.py` and `characterize_mpi.py`, and the same
unit of work as the latter: one shard of one population, 320 x 3 = 960 of them.

**`--n-parts` matters much more here.** At the measured ~2.5 s per system a full
17,890-system shard is ~12 hours on one rank, so 960 ranks means a 12-hour
walltime no matter how much of the cluster is free. Cutting each shard into
parts is what lets more ranks finish it sooner: `--n-parts 5` gives 4,800 units
of ~2.5 h. Parts are cut on systems, not row groups, so five parts really is
five near-equal fifths.

The prior library is drawn in process, from `config.SEED`, on every rank -- it
is 32 MB and 0.3 s, so there is no file to build, stage, or read from a thousand
ranks at once. The final gather compares every rank's fingerprint, which is what
turns "they all used the same library" from an argument into a check.

    python scripts/harv_mpi.py --subsample 2000 --max-units 4   # a laptop run
    mpirun python scripts/harv_mpi.py --n-parts 5               # the cluster

Set OMP_NUM_THREADS=1 in the job script: the per-system arrays are ~300 x 9, so
BLAS threads buy nothing and with tens of ranks per node they oversubscribe the
cores.
"""

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

from epochalypse import mpi
from epochalypse.harv import adapt
from epochalypse.harv import config as C
from epochalypse.harv import library as L
from epochalypse.harv.unit import run_unit, unit_costs
from epochalypse.periodogram.shards import work_units


def write_manifest(described, args, size):
    """Everything needed to interpret the output, written before any of it exists.

    A stored sample is a bare float32 and means nothing without its unit; a
    `weight` means nothing without knowing it is normalized over the whole
    library rather than over the 1,024 rows beside it; a `logZ_int` means
    nothing without knowing the padding was corrected out of it. All of that
    lives here, and nowhere else in the output.
    """
    manifest = {
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "n_ranks": size,
        "catalog_root": str(C.catalog_root()),
        "output_root": str(C.OUTPUT_ROOT),
        "populations": list(args.populations),
        "subsample": C.SUBSAMPLE,
        "min_snr": args.min_snr,
        "n_parts": args.n_parts,
        "library": described,
        "padding": {
            "buckets": list(C.EPOCH_BUCKETS),
            "pad_err_mas": adapt.PAD_ERR_MAS,
            "note": (
                "epochs are padded to a bucket so harv compiles ~9 times rather "
                "than once per epoch count; logZ_int and max_log_likelihood have "
                "the padding's constant offset removed, weights and ess never "
                "had it"
            ),
        },
        "samples": {
            "top_k": C.TOP_K,
            "dtype": C.SAMPLE_DTYPE,
            "note": (
                "weights are normalized over the whole prior library, so they sum "
                "to weight_captured and NOT to 1; any average over these draws "
                "must use them"
            ),
        },
    }
    path = C.manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--populations",
        nargs="+",
        choices=list(C.POPULATIONS),
        default=list(C.POPULATIONS),
    )
    parser.add_argument(
        "--catalog-root",
        type=Path,
        help="the generated catalog to read (contains data/)",
    )
    parser.add_argument(
        "--output-root", type=Path, help="write results here instead of <repo>/outputs"
    )
    parser.add_argument(
        "--n-parts",
        type=int,
        default=1,
        help="cut each shard into this many units; at ~2.5 s/system a whole "
        "shard is ~12 h on one rank, so this is the knob that sets walltime",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help=f"fit only ~this many systems per population (default "
        f"{C.SUBSAMPLE if C.SUBSAMPLE is not None else 'all of them'})",
    )
    parser.add_argument(
        "--n-prior-samples",
        type=int,
        default=None,
        help=f"prior library size (default {C.N_PRIOR_SAMPLES:,}); cost and ESS "
        "are both linear in it",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help=f"samples kept per system (default {C.TOP_K})",
    )
    parser.add_argument(
        "--m-max-mjup",
        type=float,
        default=None,
        help=f"largest companion the orbit-amplitude prior expects, in Jupiter "
        f"masses (default {C.M_MAX_MJUP:g}). Converted per system to a "
        f"sigma_a0 of m_max / M_star^(2/3), so one number covers every host. "
        "This sets the DETECTION THRESHOLD -- a wider prior means a larger Occam "
        "penalty on a real orbit relative to the no-orbit solution",
    )
    parser.add_argument(
        "--sigma-a0",
        type=float,
        default=None,
        help="pin sigma_a0 to a constant in AU, DISABLING the per-system mass "
        "scaling. Only for sweeps, which vary one constant across arms. Recorded "
        "in the manifest, but it does not change the library fingerprint -- it "
        "affects only the analytically marginalized Thiele-Innes priors, which "
        "are never drawn -- so give each arm its own --output-root",
    )
    parser.add_argument(
        "--error-mode",
        choices=("reported", "injected"),
        default=None,
        help="'reported' fits with the catalog's own error bars, which are "
        "SMALLER than the scatter that was injected (median ratio 1.28, tail to "
        "11) and so make the sampler overconfident by r^2 in the exponent. "
        "'injected' reconstructs the true scale per epoch and weights by it -- "
        "correct calibration at no cost in library resolution",
    )
    parser.add_argument(
        "--jitter-sigma",
        type=float,
        default=None,
        help="HalfNormal scale in mas for a LEARNED excess-variance term. Adds a "
        "fourth sampled nonlinear dimension, so effective resolution per "
        "dimension falls from M^(1/3) to M^(1/4) -- compare against a matched-M "
        "baseline before believing any improvement",
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=None,
        help="fit only systems where EVERY injected companion clears this "
        "SNR_tot. Only 6.9%% of the catalog is high-SNR, so a calibration run "
        "that only cares about detected systems otherwise spends 93%% of its "
        "budget on ones that cannot inform it. Implies --populations without "
        "0_companion, which has no companions and so no SNR",
    )
    parser.add_argument(
        "--limit", type=int, help="cap systems per unit (smoke tests only)"
    )
    parser.add_argument(
        "--max-units",
        type=int,
        help="use only the first N work units (smoke tests only). --limit alone "
        "still walks all 960 of them",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a unit whose output already exists, so a rerun only redoes "
        "the units that died",
    )
    args = parser.parse_args(argv)

    comm, rank, size = mpi.mpi_context()
    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)
    if args.output_root:
        C.set_output_root(args.output_root)
    if args.subsample is not None:
        C.set_subsample(args.subsample)
    if args.n_prior_samples is not None:
        C.N_PRIOR_SAMPLES = args.n_prior_samples
    if args.top_k is not None:
        C.TOP_K = args.top_k
    if args.sigma_a0 is not None:
        C.SIGMA_A0_AU = args.sigma_a0
    if args.m_max_mjup is not None:
        C.M_MAX_MJUP = args.m_max_mjup
    if args.error_mode is not None:
        C.ERROR_MODE = args.error_mode
    if args.jitter_sigma is not None:
        C.JITTER_SIGMA_MAS = args.jitter_sigma

    units = work_units(args.populations, args.n_parts)
    if args.max_units:
        # Truncate before the slice, so every rank agrees on the same short list.
        units = units[: args.max_units]

    started = time.time()
    prior_samples = L.draw()
    described = L.describe(prior_samples)

    if rank == 0:
        mpi.banner(
            comm,
            size,
            len(units),
            item="work units",
            catalog=C.catalog_root(),
            output=C.OUTPUT_ROOT,
            populations=", ".join(args.populations),
            library=(
                f"{C.N_PRIOR_SAMPLES:,} prior samples, top {C.TOP_K}, "
                f"seed {C.SEED}, {described['fingerprint']}"
            ),
            amplitude=(
                f"sigma_a0 pinned at {C.SIGMA_A0_AU} AU"
                if C.SIGMA_A0_AU is not None
                else f"m_max = {C.M_MAX_MJUP:g} MJup, per-system from the host mass"
            ),
            errors=(
                f"{C.ERROR_MODE}"
                + (
                    f", learned jitter ~ HalfNormal({C.JITTER_SIGMA_MAS:g} mas)"
                    if C.JITTER_SIGMA_MAS is not None
                    else ", no jitter term"
                )
            ),
            subsample=C.SUBSAMPLE if C.SUBSAMPLE is not None else "the full catalog",
            balance=(
                f"{len(units) / max(size, 1):.1f} units/rank"
                + (
                    "  <- raise --n-parts; below ~4 the largest single unit caps "
                    "how balanced the run can be"
                    if len(units) / max(size, 1) < 3.5
                    else ""
                )
            ),
            min_snr=args.min_snr if args.min_snr is not None else "no cut",
        )
        write_manifest(described, args, size)
        print(f"wrote       : {C.manifest_path().name}\n", flush=True)

    # Cost-aware assignment. Unit cost spans ~2.7x because it is linear in the
    # padded epoch count, and neither a contiguous slice nor a stride reduces
    # the VARIANCE of a rank's total -- measured, both left ~half the allocation
    # idle. Rank 0 predicts each unit's cost from the truth tables' epoch counts
    # and shares the plan; every rank then computes the same longest-first deal.
    #
    # This only pays if no single unit costs more than a rank's fair share, so
    # it wants --n-parts high enough: at 1.9 units per rank it recovers 52% ->
    # ~77%, at 3.8 it reaches ~95%.
    costs = None
    if rank == 0:
        try:
            costs = unit_costs(units, min_snr=args.min_snr)
        except Exception as error:  # noqa: BLE001 - fall back, never fail the run
            print(
                f"cost model unavailable ({error!r}); assigning by stride", flush=True
            )
    costs = mpi.broadcast(comm, costs)
    mine = (
        mpi.balance(units, costs, rank, size)
        if costs is not None
        else mpi.stride_for_rank(units, rank, size)
    )

    summaries = []
    for population, shard, n_shards, part, n_parts in mine:
        summaries.append(
            run_unit(
                population,
                shard,
                n_shards,
                part,
                n_parts,
                prior_samples=prior_samples,
                top_k=args.top_k,
                limit=args.limit,
                min_snr=args.min_snr,
                skip_existing=args.skip_existing,
                verbose=True,
                # Rank 0 only: 2048 ranks at one line per 50 systems is a
                # quarter-million lines in one log file, and one rank's rate is
                # all you need. Silence here is what made the first production
                # attempt undiagnosable -- a unit is hours long, so the
                # unit-completion line alone tells you nothing until it is too
                # late to act on.
                progress_every=50 if rank == 0 else 0,
            )
        )

    # Peak RSS, so the memory question is measured on every run rather than
    # extrapolated. It is the number that decides ranks-per-node, and at large
    # N_PRIOR_SAMPLES the library dominates it.
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 1e9 if sys.platform == "darwin" else peak / 1e6

    mine = {
        "rank": rank,
        "peak_rss_gb": peak_gb,
        "fingerprint": described["fingerprint"],
        "n_units": len(summaries),
        "n_systems": sum(s["n_systems"] for s in summaries),
        "n_failed": sum(s["n_failed"] for s in summaries),
        "seconds": time.time() - started,
    }
    print(
        f"[rank {rank:05d}] {mine['n_units']} unit(s), {mine['n_systems']:,} systems "
        f"in {mine['seconds'] / 60:.1f} min",
        flush=True,
    )

    everyone = mpi.gather(comm, mine)
    if rank == 0:
        systems = sum(s["n_systems"] for s in everyone)
        failed = sum(s["n_failed"] for s in everyone)
        slowest = max(s["seconds"] for s in everyone)
        prints = {s["fingerprint"] for s in everyone}
        print(f"\ndone: {systems:,} systems across {size} rank(s)")
        print(f"  slowest rank : {slowest / 3600:.2f} h")
        if systems:
            print(
                f"  per system   : {slowest * size / systems:.2f} s "
                f"(core-hours: {sum(s['seconds'] for s in everyone) / 3600:,.0f})"
            )
        # The one thing that would silently invalidate every comparison in the
        # output: two ranks measuring against two different priors.
        worst = max(s["peak_rss_gb"] for s in everyone)
        per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE") or 0)
        print(
            f"  peak RSS     : {worst:.1f} GB/rank"
            + (
                f"  ->  {worst * per_node:.0f} GB per node at {per_node} ranks"
                if per_node
                else ""
            )
        )
        print(
            f"  library      : {prints.pop()} on every rank"
            if len(prints) == 1
            else f"  library      : *** {len(prints)} DIFFERENT LIBRARIES: {prints} ***"
        )
        if failed:
            print(f"  failed       : {failed:,} systems (see {C.failed_dir()})")
        print("  next         : python scripts/harv_finish.py --stages census merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
