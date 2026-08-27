#!/usr/bin/env python
"""Run the periodogram across MPI ranks -- the only characterization entry point.

`mpirun python run_mpi.py` launches one copy of this script per slot in the
allocation. Each copy asks MPI two questions -- which rank am I, and how many of
us are there -- takes the corresponding contiguous slice of the work-unit list,
and characterizes it. Ranks never communicate except for one `gather` at the end
to print a summary, and no two ranks write the same file, so nothing here needs
MPI-IO or parallel HDF5. MPI is being used purely as a launcher.

This is the same SPMD shape as the generator's `run_mpi.py`, and for the same
reason, but the unit of work is different. The generator sliced the *source
list* and each rank wrote its own shard. Here the shards already exist, so the
work unit is one of them: 320 shards x 3 populations = 960 units, each ~17,890
systems. Slices are contiguous, so a rank reads a contiguous region of one
population's directory rather than scattering reads across all 320 files.

960 units caps the useful rank count at 960. `--n-parts` cuts each shard into
that many contiguous pieces if more ranks than that are available -- pieces are
cut on systems, not row groups, so four parts really is four near-equal
quarters. There is no reason to go past one rank per unit otherwise: at 960
ranks the whole catalog is a few hours, and a finer split re-reads the shard's
first row group in every part that touches it.

    python scripts/characterize_shard.py 1_companion 7   # one unit, no MPI
    python scripts/run_mpi.py --limit 50                 # one process, no MPI
    mpirun -n 8 python scripts/run_mpi.py                # 8 local processes
    srun python scripts/run_mpi.py                       # a cluster allocation

Unlike the generator there is no JAX warm-up here: `kepmodel` is NumPy and
S+LEAF, so a rank reaches steady state in under a second and short test runs
give honest timings. Cost per system is ~0.34 s on an Apple M-series core and is
nearly flat in the epoch count (320 ms at 48 epochs, 375 ms at 169) because the
16,600-iteration frequency loop dominates, not the linear algebra.

Set OMP_NUM_THREADS=1 in the job script: the per-system arrays are tiny, BLAS
threads buy nothing, and with tens of ranks per node they oversubscribe the
cores.
"""

import argparse
import json
import platform
import time
from pathlib import Path

from epochalypse import mpi
from epochalypse.periodogram import __version__
from epochalypse.periodogram import config as C
from epochalypse.periodogram import grid as G
from epochalypse.periodogram.shards import work_units
from epochalypse.periodogram.unit import run_unit
from epochalypse.periodogram.writers import write_period_grid


def write_manifest(segments, args, size):
    """Everything needed to interpret the output, written before any of it exists.

    The grid, the noise model, the classifier thresholds and the code version.
    A power array is 16,641 float32 and means nothing without the period axis
    and the decimation factor; a `top_power` means nothing without knowing
    whether jitter was fitted. Both live here, and in `period_grid.parquet`.
    """
    manifest = {
        "version": __version__,
        "written": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "n_ranks": size,
        "catalog_root": str(C.CATALOG_ROOT),
        "output_root": str(C.OUTPUT_ROOT),
        "populations": list(args.populations),
        "grid": G.describe(segments),
        "grid_config": {
            "p_min_yr": C.P_MIN,
            "p_max_yr": C.P_MAX,
            "n_segments": C.N_SEGMENTS,
            "baseline_n_periods": C.BASELINE_N_PERIODS,
        },
        "classifier": {
            "delta_bic_detect": C.DELTA_BIC_DETECT,
            "delta_power_unimodal": C.DELTA_POWER_UNIMODAL,
            "min_separation_dex": C.MIN_SEPARATION_DEX,
            "width_delta": C.WIDTH_DELTA,
            "width_constrained_dex": C.WIDTH_CONSTRAINED_DEX,
            "edge_frac": C.EDGE_FRAC,
            "period_recover_tol": C.PERIOD_RECOVER_TOL,
        },
        "power": {
            "mode": args.power or C.POWER_MODE,
            "dtype": C.POWER_DTYPE,
        },
        "high_snr_min": C.HIGH_SNR_MIN,
        "target_fp": C.TARGET_FP,
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
        help="cut each shard into this many units (only for >960 ranks)",
    )
    parser.add_argument(
        "--power",
        choices=("all", "none"),
        default=None,
        help=f"keep the raw Delta-chi^2 curves (default {C.POWER_MODE})",
    )
    parser.add_argument(
        "--limit", type=int, help="cap systems per unit (smoke tests only)"
    )
    parser.add_argument(
        "--max-units",
        type=int,
        help="use only the first N work units (smoke tests only). "
        "--limit alone still walks all 960 of them, which on one "
        "process is hours -- pass this too",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip a unit whose output already exists, so a rerun "
        "only redoes the units that died",
    )
    args = parser.parse_args(argv)

    comm, rank, size = mpi.mpi_context()
    if args.catalog_root:
        C.set_catalog_root(args.catalog_root)
    if args.output_root:
        C.set_output_root(args.output_root)

    segments = G.frequency_segments()
    periods = G.segment_periods(segments)
    units = work_units(args.populations, args.n_parts)
    if args.max_units:
        # Truncate before the slice, so every rank agrees on the same short list.
        units = units[: args.max_units]

    if rank == 0:
        described = G.describe(segments)
        mpi.banner(
            comm,
            size,
            len(units),
            item="work units",
            catalog=C.CATALOG_ROOT,
            output=C.OUTPUT_ROOT,
            populations=", ".join(args.populations),
            grid=(
                f"{described['n_periods']:,} trial periods, "
                f"{described['p_min_yr']:.2e} - {described['p_max_yr']:.0f} yr, "
                f"{described['n_segments']} segments, "
                f"dlogP <= {described['dlog_max']:.2e}"
            ),
            curves=args.power or C.POWER_MODE,
        )
        write_manifest(segments, args, size)
        write_period_grid(periods)
        print(
            f"wrote       : {C.manifest_path().name}, {C.period_grid_path().name}\n",
            flush=True,
        )

    if comm is not None:  # every rank waits for the manifest and the grid file
        comm.Barrier()

    start, stop = mpi.slice_for_rank(len(units), rank, size)
    started = time.time()
    summaries = []
    for population, shard, n_shards, part, n_parts in units[start:stop]:
        summaries.append(
            run_unit(
                population,
                shard,
                n_shards,
                part,
                n_parts,
                segments=segments,
                limit=args.limit,
                skip_existing=args.skip_existing,
                power_mode=args.power,
                verbose=True,
                progress_every=0,
            )
        )

    mine = {
        "rank": rank,
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
        print(f"\ndone: {systems:,} systems across {size} rank(s)")
        print(f"  slowest rank : {slowest / 3600:.2f} h")
        if systems:
            print(
                f"  per system   : {slowest * size / systems * 1e3:.0f} ms "
                f"(core-hours: {sum(s['seconds'] for s in everyone) / 3600:,.0f})"
            )
        if failed:
            print(f"  failed       : {failed:,} systems (see {C.failed_dir()})")
        print("  next         : python scripts/finish.py --stages calibrate census")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
