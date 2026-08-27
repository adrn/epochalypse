#!/usr/bin/env python
"""Build the epochalypse catalog, everything except the simulation itself.

    stars  ->  index  ->  [ mpirun run_mpi.py ]  ->  merge  ->  select  ->  figures

Every prior, threshold, path, seed, and figure choice lives in
`epochalypse/config.py`; this file only decides what runs.

The `simulate` step is not a stage here. It is the only expensive part, it is
the part that needs many cores, and it is `run_mpi.py`:

    python scripts/generate_catalog.py --stages stars index
    mpirun -n 1024 python scripts/run_mpi.py
    python scripts/generate_catalog.py --stages merge select figures

Usage
-----
    python scripts/generate_catalog.py                  # all stages
    python scripts/generate_catalog.py --stages figures
    python scripts/generate_catalog.py --stages figures \
        --figures population_schematic companion_gallery
    python scripts/generate_catalog.py --stages stars --overwrite
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


from epochalypse import config as C

STAGES = ("stars", "index", "merge", "select", "figures")
# populations with companions; the control has no companions to select on
WITH_COMPANIONS = [name for name, n in C.POPULATIONS.items() if n > 0]


def merge_truths(population):
    """Concatenate a population's shard truth tables into one parquet.

    The epochs stay sharded -- that is the point of the layout -- but the truth
    table is one row per system and is what analysis reads first, so it is worth
    having whole.
    """
    import pandas as pd

    parts = sorted(C.shard_dir(population).glob("truths_*.parquet"))
    if not parts:
        print(f"  {population}: no shard truth tables in {C.shard_dir(population)}")
        return
    frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    frame = frame.sort_values("gaia_source_id").reset_index(drop=True)
    out = C.truths(population)
    frame.to_parquet(out, index=False, compression=C.PARQUET_COMPRESSION)
    print(f"  {population}: {len(frame):,} systems from {len(parts)} shards -> {out}")


def select_views(population):
    """Write the high-SNR view of a population: the top slice by SNR_tot."""
    import pandas as pd

    from epochalypse.sources import select_high_snr

    merged = C.truths(population)
    if not merged.exists():
        print(f"  {population}: {merged} not found, skipping")
        return
    frame = pd.read_parquet(merged)
    selected = select_high_snr(frame)
    out = C.truths(population, high_snr=True)
    selected.to_parquet(out, index=False, compression=C.PARQUET_COMPRESSION)
    snr = selected[[c for c in ("snr_total_1", "snr_total_2")
                    if c in selected.columns]].max(axis=1)
    print(f"  {population}: SNR_tot >= {C.HIGH_SNR_MIN:g} on every companion, "
          f"{len(frame):,} -> "
          f"{len(selected):,} systems, SNR_tot >= {snr.min():.1f} "
          f"(median {snr.median():.1f})")


def run(stages, populations, *, overwrite, figures=None):
    started = time.perf_counter()

    if "stars" in stages:
        print("\n== stars -- parent stellar sample ==")
        from epochalypse.stars import build_star_catalog
        build_star_catalog(overwrite=overwrite)

    if "index" in stages:
        print("\n== index -- per-source lookup indices ==")
        from epochalypse.sources import build_indices
        build_indices(overwrite=overwrite)

    if "merge" in stages:
        print("\n== merge -- one truth table per population ==")
        for population in populations:
            merge_truths(population)

    if "select" in stages:
        print("\n== select -- high-SNR views ==")
        for population in populations:
            if C.POPULATIONS[population] > 0:
                select_views(population)

    if "figures" in stages:
        print("\n== figures ==")
        from epochalypse.figures import make_figures
        make_figures(figures)

    print(f"\ndone in {time.perf_counter() - started:.1f} s")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--populations", nargs="+", choices=list(C.POPULATIONS),
                        default=list(C.POPULATIONS),
                        help="populations to merge/select (default: all)")
    parser.add_argument("--figures", nargs="+", choices=C.FIGURES,
                        help="figures to build (default: all)")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild stars.csv / the indices instead of reusing them")
    parser.add_argument("--output-root", type=Path,
                        help="write products here instead of <repo>/outputs")
    args = parser.parse_args(argv)

    if args.output_root:
        C.set_output_root(args.output_root)

    print(f"inputs      : {C.DATA_IN}")
    print(f"outputs     : {C.OUTPUT_ROOT}")
    print(f"stages      : {', '.join(args.stages)}")
    print(f"populations : {', '.join(args.populations)}")
    print(f"seeds       : planets={C.SEED_PLANETS}, astrometry={C.SEED_ASTROMETRY}"
          "  (keyed on gaia_source_id)")

    missing = [p for p in (C.G23H_SAMPLE, C.PECAUT_MAMAJEK) if not p.exists()]
    if "index" in args.stages and not C.SCANLAW_DR4.exists():
        missing.append(C.SCANLAW_DR4)
    if missing:
        raise SystemExit("missing input files:\n  " + "\n  ".join(str(p) for p in missing))

    run(set(args.stages), args.populations, overwrite=args.overwrite,
        figures=args.figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
