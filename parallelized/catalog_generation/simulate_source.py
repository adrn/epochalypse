#!/usr/bin/env python
"""Simulate ONE source (or a contiguous block of them): the atom of the pipeline.

A source can be addressed two ways:

* ``--id N`` -- an integer task id, which indexes the frozen source list built
  by `build_indices.py`. This is the form a scheduler wants: ids run 0..N-1 and
  mean nothing beyond "the Nth star", so an MPI rank or a job-array task can be
  handed a number without knowing anything about Gaia.
* ``--gaia-id 5484066448309985152`` -- the Gaia DR3 source id itself, for when
  you want a specific star.

The integer mapping is fixed at index-build time and fingerprinted in
index_manifest.json; every run verifies the fingerprint, so a stale catalog on
one node cannot silently change which star ``--id 7`` means. Note that the
*simulation* is seeded from the Gaia source id, never from the task id: if the
list were rebuilt, the task ids would point elsewhere but each star's data would
be unchanged.

Usage
-----
    # what the scheduler runs
    python catalog_generation/simulate_source.py --id 0 --population one-companion

    # a contiguous block in one process (amortizes the ~90 s JAX warm-up:
    # one call per source pays it every time, one call per 50k sources pays once)
    python catalog_generation/simulate_source.py --id-start 0 --id-count 50000 \
        --population one-companion --write

    # a specific star, printed rather than written
    python catalog_generation/simulate_source.py --gaia-id 5484066448309985152 \
        --population no-companion one-companion two-companion

    python catalog_generation/simulate_source.py --id 0 --which-shard 512
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_catalog import POPULATIONS, build_config  # noqa: E402
from pipeline import astrometry as astro                # noqa: E402
from pipeline.sources import ScanLawStore, SourceCatalog, shard_of  # noqa: E402

# The canonical names, plus the hyphenated spellings that read more naturally on
# a command line. Both resolve to the same population.
ALIASES = {
    "no-companion": "0_companion", "zero-companion": "0_companion",
    "one-companion": "1_companion", "two-companion": "2_companion",
}
CHOICES = sorted({spec.name for spec in POPULATIONS} | set(ALIASES))


def resolve_population(name):
    return ALIASES.get(name, name)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--id", type=int, help="integer task id into the frozen source list")
    target.add_argument("--id-start", type=int, help="first task id of a contiguous block")
    target.add_argument("--gaia-id", help="Gaia DR3 source id of the host star")
    parser.add_argument("--id-count", type=int, default=1,
                        help="how many sources to run from --id-start (default 1)")
    parser.add_argument("--population", nargs="+", default=["1_companion"], choices=CHOICES,
                        help="population(s) to simulate; hyphenated aliases accepted")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--write", action="store_true",
                        help="write parquet parts under outputs/ instead of printing")
    parser.add_argument("--out-dir", type=Path,
                        help="write <system_id>_epochs.csv / _truth.json here instead")
    parser.add_argument("--which-shard", type=int, metavar="N",
                        help="print this source's shard index out of N and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args.output_root.resolve() if args.output_root else None)
    catalog = SourceCatalog(config)
    catalog.verify_checksum()

    # --- work out which sources this invocation covers ---
    if args.gaia_id is not None:
        targets = [str(args.gaia_id)]
        first_id = None
    elif args.id is not None:
        targets, first_id = [catalog.id_at(args.id)], args.id
    else:
        first_id = args.id_start
        last = min(args.id_start + args.id_count, len(catalog))
        targets = [catalog.id_at(i) for i in range(args.id_start, last)]

    if args.which_shard:
        for gaia_id in targets:
            print(f"{gaia_id} -> shard {shard_of(gaia_id, args.which_shard)} "
                  f"of {args.which_shard}")
        return 0

    populations = [config.population(resolve_population(n)) for n in args.population]
    astro.configure_jax(config.astrometry)
    scanlaw = ScanLawStore(config)

    for gaia_id in targets:
        if gaia_id not in catalog:
            raise SystemExit(f"gaia_source_id {gaia_id} is not in the source list")
        if gaia_id not in scanlaw:
            raise SystemExit(f"no DR4 scan law for gaia_source_id {gaia_id}")

    if args.write:
        # one parquet part per (population, block), named by the block's first id
        started, totals = time.time(), []
        for spec in populations:
            with astro.ShardWriter(config, spec, first_id or 0, len(catalog),
                                   tag="ids") as writer:
                for gaia_id in targets:
                    epochs, truth = astro.simulate_source(
                        config, spec, gaia_id, catalog=catalog, scanlaw=scanlaw)
                    writer.add(epochs, truth)
            totals.append((spec.name, writer.n_systems, writer.epochs_path))
        for name, n, path in totals:
            print(f"  {name:<14} {n:>7,} systems -> {path}")
        print(f"  {len(targets):,} sources in {time.time() - started:.1f} s")
        return 0

    for spec in populations:
        for gaia_id in targets:
            epochs, truth = astro.simulate_source(config, spec, gaia_id,
                                                  catalog=catalog, scanlaw=scanlaw)
            if not args.quiet:
                print(f"\n=== {truth['system_id']} ===")
                print(f"  host      : M = {truth['mass_st_msun']:.3f} Msun, "
                      f"R = {truth['radius_st_rsun']:.3f} Rsun, "
                      f"parallax = {truth['parallax_mas']:.3f} mas")
                print(f"  epochs    : {len(epochs)} FoV transits, "
                      f"sigma_single = {truth['sigma_single_mas']:.4f} mas")
                print(f"  seeds     : system {truth['system_seed']} "
                      f"(noise {truth['noise_seed']}, obs {truth['observation_seed']})")
                for k in range(1, truth["n_planets"] + 1):
                    print(f"  companion {k}: M = {truth[f'mass_pl_{k}']:.4g} Mjup, "
                          f"a = {truth[f'sma_{k}']:.4g} AU, "
                          f"P = {truth[f'period_{k}']:.4g} yr, "
                          f"e = {truth[f'ecc_{k}']:.3f}, "
                          f"SNR_tot = {truth[f'snr_total_{k}']:.3g}")
                if truth["n_planets"] == 0:
                    print("  companion : none (noise-only control)")

            if args.out_dir:
                import json

                args.out_dir.mkdir(parents=True, exist_ok=True)
                epochs_path = args.out_dir / f"{truth['system_id']}_epochs.csv"
                truth_path = args.out_dir / f"{truth['system_id']}_truth.json"
                epochs.to_csv(epochs_path, index=False)
                truth_path.write_text(json.dumps(truth, indent=2, default=str))
                print(f"  wrote {epochs_path}")
                print(f"  wrote {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
