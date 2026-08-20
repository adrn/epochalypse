#!/usr/bin/env python
"""Simulate ONE Gaia source: the atom of the parallel pipeline.

Given a Gaia source id and a population (0, 1, or 2 companions), this looks the
star up, draws its companions, simulates its DR4 epoch astrometry, and writes or
prints the result. Everything is seeded from the source id, so this call is a
pure function of (config, population, gaia_source_id): running it alone, inside
a shard, or a year later on another machine gives byte-identical output.

Usage
-----
    # inspect one system
    python catalog_generation/simulate_source.py --gaia-id 4116504967401881600 \
        --population 1_companion

    # write it out
    python catalog_generation/simulate_source.py --gaia-id 4116504967401881600 \
        --population 2_companion --out-dir /tmp/one_source

    # all three populations for the same star
    python catalog_generation/simulate_source.py --gaia-id 4116504967401881600 \
        --population 0_companion 1_companion 2_companion

    # what would this source's shard be, out of 512?
    python catalog_generation/simulate_source.py --gaia-id 4116504967401881600 --which-shard 512

Requires the per-source indices (`python catalog_generation/build_indices.py`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_catalog import POPULATIONS, build_config  # noqa: E402
from pipeline import astrometry as astro                # noqa: E402
from pipeline.sources import ScanLawStore, SourceCatalog, shard_of  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gaia-id", required=True,
                        help="Gaia DR3 source id of the host star")
    parser.add_argument("--population", nargs="+", default=["1_companion"],
                        choices=[spec.name for spec in POPULATIONS],
                        help="population(s) to simulate for this source")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="catalog root (default: the repo's outputs/)")
    parser.add_argument("--out-dir", type=Path,
                        help="write <system_id>_epochs.csv / _truth.json here")
    parser.add_argument("--which-shard", type=int, metavar="N",
                        help="print this source's shard index out of N and exit")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args.output_root.resolve() if args.output_root else None)

    if args.which_shard:
        print(f"{args.gaia_id} -> shard {shard_of(args.gaia_id, args.which_shard)} "
              f"of {args.which_shard}")
        return 0

    astro.configure_jax(config.astrometry)
    catalog = SourceCatalog(config)
    scanlaw = ScanLawStore(config)

    if args.gaia_id not in catalog:
        raise SystemExit(f"gaia_source_id {args.gaia_id} is not in {config.paths.stars_csv}")
    if args.gaia_id not in scanlaw:
        raise SystemExit(f"no DR4 scan law for gaia_source_id {args.gaia_id}")

    for name in args.population:
        spec = config.population(name)
        epochs, truth = astro.simulate_source(config, spec, args.gaia_id,
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
                      f"a = {truth[f'sma_{k}']:.4g} AU, P = {truth[f'period_{k}']:.4g} yr, "
                      f"e = {truth[f'ecc_{k}']:.3f}, "
                      f"alpha = {truth[f'alpha_mas_{k}']:.4g} mas, "
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
