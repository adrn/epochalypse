#!/usr/bin/env python
"""Build the epochalypse catalog-generation figures from an existing catalog.

This is the standalone counterpart to the `figures` stage of
`generate_catalog.py`. It runs the same code over the same configuration --
paths, palette, formats, and figure list are imported from `generate_catalog`,
so there is one place to change them and the two entry points cannot drift
apart. Nothing is simulated here: the figures are drawn from `stars.csv` and
the population CSVs that stage 1 and 2 already wrote.

Figures produced (the catalog-generation figures from the paper):

    star_sky_scanlaw_<frame>        parent sample over the DR4 scan law
    population_schematic            selection funnel + population branching
    pop_diagnostics_1planet         one-companion prior distributions
    pop_diagnostics_2planet         two-companion prior distributions
    companion_gallery               sample on-sky orbits, one row per population
    simulated_planets_mass_period   mass vs. period, coloured by signature alpha

Usage
-----
    # everything, from and into the repo's outputs/
    python catalog_generation/make_figures.py

    # a subset
    python catalog_generation/make_figures.py \
        --figures population_schematic companion_gallery

    # read a catalog from one place, write figures somewhere else
    python catalog_generation/make_figures.py \
        --data-root outputs/data --figure-dir /tmp/paper_figures

    python catalog_generation/make_figures.py --list   # what would be built
    python catalog_generation/make_figures.py --no-usetex   # no TeX installed

Requires `healpy` (sky map only) and, unless --no-usetex is passed, a working
LaTeX installation.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_catalog import FIGURES, OUTPUT_ROOT, ROOT, build_config  # noqa: E402
from pipeline import figures as figure_stage                           # noqa: E402


# What each figure needs on disk, so a missing catalog is reported up front
# rather than as a traceback halfway through a run.
REQUIREMENTS = {
    "star_sky_scanlaw": ("stars", "gost"),
    "population_schematic": ("stars", "populations", "g23h"),
    "pop_diagnostics_1planet": ("one_companion",),
    "pop_diagnostics_2planet": ("two_companion",),
    "companion_gallery": ("populations",),
    "simulated_planets_mass_period": ("stars", "populations"),
}


def resolve_config(args):
    """The pipeline configuration, with the command-line overrides applied."""
    config = build_config(args.output_root.resolve())

    paths = config.paths
    if args.data_root is not None:
        data_root = args.data_root.resolve()
        paths = replace(paths, data_dir=data_root, stars_csv=data_root / "stars.csv")
    if args.figure_dir is not None:
        paths = replace(paths, figure_dir=args.figure_dir.resolve())
    config = replace(config, paths=paths)

    settings = config.figures
    if args.no_usetex:
        settings = replace(settings, usetex=False)
    if args.formats:
        settings = replace(settings, formats=tuple(args.formats))
    return replace(config, figures=settings)


def missing_inputs(config, figure):
    """Inputs `figure` needs that are not on disk, as human-readable strings."""
    paths = config.paths
    population_csvs = {
        spec.n_companions: paths.population_csv(spec.name)
        for spec in config.populations if spec.n_companions > 0
    }
    checks = {
        "stars": [paths.stars_csv],
        "gost": [paths.gost_fov_map],
        "g23h": [paths.g23h_sample],
        "populations": [paths.population_csv(spec.name)
                        for spec in config.populations if spec.n_companions > 0],
        "one_companion": [population_csvs.get(1)],
        "two_companion": [population_csvs.get(2)],
    }
    needed = []
    for requirement in REQUIREMENTS[figure]:
        needed += [path for path in checks[requirement] if path is not None]
    return [str(path) for path in needed if not path.exists()]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--figures", nargs="+", choices=FIGURES.figures,
                        help="figures to build (default: all of them)")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="catalog/figure root; data is read from "
                             "<root>/data and figures written to <root>/figures")
    parser.add_argument("--data-root", type=Path,
                        help="read the catalog from here instead of <output-root>/data")
    parser.add_argument("--figure-dir", type=Path,
                        help="write figures here instead of <output-root>/figures")
    parser.add_argument("--formats", nargs="+", metavar="EXT",
                        help="file formats to write (default: pdf png)")
    parser.add_argument("--no-usetex", action="store_true",
                        help="render without LaTeX (for machines with no TeX install)")
    parser.add_argument("--list", action="store_true",
                        help="report what would be built, and what is missing, then exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = resolve_config(args)
    selected = list(args.figures or config.figures.figures)

    print(f"repo root  : {ROOT}")
    print(f"catalog    : {config.paths.data_dir}")
    print(f"figures    : {config.paths.figure_dir}")
    print(f"formats    : {', '.join(config.figures.formats)}"
          f"{'' if config.figures.usetex else '   (usetex off)'}")
    print()

    buildable, blocked = [], {}
    for figure in selected:
        missing = missing_inputs(config, figure)
        if missing:
            blocked[figure] = missing
        else:
            buildable.append(figure)
        status = "ok" if not missing else f"missing {len(missing)} input(s)"
        print(f"  {figure:32s} {status}")
        for path in missing:
            print(f"      - {path}")

    if not buildable:
        print("\nnothing to build: generate the catalog first with "
              "`python catalog_generation/generate_catalog.py`")
        return 1
    if args.list:
        return 0

    started = time.time()
    written = figure_stage.make_figures(config, buildable)
    print(f"\nwrote {len(written)} file(s) in {time.time() - started:.0f} s "
          f"-> {config.paths.figure_dir}")
    if blocked:
        print(f"skipped {len(blocked)}: {', '.join(blocked)} (see above)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
