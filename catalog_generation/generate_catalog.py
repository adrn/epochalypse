#!/usr/bin/env python
"""Generate the epochalypse catalog end to end.

    stars  ->  planets  ->  figures  ->  astrometry  ->  export  [->  repack]

Every input file, every prior, every threshold, every seed, and every figure
choice is declared in the CONFIGURATION block directly below. The stage
implementations in `pipeline/` take those values as arguments and hard-code
nothing, so this block is the complete specification of what a run produces.

Usage
-----
    # everything, all five populations
    python catalog_generation/generate_catalog.py

    # rebuild only the high-SNR populations, skipping stage 1
    python catalog_generation/generate_catalog.py \
        --stages planets astrometry export \
        --populations 1_companion_detectable 2_companion_detectable

    # just redraw the paper's catalog-generation figures
    python catalog_generation/generate_catalog.py --stages figures
    python catalog_generation/generate_catalog.py --stages figures \
        --figures population_schematic companion_gallery

    # 20-star smoke test of the whole pipeline into a scratch directory
    python catalog_generation/generate_catalog.py --limit 20 \
        --output-root /tmp/epochalypse_smoke --overwrite

    python catalog_generation/generate_catalog.py --list   # show the plan only

Every run prints a wall-clock summary at the end, broken down by stage and by
population; --timing-csv PATH also writes that breakdown to disk.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

# The repo root holds src/ and outputs/; resolve it from this file so the
# script runs identically from anywhere.
ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "src" / "epochalypse_fitting.py").exists())
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

# Physical and mission constants, all derived from astropy in one place and
# shared with the analysis code in src/ -- never restated here.
from epochalypse_constants import (DAYS_PER_YEAR, DR4_BASELINE_YEARS,   # noqa: E402
                                   GAIA_EPOCH_TCB_JD, MARS_IN_MJUP,
                                   MAX_COMPANION_MASS_MJUP, MJUP_IN_MSUN,
                                   RSUN_IN_AU)
from pipeline.config import (AstrometrySettings, CatalogConfig, FigureSettings,  # noqa: E402
                             Paths, PlanetPriors, PopulationSpec, Seeds,
                             StarSelection)

# ==========================================================================
# CONFIGURATION -- every choice the catalog depends on
# ==========================================================================

# --- inputs (static; not produced by this pipeline) -----------------------
DATA_DIR_IN = ROOT / "data"
G23H_SAMPLE = DATA_DIR_IN / "g23h_epochalypse_stars" / "G23H_sample_subset.arrow"
SCANLAW_DR4 = DATA_DIR_IN / "g23h_epochalypse_stars" / "sample_scanlaw_dr4.arrow"
PECAUT_MAMAJEK = DATA_DIR_IN / "pecaut_mamajek.txt"
GOST_FOV_MAP = DATA_DIR_IN / "gost_fov_counts_dr4.fits"   # figures only

# --- outputs --------------------------------------------------------------
OUTPUT_ROOT = ROOT / "outputs"          # overridable with --output-root

# --- stellar sample -------------------------------------------------------
STAR_SELECTION = StarSelection(
    parallax_col="parallax",
    gmag_col="phot_g_mean_mag_dr3",
    source_id_col="gaia_source_id",
    # drop stars whose absolute G falls outside the Pecaut & Mamajek sequence
    require_mass_radius=True,
    # drop the handful of high-RUWE binaries with no sig_AL noise model
    require_sigma_al=True,
)

# --- companion priors -----------------------------------------------------
# Unit conversions come from `epochalypse_constants` (astropy); the numbers
# below are the choices.
PLANET_PRIORS = PlanetPriors(
    # semi-major axis: log-uniform. The floor is deliberately below anything
    # physical -- the binding inner limit is the per-star Roche-lobe screen
    # below, which cuts each system at its own separation and so leaves a
    # smeared inner edge rather than a wall at a_min_au.
    a_min_au=0.001,
    a_max_au=100.0,

    # innermost separation: reject draws where the star would overflow its
    # Roche lobe (Eggleton 1983); 1.0 = the bare limit, no safety margin
    enforce_roche_lobe=True,
    roche_lobe_safety_factor=1.0,

    # companion mass: log-uniform, Mars mass up to the hydrogen-burning limit
    mass_min_mjup=MARS_IN_MJUP,             # 1 M_Mars = 3.3668e-04 M_Jup
    mass_max_mjup=MAX_COMPANION_MASS_MJUP,  # 80 M_Jup

    # eccentricity: uniform
    ecc_min=0.0,
    ecc_max=0.99,

    # orientation: isotropic orbits, all angles uniform over the full circle
    isotropic_inclination=True,
    angle_min_deg=0.0,
    angle_max_deg=360.0,
    # half of the two-companion systems are injected coplanar
    coplanar_probability=0.5,

    # detectability filter (applies only to the high-SNR populations)
    baseline_years=DR4_BASELINE_YEARS,   # 5.5 yr; sets a_crit
    snr_total_min=10.0,        # keep companions with SNR_total >= this
    snr_draw_batch=4096,
    snr_max_draws=8_000_000,

    # two-companion stability screen
    hill_stability_factor=2.0,      # unstable if delta < 2 sqrt(3) Hill radii
    resonance_orders=(1, 2),        # check the 2:1 and 3:2 resonances
    resonance_tolerance=0.05,       # within 5% in period ratio counts as near
    max_stability_retries=1000,

    # one Msun-per-Mjup for the signature alpha, Kepler's third law, and the
    # stability mass ratios (was three slightly different hand-typed values)
    mjup_in_msun=MJUP_IN_MSUN,      # 9.545942e-04
)

# --- epoch astrometry -----------------------------------------------------
ASTROMETRY = AstrometrySettings(
    gaia_epoch_tcb_jd=GAIA_EPOCH_TCB_JD,  # 2457936.875 JD (TCB)
    days_per_year=DAYS_PER_YEAR,          # 365.25, the Julian year
    enable_float64=True,             # float32 draws would change the noise realization
    rsun_to_au=RSUN_IN_AU,           # 4.650467e-03
    mjup_to_msun=MJUP_IN_MSUN,       # population CSV (Mjup) -> simulator (Msun)
    n_dof_five_param=5,
    n_dof_other=6,
    params_solved_five_param=31,
    noise_jitter_frac=0.1,           # shared 10% per-epoch uncertainty jitter
    alpha0_mas=0.0,
    delta0_mas=0.0,
    hdf5_compression="gzip",
)

# --- figures --------------------------------------------------------------
# Master switch for the figure stage. False drops `figures` from the default
# run; the stage still runs if it is named explicitly with --stages figures,
# and --no-figures forces it off whatever this says.
MAKE_FIGURES = True

# The catalog-generation figures from the paper. The analysis figures
# (characterizability maps, LW25 comparison) need the periodogram
# characterization and stay in analysis/.
FIGURES = FigureSettings(
    figures=(
        "star_sky_scanlaw",              # parent sample over the DR4 scan law
        "population_schematic",          # selection funnel + population branching
        "pop_diagnostics_1planet",       # one-companion prior distributions
        "pop_diagnostics_2planet",       # two-companion prior distributions
        "companion_gallery",             # sample on-sky orbits per population
        "simulated_planets_mass_period",  # mass vs. period, coloured by alpha
    ),
    formats=("pdf", "png"),
    png_dpi=300,
    usetex=True,                 # set False if the TeX install is unavailable
    font_family="serif",
    serif_font="Computer Modern",

    # palette: blue = random (unbiased prior), rose = high-SNR
    random_color="#050CDB",
    high_snr_color="#DC144D",
    inner_color="#01019D",
    outer_color="#BB3DF1",
    ink_color="#1a1a1a",
    control_color="#D9DEE3",
    funnel_color="#C4D2DE",
    parent_color="#A7BFD8",
    schematic_random_color="#BBC0F0",
    schematic_high_snr_color="#F3B9C6",

    # sky map (the paper uses the equatorial panel)
    sky_frames=("equatorial", "ecliptic"),
    skymap_figsize=(10.0, 6.0),
    transit_vmin=0.0,
    transit_vmax=200.0,
    distance_vmax_pc=250.0,
    star_cmap_clip=0.1,
    mass_marker_floor=1.5,
    mass_marker_scale=8.0,
    mass_legend_msun=(0.1, 0.5, 1.0, 2.0),

    # gallery
    gallery_n_per_row=10,
    gallery_seed=18,

    # schematic: the two-companion high-SNR shortfall, split into
    # (no stable pair in the retry budget, no companion above the S/N floor)
    schematic_two_companion_drop_split=(1, 1),
    mars_mass_mjup=MARS_IN_MJUP,     # quotes the mass prior in Mars masses
)

# --- seeds ----------------------------------------------------------------
# Per-population and per-system streams are derived from these by hashing, so
# a rerun of any subset reproduces exactly the same numbers.
SEEDS = Seeds(planets=42, astrometry=45)

# --- populations ----------------------------------------------------------
POPULATIONS = (
    PopulationSpec("0_companion", "companion-free control",
                   n_companions=0, filter_snr=False, seed_key=None),
    PopulationSpec("1_companion_agnostic", "one companion, random",
                   n_companions=1, filter_snr=False, seed_key="1planet_nosnrfilter"),
    PopulationSpec("1_companion_detectable", "one companion, high-SNR",
                   n_companions=1, filter_snr=True, seed_key="1planet_snrfilter"),
    PopulationSpec("2_companion_agnostic", "two companions, random",
                   n_companions=2, filter_snr=False, seed_key="2planet_nosnrfilter"),
    PopulationSpec("2_companion_detectable", "two companions, high-SNR",
                   n_companions=2, filter_snr=True, seed_key="2planet_snrfilter"),
)

# Pipeline stages, in execution order. `figures` runs straight after `planets`
# because it needs only stages 1-2, so the paper figures land without waiting on
# the hours of epoch simulation. `repack` is opt-in.
STAGES = ("stars", "planets", "figures", "astrometry", "export", "repack")
DEFAULT_STAGES = tuple(stage for stage in STAGES
                       if stage != "repack" and (stage != "figures" or MAKE_FIGURES))

# ==========================================================================
# END OF CONFIGURATION
# ==========================================================================


def build_config(output_root: Path) -> CatalogConfig:
    """Assemble the configuration above into one object for the stages."""
    data_dir = output_root / "data"
    return CatalogConfig(
        paths=Paths(
            g23h_sample=G23H_SAMPLE,
            scanlaw_dr4=SCANLAW_DR4,
            pecaut_mamajek=PECAUT_MAMAJEK,
            gost_fov_map=GOST_FOV_MAP,
            data_dir=data_dir,
            figure_dir=output_root / "figures",
            stars_csv=data_dir / "stars.csv",
            epochs_dir=data_dir / "simulated_astrometry",
            injected_solutions_csv=data_dir / "injected_solutions_all.csv",
            bundle_h5=data_dir / "simulated_astrometry_bundle.h5",
        ),
        stars=STAR_SELECTION,
        priors=PLANET_PRIORS,
        astrometry=ASTROMETRY,
        figures=FIGURES,
        seeds=SEEDS,
        populations=POPULATIONS,
    )


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def format_duration(seconds):
    """Compact, readable duration: '42 s', '7 min 12 s', '2 h 05 min'."""
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(int(round(seconds)), 60)
    if minutes < 90:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


class WallClock:
    """Times the run: one entry per stage, plus per-population entries inside.

    A full generation is hours long and unevenly distributed across stages, so
    the summary at the end is what tells you where the time actually went (and
    what to expect the next time you rerun one population).
    """

    def __init__(self):
        self.start_wall = datetime.now()
        self.start = time.perf_counter()
        self.stages = []          # [{"name", "seconds", "items": [(name, seconds)]}]
        self._current = None

    @contextmanager
    def stage(self, name):
        entry = {"name": name, "seconds": 0.0, "items": []}
        self.stages.append(entry)
        previous, self._current = self._current, entry
        started = time.perf_counter()
        try:
            yield entry
        finally:
            entry["seconds"] = time.perf_counter() - started
            self._current = previous
            print(f"\n  -- {name} took {format_duration(entry['seconds'])}")

    @contextmanager
    def item(self, name):
        """Time one unit of work inside the current stage (a population, say)."""
        started = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            if self._current is not None:
                self._current["items"].append((name, seconds))
            print(f"  -- {name} took {format_duration(seconds)}")

    @property
    def elapsed(self):
        return time.perf_counter() - self.start

    def report(self):
        """Print the timing summary."""
        finished = datetime.now()
        banner("WALL CLOCK")
        print(f"  started   {self.start_wall:%Y-%m-%d %H:%M:%S}")
        print(f"  finished  {finished:%Y-%m-%d %H:%M:%S}")
        print("  " + "-" * 62)
        for entry in self.stages:
            print(f"  {entry['name']:<42s}{format_duration(entry['seconds']):>18s}")
            for name, seconds in entry["items"]:
                share = f"{100 * seconds / entry['seconds']:.0f}%" if entry["seconds"] else ""
                print(f"      {name:<34s}{format_duration(seconds):>16s}  {share:>4s}")
        print("  " + "-" * 62)
        print(f"  {'total':<42s}{format_duration(self.elapsed):>18s}")

    def write_csv(self, path):
        """Save the same numbers as one row per stage/item, for the record."""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["started", "stage", "item", "seconds"])
            stamp = f"{self.start_wall:%Y-%m-%d %H:%M:%S}"
            for entry in self.stages:
                writer.writerow([stamp, entry["name"], "", f"{entry['seconds']:.3f}"])
                for name, seconds in entry["items"]:
                    writer.writerow([stamp, entry["name"], name, f"{seconds:.3f}"])
            writer.writerow([stamp, "total", "", f"{self.elapsed:.3f}"])
        print(f"  timing written to {path}")


def run(config: CatalogConfig, stages, populations, *, overwrite, limit,
        figures=None, timing_csv=None):
    """Run the requested stages over the requested populations."""
    from pipeline import astrometry as astro_stage
    from pipeline import planets as planet_stage
    from pipeline import stars as star_stage

    clock = WallClock()

    if "stars" in stages:
        banner("STAGE  stars -- parent stellar sample")
        with clock.stage("stars"):
            star_stage.build_star_catalog(config, overwrite=overwrite)

    if "planets" in stages:
        banner("STAGE  planets -- companion populations")
        with clock.stage("planets"):
            for spec in populations:
                if spec.n_companions == 0:
                    print(f"\n[{spec.name}] control population, no companions to draw")
                    continue
                print(f"\n[{spec.name}] {spec.label}")
                with clock.item(spec.name):
                    planet_stage.generate_population(config, spec,
                                                     overwrite=overwrite, limit=limit)

    if "figures" in stages:
        banner("STAGE  figures -- catalog-generation figures")
        from pipeline import figures as figure_stage
        with clock.stage("figures"):
            figure_stage.make_figures(config, figures)

    if "astrometry" in stages:
        banner("STAGE  astrometry -- per-epoch along-scan measurements")
        with clock.stage("astrometry"):
            scanlaw_df, sources_df = astro_stage.load_reference_tables(config)
            for spec in populations:
                print(f"\n[{spec.name}] {spec.label}")
                with clock.item(spec.name):
                    astro_stage.simulate_population(config, spec, scanlaw_df=scanlaw_df,
                                                    sources_df=sources_df, limit=limit)
            print("\ncombining truth tables")
            with clock.item("combine truth tables"):
                astro_stage.combine_truth_tables(config, config.populations)

    if "export" in stages:
        banner("STAGE  export -- one HDF5 file per population")
        with clock.stage("export"):
            for spec in populations:
                print(f"\n[{spec.name}] {spec.label}")
                with clock.item(spec.name):
                    astro_stage.export_population_hdf5(config, spec, overwrite=overwrite)

    if "repack" in stages:
        banner("OPTIONAL  repack -- single shareable bundle")
        import subprocess
        script = Path(__file__).resolve().parent / "repack_simulated_astrometry.py"
        with clock.stage("repack"):
            subprocess.run([sys.executable, str(script), "--out",
                            str(config.paths.bundle_h5)], check=True)

    clock.report()
    if timing_csv:
        clock.write_csv(timing_csv)
    return clock


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stages", nargs="+", choices=STAGES,
                        default=list(DEFAULT_STAGES),
                        help="pipeline stages to run (default: all but repack)")
    parser.add_argument("--populations", nargs="+",
                        choices=[spec.name for spec in POPULATIONS],
                        help="populations to generate (default: all)")
    parser.add_argument("--figures", nargs="+", choices=FIGURES.figures,
                        help="figures to build in the figures stage (default: all)")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip the figures stage even if it was requested")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT,
                        help="write products under this directory instead of outputs/")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing products instead of reusing/refusing")
    parser.add_argument("--limit", type=int,
                        help="cap the number of stars/systems per population "
                             "(smoke tests only -- it changes the catalog)")
    parser.add_argument("--timing-csv", type=Path, metavar="PATH",
                        help="also write the wall-clock breakdown to this CSV")
    parser.add_argument("--list", action="store_true",
                        help="print the plan and exit without running anything")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args.output_root.resolve())
    populations = config.select(args.populations)

    stages = list(args.stages)
    if args.no_figures and "figures" in stages:
        stages.remove("figures")
        print("note: --no-figures, skipping the figures stage")

    print(f"repo root     : {ROOT}")
    print(f"inputs        : {DATA_DIR_IN}")
    print(f"outputs       : {args.output_root}")
    print(f"stages        : {', '.join(stages)}")
    print(f"populations   : {', '.join(spec.name for spec in populations)}")
    if "figures" in stages:
        print(f"figures       : {', '.join(args.figures or FIGURES.figures)}")
    print(f"seeds         : planets={SEEDS.planets}, astrometry={SEEDS.astrometry}")
    if args.limit:
        print(f"LIMIT         : {args.limit} (smoke test -- not a release catalog)")

    if args.list:
        return 0

    missing = [path for path in (G23H_SAMPLE, SCANLAW_DR4, PECAUT_MAMAJEK)
               if not path.exists()]
    if missing:
        raise SystemExit("missing input files:\n  " + "\n  ".join(str(p) for p in missing))

    run(config, set(stages), populations, overwrite=args.overwrite,
        limit=args.limit, figures=args.figures, timing_csv=args.timing_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
