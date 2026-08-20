"""Configuration schema for the epochalypse catalog-generation pipeline.

This module defines the *shape* of the configuration only -- no scientific
choices live here. Every value (paths, priors, thresholds, seeds) is spelled out
at the top of `generate_catalog.py`, so that one screen shows exactly what
catalog a run will produce.

The dataclass fields deliberately carry no defaults: a missing choice is a
`TypeError` at construction rather than a silent fallback buried in a module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------
# Populations
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PopulationSpec:
    """One population of simulated systems.

    name
        On-disk key, e.g. ``1_companion_detectable``. Used for the population
        CSV, the per-system epoch directory, and the exported HDF5 file.
    label
        Display label used in prose/figures ("random" / "high-SNR"), kept
        separate from `name` because the two vocabularies differ.
    n_companions
        0, 1, or 2. The 0-companion control is drawn straight from the stellar
        catalog and has no population CSV of its own.
    filter_snr
        If True, companions are rejection-sampled until the period-suppressed
        *total* detectability S/N clears `PlanetPriors.snr_total_min`.
    seed_key
        String hashed with the master seed to derive this population's RNG
        stream. Kept at the historical `<n>planet_<snrfilter|nosnrfilter>`
        spelling so regenerating reproduces the released draws exactly; the
        population rename was a rename, not a resimulation.
    """

    name: str
    label: str
    n_companions: int
    filter_snr: bool
    seed_key: str | None

    @property
    def csv_name(self) -> str:
        return f"{self.name}.csv"


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    """Every file the pipeline reads or writes."""

    # inputs (static, not produced by this repo)
    g23h_sample: Path            # parent sample, arrow
    scanlaw_dr4: Path            # DR4 scan law, one row per FoV transit, arrow
    pecaut_mamajek: Path         # Pecaut & Mamajek (2013) main-sequence table
    gost_fov_map: Path           # GOST DR4 FoV-transit healpix map, figures only

    # outputs
    data_dir: Path               # outputs/data
    figure_dir: Path             # outputs/figures
    stars_csv: Path              # stage 1 product
    epochs_dir: Path             # per-system epoch CSVs, one subdir per population
    injected_solutions_csv: Path  # truth table across all populations
    bundle_h5: Path              # optional single-file repack

    def systems_h5(self, population: str) -> Path:
        return self.data_dir / f"simulated_astrometry_{population}_systems.h5"

    def population_csv(self, population: str) -> Path:
        return self.data_dir / f"{population}.csv"


# --------------------------------------------------------------------------
# Stellar sample
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StarSelection:
    """Choices made when building the parent stellar sample."""

    parallax_col: str
    gmag_col: str
    source_id_col: str
    # Interpolating mass/radius off the Pecaut & Mamajek sequence in absolute G;
    # stars outside the table's M_G range are dropped rather than extrapolated.
    require_mass_radius: bool
    # A handful of high-RUWE binaries carry no per-CCD AL noise calibration
    # (sig_AL is NaN) and so have no usable noise model. Dropping them here
    # keeps one clean parent sample shared by every population.
    require_sigma_al: bool


# --------------------------------------------------------------------------
# Companion priors
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanetPriors:
    """Every choice about the injected companions.

    Semi-major axis and mass are log-uniform; eccentricity uniform; orbits
    isotropic (uniform in cos i, with the nodes/arguments/mean anomalies
    uniform in angle).
    """

    # --- semi-major axis: log-uniform in [a_min, a_max] AU ---
    a_min_au: float
    a_max_au: float

    # --- innermost separation: the star must fit inside its own Roche lobe ---
    # A configuration with R_star > R_L,star is a contact/mass-transferring
    # binary, not a star with a companion, so it is rejected as impossible:
    #     a > factor * R_star / ell(M_star/M_p),  ell from Eggleton (1983).
    # The floor works out to 1.2-2.6 R_star across this catalog's mass ratios.
    # It needs no companion radius, so unlike the classical (companion-side)
    # Roche limit it imports no mass-radius model. Being a function of both
    # R_star and the mass ratio, it cuts each system at a different separation,
    # which smears the inner edge of the population instead of stacking it into
    # a vertical line at a_min_au.
    enforce_roche_lobe: bool
    roche_lobe_safety_factor: float   # 1.0 = the bare lobe-filling limit

    # --- companion mass: log-uniform in [m_min, m_max] Jupiter masses ---
    mass_min_mjup: float
    mass_max_mjup: float

    # --- eccentricity: uniform in [e_min, e_max] ---
    ecc_min: float
    ecc_max: float

    # --- angles ---
    isotropic_inclination: bool   # uniform in cos i over [-1, 1]
    angle_min_deg: float          # Omega, omega, mean anomaly: uniform
    angle_max_deg: float
    # In two-companion systems, a coin flip decides whether the pair is
    # coplanar (shared inclination and ascending node) or drawn independently.
    coplanar_probability: float

    # --- detectability (used only when a population sets filter_snr) ---
    baseline_years: float         # DR4 observing baseline; sets a_crit
    snr_total_min: float          # keep companions with SNR_total >= this
    snr_draw_batch: int           # proposals per rejection-sampling batch
    snr_max_draws: int            # give up on a star after this many proposals

    # --- two-companion stability screen ---
    hill_stability_factor: float  # unstable if delta < factor * sqrt(3)
    resonance_orders: tuple[int, ...]   # first-order (j+1):j resonances checked
    resonance_tolerance: float          # fractional tolerance on P2/P1
    max_stability_retries: int          # attempts before a star is skipped

    # Msun per Mjup, used for the astrometric signature alpha, the total system
    # mass in Kepler's third law, and the mass ratios in the stability screen.
    # One value for all three (see `epochalypse_constants`).
    mjup_in_msun: float


# --------------------------------------------------------------------------
# Epoch astrometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AstrometrySettings:
    """Choices made when simulating the per-epoch along-scan measurements."""

    gaia_epoch_tcb_jd: float      # time origin; epochs are centred on this
    days_per_year: float
    # JAX must run in float64: the random draws (and hence the injected noise
    # realization) differ from the released catalog at float32 precision.
    enable_float64: bool
    rsun_to_au: float
    mjup_to_msun: float           # population CSV (Mjup) -> simulator (Msun)
    # Degrees of freedom in the Gaia astrometric solution: 5 for a
    # five-parameter solution (astrometric_params_solved_dr3 == 31), else 6.
    n_dof_five_param: int
    n_dof_other: int
    params_solved_five_param: int
    # Per-epoch uncertainties get a shared multiplicative jitter,
    # 1 + noise_jitter_frac * N(0, 1), applied to both sigma_UEVA and the
    # reported sigma so the two stay consistent epoch by epoch.
    noise_jitter_frac: float
    # Optional constant offsets of the reference position [mas].
    alpha0_mas: float
    delta0_mas: float
    # HDF5 export
    hdf5_compression: str


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FigureSettings:
    """Everything the catalog-generation figures decide for themselves.

    Numbers quoted in annotations (sample counts, prior ranges, the S/N floor,
    the coplanar fraction) are not configured here: they are read from the
    priors and the catalogs on disk so they cannot drift out of date.
    """

    figures: tuple[str, ...]     # which figures to build, in order
    formats: tuple[str, ...]     # file suffixes, e.g. ("pdf", "png")
    png_dpi: int
    usetex: bool                 # LaTeX typesetting; needs a working TeX install
    font_family: str
    serif_font: str

    # --- palette ---
    random_color: str            # the random (unbiased-prior) populations
    high_snr_color: str          # the high-SNR (detectability-filtered) ones
    inner_color: str             # inner companion, gallery
    outer_color: str             # outer companion, gallery
    ink_color: str               # schematic text/arrows
    control_color: str           # schematic: companion-free control box
    funnel_color: str            # schematic: selection-funnel boxes
    parent_color: str            # schematic: parent-sample box
    schematic_random_color: str  # schematic box fills, lighter than the series
    schematic_high_snr_color: str

    # --- sky map ---
    sky_frames: tuple[str, ...]  # display frames: "equatorial" and/or "ecliptic"
    skymap_figsize: tuple[float, float]
    transit_vmin: float          # FoV-transit colour scale
    transit_vmax: float
    distance_vmax_pc: float      # distance colour scale
    star_cmap_clip: float        # clip this fraction off plasma's dark end
    mass_marker_floor: float     # marker area = floor + scale * (mass / Msun)
    mass_marker_scale: float
    mass_legend_msun: tuple[float, ...]

    # --- gallery ---
    gallery_n_per_row: int
    gallery_seed: int            # sampling seed; figure-only, not the catalog's

    # --- schematic ---
    # The two-companion high-SNR shortfall splits into "no stable pair in the
    # retry budget" and "no companion clears the S/N floor". That split is only
    # in the generation log, not the CSVs, so it is stated here; if it stops
    # matching the catalog the figure falls back to a combined count.
    schematic_two_companion_drop_split: tuple[int, int]
    # Mars mass in Jupiter masses, so the schematic can quote the bottom of the
    # mass prior in Mars masses the way the paper does.
    mars_mass_mjup: float


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Seeds:
    """Master seeds. Per-population and per-system streams are derived from
    these by hashing (blake2s), so they are stable across processes and
    independent of Python's hash randomization."""

    planets: int
    astrometry: int


# --------------------------------------------------------------------------
# Everything together
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CatalogConfig:
    paths: Paths
    stars: StarSelection
    priors: PlanetPriors
    astrometry: AstrometrySettings
    figures: FigureSettings
    seeds: Seeds
    populations: tuple[PopulationSpec, ...] = field(default_factory=tuple)

    def population(self, name: str) -> PopulationSpec:
        for spec in self.populations:
            if spec.name == name:
                return spec
        known = ", ".join(spec.name for spec in self.populations)
        raise KeyError(f"unknown population {name!r}; configured: {known}")

    def select(self, names: list[str] | None) -> tuple[PopulationSpec, ...]:
        """Populations to run: all of them, or the named subset in config order."""
        if not names:
            return self.populations
        wanted = set(names)
        unknown = wanted.difference(spec.name for spec in self.populations)
        if unknown:
            known = ", ".join(spec.name for spec in self.populations)
            raise KeyError(f"unknown population(s): {sorted(unknown)}; configured: {known}")
        return tuple(spec for spec in self.populations if spec.name in wanted)
