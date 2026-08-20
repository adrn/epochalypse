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

    Populations are of two kinds:

    * **generated** (``derive_from is None``) -- drawn and simulated. All three
      are drawn from the unbiased prior; there is no detectability rejection.
    * **derived** (``derive_from`` names a generated population) -- a view of a
      generated population, selected after the fact as its top
      ``high_snr_fraction`` by recorded ``SNR_tot``. Nothing is simulated for a
      derived population; it costs one quantile of a column.

    That split is the point of the redesign: at ~4 million stars, rejection
    sampling to a fixed S/N threshold is both expensive and irreversible, while
    a quantile selection is cheap and leaves the threshold an analysis choice.
    """

    name: str
    label: str
    n_companions: int
    derive_from: str | None = None
    high_snr_fraction: float | None = None

    @property
    def is_generated(self) -> bool:
        return self.derive_from is None

    @property
    def csv_name(self) -> str:
        return f"{self.name}.csv"


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
    stars_csv: Path              # stage 1 product (parent stellar sample)
    epochs_dir: Path             # sharded epoch tables, one subdir per population
    injected_solutions_csv: Path  # truth table across all populations
    bundle_h5: Path              # optional single-file repack

    # per-source lookup indices, built once and memory-mapped by every worker
    # (a 4-million-star scan law is far too large to load per process)
    index_dir: Path

    def systems_h5(self, population: str) -> Path:
        return self.data_dir / f"simulated_astrometry_{population}_systems.h5"

    def population_csv(self, population: str) -> Path:
        return self.data_dir / f"{population}.csv"

    def shard_dir(self, population: str) -> Path:
        return self.epochs_dir / population

    def shard_epochs(self, population: str, shard: int, n_shards: int) -> Path:
        return self.shard_dir(population) / f"epochs_{shard:05d}_of_{n_shards:05d}.parquet"

    def shard_truths(self, population: str, shard: int, n_shards: int) -> Path:
        return self.shard_dir(population) / f"truths_{shard:05d}_of_{n_shards:05d}.parquet"


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

    # --- detectability metric (recorded per companion, never used to reject) ---
    # SNR_tot = sqrt(N_DR4) * (alpha / sigma_single) / (1 + (P/T_baseline)^2)
    # is written to the catalog so a high-S/N sample can be selected downstream.
    baseline_years: float         # DR4 observing baseline; sets a_crit
    # proposals per batch in the (Roche / stability) rejection loop
    draw_batch: int
    max_draws: int                # give up on a star after this many proposals

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
    # fraction of a random population kept as its high-SNR view; quoted in the
    # schematic so the figure states how the selection was made
    high_snr_fraction: float
    # Mars mass in Jupiter masses, so the schematic can quote the bottom of the
    # mass prior in Mars masses the way the paper does.
    mars_mass_mjup: float


# --------------------------------------------------------------------------
# Parallel execution
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ShardingSettings:
    """How the source list is split across independent workers.

    A source is assigned to a shard by hashing its Gaia source id, so the
    partition is a pure function of the id: no worker needs to know the global
    ordering, shards can be run in any order on any machine, and re-running one
    shard reproduces it exactly.
    """

    n_shards: int                # default partition count
    compression: str             # parquet codec, e.g. "zstd"
    # flush the buffer every N systems so a worker's peak memory stays bounded
    # no matter how many sources land in its shard
    flush_every: int


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Seeds:
    """Master seeds.

    Every per-system stream is derived as
    ``blake2s(master : population : gaia_source_id)`` -- keyed on the *source
    id*, never on a row index. That is what makes the pipeline parallelizable:
    a star's companions and noise realization depend only on its own id, so any
    subset of stars can be generated in any order, split across any number of
    workers, and re-run individually, all reproducing the same catalog."""

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
    sharding: ShardingSettings
    seeds: Seeds
    populations: tuple[PopulationSpec, ...] = field(default_factory=tuple)

    def population(self, name: str) -> PopulationSpec:
        for spec in self.populations:
            if spec.name == name:
                return spec
        known = ", ".join(spec.name for spec in self.populations)
        raise KeyError(f"unknown population {name!r}; configured: {known}")

    @property
    def generated(self) -> tuple[PopulationSpec, ...]:
        """Populations that are actually drawn and simulated."""
        return tuple(spec for spec in self.populations if spec.is_generated)

    @property
    def derived(self) -> tuple[PopulationSpec, ...]:
        """Populations that are a selection over a generated one."""
        return tuple(spec for spec in self.populations if not spec.is_generated)

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
