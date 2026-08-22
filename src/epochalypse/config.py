"""Every choice the catalog depends on -- one screen, no indirection.

Paths, priors, thresholds, seeds, and figure settings are plain module
constants. Read them as `config.A_MIN_AU`; there is nothing to construct and
nothing to pass down. The output paths are functions because `--output-root`
can move them; everything else is a value.

Physical constants come from `epochalypse.constants` (astropy), never typed
in here, so no two call sites can disagree in the fifth decimal.
"""
from __future__ import annotations

from pathlib import Path

from .constants import (DAYS_PER_YEAR, DR4_BASELINE_YEARS, GAIA_EPOCH_TCB_JD,
                        MARS_IN_MJUP, MAX_COMPANION_MASS_MJUP, MJUP_IN_MSUN,
                        RSUN_IN_AU)

# The repo root: src/epochalypse/config.py -> ../../ . Inputs and outputs are
# resolved relative to it, so an editable checkout finds data/ and outputs/
# wherever it is. `--output-root` overrides the output half.
ROOT = Path(__file__).resolve().parents[2]

# ==========================================================================
# Inputs -- static, not produced here
# ==========================================================================
DATA_IN = ROOT / "data"
G23H_SAMPLE = DATA_IN / "g23h_epochalypse_stars" / "G23H_within_250pc.arrow"
SCANLAW_DR4 = (DATA_IN / "g23h_epochalypse_stars"
               / "scanlaw_dr4_within_250pc_hpx64_transit_loss10.arrow")
# The 16k-star G23H_sample_subset.arrow is the committed smoke-test sample;
# point G23H_SAMPLE at it to exercise the pipeline without the full inputs.
PECAUT_MAMAJEK = DATA_IN / "pecaut_mamajek.txt"
GOST_FOV_MAP = DATA_IN / "gost_fov_counts_dr4.fits"      # sky-map figure only

# ==========================================================================
# Outputs
# ==========================================================================
OUTPUT_ROOT = ROOT / "outputs"


def set_output_root(path) -> None:
    """Point every output path somewhere else (`--output-root`, smoke tests).

    Called once per process before any work; each MPI rank parses its own
    argv, so there is no shared state to keep in step.
    """
    global OUTPUT_ROOT
    OUTPUT_ROOT = Path(path).resolve()


def data_dir():   return OUTPUT_ROOT / "data"
def figure_dir(): return OUTPUT_ROOT / "figures"
def stars_csv():  return data_dir() / "stars.csv"
def index_dir():  return data_dir() / "index"
def skipped_dir(): return data_dir() / "skipped"


def shard_dir(population):
    return data_dir() / "simulated_astrometry" / population


def shard_epochs(population, rank, n_ranks):
    return shard_dir(population) / f"epochs_rank{rank:05d}_of_{n_ranks:05d}.parquet"


def shard_truths(population, rank, n_ranks):
    return shard_dir(population) / f"truths_rank{rank:05d}_of_{n_ranks:05d}.parquet"


def truths(population, high_snr=False):
    """The merged one-row-per-system truth table for a population."""
    suffix = "_high_snr" if high_snr else ""
    return data_dir() / f"injected_solutions_{population}{suffix}.parquet"


# ==========================================================================
# Populations
# ==========================================================================
# Simulated populations: name -> number of injected companions. All three are
# drawn from the unbiased prior; there is no detectability rejection.
POPULATIONS = {"0_companion": 0, "1_companion": 1, "2_companion": 2}

# The high-SNR sample is not generated. It is the top slice of a random
# population by recorded SNR_tot, so re-selecting costs seconds and the
# threshold stays an analysis choice rather than being baked into the data.
HIGH_SNR_FRACTION = 0.01

# Figure panels: (population, high-SNR?, label). The companion-free control
# has nothing to plot.
PANELS = (("1_companion", False, "one companion, random"),
          ("2_companion", False, "two companions, random"),
          ("1_companion", True, "one companion, high-SNR"),
          ("2_companion", True, "two companions, high-SNR"))

# ==========================================================================
# Seeds
# ==========================================================================
# Every per-system stream is blake2s(master : population : gaia_source_id),
# keyed on the *source id*, never on a row index. That is what makes the
# pipeline parallelizable: a star's companions and noise realization depend
# only on its own id, so any subset can run in any order on any number of
# ranks and reproduce the same catalog.
SEED_PLANETS = 42
SEED_ASTROMETRY = 45

# ==========================================================================
# Stellar sample
# ==========================================================================
PARALLAX_COL = "parallax"
GMAG_COL = "phot_g_mean_mag_dr3"
SOURCE_ID_COL = "gaia_source_id"

# ==========================================================================
# Companion priors
# ==========================================================================
# Semi-major axis: log-uniform. The floor is deliberately below anything
# physical -- the binding inner limit is the per-star Roche-lobe screen below,
# which cuts each system at its own separation and so leaves a smeared inner
# edge rather than a wall at A_MIN_AU.
A_MIN_AU = 0.001
A_MAX_AU = 100.0

# Innermost separation: the star must fit inside its own Roche lobe, else the
# configuration is a contact binary rather than a star with a companion:
#     a > R_star / ell(M_star/M_p),  ell from Eggleton (1983).
# Works out to 1.2-2.6 R_star across this catalog's mass ratios, and needs no
# companion radius, so it imports no mass-radius model.
ROCHE_SAFETY_FACTOR = 1.0        # 1.0 = the bare lobe-filling limit

# Companion mass: log-uniform, Mars mass to the hydrogen-burning limit.
MASS_MIN_MJUP = MARS_IN_MJUP              # 1 M_Mars = 3.3668e-04 M_Jup
MASS_MAX_MJUP = MAX_COMPANION_MASS_MJUP   # 80 M_Jup

# Eccentricity: uniform. Angles: isotropic orbits (uniform in cos i, with the
# nodes, arguments, and mean anomalies uniform over the full circle).
ECC_MIN = 0.0
ECC_MAX = 0.99

# In two-companion systems, a coin flip decides whether the pair is coplanar
# (shared inclination and ascending node) or drawn independently.
COPLANAR_PROBABILITY = 0.5

# Detectability metric, recorded per companion and never used to reject:
#   SNR_tot = sqrt(N_DR4) * (alpha / sigma_single) / (1 + (a/a_crit)^3)
BASELINE_YEARS = DR4_BASELINE_YEARS      # 5.5 yr; sets a_crit

# Two-companion stability screen.
HILL_STABILITY_FACTOR = 2.0        # unstable if delta < 2 sqrt(3) Hill radii
RESONANCE_ORDERS = (1, 2)          # check the 2:1 and 3:2 commensurabilities
RESONANCE_TOLERANCE = 0.05         # within 5% in period ratio counts as near
MAX_STABILITY_RETRIES = 1000       # attempts before a star is skipped

# ==========================================================================
# Epoch astrometry
# ==========================================================================
# Degrees of freedom in the Gaia astrometric solution: 5 for a five-parameter
# solution (astrometric_params_solved_dr3 == 31), else 6.
N_DOF_FIVE_PARAM = 5
N_DOF_OTHER = 6
PARAMS_SOLVED_FIVE_PARAM = 31

# Per-epoch uncertainties get a shared multiplicative jitter,
# 1 + NOISE_JITTER_FRAC * N(0, 1), applied to both sigma_UEVA and the reported
# sigma so the two stay consistent epoch by epoch.
NOISE_JITTER_FRAC = 0.1

# ==========================================================================
# Parallel output
# ==========================================================================
PARQUET_COMPRESSION = "zstd"
FLUSH_EVERY = 2000        # systems buffered before a parquet row-group flush

# ==========================================================================
# Figures
# ==========================================================================
FIGURES = (
    "star_sky_scanlaw",               # parent sample over the DR4 scan law
    "population_schematic",          # selection funnel + population branching
    "pop_diagnostics_1planet",       # one-companion: random vs high-SNR
    "pop_diagnostics_2planet",       # two-companion: random vs high-SNR
    "companion_gallery",             # sample on-sky orbits per population
    "simulated_planets_mass_period",  # mass vs. period, coloured by alpha
)
FORMATS = ("pdf", "png")
PNG_DPI = 300
USETEX = True                # set False if the TeX install is unavailable
FONT_FAMILY = "serif"
SERIF_FONT = "Computer Modern"

# palette: blue = random (unbiased prior), rose = high-SNR
RANDOM_COLOR = "#050CDB"
HIGH_SNR_COLOR = "#DC144D"
INNER_COLOR = "#01019D"          # inner companion, gallery
OUTER_COLOR = "#BB3DF1"          # outer companion, gallery
INK_COLOR = "#1a1a1a"            # schematic text/arrows
CONTROL_COLOR = "#D9DEE3"        # schematic: companion-free control box
FUNNEL_COLOR = "#C4D2DE"         # schematic: selection-funnel boxes
PARENT_COLOR = "#A7BFD8"         # schematic: parent-sample box
SCHEMATIC_RANDOM_COLOR = "#BBC0F0"
SCHEMATIC_HIGH_SNR_COLOR = "#F3B9C6"

# sky map (the paper uses the equatorial panel)
SKY_FRAMES = ("equatorial", "ecliptic")
SKYMAP_FIGSIZE = (10.0, 6.0)
TRANSIT_VMIN = 0.0               # FoV-transit colour scale
TRANSIT_VMAX = 200.0
DISTANCE_VMAX_PC = 250.0
STAR_CMAP_CLIP = 0.1             # clip this fraction off plasma's dark end
MASS_MARKER_FLOOR = 1.5          # marker area = floor + scale * (mass / Msun)
MASS_MARKER_SCALE = 8.0
MASS_LEGEND_MSUN = (0.1, 0.5, 1.0, 2.0)

# gallery
GALLERY_N_PER_ROW = 10
GALLERY_SEED = 18                # figure-only sampling seed, not the catalog's

# Mars mass in Jupiter masses, so the schematic can quote the bottom of the
# mass prior in Mars masses the way the paper does.
MARS_MASS_MJUP = MARS_IN_MJUP
