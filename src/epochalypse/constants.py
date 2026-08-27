"""Physical and mission constants, defined once for the whole repository.

Everything physical is derived from `astropy.constants` / `astropy.units` rather
than typed in, so there is a single authority and no room for two call sites to
disagree in the fifth decimal. The catalog-generation pipeline
(`scripts/`) and the analysis code (`src/`) both import from here.

The printed values are the astropy ones at the time of writing; they are
comments, not definitions -- the expressions are what run.
"""

import astropy.units as u

# --------------------------------------------------------------------------
# Mass conversions
# --------------------------------------------------------------------------
MJUP_IN_MSUN = float((1 * u.M_jup).to(u.M_sun).value)  # 9.545942e-04
MEARTH_IN_MSUN = float((1 * u.M_earth).to(u.M_sun).value)  # 3.003489e-06
MEARTH_IN_MJUP = float((1 * u.M_earth).to(u.M_jup).value)  # 3.146553e-03

# astropy carries no Mars mass, so the one non-astropy number here is the
# Mars/Earth mass ratio (NASA planetary fact sheet, 6.417e23 / 5.972e24).
MARS_IN_MEARTH = 0.107
MARS_IN_MJUP = MARS_IN_MEARTH * MEARTH_IN_MJUP  # 3.366759e-04
MARS_IN_MSUN = MARS_IN_MEARTH * MEARTH_IN_MSUN  # 3.213733e-07

# --------------------------------------------------------------------------
# Length and time conversions
# --------------------------------------------------------------------------
RSUN_IN_AU = float((1 * u.R_sun).to(u.au).value)  # 4.650467e-03
DAYS_PER_YEAR = float((1 * u.yr).to(u.day).value)  # 365.25 (Julian)

# --------------------------------------------------------------------------
# Gaia mission parameters
# --------------------------------------------------------------------------
# DR4 reference epoch: epoch astrometry is centred here (JD, TCB).
GAIA_EPOCH_TCB_JD = 2457936.875
# DR4 observing baseline, ~66 months. Sets a_crit in the detectability metric
# and the baseline marker in the figures.
DR4_BASELINE_YEARS = 5.5

# --------------------------------------------------------------------------
# Companion-mass prior bounds, in solar masses
# --------------------------------------------------------------------------
# The upper bound is the hydrogen-burning limit used throughout the catalog;
# `epochalypse.config` states the same bound in Jupiter masses.
MAX_COMPANION_MASS_MJUP = 80.0
MAX_COMPANION_MASS_MSUN = MAX_COMPANION_MASS_MJUP * MJUP_IN_MSUN

# --------------------------------------------------------------------------
# Backwards-compatible aliases
# --------------------------------------------------------------------------
# The analysis modules and notebooks refer to these older names.
MJUP_TO_MSUN = MJUP_IN_MSUN
MARS_TO_MSUN = MARS_IN_MSUN
RSUN_TO_AU = RSUN_IN_AU
