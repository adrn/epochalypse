"""Physical and mission constants, derived from astropy rather than typed in.

A copy of the repository's `src/epochalypse_constants.py`, restricted to the
seven values this package actually uses. It is duplicated rather than imported
so that `periodograms/` is a standalone tree that can be dropped into the
cluster on its own; the expressions are the same ones, so the two cannot drift
in the fifth decimal.

The printed values are astropy's at the time of writing -- comments, not
definitions. The expressions are what run.
"""
from __future__ import annotations

import astropy.units as u

# --------------------------------------------------------------------------
# Mass conversions
# --------------------------------------------------------------------------
MJUP_IN_MSUN = float((1 * u.M_jup).to(u.M_sun).value)        # 9.545942e-04

# --------------------------------------------------------------------------
# Length and time conversions
# --------------------------------------------------------------------------
RSUN_IN_AU = float((1 * u.R_sun).to(u.au).value)             # 4.650467e-03
DAYS_PER_YEAR = float((1 * u.yr).to(u.day).value)            # 365.25 (Julian)

# --------------------------------------------------------------------------
# Gaia mission parameters
# --------------------------------------------------------------------------
# DR4 reference epoch: epoch astrometry is centred here (JD, TCB). Trial
# periods are in years because `epoch_arrays` divides the offset from this
# epoch by DAYS_PER_YEAR.
GAIA_EPOCH_TCB_JD = 2457936.875

# DR4 observing baseline, ~66 months. The dashed line in every figure, and the
# threshold in `period_reliable`.
DR4_BASELINE_YEARS = 5.5
