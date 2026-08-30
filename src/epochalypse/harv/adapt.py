"""Epochalypse epoch arrays -> harv `GaiaAstrometryData`.

The one genuinely new mapping in this stage. It takes exactly what
`periodogram.shards.ShardReader` yields -- `(t, psi, pf, y, yerr)`, already
sorted by observation time, with `t` in years from the DR4 reference epoch --
so the two analysis stages read the catalog through one reader and share one
time convention. harv takes unxt quantities, so years need no conversion: the
model only ever uses `t - t_ref`, and both come from the same array.

**Padding.** harv JITs per epoch count. The catalog spans 44-298 epochs, so
without padding every distinct count is a fresh compile, 17.2 M times over. Each
system is padded up to its bucket by three changes, each necessary and none
obvious:

* padded rows get a **large finite** uncertainty, not `inf`. Infinity zeroes the
  chi^2 contribution as intended, but the Gaussian normalization carries a
  `log(sigma)` term that diverges with it, and `logZ` comes back `-inf`. A
  finite `PAD_ERR_MAS` leaves the chi^2 contribution at ~1e-12 of a real epoch
  and adds only a *constant* to every prior sample's log-likelihood -- which
  cancels exactly when the importance weights are normalized. Weights, top-K
  selection, and ESS are therefore identical to the unpadded run. `logZ_int` and
  `max_log_likelihood` are absolute, so they need `pad_log_offset` subtracted.
* `t_ref` is passed explicitly, computed from the **real** epochs. harv derives
  it as `mean(time)` when it is not given, so padding would otherwise drag the
  model's time origin and change the answer.
* the parameterization is built from the **unpadded** data, because
  `from_data` sets `a_floor = med(sigma_AL)/sqrt(N)` and padded rows would move
  both the median and N.
"""

from __future__ import annotations

import numpy as np
from unxt import Quantity as Q

from . import config as C

# Padded-row uncertainty. Large enough that a padded epoch cannot constrain
# anything (the signal is of order 1 mas and the real uncertainties are ~0.06),
# finite so the Gaussian normalization stays finite. Its only effect is a
# constant log-likelihood offset, which `pad_log_offset` names exactly.
PAD_ERR_MAS = 1.0e6


def pad_log_offset(n_epochs, n_padded):
    """The constant every log-likelihood gains from padding to `n_padded`.

    Each padded row contributes only the Gaussian normalization
    `-0.5*log(2*pi*PAD_ERR_MAS**2)` -- its chi^2 term is ~1e-12 of a real
    epoch's. That is the same constant for every prior sample, so it cancels in
    the importance weights, the top-K selection, and the ESS. It does **not**
    cancel in `logZ_int` or `max_log_likelihood`, which are absolute. Subtract
    this from both, or two systems in different epoch buckets are not comparable
    -- the buckets span 64 to 320 rows, so the offset spans ~4,000 nats.
    """
    return (int(n_padded) - int(n_epochs)) * -0.5 * np.log(2.0 * np.pi * PAD_ERR_MAS**2)


def pad_arrays(t, psi, pf, y, yerr, n_padded):
    """Pad one system's epoch arrays up to `n_padded`, neutrally.

    The padded rows repeat the first epoch's time and geometry rather than
    zeros, so nothing sits at an arbitrary time origin if a padded row is ever
    read. Their uncertainty is `PAD_ERR_MAS`, not infinity -- see the module
    docstring.
    """
    n = len(t)
    if n > n_padded:
        msg = f"{n} epochs will not fit in a {n_padded}-row bucket"
        raise ValueError(msg)
    if n == n_padded:
        return t, psi, pf, y, yerr
    pad = n_padded - n
    return (
        np.concatenate([t, np.full(pad, t[0])]),
        np.concatenate([psi, np.full(pad, psi[0])]),
        np.concatenate([pf, np.full(pad, pf[0])]),
        np.concatenate([y, np.zeros(pad)]),
        np.concatenate([yerr, np.full(pad, PAD_ERR_MAS)]),
    )


def injected_errors(yerr, sigma_single_mas):
    """The uncertainties the noise was actually drawn from.

    The generator injects at `sigma_UEVA,single` and reports `sigma_formal`,
    applying the SAME per-epoch jitter draw `xi_i` to both. Their ratio is
    therefore the constant `sigma_UEVA/sigma_formal` at every epoch, and
    `median(yerr)` estimates `sigma_formal` because `E[1 + 0.1 xi] = 1`. So the
    injected scale is recoverable exactly, epoch by epoch, from what the shard
    already stores.

    Returns `yerr` unchanged when the mode is "reported" or no scale is given.
    """
    if C.ERROR_MODE != "injected" or sigma_single_mas is None:
        return yerr
    yerr = np.asarray(yerr, dtype=np.float64)
    reported = float(np.median(yerr))
    if not np.isfinite(reported) or reported <= 0:
        return yerr
    return yerr * (float(sigma_single_mas) / reported)


def to_harv(t, psi, pf, y, yerr, *, pad=True):
    """One system's epochs as a harv `GaiaAstrometryData`.

    Arrays are what `ShardReader.iter_systems` yields. With `pad`, the system is
    padded up to its `config.bucket_for` size so the JIT cache is keyed on one
    of ~9 shapes rather than on every epoch count in the catalog.
    """
    from harv import GaiaAstrometryData

    t, psi, pf, y, yerr = (
        np.asarray(a, dtype=np.float64) for a in (t, psi, pf, y, yerr)
    )
    t_ref = float(np.mean(t))  # from the REAL epochs, before padding shifts the mean

    if pad:
        t, psi, pf, y, yerr = pad_arrays(t, psi, pf, y, yerr, C.bucket_for(len(t)))

    return GaiaAstrometryData(
        time=Q(t, "yr"),
        al_position=Q(y, "mas"),
        al_position_err=Q(yerr, "mas"),
        scan_angle=Q(psi, "rad"),
        parallax_factor=pf,
        t_ref=Q(t_ref, "yr"),
    )


def prepare(t, psi, pf, y, yerr, sigma_single_mas=None):
    """One system as `(data, parameterization, n_epochs)`.

    The data is padded to its bucket; the parameterization is built from the
    unpadded data, for the reason in the module docstring. Building the data
    twice costs microseconds against a ~2.5 s fit.

    `sigma_single_mas` switches the uncertainties to the injected scale under
    `config.ERROR_MODE = "injected"`. It is applied BEFORE the parameterization
    is built, so `a_floor = med(sigma)/sqrt(N)` reflects the precision the data
    actually have rather than the precision they claim.
    """
    from .library import parameterization

    yerr = injected_errors(yerr, sigma_single_mas)
    return (
        to_harv(t, psi, pf, y, yerr, pad=True),
        parameterization(to_harv(t, psi, pf, y, yerr, pad=False)),
        len(t),
    )


def real_rows(yerr):
    """Which rows are real epochs and which are padding.

    Padding is identified by its uncertainty rather than by an epoch count, so
    a caller that only has the arrays -- a figure reading `data.al_position_err`
    back out of a `GaiaAstrometryData` -- needs nothing else to strip it.
    """
    return np.asarray(yerr, dtype=np.float64) < PAD_ERR_MAS / 2


def chi2(residual, yerr):
    """`sum((residual/sigma)^2)` over the real epochs only.

    Padded rows carry `PAD_ERR_MAS`, so including them would add ~1e-12 per row
    -- harmless, but it also makes the per-system value depend on which bucket
    the system landed in, which is exactly the comparability `pad_log_offset`
    exists to protect.
    """
    keep = real_rows(yerr)
    r = np.asarray(residual, dtype=np.float64)[keep]
    s = np.asarray(yerr, dtype=np.float64)[keep]
    return float(np.sum((r / s) ** 2))


def linear_solution(design, al, yerr, n_columns=None):
    """Best-fit linear parameters at fixed orbit: weighted least squares.

    With `n_columns=5` this is the **no-orbit** solution -- the first five
    columns of harv's design matrix (`ra0`, `dec0`, `pmra`, `pmdec`, `parallax`)
    do not depend on the nonlinear orbital parameters, so a companion has to beat
    it. With all nine it is the best orbit fit at that period, eccentricity and
    phase.

    `chi2_null - chi2_best` between the two is then a proper likelihood ratio:
    nested models, both at their own maximum, so the difference is >= 0 always
    and a violation means the design matrix or the padding mask is wrong.

    **Both sides must be maxima.** harv returns the marginalized linear
    parameters as a *draw* from their conditional Gaussian, not as its mean
    (`use_mean=False`), so scoring the drawn theta against a least-squares null
    compares a posterior sample with an optimum and can make a real orbit look
    worse than no orbit at all. That is not a small effect: on a railed system it
    produced a delta chi-square of -13.
    """
    keep = real_rows(yerr)
    a = np.asarray(design, dtype=np.float64)[keep, : n_columns or None]
    b = np.asarray(al, dtype=np.float64)[keep]
    s = np.asarray(yerr, dtype=np.float64)[keep]
    theta, *_ = np.linalg.lstsq(a / s[:, None], b / s, rcond=None)
    return theta, float(np.sum(((b - a @ theta) / s) ** 2))
