"""The one prior library every system in the catalog is measured against.

Drawn in process, from a fixed seed, on every rank. There is no cache file and
no build stage, because there is nothing left for one to do: with the
Thiele-Innes parameterization the library is four columns -- `period`,
`eccentricity`, `phase_peri`, `parallax` -- which is **32 MB at
`N_PRIOR_SAMPLES = 10^6` and 0.3 s to draw**. A file would add a stage, a
barrier before it, and a thousand ranks reading one path off ceph at once, in
exchange for what `jr.key(config.SEED)` already guarantees.

The other eight linear parameters (`ra0`, `dec0`, `pmra`, `pmdec`, and the four
Thiele-Innes constants) are Gaussian, so harv marginalizes them analytically
inside the likelihood and draws them conditionally per system. They are not in
the library because they are not sampled from the prior at all. `parallax` is
`HalfNormal`, which is not Gaussian, so it survives as an explicitly sampled
column.

`describe()` is what makes "every rank used the same library" checkable rather
than assumed: it carries the settings, the seed, and a hash of the drawn arrays.
"""

from __future__ import annotations

import hashlib

import numpy as np
from unxt import Quantity as Q

from . import config as C

# The unit of every column `writers.SampleWriter` stores, for the run manifest.
# The parquet holds bare float32, because a unit string identical for all 17.2 M
# rows belongs in the manifest and not in the schema. Hard-coded rather than
# probed because the manifest is written before the first system is fitted --
# and asserted against a real fit in `tests/test_harv.py`, so it cannot drift.
SAMPLE_UNITS = {
    "period": "yr",
    "eccentricity": "",
    "phase_peri": "",
    "parallax": "mas",
    "ra0": "mas",
    "dec0": "mas",
    "pmra": "mas / yr",
    "pmdec": "mas / yr",
    "ti_A": "mas",
    "ti_B": "mas",
    "ti_F": "mas",
    "ti_G": "mas",
    "weight": "",
    "ln_likelihood": "",
    "ln_prior": "",
}


def parameterization(data=None):
    """The orbital parameterization, optionally floored to one system's data.

    With `data`, this is `ThieleInnesGaiaAstrometry.from_data`, which sets
    `a_floor = med(sigma_AL)/sqrt(N)` and turns on the Jacobian correction --
    the correction that makes a flat prior on the Thiele-Innes constants
    equivalent to the Campbell prior. Pass the **unpadded** data: padding adds
    rows at `adapt.PAD_ERR_MAS`, which moves both the median and N.

    Without `data`, the plain instance. That is only ever used to build the
    prior, which does not read `a_floor` -- `default_prior` sets the four
    Thiele-Innes scales from `sigma_a0` and `P0` alone. That independence is
    what lets one library serve every system, and `tests/test_harv.py` asserts
    it rather than trusting this comment.
    """
    if not C.USE_THIELE_INNES:
        from harv.models.parameterizations.gaia import StandardGaiaAstrometry

        return StandardGaiaAstrometry()

    from harv.models.parameterizations.gaia import ThieleInnesGaiaAstrometry

    if data is None:
        return ThieleInnesGaiaAstrometry()
    return ThieleInnesGaiaAstrometry.from_data(data)


def eccentricity_prior():
    """The eccentricity prior, overriding harv's Kipping default.

    See `config.ECC_LOC` for why: Kipping under-samples the injected
    Uniform(0, 0.99) by 14x above e = 0.7, and measured period recovery falls
    with eccentricity because of it.
    """
    import numpyro.distributions as dist

    return dist.TruncatedNormal(C.ECC_LOC, C.ECC_SCALE, low=0.0, high=1.0)


def sigma_a0_au(m_star_msun=None):
    """The orbit-amplitude prior scale, as the largest companion it expects.

    At the reference period, a companion of mass `m` around a star of mass `M`
    displaces the photocentre by `a0 = a m/(M+m)` with `a = (M P0^2)^(1/3)`,
    which for `m << M` is `m / M^(2/3)`. So one companion-mass ceiling
    (`config.M_MAX_MJUP`) gives the right scale for every star, which is what
    harv's own `(P/P0)^(2/3) x parallax` scaling is reaching for -- it keeps the
    prior constant in companion mass at *fixed* primary mass, and the `M^(2/3)`
    here removes the "fixed" part.

    `config.SIGMA_A0_AU` pins a constant instead, for sweeps.
    """
    if C.SIGMA_A0_AU is not None:
        return float(C.SIGMA_A0_AU)
    if m_star_msun is None:
        msg = (
            "sigma_a0 is derived per system from the host mass; pass "
            "m_star_msun, or pin config.SIGMA_A0_AU to a constant"
        )
        raise ValueError(msg)
    m_max = C.M_MAX_MJUP * C.MJUP_IN_MSUN
    return m_max / float(m_star_msun) ** (2.0 / 3.0)


def prior(par=None, m_star_msun=None):
    """The catalog-wide prior, with the amplitude scale set from the host mass.

    Everything except `sigma_a0` is a catalog constant. `sigma_a0` is per system
    and costs nothing: it shapes only the analytically marginalized Thiele-Innes
    priors, which are never drawn into the shared library, so it changes neither
    the fingerprint nor the JIT cache.
    """
    par = parameterization() if par is None else par
    return par.default_prior(
        eccentricity=eccentricity_prior(),
        period_min=Q(C.PERIOD_MIN_YR, "yr"),
        period_max=Q(C.PERIOD_MAX_YR, "yr"),
        sigma_pos=Q(C.SIGMA_POS_MAS, "mas"),
        sigma_vtan=Q(C.SIGMA_VTAN_KMS, "km/s"),
        sigma_parallax=Q(C.SIGMA_PARALLAX_MAS, "mas"),
        sigma_a0=Q(sigma_a0_au(m_star_msun), "AU"),
        P0=Q(C.P0_YR, "yr"),
    )


def model(par=None):
    """The Gaia astrometry model. Per system when `par` came from that system."""
    from harv import GaiaAstrometryModel

    return GaiaAstrometryModel(
        parameterization=parameterization() if par is None else par
    )


def draw(n=None, seed=None):
    """The prior library: `n` draws under `seed`, identical on every rank."""
    import jax.random as jr

    n = C.N_PRIOR_SAMPLES if n is None else int(n)
    seed = C.SEED if seed is None else int(seed)
    par = parameterization()
    # The host mass here is arbitrary and the draws do not depend on it: it only
    # sets `sigma_a0`, which shapes the four Thiele-Innes priors, and those are
    # analytically marginalized rather than sampled. `tests/test_harv.py` asserts
    # the library is bit-identical across host masses rather than trusting this.
    return prior(par, m_star_msun=1.0).sample(jr.key(seed), n, model=model(par))


def fingerprint(library):
    """A hash of every drawn value, so two ranks can be compared in one string.

    Over the parameter names in sorted order and their raw values, so it is
    stable against dict ordering but not against a changed draw.

    It covers the DRAWS, not every setting. `sigma_a0` and `sigma_pos` shape the
    analytically marginalized linear priors, which are never sampled into the
    library, so changing them leaves this hash identical. The full settings are
    in `describe()` and so in the run manifest -- read those, not this, to tell
    two runs apart.
    """
    digest = hashlib.blake2s(digest_size=16)
    columns = {**library.nonlinear, **library.linear}
    for name in sorted(columns):
        digest.update(name.encode())
        digest.update(np.asarray(columns[name].value, dtype=np.float64).tobytes())
    return digest.hexdigest()


def describe(library=None):
    """Everything that defines the library, for the run manifest."""
    described = {
        "units": SAMPLE_UNITS,
        "n_prior_samples": C.N_PRIOR_SAMPLES,
        "seed": C.SEED,
        "top_k": C.TOP_K,
        "batch_size": C.BATCH_SIZE,
        "parameterization": type(parameterization()).__name__,
        "period_min_yr": C.PERIOD_MIN_YR,
        "period_max_yr": C.PERIOD_MAX_YR,
        "sigma_parallax_mas": C.SIGMA_PARALLAX_MAS,
        "sigma_pos_mas": C.SIGMA_POS_MAS,
        "sigma_vtan_kms": C.SIGMA_VTAN_KMS,
        "m_max_mjup": C.M_MAX_MJUP,
        "sigma_a0_au": C.SIGMA_A0_AU,  # None -> derived per system from the host mass
        "p0_yr": C.P0_YR,
        # Not harv's default -- a run's fingerprint changes with it, and a
        # recovery number is not comparable across two different ones.
        "eccentricity_prior": f"TruncatedNormal({C.ECC_LOC}, {C.ECC_SCALE}, 0, 1)",
    }
    if library is not None:
        described["sampled"] = sorted({**library.nonlinear, **library.linear})
        described["fingerprint"] = fingerprint(library)
    return described
