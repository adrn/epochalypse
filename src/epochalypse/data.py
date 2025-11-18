"""
Classes for representing and managing different types of epoch data.

TODO: Need to think about the API here. Do we want different container classes for
different kinds of epoch data? Might make sense, since the photometric vs astrometric vs
radial velocity data will have different times and metadata.
"""

import equinox as eqx

from epochalypse.custom_types import NAngle, NFloatArray, NIntArray, NTime, NVelocity

__all__ = ["GaiaEpochAstrometry", "EpochRV"]


class GaiaEpochAstrometry(eqx.Module):
    """Container for along-scan (AL) epoch astrometry.

    Attributes
    ----------
    time : Quantity['time'], shape (n,)
        Barycentric TCB times of observations.
    al_position : Quantity['angle'], shape (n,)
        Observed along-scan position values.
    al_position_error : Quantity['angle'], shape (n,)
        1-sigma uncertainty for al_position.
    scan_angle : Quantity['angle'], shape (n,)
        Per-CCD scan position angle theta.
    parallax_factor : Array, shape (n,)
        AL projection of parallax factors. Angle of AL displacement per mas of parallax.
    transit_index : Array, shape (n,), optional
        Integer index 0...K-1 indicating which FoV transit each observation belongs to.
        Default is None.
    """

    time: NTime
    al_position: NAngle
    al_position_error: NAngle
    scan_angle: NAngle
    parallax_factor: NFloatArray
    transit_index: NIntArray | None = None


class EpochRV(eqx.Module):
    """Container for radial velocity epoch measurements.

    Attributes
    ----------
    time : Quantity['time'], shape (n,)
        Observation times.
    rv : Quantity['speed'], shape (n,)
        Radial velocities.
    rv_error : Quantity['speed'], shape (n,)
        1-sigma uncertainties on radial velocities.
    """

    time: NTime
    rv: NVelocity
    rv_error: NVelocity
    # instrument: NIntArray | None = None - to be more general?
