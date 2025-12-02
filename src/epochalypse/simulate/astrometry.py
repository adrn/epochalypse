"""Generate simulated epoch astrometry."""

import jax
import jax.random as jr
import quaxed.numpy as jnp
from unxt import Quantity

from epochalypse.simulate.scanlaw import GaiaReducedCommandedScanLaw
from epochalypse.simulate.source import AbstractSource


def simulate_gaia_al(
    source: AbstractSource,
    scanlaw: GaiaReducedCommandedScanLaw,
    noise_scale: Quantity["angle"] | None = None,
    rng_key: jax.Array | None = None,
) -> dict:
    """
    Simulate Gaia along-scan epoch astrometry.

    Parameters
    ----------
    source
        Source model (LinearMotionSource or KeplerianSource)
    scanlaw
        Gaia scan law instance
    noise_scale
        Optional: Gaussian noise to add to y_al measurements
    rng_key
        Optional: JAX random key for noise generation

    Returns
    -------
    dict with times, y_al, scan_angle, parallax_factor
    """
    # Query scan law at reference position
    ra0 = source.pos0.lon
    dec0 = source.pos0.lat
    scans = scanlaw.query(ra0, dec0)
    meta = scanlaw.read_metadata()

    # TODO: work in Gaia JD relative time system?
    times = Quantity(scans["time_bjd"] + meta["gaia_time_origin_bjd"], "day")
    scan_angle = Quantity(scans["scan_angle_deg"], "deg")
    parallax_factor = scans["parallax_factor_al"]

    # Get sky offsets at each epoch
    d_ra_cosdec, d_dec = source.offset_sky(times)

    # Along-scan measurement
    y_al = (
        d_ra_cosdec * jnp.sin(scan_angle)
        + d_dec * jnp.cos(scan_angle)
        + source.parallax * parallax_factor
    )

    # Add noise if requested
    if noise_scale is not None and rng_key is not None:
        noise = noise_scale * jr.normal(rng_key, shape=y_al.shape)
        y_al = y_al + noise

    return {
        "times": times,
        "y_al": y_al,
        "scan_angle": scan_angle,
        "parallax_factor": parallax_factor,
    }
