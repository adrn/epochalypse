import quaxed.numpy as jnp
from jaxoplanet.core.kepler import kepler
from jaxtyping import Array, Float
from unxt import Quantity, ustrip


def compute_true_anomaly_components(
    time: Quantity["time"],
    period: Quantity["time"],
    eccentricity: Float[Array, ""],
    t_peri: Quantity["time"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute true anomaly at given times.

    Parameters
    ----------
    time
        Observation times, shape (n,)
    period
        Orbital period
    eccentricity
        Orbital eccentricity
    t_peri
        Time of pericenter passage

    Returns
    -------
    sin_f, cos_f
        True anomaly components, each shape (n,)
    """
    M = ustrip("", 2 * jnp.pi * (time - t_peri) / period)
    return kepler(M, eccentricity)
