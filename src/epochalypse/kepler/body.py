"""Keplerian orbit implementation with units support and JAX compatibility."""

import astropy.units as apyu
import equinox as eqx
import jax
import quaxed.numpy as jnp
from astropy.constants import G as G_astropy  # noqa: N811
from jaxoplanet.core.kepler import kepler
from unxt import Quantity, ustrip

from epochalypse.kepler.orientation import KeplerianOrientation

G = Quantity.from_(G_astropy)

# TODO: make this a configurable thing
ECC_ZERO_TOL = 1e-10


class KeplerianBody(eqx.Module):
    """Orbital parameters of a Keplerian body (companion).

    This class represents the orbital parameters of a companion or second body (relative
    to some barycenter) with support for units via unxt. So, all parameters represent
    the orbital elements of a specific body relative to the barycenter.

    The primary parameterization uses:
    - Period (P)
    - Eccentricity (e)
    - Semi-major axis (a) or ``semi_major_axis``
    - Time of pericenter or ``t_peri``

    Alternative constructors support different parameterizations.

    Parameters
    ----------
    period
        Orbital period.
    eccentricity
        Orbital eccentricity.
    semi_major_axis
        The semi-major axis of the body relative to its system barycenter.
    t_peri
        Time of pericenter passage.
    orientation
        Optional: Orientation of the orbit.

    """

    # TODO: decide on API here - use eqx.field(converter=Quantity["time"].from_)?
    period: Quantity["time"]
    eccentricity: Quantity["dimensionless"]
    semi_major_axis: Quantity["length"]
    t_peri: Quantity["time"]
    orientation: KeplerianOrientation = KeplerianOrientation()

    def __check_init__(self) -> None:
        if not (0.0 <= self.eccentricity < 1.0):
            raise ValueError(
                "Eccentricity must be in the range [0, 1) for bound orbits"
            )

        # Check that either all dimensional inputs are specified with units, or are all
        # dimensionless - no mixing of dimensionless and dimensional
        checks = [
            x.unit.decompose().is_equivalent(apyu.one) if hasattr(x, "unit") else True
            for x in [self.period, self.semi_major_axis, self.t_peri]
        ]
        if any(checks) and not all(checks):
            raise ValueError(
                "Either all or none of period, semi_major_axis, and t_peri must have "
                "units"
            )

    # ========================================================================
    # Alternative constructors
    #

    @classmethod
    def from_masses(
        cls,
        period: Quantity["time"],
        eccentricity: Quantity["dimensionless"],
        m_companion: Quantity["mass"],
        m_primary: Quantity["mass"],
        t_peri: Quantity["time"],
        orientation: KeplerianOrientation | None = None,
    ) -> "KeplerianBody":
        r"""Construct companion's barycentric orbit from masses and period.

        Computes the companion's barycentric semi-major axis from Kepler's 3rd law:
        1. Compute relative orbit: a_rel = (G (m_1 + m_2) P^2 / 4 \pi^2)^(1/3)
        2. Convert to barycentric: a_body = a_rel * m_1 / (m_1 + m_2)

        Parameters
        ----------
        period
            Orbital period.
        eccentricity
            Orbital eccentricity.
        m_companion
            Companion mass (this body).
        m_primary
            Primary (central body) mass.
        t_peri
            Time of pericenter passage.
        orientation
            Optional: Orientation of the orbit.

        Returns
        -------
        orbit: KeplerianBody
            The companion's orbit about the barycenter
        """
        m_tot = m_primary + m_companion
        a_rel = jnp.cbrt((G * m_tot * period**2) / (4 * jnp.pi**2))
        a_body = a_rel * (m_primary / m_tot)

        kw = {}
        if orientation is not None:
            kw["orientation"] = orientation

        return cls(
            period=period,
            eccentricity=eccentricity,
            semi_major_axis=a_body,
            t_peri=t_peri,
            **kw,
        )

    # ========================================================================
    # Other methods
    #
    def get_mass(self, m_primary: Quantity["mass"]) -> Quantity["mass"]:
        """Compute companion mass given primary mass and barycentric semi-major axis."""
        num = G * m_primary**3 * self.period**2
        den = 4 * jnp.pi**2 * self.semi_major_axis**3
        m_tot = jnp.sqrt(num / den)
        return m_tot - m_primary

    def get_position(
        self, time: Quantity["time"], orientation: KeplerianOrientation | None = None
    ) -> Quantity["length"]:
        """Get 3D position of the body in its orbit at given time(s).

        By definition and convention of this class, this is the position of the body
        relative to the system barycenter, accounting for the orbit orientation.
        """
        # Mean anomaly
        M = ustrip("", 2 * jnp.pi * (time - self.t_peri) / self.period)

        # Eccentric anomaly using jaxoplanet kepler solver
        sin_cos_f = jax.lax.cond(
            jnp.isclose(self.eccentricity, 0.0, atol=ECC_ZERO_TOL),
            lambda: (jnp.sin(M), jnp.cos(M)),
            lambda: kepler(M, self.eccentricity),
        )

        # Distance from focus
        r = (
            self.semi_major_axis
            * (1 - self.eccentricity**2)
            / (1 + self.eccentricity * sin_cos_f[1])
        )

        # Position in orbital plane
        x_orb = r * sin_cos_f[1]
        y_orb = r * sin_cos_f[0]
        xyz_orb = jnp.stack([x_orb, y_orb, jnp.zeros_like(x_orb)], axis=0)

        orientation = self.orientation if orientation is None else orientation

        # Rotate to observer frame
        # TODO: identify if rotation is close to identity and skip, for performance
        return jnp.einsum("ij,j...->i...", orientation.rotation_matrix, xyz_orb)

    def get_velocity(
        self, time: Quantity["time"], orientation: KeplerianOrientation | None = None
    ) -> Quantity["speed"]:
        """Get 3D velocity of the body relative to the system barycenter."""
        # Mean anomaly (dimensionless)
        M = ustrip("", 2 * jnp.pi * (time - self.t_peri) / self.period)

        # True anomaly (sin f, cos f); circular shortcut consistent with get_position
        sin_f, cos_f = jax.lax.cond(
            jnp.isclose(self.eccentricity, 0.0, atol=ECC_ZERO_TOL),
            lambda: (jnp.sin(M), jnp.cos(M)),
            lambda: kepler(M, self.eccentricity),
        )

        a = self.semi_major_axis
        e = self.eccentricity
        n = (2 * jnp.pi) / self.period

        # Radius and kinematic rates
        r = a * (1 - e**2) / (1 + e * cos_f)

        def _vel_circular():  # type: ignore[no-untyped-def] # noqa: ANN202
            vx = (-n * a) * sin_f
            vy = (n * a) * cos_f
            return vx, vy

        def _vel_eccentric():  # type: ignore[no-untyped-def] # noqa: ANN202
            rdot = n * a * e * sin_f / jnp.sqrt(1 - e**2)
            fdot = n * (1 + e * cos_f) ** 2 / (1 - e**2) ** 1.5
            vx = rdot * cos_f - r * fdot * sin_f
            vy = rdot * sin_f + r * fdot * cos_f
            return vx, vy

        vx_orb, vy_orb = jax.lax.cond(
            jnp.isclose(e, 0.0, atol=ECC_ZERO_TOL), _vel_circular, _vel_eccentric
        )
        vz_orb = jnp.zeros_like(vx_orb)
        vel_orb = jnp.stack([vx_orb, vy_orb, vz_orb], axis=0)

        orientation = self.orientation if orientation is None else orientation
        return jnp.einsum("ij,j...->i...", orientation.rotation_matrix, vel_orb)
