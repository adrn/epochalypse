"""Keplerian orbit implementation with units support and JAX compatibility."""

import equinox as eqx
from astropy.constants import G as G_astropy  # noqa: N811
from unxt import Quantity

from epochalypse.kepler.body import KeplerianBody

G = Quantity.from_(G_astropy)


class AbstractNBodySystem(eqx.Module):
    """Abstract base class for Keplerian N-body systems."""

    @property
    def n_bodies(self) -> int:
        """Total number of bodies (primary + companions)."""
        raise NotImplementedError

    @property
    def m_total(self) -> Quantity["mass"]:
        """Total system mass."""
        raise NotImplementedError


class TwoBodySystem(AbstractNBodySystem):
    """A system with a primary body and one companion.

    Bodies are indexed as:
    - 0: Primary body
    - 1: Companion
    """

    m_primary: Quantity["mass"]
    companion: KeplerianBody

    # ========================================================================
    # Properties
    #

    @property
    def n_bodies(self) -> int:
        """Total number of bodies (primary + companions)."""
        return 2

    @property
    def m_companion(self) -> Quantity["mass"]:
        """Companion mass."""
        return self.companion.get_mass(self.m_primary)

    @property
    def m_total(self) -> Quantity["mass"]:
        """Total system mass."""
        return self.m_primary + self.m_companion

    # ========================================================================
    # Methods
    #

    def position_barycentric(
        self, time: Quantity["time"], body_idx: int
    ) -> Quantity["length"]:
        """Get barycentric position of specified body at given time(s).

        Parameters
        ----------
        time
            Time(s) to evaluate
        body_idx
            Index of body (0=primary, 1=companion)

        Returns
        -------
        pos
            3D position vector(s) of specified body relative to barycenter
        """
        r2 = self.companion.get_position(time)

        if body_idx == 1:
            return r2  # companion about barycenter

        if body_idx == 0:
            return -(self.m_companion / self.m_primary) * r2

        raise IndexError("body_idx must be 0 (primary) or 1 (companion)")

    def position_relative(self, time: Quantity["time"]) -> Quantity["length"]:
        """Position of companion relative to primary.

        Parameters
        ----------
        time
            Time(s) to evaluate

        Returns
        -------
        pos
            3D position vector(s) of companion relative to primary
        """
        r2 = self.position_barycentric(time, 1)
        return r2 * (1 + self.m_companion / self.m_primary)

    def velocity_barycentric(
        self, time: Quantity["time"], body_idx: int
    ) -> Quantity["speed"]:
        """Get barycentric velocity of specified body at given time(s).

        Parameters
        ----------
        time
            Time(s) to evaluate
        body_idx
            Index of body (0=primary, 1=companion)

        Returns
        -------
        vel
            3D velocity vector(s) of specified body relative to barycenter
        """
        v2 = self.companion.get_velocity(time)  # companion barycentric velocity

        if body_idx == 1:
            return v2

        if body_idx == 0:
            return -(self.m_companion / self.m_primary) * v2

        raise IndexError("body_idx must be 0 (primary) or 1 (companion)")

    def velocity_relative(self, time: Quantity["time"]) -> Quantity["speed"]:
        """Velocity of companion relative to primary.

        Parameters
        ----------
        time
            Time(s) to evaluate

        Returns
        -------
        vel
            3D velocity vector(s) of companion relative to primary
        """
        v2 = self.velocity_barycentric(time, 1)
        return v2 * (1 + self.m_companion / self.m_primary)
