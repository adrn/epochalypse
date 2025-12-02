"""Source motion models for simulating epoch astrometry."""

__all__ = ["AbstractSource"]

from typing import TYPE_CHECKING, Any

import coordinax as cx
import equinox as eqx
import quaxed.numpy as jnp
from unxt import Quantity

if TYPE_CHECKING:
    from epochalypse.kepler import TwoBodySystem


class AbstractSource(eqx.Module):
    """Abstract base class for astrometric source models."""

    ref_epoch: Quantity["time"]  # reference time
    pos0: cx.vecs.AbstractPos
    vel0: cx.vecs.AbstractVel

    def pos_at_time(self, _: Quantity["time"]) -> cx.vecs.AbstractPos:
        """Compute position at given times.

        This default implementation returns the reference position (i.e., no motion).
        """
        return self.pos0

    def vel_at_time(self, _: Quantity["time"]) -> cx.vecs.AbstractVel:
        """Compute velocity at given times.

        This default implementation returns the reference velocity (i.e., no
        acceleration).
        """
        return self.vel0

    def offset_sky(self, times: Quantity["time"], ref_pos: cx.vecs.AbstractPos) -> Any:
        """Compute (d_ra * cos(dec), d_dec) offset from reference position at times.

        Returns
        -------
        d_lon_coslat
            Longitude offset, in angular units
        d_lat
            Latitude offset, in angular units
        """
        pos_t = self.pos_at_time(times)

        # TODO: convert to offset class with origin at ref_pos
        # Rotate to ref_pos, convert to lon/lat

        # Old:
        # Compute angular offsets
        # d_ra_cosdec = (pos_t.lon - self.pos0.lon) * jnp.cos(pos_t.lat)
        # d_dec = pos_t.lat - self.pos0.lat

        # # TODO: coordinax should have an offset class
        # return d_ra_cosdec, d_dec


class LinearMotion3DSource(AbstractSource):
    """Source with linear 3D motion (single star)."""

    pos0: cx.vecs.AbstractPos3D
    vel0: cx.vecs.AbstractVel3D

    def pos_at_time(self, times: Quantity["time"]) -> cx.vecs.AbstractPos3D:
        """Compute 3D position at given times.

        Parameters
        ----------
        times
            Times at which to compute 3D position. This should be in the same time
            format and system as ``ref_epoch``.
        """
        dt = times - self.ref_epoch

        # Reference position in Cartesian
        xyz = self.pos0.represent_as(cx.CartesianPos3D)
        vxyz = self.vel0.represent_as(cx.CartesianVel3D)

        # Linear propagation
        return cx.CartesianPos3D(
            x=xyz.x + vxyz.d_x * dt,
            y=xyz.y + vxyz.d_y * dt,
            z=xyz.z + vxyz.d_z * dt,
        )


class LinearMotionSmallAngleSource(AbstractSource):
    """Source with linear motion (small-angle approximation, single star)."""

    # TODO:
    # pos0: cx.vecs.TwoSphereLonLatPos = eqx.field(converter=lambda x: cx.vconvert(cx.vecs.TwoSphereLonLatPos, x)))
    # vel0: cx.vecs.TwoSphereLonCosLatVel

    # TODO:
    def pos_at_time(self, times: Quantity["time"]) -> cx.vecs.AbstractPos3D:
        """Compute position at given times.

        Parameters
        ----------
        times
            Times at which to compute 3D position. This should be in the same time
            format and system as ``ref_epoch``.
        """
        dt = times - self.ref_epoch

        # Reference position in Cartesian
        xyz = self.pos0.represent_as(cx.CartesianPos3D)
        vxyz = self.vel0.represent_as(cx.CartesianVel3D)

        # Linear propagation
        return cx.CartesianPos3D(
            x=xyz.x + vxyz.d_x * dt,
            y=xyz.y + vxyz.d_y * dt,
            z=xyz.z + vxyz.d_z * dt,
        )

    # TODO: simple small-angle offset implementation
    def offset_sky(self, times: Quantity["time"], ref_pos: cx.vecs.AbstractPos) -> Any:
        """Compute sky offset via 2D linear propagation (small-angle approx).

        Parameters
        ----------
        times
            Times at which to compute sky offset. This should be in the same time format
            and system as ``ref_epoch``.
        """
        # Project back to spherical
        pos_t = xyz_t.represent_as(cx.LonLatSphericalPos)

        # Compute angular offsets
        d_ra_cosdec = (pos_t.lon - self.pos0.lon) * jnp.cos(pos_t.lat)
        d_dec = pos_t.lat - self.pos0.lat

        # TODO: coordinax should have an offset class
        return d_ra_cosdec, d_dec


class Accelerating3DSource(AbstractSource):
    """Source with a Keplerian companion (binary star or exoplanet)."""

    pos0: cx.AbstractPos3D  # barycenter (lon=ra, lat=dec, distance) at ref_epoch
    vel0: cx.AbstractVel3D  # barycenter 3D velocity
    system: (
        "TwoBodySystem"  # TODO: generalize - TwoBody, constant acceleration, or trend?
    )

    def offset_sky(self, times: Quantity["time"]) -> tuple[Quantity, Quantity]:
        """Compute sky offset of primary star.

        Parameters
        ----------
        times
            Times at which to compute sky offset. This should be in the same time format
            and system as ``ref_epoch``.
        """
        barycenter = SingleStarSource(
            pos0=self.pos0, vel=self.vel, ref_epoch=self.ref_epoch
        )
        xyz_bary_t = barycenter.offset_3d(times)

        # Primary offset from barycenter (in physical units)
        # position_barycentric returns shape (3, n_times)
        # TODO: KeplerianBody return a CartesianPos3D?
        primary_offset = self.system.position_barycentric(times, body_idx=0)

        # Add to barycenter position
        # TODO: why not just add the CaryesianPos3D objects?
        xyz_primary_t = cx.CartesianPos3D(
            x=xyz_bary_t.x + primary_offset[0],
            y=xyz_bary_t.y + primary_offset[1],
            z=xyz_bary_t.z + primary_offset[2],
        )

        # Project to sky
        pos_t = xyz_primary_t.represent_as(cx.LonLatSphericalPos)

        # Angular offsets from reference position
        d_ra_cosdec = (pos_t.lon - self.pos0.lon) * jnp.cos(pos_t.lat)
        d_dec = pos_t.lat - self.pos0.lat

        # TODO: coordinax should have an offset class
        return d_ra_cosdec, d_dec
