"""Keplerian orbit implementation with units support and JAX compatibility."""

import equinox as eqx
import jax
import quaxed.numpy as jnp
from unxt import Quantity, ustrip


class KeplerianOrientation(eqx.Module):
    """Orientation of a Keplerian orbit in 3D space.

    Stores the three Euler angles that define how the orbital plane
    is oriented relative to the observer's reference frame:
    - Inclination (i): tilt of orbital plane from sky plane
    - Longitude of ascending node (Ω): where orbit crosses sky plane
    - Argument of pericenter (ω): orientation of ellipse within orbital plane

    Angles are stored as sin/cos pairs for numerical stability.
    """

    # sin/cos of argument of pericenter (ω)
    sin_arg_peri: float = 0.0
    cos_arg_peri: float = 1.0

    # sin/cos of longitude of ascending node (Ω)
    sin_lon_asc_node: float = 0.0
    cos_lon_asc_node: float = 1.0

    # sin/cos of inclination (i)
    sin_i: float = 0.0
    cos_i: float = 1.0

    def __check_init__(self) -> None:
        x = jnp.array(self.sin_arg_peri**2 + self.cos_arg_peri**2)
        eqx.error_if(
            x,
            jnp.logical_not(
                jnp.isclose(
                    x,
                    1.0,
                    atol=jnp.finfo(float).eps,  # type: ignore[no-untyped-call]
                )
            ),
            "Argument of pericenter sin/cos values are not normalized",
        )

        x = jnp.array(self.sin_lon_asc_node**2 + self.cos_lon_asc_node**2)
        eqx.error_if(
            x,
            jnp.logical_not(
                jnp.isclose(
                    x,
                    1.0,
                    atol=jnp.finfo(float).eps,  # type: ignore[no-untyped-call]
                )
            ),
            "Longitude of ascending node sin/cos values are not normalized",
        )

        x = jnp.array(self.sin_i**2 + self.cos_i**2)
        eqx.error_if(
            x,
            jnp.logical_not(
                jnp.isclose(
                    x,
                    1.0,
                    atol=jnp.finfo(float).eps,  # type: ignore[no-untyped-call]
                )
            ),
            "Inclination sin/cos values are not normalized",
        )

    @classmethod
    def from_angles(
        cls,
        /,
        arg_peri: Quantity["angle"] = Quantity(0, "rad"),
        lon_asc_node: Quantity["angle"] = Quantity(0, "rad"),
        inclination: Quantity["angle"] = Quantity(0, "rad"),
    ) -> "KeplerianOrientation":
        """Construct from angle values."""
        return cls(
            sin_arg_peri=jnp.sin(Quantity.from_(arg_peri)),
            cos_arg_peri=jnp.cos(Quantity.from_(arg_peri)),
            sin_lon_asc_node=jnp.sin(Quantity.from_(lon_asc_node)),
            cos_lon_asc_node=jnp.cos(Quantity.from_(lon_asc_node)),
            sin_i=jnp.sin(Quantity.from_(inclination)),
            cos_i=jnp.cos(Quantity.from_(inclination)),
        )

    @classmethod
    def from_thiele_innes(
        cls,
        A: Quantity["length"],
        B: Quantity["length"],
        F: Quantity["length"],
        G: Quantity["length"],
    ) -> tuple["KeplerianOrientation", Quantity["length"]]:
        """Construct orientation from Thiele-Innes constants.

        Inverts the Thiele-Innes constants to recover (ω, Ω, i, a).
        This loosely follows Appendix A of https://arxiv.org/abs/2206.05726
        but my convention for angles is different. For this implementation:

        0 < i < π/2
        0 < ω < 2π
        0 < Ω < 2π

        whereas the paper linked above assumes:

        0 < i < π
        0 < ω < 2π
        0 < Ω < π

        Returns
        -------
        orientation
            KeplerianOrientation object
        semi_major_axis
            Recovered semi-major axis
        """
        A = Quantity.from_(A)
        B = Quantity.from_(B)
        F = Quantity.from_(F)
        G = Quantity.from_(G)

        u_ = (A**2 + B**2 + F**2 + G**2) / 2.0
        v_ = A * G - B * F

        inner = (u_ + v_) * (u_ - v_)
        # Guard against small negative from roundoff
        inner = jnp.where(inner < 0.0, 0.0, inner)
        a = jnp.sqrt(u_ + jnp.sqrt(inner))

        # From algebraic manipulation of T-I
        cos_i = ustrip("", v_ / a**2)
        cos_i = jnp.clip(cos_i, -1.0, 1.0)

        sin_i_squared = 1.0 - cos_i**2
        sin_i = jnp.sqrt(jnp.where(sin_i_squared < 0.0, 0.0, sin_i_squared))
        i = jnp.arctan2(sin_i, cos_i)  # i in [0, π]

        # Sums & differences of angles (Binnendijk paper referenced in above paper)
        # sp = (ω + Ω), sm = (ω - Ω)
        sp = ustrip("rad", jnp.arctan2(B - F, A + G))
        sm = ustrip("rad", jnp.arctan2(B + F, G - A))

        omega = jnp.mod(0.5 * (sp + sm), 2 * jnp.pi)
        Omega = jnp.mod(0.5 * (sp - sm), 2 * jnp.pi)

        # Enforce 0 < i < π/2, Ω ∈ [0, 2π) ---
        # If i > π/2, use the symmetry:
        # (i, Ω, ω) -> (π - i, Ω + π, ω + π)
        mask = i > (0.5 * jnp.pi)

        i = jnp.where(mask, jnp.pi - i, i)
        Omega = jnp.where(mask, jnp.mod(Omega + jnp.pi, 2 * jnp.pi), Omega)
        omega = jnp.where(mask, jnp.mod(omega + jnp.pi, 2 * jnp.pi), omega)

        return (
            cls.from_angles(
                arg_peri=Quantity(omega, "rad"),
                lon_asc_node=Quantity(Omega, "rad"),
                inclination=Quantity(i, "rad"),
            ),
            a,
        )

    @property
    def arg_peri(self) -> Quantity["angle"]:
        """Argument of pericenter (ω)."""
        return jnp.arctan2(self.sin_arg_peri, self.cos_arg_peri)

    @property
    def lon_asc_node(self) -> Quantity["angle"]:
        """Longitude of ascending node (Ω)."""
        return jnp.arctan2(self.sin_lon_asc_node, self.cos_lon_asc_node)

    @property
    def inclination(self) -> Quantity["angle"]:
        """Inclination (i)."""
        return jnp.arctan2(self.sin_i, self.cos_i)

    @property
    def rotation_matrix(self) -> jax.Array:
        """Compute rotation matrix from orbital plane to observer frame.

        Returns the rotation matrix R such that:
        r_observer_frame = R @ r_orbital_frame

        The rotation is composed of three sequential rotations:
        1. R_z(ω): Rotate by argument of pericenter, ω, in orbital plane
        2. R_x(i): Rotate by inclination, i, to tilt orbital plane
        3. R_z(Ω): Rotate by longitude of ascending node, Ω, on sky plane

        The full rotation matrix is therefore:
        R = R_z(Ω) @ R_x(i) @ R_z(ω)

        We build the matrix directly from the sin/cos pairs for numerical stability and
        speed, but using the notation below, it is equivalent to:

        R1 = jnp.array([[c_w, -s_w, 0], [s_w, c_w, 0], [0, 0, 1]])
        R2 = jnp.array([[1., 0, 0], [0, c_i, -s_i], [0, s_i, c_i]])
        R3 = jnp.array([[c_W, -s_W, 0], [s_W, c_W, 0], [0, 0, 1]])
        R = R3 @ R2 @ R1

        Or, alternatively:
        omega = arg_peri.to_value("rad")
        Omega = lon_asc_node.to_value("rad")
        i = inclination.to_value("rad")
        R = Rotation.from_euler('ZXZ', [Omega, i, omega]).as_matrix()

        """
        s_w = self.sin_arg_peri
        c_w = self.cos_arg_peri
        s_W = self.sin_lon_asc_node
        c_W = self.cos_lon_asc_node
        s_i = self.sin_i
        c_i = self.cos_i

        # Write out all terms explicitly (for speed)
        r11 = c_W * c_w - s_W * c_i * s_w
        r12 = -c_W * s_w - s_W * c_i * c_w
        r13 = s_W * s_i
        r21 = s_W * c_w + c_W * c_i * s_w
        r22 = -s_W * s_w + c_W * c_i * c_w
        r23 = -c_W * s_i
        r31 = s_i * s_w
        r32 = s_i * c_w
        r33 = c_i

        return jnp.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])

    def thiele_innes_constants(
        self, semi_major_axis: Quantity["length"]
    ) -> tuple[Quantity["length"], ...]:
        """Compute Thiele-Innes constants (A, B, F, G).

        These constants linearize the relationship between orbital position
        and sky-plane projection. See Appendix A of: https://arxiv.org/abs/2206.05726

        Parameters
        ----------
        semi_major_axis
            Semi-major axis of the orbit

        Returns
        -------
        A, B, F, G
            The four Thiele-Innes constants.
        """
        a = Quantity.from_(semi_major_axis)
        s_w = self.sin_arg_peri
        c_w = self.cos_arg_peri
        s_W = self.sin_lon_asc_node
        c_W = self.cos_lon_asc_node
        c_i = self.cos_i

        # Eq A.1 of https://arxiv.org/abs/2206.05726
        A = a * (c_w * c_W - s_w * s_W * c_i)
        B = a * (c_w * s_W + s_w * c_W * c_i)
        F = -a * (s_w * c_W + c_w * s_W * c_i)
        G = -a * (s_w * s_W - c_w * c_W * c_i)

        return A, B, F, G
