"""Unit tests for :mod:`epochalypse.kepler.orientation`."""

import dataclasses

import jax
import pytest
import quaxed.numpy as jnp
from jax import config as jax_config
from unxt import Quantity, ustrip

from epochalypse.kepler.orientation import KeplerianOrientation


def _check_thiele_innes_round_trip(
    orientation: KeplerianOrientation,
    semi_major_axis: Quantity,
    expected_orientation: KeplerianOrientation | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-8,
) -> None:
    """Check that Thiele-Innes constants round-trip correctly.

    Parameters
    ----------
    orientation
        Input orientation to test
    semi_major_axis
        Semi-major axis of the orbit
    expected_orientation
        Expected orientation after round-trip. If None, will compare rotation
        matrices instead (more robust to angle ambiguities).
    rtol
        Relative tolerance for comparisons
    atol
        Absolute tolerance for comparisons
    """
    ti_constants = orientation.thiele_innes_constants(semi_major_axis)
    roundtrip_orientation, roundtrip_a = KeplerianOrientation.from_thiele_innes(
        *ti_constants
    )

    # Check semi-major axis is preserved
    assert jnp.allclose(
        ustrip("au", roundtrip_a), ustrip("au", semi_major_axis), rtol=rtol, atol=atol
    )

    # Check physical orientation matches
    if expected_orientation is None:
        # Compare rotation matrices - more robust than comparing angles since there can
        # be angle ambiguities that give the same rotation

        match_primary = jnp.allclose(
            roundtrip_orientation.rotation_matrix,
            orientation.rotation_matrix,
            rtol=rtol,
            atol=atol,
        )

        # Check symmetric solution: Ω -> Ω+π, ω -> ω+π Thiele-Innes constants are
        # invariant under this transformation, so we can't distinguish without radial
        # velocity data
        sym_orientation = KeplerianOrientation.from_angles(
            arg_peri=orientation.arg_peri + Quantity(jnp.pi, "rad"),
            lon_asc_node=orientation.lon_asc_node + Quantity(jnp.pi, "rad"),
            inclination=orientation.inclination,
        )

        match_sym = jnp.allclose(
            roundtrip_orientation.rotation_matrix,
            sym_orientation.rotation_matrix,
            rtol=rtol,
            atol=atol,
        )

        assert (
            match_primary or match_sym
        ), "Recovered orientation does not match original or symmetric solution."
    else:
        # For specific expected orientation (e.g., symmetry tests), compare components
        assert jnp.allclose(
            jnp.array([x[0] for x in dataclasses.astuple(roundtrip_orientation)]),
            jnp.array([x[0] for x in dataclasses.astuple(expected_orientation)]),
            rtol=rtol,
            atol=atol,
        )

    # Check Thiele-Innes constants are preserved
    original_constants = orientation.thiele_innes_constants(semi_major_axis)
    roundtrip_constants = roundtrip_orientation.thiele_innes_constants(roundtrip_a)
    assert jnp.allclose(
        jnp.stack([ustrip("AU", val) for val in original_constants]),
        jnp.stack([ustrip("AU", val) for val in roundtrip_constants]),
        rtol=rtol,
        atol=atol,
    )


@pytest.fixture(params=["float32", "float64"])
def dtype(request):
    """Parametrize tests over float32 and float64."""
    original_value = jax_config.read("jax_enable_x64")
    jax_config.update("jax_enable_x64", request.param == "float64")
    yield request.param
    jax_config.update("jax_enable_x64", original_value)


@pytest.mark.parametrize(
    ("arg_peri", "lon_asc_node", "inclination", "semi_major_axis"),
    [
        (0.0, 0.0, 0.0, 5.0),  # zero angles
        (1.5, 2.0, jnp.pi / 2, 3.5),  # max inclination
        (
            2 * jnp.pi - 0.01,
            2 * jnp.pi - 0.02,
            jnp.pi / 2 - 0.001,
            2.8,
        ),  # angles near 2π and π/2
        (1.2, 0.5, 0.3, 4.0),  # arbitrary angles
    ],
)
def test_thiele_innes_round_trip_edge_cases(
    arg_peri: float,
    lon_asc_node: float,
    inclination: float,
    semi_major_axis: float,
    dtype: str,
) -> None:
    """Test round-trip for edge cases and typical values."""
    # Set appropriate tolerance based on dtype
    # Note: Multiple arctan2 operations accumulate numerical errors
    rtol = 5e-4 if dtype == "float32" else 1e-6

    orientation = KeplerianOrientation.from_angles(
        arg_peri=Quantity(arg_peri, "rad"),
        lon_asc_node=Quantity(lon_asc_node, "rad"),
        inclination=Quantity(inclination, "rad"),
    )
    _check_thiele_innes_round_trip(
        orientation, Quantity(semi_major_axis, "AU"), rtol=rtol
    )


@pytest.mark.parametrize("seed", [42, 123, 456])
def test_thiele_innes_round_trip_random_low_inclination(seed: int, dtype: str) -> None:
    """Test round-trip with random angles where inclination < π/2."""
    # Set appropriate tolerance based on dtype
    # Note: Multiple arctan2 operations accumulate numerical errors
    # For float32, low inclination orbits suffer from catastrophic cancellation
    # in the recovery of i, leading to larger errors.
    rtol = 2e-2 if dtype == "float32" else 1e-6
    atol = 2e-2 if dtype == "float32" else 1e-8

    key = jax.random.PRNGKey(seed)

    # Test multiple random configurations
    for _ in range(5):
        key, subkey = jax.random.split(key)
        random_vals = jax.random.uniform(subkey, shape=(4,))

        # Generate random angles within valid ranges
        arg_peri = random_vals[0] * 2 * jnp.pi
        lon_asc_node = random_vals[1] * 2 * jnp.pi
        inclination = random_vals[2] * jnp.pi / 2
        semi_major_axis = Quantity(jnp.asarray(1.0 + random_vals[3] * 10), "AU")

        orientation = KeplerianOrientation.from_angles(
            arg_peri=Quantity(arg_peri, "rad"),
            lon_asc_node=Quantity(lon_asc_node, "rad"),
            inclination=Quantity(inclination, "rad"),
        )

        _check_thiele_innes_round_trip(
            orientation, semi_major_axis, rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("seed", [42, 99])
def test_thiele_innes_round_trip_random_high_inclination(seed: int, dtype: str) -> None:
    """Test round-trip with random angles where inclination > π/2.

    We no longer enforce i <= pi/2, so we expect to recover the original inclination
    (subject to the standard Thiele-Innes ambiguity).
    """
    # Set appropriate tolerance based on dtype
    # Note: Multiple arctan2 operations accumulate numerical errors
    rtol = 5e-4 if dtype == "float32" else 1e-6

    key = jax.random.PRNGKey(seed)

    # Test multiple random configurations with i > π/2
    for _ in range(5):
        key, subkey = jax.random.split(key)
        random_vals = jax.random.uniform(subkey, shape=(4,))

        # Generate random angles with inclination in (π/2, π)
        arg_peri = random_vals[0] * 2 * jnp.pi
        lon_asc_node = random_vals[1] * 2 * jnp.pi
        inclination = jnp.pi / 2 + random_vals[2] * jnp.pi / 2
        semi_major_axis = Quantity(1.0 + random_vals[3] * 5, "AU")

        orientation = KeplerianOrientation.from_angles(
            arg_peri=Quantity(arg_peri, "rad"),
            lon_asc_node=Quantity(lon_asc_node, "rad"),
            inclination=Quantity(inclination, "rad"),
        )

        _check_thiele_innes_round_trip(orientation, semi_major_axis, rtol=rtol)


def test_thiele_innes_high_inclination_preservation(dtype: str) -> None:
    """Test that inclination > π/2 is preserved (not forced to < π/2)."""
    # Set appropriate tolerance based on dtype
    # Note: Multiple arctan2 operations accumulate numerical errors
    rtol = 5e-4 if dtype == "float32" else 1e-6

    semi_major_axis = Quantity(7.2, "AU")

    # Use inclination > π/2
    high_inclination = KeplerianOrientation.from_angles(
        arg_peri=Quantity(0.8, "rad"),
        lon_asc_node=Quantity(2.3, "rad"),
        inclination=Quantity(2.5, "rad"),  # > π/2
    )

    # We expect to recover the original orientation (subject to depth ambiguity)
    _check_thiele_innes_round_trip(high_inclination, semi_major_axis, rtol=rtol)
