"""Custom types used in epochalypse."""

import jax
from jaxtyping import Float, Int, Real
from unxt import Quantity

NAngle = Real[Quantity["angle"], "n"]  # type: ignore[type-arg]
NTime = Real[Quantity["time"], "n"]  # type: ignore[type-arg]
NVelocity = Real[Quantity["speed"], "n"]  # type: ignore[type-arg]
NFloatArray = Float[jax.Array, "n"]
NIntArray = Int[jax.Array, "n"]
