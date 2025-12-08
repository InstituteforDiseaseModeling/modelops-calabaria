"""Transform system for optimization coordinates.

Transforms provide bijective mappings between natural parameter space
and optimization space. They are applied ONLY to free parameters and
do NOT change dimensionality - only the coordinate system.

This is distinct from scenarios (which patch values) and reparameterizations
(which can change dimensionality, not in MVP).
"""

import math
from typing import Protocol, Tuple
from dataclasses import dataclass


class Transform(Protocol):
    """Protocol for parameter transforms.

    Transforms provide forward (natural → optimization) and
    backward (optimization → natural) mappings.
    """

    def forward(self, x: float) -> float:
        """Transform from natural space to optimization space.

        Args:
            x: Value in natural parameter space

        Returns:
            Value in optimization space

        Raises:
            ValueError: If x is outside valid domain
        """
        ...

    def backward(self, y: float) -> float:
        """Transform from optimization space to natural space.

        Args:
            y: Value in optimization space

        Returns:
            Value in natural parameter space
        """
        ...

    def bounds(self, natural_bounds: Tuple[float, float], transformed: bool = False) -> Tuple[float, float]:
        """Get bounds in natural or transformed space.

        Args:
            natural_bounds: (min, max) in natural space
            transformed: If True, return bounds in transformed space

        Returns:
            Bounds in requested space
        """
        ...


@dataclass(frozen=True)
class Identity:
    """Identity transform (no-op).

    Useful as default or when no transform is needed.
    """

    def forward(self, x: float) -> float:
        """Return value unchanged."""
        return float(x)

    def backward(self, y: float) -> float:
        """Return value unchanged."""
        return float(y)

    def bounds(self, natural_bounds: Tuple[float, float], transformed: bool = False) -> Tuple[float, float]:
        """Bounds are unchanged by identity transform."""
        return natural_bounds


@dataclass(frozen=True)
class LogTransform:
    """Logarithmic transform for positive parameters.

    Maps (0, ∞) → (-∞, ∞) for unconstrained optimization.
    """

    def forward(self, x: float) -> float:
        """Natural → log space."""
        if x <= 0:
            raise ValueError(f"LogTransform requires x > 0, got {x}")
        return math.log(x)

    def backward(self, y: float) -> float:
        """Log space → natural."""
        return math.exp(y)

    def bounds(self, natural_bounds: Tuple[float, float], transformed: bool = False) -> Tuple[float, float]:
        """Transform bounds to/from log space."""
        min_val, max_val = natural_bounds
        if min_val <= 0:
            raise ValueError(f"LogTransform requires positive bounds, got [{min_val}, {max_val}]")

        if not transformed:
            return natural_bounds
        return (math.log(min_val), math.log(max_val))


@dataclass(frozen=True)
class LogitTransform:
    """Logit transform for [0,1] bounded parameters.

    Maps [0, 1] → (-∞, ∞) for unconstrained optimization.
    Note: Undefined at exact 0 and 1; use AffineSqueezedLogit for robustness.
    """

    def forward(self, x: float) -> float:
        """Natural [0,1] → unbounded."""
        if not (0.0 < x < 1.0):
            raise ValueError(f"LogitTransform requires 0 < x < 1, got {x}")
        return math.log(x / (1.0 - x))

    def backward(self, y: float) -> float:
        """Unbounded → natural [0,1]."""
        return 1.0 / (1.0 + math.exp(-y))

    def bounds(self, natural_bounds: Tuple[float, float], transformed: bool = False) -> Tuple[float, float]:
        """Transform bounds to/from logit space."""
        min_val, max_val = natural_bounds
        if not (0.0 <= min_val < max_val <= 1.0):
            raise ValueError(f"LogitTransform requires bounds in [0,1], got [{min_val}, {max_val}]")

        if not transformed:
            return natural_bounds

        # In transformed space, bounds approach ±∞
        # Use large finite values for practical optimization
        return (-10.0, 10.0)  # Corresponds roughly to [0.00005, 0.99995]


@dataclass(frozen=True)
class AffineSqueezedLogit:
    """Robust logit transform for [0,1] bounded parameters.

    Squeezes the domain slightly away from 0 and 1 using epsilon,
    then applies logit transform. This avoids numerical issues at boundaries.

    The transform maps [0,1] → (-∞, ∞) via:
    1. Affine squeeze: x → eps + (1-2*eps)*x  (maps [0,1] → [eps, 1-eps])
    2. Logit: p → log(p/(1-p))

    This is the recommended transform for [0,1] bounded parameters in optimization.
    """
    eps: float = 1e-6

    def __post_init__(self):
        """Validate epsilon parameter."""
        if not (0 < self.eps < 0.5):
            raise ValueError(f"eps must be in (0, 0.5), got {self.eps}")

    def forward(self, x: float) -> float:
        """Natural [0,1] → unbounded for optimizer."""
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"AffineSqueezedLogit requires 0≤x≤1, got {x}")

        # Squeeze away from boundaries
        p = self.eps + (1.0 - 2.0 * self.eps) * x

        # Apply logit
        return math.log(p / (1.0 - p))

    def backward(self, y: float) -> float:
        """Unbounded → natural [0,1]."""
        # Inverse logit (sigmoid)
        s = 1.0 / (1.0 + math.exp(-y))

        # Inverse affine squeeze
        x = (s - self.eps) / (1.0 - 2.0 * self.eps)

        # Clamp to [0, 1] to handle numerical edge cases
        # This ensures we never return values outside bounds due to floating point errors
        return max(0.0, min(1.0, x))

    def bounds(self, natural_bounds: Tuple[float, float], transformed: bool = False) -> Tuple[float, float]:
        """Transform bounds to/from squeezed logit space."""
        min_val, max_val = natural_bounds
        if not (0.0 <= min_val < max_val <= 1.0):
            raise ValueError(f"AffineSqueezedLogit requires bounds in [0,1], got [{min_val}, {max_val}]")

        if not transformed:
            return natural_bounds

        # Transform the bounds through forward mapping
        return (self.forward(min_val), self.forward(max_val))
