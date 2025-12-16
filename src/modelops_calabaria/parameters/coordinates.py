"""CoordinateSystem: Packaging view + transforms for inference coordinates.

This module implements the CoordinateSystem abstraction that packages together:
- A ParameterView (which parameters are free vs fixed)
- Transform mappings (coordinate transformations for free parameters)

The CoordinateSystem defines bidirectional mappings:
- Z_V ↔ P_V ↔ M

Where:
- Z_V: Inference coordinate space (unconstrained, e.g. log-transformed)
- P_V: Active free parameter space (natural coordinates)
- M: Full model parameter space

This is the core abstraction for optimization and calibration workflows.
"""

from dataclasses import dataclass, field
from typing import Dict, Mapping, Tuple, Union
from types import MappingProxyType
import numpy as np

from .view import ParameterView
from .transforms import Transform, Identity
from .types import ParameterSet


@dataclass(frozen=True)
class CoordinateSystem:
    """Packages view + transforms for inference coordinates.

    Defines the coordinate transformations:
    - Z_V → P_V → M (downward: to_M)
    - M → P_V → Z_V (upward: from_M)

    Where:
    - Z_V: Inference coordinate space (unconstrained, e.g. log-transformed)
    - P_V: Active free parameter space (natural coordinates)
    - M: Full model parameter space

    The vector z ∈ Z_V has length len(view.free) and follows view.free ordering.

    Attributes:
        view: ParameterView defining fixed/free parameters
        transforms: Mapping from parameter names to Transform instances

    Example:
        >>> view = ParameterView.from_fixed(space, gamma=0.1)
        >>> coords = CoordinateSystem(view, {"beta": LogTransform()})
        >>> # Now coords can map z vectors to complete ParameterSets
        >>> z = np.array([0.0])  # log(beta) = 0 → beta = 1.0
        >>> params = coords.to_M(z)  # Complete with beta=1.0, gamma=0.1
    """
    view: ParameterView
    transforms: Mapping[str, Transform] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and freeze the transforms mapping."""
        # Make transforms immutable
        object.__setattr__(self, 'transforms', MappingProxyType(dict(self.transforms)))

        # Validate transforms are only for free parameters
        transform_names = set(self.transforms.keys())
        free_names = set(self.view.free)

        if not transform_names.issubset(free_names):
            unknown = sorted(transform_names - free_names)
            raise ValueError(
                f"Transforms specified for non-free parameters: {unknown}. "
                f"Free parameters: {sorted(free_names)}"
            )

        # Validate all transforms implement the Transform protocol
        for name, transform in self.transforms.items():
            if not (hasattr(transform, 'forward') and
                    hasattr(transform, 'backward') and
                    hasattr(transform, 'bounds')):
                raise TypeError(
                    f"Transform for '{name}' must implement Transform protocol "
                    f"(forward, backward, bounds methods)"
                )

    def _get_transform(self, param_name: str) -> Transform:
        """Get transform for parameter (Identity if not specified)."""
        return self.transforms.get(param_name, Identity())

    def to_M(self, z: np.ndarray) -> ParameterSet:
        """Downward map: Z_V → P_V → M

        Pipeline:
        1. Apply inverse transforms: z[i] → p[free[i]] (Z_V → P_V)
        2. Bind free params with fixed: P_V → M

        Args:
            z: Vector in inference space, length = len(view.free)

        Returns:
            Complete ParameterSet in M-space

        Raises:
            ValueError: If z has wrong dimensionality
        """
        if len(z) != self.dim:
            raise ValueError(
                f"Expected z vector of length {self.dim}, got {len(z)}"
            )

        # Step 1: Z_V → P_V (apply backward/inverse transforms)
        free_values = {}
        for i, param_name in enumerate(self.view.free):
            transform = self._get_transform(param_name)
            z_val = float(z[i])
            p_val = transform.backward(z_val)  # Z → P (inverse)
            free_values[param_name] = p_val

        # Step 2: P_V → M (bind with fixed parameters)
        return self.view.bind(**free_values)

    def from_M(self, params: ParameterSet) -> np.ndarray:
        """Upward map: M → P_V → Z_V

        Pipeline:
        1. Project to free params: M → P_V
        2. Apply forward transforms: p[free[i]] → z[i] (P_V → Z_V)

        Args:
            params: Complete ParameterSet

        Returns:
            Vector in inference space, length = len(view.free)

        Raises:
            ValueError: If params is for different space
        """
        # Step 1: M → P_V (project to free parameters)
        free_dict = self.view.project(params)

        # Step 2: P_V → Z_V (apply forward transforms)
        z = np.zeros(self.dim)
        for i, param_name in enumerate(self.view.free):
            transform = self._get_transform(param_name)
            p_val = free_dict[param_name]
            z_val = transform.forward(p_val)  # P → Z (forward)
            z[i] = z_val

        return z

    def bounds_transformed(self) -> np.ndarray:
        """Get parameter bounds in transformed (Z_V) space.

        Applies transforms to natural bounds to get bounds in inference space.

        Returns:
            (n_free, 2) array of [lower, upper] bounds in Z_V space
        """
        bounds = np.zeros((self.dim, 2))

        for i, param_name in enumerate(self.view.free):
            # Get natural bounds from space
            spec = self.view.space.get_spec(param_name)
            natural_bounds = (spec.lower, spec.upper)

            # Apply transform to get Z_V bounds
            transform = self._get_transform(param_name)
            z_bounds = transform.bounds(natural_bounds, transformed=True)
            bounds[i] = z_bounds

        return bounds

    @property
    def dim(self) -> int:
        """Dimensionality of inference space (number of free parameters)."""
        return len(self.view.free)

    @property
    def param_names(self) -> Tuple[str, ...]:
        """Ordered parameter names (matches z vector dimensions)."""
        return self.view.free

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        n_transforms = len(self.transforms)
        transform_preview = list(self.transforms.keys())[:3]
        if len(self.transforms) > 3:
            transform_preview.append("...")
        return (
            f"CoordinateSystem(dim={self.dim}, "
            f"transforms={n_transforms}{transform_preview})"
        )
