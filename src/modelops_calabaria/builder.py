"""Fluent builder API for creating ModelSimulator instances.

This module provides SimulatorBuilder, a fluent API for constructing
ModelSimulator instances with a clean, expressive syntax:

    sim = (model
           .as_sim("baseline")
           .fix(gamma=0.1)
           .with_transforms(beta="log")
           .build())

The builder handles:
- Scenario selection
- Parameter fixing (which parameters are free vs fixed)
- Transform specification (coordinate system setup)
- ModelSimulator construction
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Union

from .parameters import (
    ParameterView,
    CoordinateSystem,
    Transform,
    Identity,
    LogTransform,
    LogitTransform,
    AffineSqueezedLogit,
    Scalar,
)
from .simulator import ModelSimulator
from .base_model import BaseModel


# Transform name registry for string-based transform specification
_TRANSFORM_REGISTRY = {
    "identity": Identity,
    "log": LogTransform,
    "logit": AffineSqueezedLogit,  # Recommended for [0,1] parameters
}


def _resolve_transform(spec: Union[str, Transform]) -> Transform:
    """Resolve transform specification to Transform instance.

    Args:
        spec: Either a string name ("log", "logit", "identity") or a Transform instance

    Returns:
        Transform instance

    Raises:
        ValueError: If string name is unknown

    Example:
        >>> _resolve_transform("log")
        LogTransform()
        >>> _resolve_transform(LogTransform())
        LogTransform()
    """
    if isinstance(spec, str):
        spec_lower = spec.lower()
        if spec_lower not in _TRANSFORM_REGISTRY:
            available = ", ".join(sorted(_TRANSFORM_REGISTRY.keys()))
            raise ValueError(
                f"Unknown transform '{spec}'. "
                f"Available: {available}"
            )
        transform_class = _TRANSFORM_REGISTRY[spec_lower]
        return transform_class()
    else:
        # Assume it's already a Transform instance
        return spec


@dataclass
class SimulatorBuilder:
    """Fluent builder for ModelSimulator instances.

    The builder is immutable - each method returns a new builder instance
    with updated state. This allows for clean, composable API usage.

    Example:
        >>> # Start from a model
        >>> model = StochasticSEIR(...)
        >>>
        >>> # Build simulator with fluent API
        >>> sim = (model
        ...        .as_sim("lockdown")           # Select scenario
        ...        .fix(gamma=0.1, population=1000)  # Fix some params
        ...        .with_transforms(beta="log")  # Transform others
        ...        .build())                      # Create ModelSimulator
        >>>
        >>> # Now use simulator
        >>> z = np.array([0.5])  # Just beta (others fixed)
        >>> outputs = sim(z, seed=42)

    Attributes:
        _model: The BaseModel to simulate
        _scenario: Scenario name to apply
        _fixed: Dict of fixed parameter values
        _transforms: Dict of transforms for free parameters
    """
    _model: BaseModel
    _scenario: str = "baseline"
    _fixed: Dict[str, Scalar] = field(default_factory=dict)
    _transforms: Dict[str, Transform] = field(default_factory=dict)

    def fix(self, **params: Scalar) -> 'SimulatorBuilder':
        """Fix parameters (add to fixed set).

        Fixed parameters are not varied during calibration/optimization.
        Only the remaining free parameters appear in the z vector.

        Args:
            **params: Parameters to fix with their values

        Returns:
            New builder with updated fixed parameters

        Example:
            >>> builder = builder.fix(gamma=0.1, population=1000)
            >>> # Now gamma and population are fixed
            >>> # Only remaining params are free
        """
        new_fixed = {**self._fixed, **params}
        return replace(self, _fixed=new_fixed)

    def with_transforms(self, **transforms: Union[str, Transform]) -> 'SimulatorBuilder':
        """Specify transforms for free parameters.

        Transforms map between natural parameter space and inference space.
        Common transforms:
        - "log": For positive parameters (rates, counts)
        - "logit": For probability parameters [0,1]
        - "identity": No transformation (default)

        Args:
            **transforms: Parameter names to transform specifications
                Can be string names ("log", "logit") or Transform instances

        Returns:
            New builder with updated transforms

        Raises:
            ValueError: If transform name is unknown

        Example:
            >>> # String-based (convenient)
            >>> builder = builder.with_transforms(
            ...     beta="log",
            ...     alpha="logit"
            ... )
            >>>
            >>> # Or with Transform instances (more control)
            >>> builder = builder.with_transforms(
            ...     beta=LogTransform(),
            ...     alpha=AffineSqueezedLogit(eps=1e-5)
            ... )
        """
        # Resolve all transform specifications
        resolved = {
            name: _resolve_transform(spec)
            for name, spec in transforms.items()
        }
        new_transforms = {**self._transforms, **resolved}
        return replace(self, _transforms=new_transforms)

    def build(self) -> ModelSimulator:
        """Build the final ModelSimulator.

        Creates:
        1. ParameterView from fixed parameters
        2. CoordinateSystem with view + transforms
        3. ModelSimulator wrapping model + scenario + coords

        Configuration comes from model.base_config (c₀) - it's not
        manipulated by the builder.

        Returns:
            ModelSimulator ready for execution

        Raises:
            ValueError: If validation fails (e.g., transform for fixed param)

        Example:
            >>> sim = builder.build()
            >>> # Now can execute
            >>> outputs = sim(z, seed=42)
        """
        # Create parameter view (defines free vs fixed)
        view = ParameterView.from_fixed(self._model.space, **self._fixed)

        # Create coordinate system (view + transforms)
        coords = CoordinateSystem(view, self._transforms)

        # Create and return simulator
        return ModelSimulator(self._model, self._scenario, coords)

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        n_fixed = len(self._fixed)
        n_transforms = len(self._transforms)
        return (
            f"SimulatorBuilder("
            f"model={self._model.__class__.__name__}, "
            f"scenario='{self._scenario}', "
            f"fixed={n_fixed}, "
            f"transforms={n_transforms})"
        )
