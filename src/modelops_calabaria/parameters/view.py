"""ParameterView for P-space operations in calibration.

This module implements ParameterView, which defines a partial valuation
of the parameter space that induces a calibration subspace P_v.

The view provides the critical bind() operation (embedding ι: P_v → M)
that bridges from calibration space back to the full model parameter space.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
from types import MappingProxyType

from .types import Scalar, ParameterSpace, ParameterSet


@dataclass(frozen=True)
class ParameterView:
    """Partial valuation of parameter space inducing calibration subspace.

    A ParameterView defines which parameters are fixed (with values) and
    which are free (to be explored/calibrated). The fixed mapping is the
    essential data; free parameters are derived as the complement.

    This provides:
    - Projection π: M → P_v (extract free coordinates)
    - Embedding ι (bind): P_v → M (combine free values with fixed)

    All operations return new views (immutability).

    Attributes:
        space: The complete parameter space M
        fixed: Mapping of fixed parameter names to values
    """
    space: ParameterSpace
    fixed: Dict[str, Scalar] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and freeze the fixed mapping."""
        # Make fixed mapping immutable
        object.__setattr__(self, 'fixed', MappingProxyType(dict(self.fixed)))

        # Validate fixed keys exist in space
        all_names = set(self.space.names())
        fixed_names = set(self.fixed.keys())

        if not fixed_names.issubset(all_names):
            unknown = sorted(fixed_names - all_names)
            raise ValueError(f"Unknown fixed parameters: {unknown}. Available: {sorted(all_names)}")

        # Validate fixed values are numeric (full validation happens at bind time)
        for k, v in self.fixed.items():
            if not isinstance(v, (int, float)):
                raise TypeError(f"Fixed value for {k} must be numeric, got {type(v).__name__}")

    @property
    def free(self) -> Tuple[str, ...]:
        """Derive free parameters as complement of fixed.

        Returns:
            Ordered tuple of free parameter names
        """
        return tuple(n for n in self.space.names() if n not in self.fixed)

    @classmethod
    def all_free(cls, space: ParameterSpace) -> "ParameterView":
        """Create view with all parameters free.

        Args:
            space: The parameter space

        Returns:
            View with no fixed parameters
        """
        return cls(space=space, fixed={})

    @classmethod
    def from_fixed(cls, space: ParameterSpace, **fixed: Scalar) -> "ParameterView":
        """Create view with specified parameters fixed.

        Args:
            space: The parameter space
            **fixed: Parameters to fix with their values

        Returns:
            View with specified fixed parameters
        """
        return cls(space=space, fixed=fixed)

    def fix(self, **kv: Scalar) -> "ParameterView":
        """Return new view with additional parameters fixed.

        This method is immutable - it returns a new view rather than
        modifying the existing one.

        Args:
            **kv: Additional parameters to fix

        Returns:
            New view with merged fixed parameters
        """
        merged = {**self.fixed, **kv}
        return ParameterView(self.space, merged)

    def bind(self, **free_values: Scalar) -> ParameterSet:
        """Create complete ParameterSet from free values (embedding ι: P_v → M).

        This is THE CRITICAL BRIDGE from P-space to M-space.
        Combines the fixed mapping with provided free values to create
        a complete, validated ParameterSet.

        Args:
            **free_values: Values for all free parameters

        Returns:
            Complete ParameterSet in M-space

        Raises:
            ValueError: If free_values doesn't match expected free parameters
        """
        provided = set(free_values.keys())
        expected = set(self.free)

        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            msgs = []
            if missing:
                msgs.append(f"missing: {sorted(missing)}")
            if extra:
                msgs.append(f"unexpected: {sorted(extra)}")
            raise ValueError(f"bind() error: {'; '.join(msgs)}")

        # Create complete M from fixed + free
        complete = {**self.fixed, **free_values}
        return ParameterSet(self.space, complete)

    def project(self, params: ParameterSet) -> Dict[str, Scalar]:
        """Extract free coordinates from complete ParameterSet (projection π: M → P_v).

        This is the dual of bind() - it extracts only the free parameter
        values from a complete ParameterSet, in the order defined by self.free.

        Args:
            params: Complete ParameterSet to project

        Returns:
            Ordered dict of free parameter values

        Raises:
            ValueError: If params is for different space
        """
        if params.space is not self.space:
            raise ValueError("Cannot project ParameterSet from different space")

        # Extract free coordinates in order
        return {name: params[name] for name in self.free}

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        n_fixed = len(self.fixed)
        n_free = len(self.free)
        fixed_preview = list(self.fixed.keys())[:3]
        if len(self.fixed) > 3:
            fixed_preview.append("...")
        return f"ParameterView({n_fixed} fixed{fixed_preview}, {n_free} free)"