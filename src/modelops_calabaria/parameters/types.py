"""Core parameter types for the Grammar of Parameters.

This module implements the fundamental types for parameter management:
- Scalar: Numeric parameter values (float or int)
- ParameterSpec: Specification of a single parameter with bounds
- ParameterSpace: Complete set of parameter specifications (M-space)
- ParameterSet: Immutable, validated assignment of values in M-space

All types are immutable to prevent state bugs and ensure reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Union, Any, Optional
from types import MappingProxyType
import math

# Numeric parameters only - no bool, no str
Scalar = Union[float, int]


@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a single numeric parameter.

    Defines the name, bounds, type, and documentation for a parameter.
    All parameters must be numeric (float or int).

    Attributes:
        name: Parameter identifier
        min: Lower bound (inclusive)
        max: Upper bound (inclusive)
        kind: Type hint ("float" or "int")
        doc: Human-readable description
    """
    name: str
    min: Scalar
    max: Scalar
    kind: str = "float"  # "float" or "int"
    doc: str = ""

    def __post_init__(self):
        """Validate parameter specification."""
        if self.kind not in ("float", "int"):
            raise ValueError(f"Parameter kind must be 'float' or 'int', got {self.kind}")

        if self.min > self.max:
            raise ValueError(f"Parameter {self.name}: min ({self.min}) > max ({self.max})")

        # Ensure int parameters have integer bounds
        if self.kind == "int":
            if not isinstance(self.min, int) or not isinstance(self.max, int):
                raise ValueError(f"Integer parameter {self.name} must have integer bounds")

    def validate_value(self, value: Scalar) -> None:
        """Validate a value against this specification.

        Args:
            value: The value to validate

        Raises:
            TypeError: If value type doesn't match kind
            ValueError: If value is out of bounds or non-finite
        """
        # Explicitly reject booleans (they're technically int subclass in Python)
        if isinstance(value, bool):
            raise TypeError(f"Parameter {self.name} requires {'int' if self.kind == 'int' else 'numeric'} value, got bool")

        # Check for NaN and Inf explicitly
        if isinstance(value, float):
            if math.isnan(value):
                raise ValueError(f"Parameter {self.name} cannot be NaN")
            if math.isinf(value):
                raise ValueError(f"Parameter {self.name} cannot be infinite")

        if self.kind == "int":
            # Accept integer-like floats (e.g., 3.0) but reject non-integer floats (e.g., 3.9)
            if isinstance(value, float):
                if value != math.floor(value):
                    raise TypeError(f"Parameter {self.name} requires integer value, got float {value}")
                # Convert integer-like float to int for validation
                value = int(value)
            elif not isinstance(value, int):
                raise TypeError(f"Parameter {self.name} requires int, got {type(value).__name__}")

        elif self.kind == "float":
            if not isinstance(value, (int, float)):
                raise TypeError(f"Parameter {self.name} requires numeric value, got {type(value).__name__}")

        if not (self.min <= value <= self.max):
            raise ValueError(f"Parameter {self.name}={value} outside bounds [{self.min}, {self.max}]")


@dataclass(frozen=True)
class ParameterSpace:
    """Complete specification of model parameter space (M-space).

    Defines all parameters a model accepts, their types, and bounds.
    The space is immutable - parameters cannot be added or removed after creation.

    Attributes:
        specs: List of parameter specifications
    """
    specs: List[ParameterSpec] = field(default_factory=list)

    def __post_init__(self):
        """Validate space and build lookup structures."""
        # Check for duplicate names
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate parameter names: {set(duplicates)}")

        # Create immutable lookup dict
        spec_dict = {spec.name: spec for spec in self.specs}
        object.__setattr__(self, '_spec_dict', MappingProxyType(spec_dict))

        # Freeze the specs list
        object.__setattr__(self, 'specs', tuple(self.specs))

    def names(self) -> List[str]:
        """Get ordered list of parameter names."""
        return [spec.name for spec in self.specs]

    def get_spec(self, name: str) -> ParameterSpec:
        """Get specification for a parameter.

        Args:
            name: Parameter name

        Returns:
            The parameter specification

        Raises:
            KeyError: If parameter not in space
        """
        if name not in self._spec_dict:
            raise KeyError(f"Unknown parameter: {name}. Available: {sorted(self._spec_dict.keys())}")
        return self._spec_dict[name]

    def __contains__(self, name: str) -> bool:
        """Check if parameter is in space."""
        return name in self._spec_dict

    def __len__(self) -> int:
        """Number of parameters in space."""
        return len(self.specs)

    def to_dict(self) -> Dict[str, Any]:
        """Export parameter space as serializable dictionary.

        Returns:
            Dictionary with parameter specifications in JSON-serializable format
        """
        return {
            "parameters": [
                {
                    "name": spec.name,
                    "min": spec.min,
                    "max": spec.max,
                    "kind": spec.kind,
                    "doc": spec.doc
                }
                for spec in self.specs
            ]
        }


@dataclass(frozen=True)
class ParameterSet:
    """Immutable, complete assignment of values in M-space.

    A ParameterSet represents a single point in the model's parameter space.
    It must specify a value for every parameter and all values are validated
    against the space specification at construction time.

    This is THE ONLY way to pass parameters to BaseModel.simulate(),
    enforcing complete specification and preventing partial parameter bugs.

    Attributes:
        space: The parameter space this set belongs to
        values: Complete mapping of parameter names to values
    """
    space: ParameterSpace
    values: Dict[str, Scalar]

    def __post_init__(self):
        """Validate completeness and freeze values."""
        # Check completeness - all parameters must be specified
        required = set(self.space.names())
        provided = set(self.values.keys())

        # Check for unknown parameters first (catches typos)
        extra = provided - required
        if extra:
            # This catches typos - unknown parameters fail loudly
            raise ValueError(f"Unknown parameters: {sorted(extra)}. Available: {sorted(required)}")

        # Then check for missing parameters
        missing = required - provided
        if missing:
            raise ValueError(f"Missing required parameters: {sorted(missing)}")

        # Validate and potentially convert each value
        validated_values = {}
        for name, value in self.values.items():
            spec = self.space.get_spec(name)
            try:
                # Validate the value
                spec.validate_value(value)

                # Convert integer-like floats to int for int parameters
                if spec.kind == "int" and isinstance(value, float):
                    validated_values[name] = int(value)
                else:
                    validated_values[name] = value

            except (TypeError, ValueError) as e:
                # Re-raise with more context
                raise ValueError(f"Validation failed for {name}: {e}") from e

        # Freeze the validated values dictionary
        object.__setattr__(self, 'values', MappingProxyType(validated_values))

    def __getitem__(self, name: str) -> Scalar:
        """Get parameter value by name.

        Args:
            name: Parameter name

        Returns:
            The parameter value

        Raises:
            KeyError: If parameter not in set
        """
        if name not in self.values:
            raise KeyError(f"Parameter {name} not in set. Available: {sorted(self.values.keys())}")
        return self.values[name]

    def with_updates(self, **updates: Scalar) -> 'ParameterSet':
        """Create new ParameterSet with updated values.

        This method creates a new immutable ParameterSet with the specified
        updates, leaving the original unchanged.

        Args:
            **updates: Parameter values to update

        Returns:
            New ParameterSet with updates applied

        Raises:
            ValueError: If updates contain unknown parameters or invalid values
        """
        new_values = dict(self.values)
        new_values.update(updates)
        return ParameterSet(self.space, new_values)

    def to_dict(self) -> Dict[str, Scalar]:
        """Export values as regular dict (for serialization)."""
        return dict(self.values)

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        items = [f"{k}={v}" for k, v in sorted(self.values.items())]
        return f"ParameterSet({', '.join(items[:3])}{'...' if len(items) > 3 else ''})"

    @classmethod
    def new(cls, space: ParameterSpace, **kwargs: Scalar) -> 'ParameterSet':
        """Factory method for creating ParameterSet with keyword arguments.

        This is a convenience method for creating parameter sets without
        needing to pass a dictionary.

        Args:
            space: The parameter space
            **kwargs: Parameter values as keyword arguments

        Returns:
            New ParameterSet instance

        Example:
            >>> pset = ParameterSet.new(space, alpha=0.5, beta=0.3, steps=50)
        """
        return cls(space, kwargs)