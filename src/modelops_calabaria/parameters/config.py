"""Configuration space types for runtime settings.

This module implements configuration management for model runtime settings
that are NOT calibrated (unlike parameters in M-space). Configuration values
include things like simulation timesteps, output frequencies, etc.

Key types:
- ConfigSpec: Specification for a single configuration value
- ConfigurationSpace: Complete set of configuration specifications (C-space)
- ConfigurationSet: Immutable, validated assignment of configuration values

All types are immutable to prevent state bugs and ensure reproducibility.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any
from types import MappingProxyType


@dataclass(frozen=True)
class ConfigSpec:
    """Specification for a single configuration parameter.

    Simpler than ParameterSpec - just name, default, and documentation.
    Type information is implicit from the default value (e.g., 0.1 is float).

    Attributes:
        name: Configuration identifier
        default: Default value (type inferred from this)
        doc: Human-readable description
    """
    name: str
    default: Any
    doc: str = ""

    def __post_init__(self):
        """Validate configuration specification."""
        if not self.name:
            raise ValueError("ConfigSpec name cannot be empty")


@dataclass(frozen=True)
class ConfigurationSpace:
    """Complete specification of model configuration space (C-space).

    Defines all configuration values a model accepts and their defaults.
    The space is immutable - configurations cannot be added or removed after creation.

    Attributes:
        specs: List of configuration specifications
    """
    specs: List[ConfigSpec] = field(default_factory=list)

    def __post_init__(self):
        """Validate space and build lookup structures."""
        # Check for duplicate names
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate configuration names: {set(duplicates)}")

        # Create immutable lookup dict
        spec_dict = {spec.name: spec for spec in self.specs}
        object.__setattr__(self, '_spec_dict', MappingProxyType(spec_dict))

        # Freeze the specs list
        object.__setattr__(self, 'specs', tuple(self.specs))

    def names(self) -> List[str]:
        """Get ordered list of configuration names."""
        return [spec.name for spec in self.specs]

    def get_spec(self, name: str) -> ConfigSpec:
        """Get specification for a configuration value.

        Args:
            name: Configuration name

        Returns:
            The configuration specification

        Raises:
            KeyError: If configuration not in space
        """
        if name not in self._spec_dict:
            raise KeyError(f"Unknown configuration: {name}. Available: {sorted(self._spec_dict.keys())}")
        return self._spec_dict[name]

    def __contains__(self, name: str) -> bool:
        """Check if configuration is in space."""
        return name in self._spec_dict

    def __len__(self) -> int:
        """Number of configurations in space."""
        return len(self.specs)

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration space as serializable dictionary.

        Returns:
            Dictionary with configuration specifications in JSON-serializable format
        """
        return {
            "configurations": [
                {
                    "name": spec.name,
                    "default": spec.default,
                    "doc": spec.doc
                }
                for spec in self.specs
            ]
        }


@dataclass(frozen=True)
class ConfigurationSet:
    """Immutable, complete assignment of configuration values in C-space.

    A ConfigurationSet represents a specific configuration for model runtime.
    It must specify a value for every configuration in the space.

    This enforces complete specification and prevents partial configuration bugs.

    Attributes:
        space: The configuration space this set belongs to
        values: Complete mapping of configuration names to values
    """
    space: ConfigurationSpace
    values: Dict[str, Any]

    def __post_init__(self):
        """Validate completeness and freeze values."""
        # Check completeness - all configurations must be specified
        required = set(self.space.names())
        provided = set(self.values.keys())

        # Check for unknown configurations first (catches typos)
        extra = provided - required
        if extra:
            raise ValueError(f"Unknown configurations: {sorted(extra)}. Available: {sorted(required)}")

        # Then check for missing configurations
        missing = required - provided
        if missing:
            raise ValueError(f"Missing required configurations: {sorted(missing)}")

        # Freeze the values dictionary
        object.__setattr__(self, 'values', MappingProxyType(dict(self.values)))

    def __getitem__(self, name: str) -> Any:
        """Get configuration value by name.

        Args:
            name: Configuration name

        Returns:
            The configuration value

        Raises:
            KeyError: If configuration not in set
        """
        if name not in self.values:
            raise KeyError(f"Configuration {name} not in set. Available: {sorted(self.values.keys())}")
        return self.values[name]

    def with_updates(self, **updates: Any) -> 'ConfigurationSet':
        """Create new ConfigurationSet with updated values.

        This method creates a new immutable ConfigurationSet with the specified
        updates, leaving the original unchanged.

        Args:
            **updates: Configuration values to update

        Returns:
            New ConfigurationSet with updates applied

        Raises:
            ValueError: If updates contain unknown configurations
        """
        new_values = dict(self.values)
        new_values.update(updates)
        return ConfigurationSet(self.space, new_values)

    def to_dict(self) -> Dict[str, Any]:
        """Export values as regular dict (for serialization)."""
        return dict(self.values)

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        items = [f"{k}={v}" for k, v in sorted(self.values.items())]
        return f"ConfigurationSet({', '.join(items[:3])}{'...' if len(items) > 3 else ''})"

    @classmethod
    def new(cls, space: ConfigurationSpace, **kwargs: Any) -> 'ConfigurationSet':
        """Factory method for creating ConfigurationSet with keyword arguments.

        This is a convenience method for creating configuration sets without
        needing to pass a dictionary.

        Args:
            space: The configuration space
            **kwargs: Configuration values as keyword arguments

        Returns:
            New ConfigurationSet instance

        Example:
            >>> config = ConfigurationSet.new(space, dt=0.1, output_freq=1.0)
        """
        return cls(space, kwargs)
