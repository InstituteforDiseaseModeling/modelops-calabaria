"""Scenario system for model variants.

Scenarios are pure data specifications that define model variants through
patches. They can fix parameters and modify configuration but cannot
transform parameters (transforms are for optimization only).

Key principles:
- Scenarios are immutable data (not behavior)
- Patches compose deterministically (last-write-wins or strict)
- Can FIX parameters but NOT transform them
- Applied at simulation time, not configuration time
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Mapping, Optional, Literal, List, Set
from types import MappingProxyType

from .parameters import Scalar, ParameterSet, ParameterSpace


@dataclass(frozen=True)
class ScenarioSpec:
    """Immutable specification of a model scenario.

    A scenario is a named collection of parameter and configuration patches
    that define a model variant. Scenarios can fix parameters to specific
    values and modify configuration settings.

    Scenarios are pure data - they don't contain logic or transforms.
    Multiple scenarios can be composed with configurable conflict resolution.

    Attributes:
        name: Unique scenario identifier
        doc: Human-readable description
        param_patches: Parameter values to fix (name -> value)
        config_patches: Configuration settings to modify (key -> value)
        conflict_policy: How to handle overlapping patches ("lww" or "strict")
        allow_overlap: Parameters allowed to overlap without conflict error
    """
    name: str
    doc: str = ""
    param_patches: Dict[str, Scalar] = field(default_factory=dict)
    config_patches: Dict[str, Any] = field(default_factory=dict)
    conflict_policy: Literal["lww", "strict"] = "lww"  # last-write-wins or strict
    allow_overlap: Optional[Tuple[str, ...]] = None  # params allowed to overlap

    def __post_init__(self):
        """Make patches immutable."""
        # Convert to immutable mappings
        object.__setattr__(self, 'param_patches',
                          MappingProxyType(dict(self.param_patches)))
        object.__setattr__(self, 'config_patches',
                          MappingProxyType(dict(self.config_patches)))

        # Ensure allow_overlap is a tuple (immutable)
        if self.allow_overlap is not None:
            object.__setattr__(self, 'allow_overlap',
                              tuple(self.allow_overlap))

    def apply(self, params: ParameterSet, config: Mapping) -> Tuple[ParameterSet, Mapping]:
        """Apply patches to parameters and configuration.

        Validates that parameter patches reference valid parameters and
        values are within bounds. Config patches are applied without validation.

        This method is pure - it returns new objects without modifying inputs.

        Args:
            params: Current parameter set
            config: Current configuration

        Returns:
            Tuple of (updated_params, updated_config)

        Raises:
            ValueError: If parameter patches reference unknown parameters or
                       values are out of bounds
        """
        # Access space from ParameterSet
        space = params.space

        # Validate param_patches
        if self.param_patches:
            # Check all parameters exist and values are valid
            for param_name, value in self.param_patches.items():
                # Check parameter exists
                if param_name not in space:
                    available = sorted(space.names())
                    raise ValueError(
                        f"Scenario '{self.name}' references unknown parameter: {param_name}. "
                        f"Available: {available}"
                    )

                # Validate value (will check type and bounds)
                spec = space.get_spec(param_name)
                try:
                    spec.validate_value(value)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Scenario '{self.name}' invalid value for {param_name}: {e}"
                    ) from e

        # Apply parameter patches (creates new ParameterSet)
        if self.param_patches:
            new_params = params.with_updates(**self.param_patches)
        else:
            new_params = params

        # Apply config patches (creates new immutable mapping)
        if self.config_patches:
            new_config = MappingProxyType({**config, **self.config_patches})
        else:
            new_config = config

        return new_params, new_config

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        n_params = len(self.param_patches)
        n_config = len(self.config_patches)
        return f"ScenarioSpec('{self.name}', {n_params} params, {n_config} config)"


def compose_scenarios(
    scenarios: List[ScenarioSpec],
    params: ParameterSet,
    config: Mapping
) -> Tuple[ParameterSet, Mapping, Dict[str, str]]:
    """Apply multiple scenarios in order with conflict detection.

    Scenarios are applied sequentially with last-write-wins semantics
    by default. Individual scenarios can use conflict_policy="strict"
    to enforce conflict detection.

    Args:
        scenarios: Ordered list of scenarios to apply
        params: Base parameter set
        config: Base configuration

    Returns:
        Tuple of (final_params, final_config, param_sources)
        where param_sources maps parameter names to scenario names

    Raises:
        ValueError: If strict mode and conflicts detected
    """
    current_params = params
    current_config = config
    param_sources: Dict[str, str] = {}  # Track which scenario set each param

    for spec in scenarios:
        # Check for conflicts if any scenario uses strict mode
        if spec.conflict_policy == "strict" and spec.param_patches:
            for param_name in spec.param_patches:
                if param_name in param_sources:
                    # Check if overlap is allowed
                    if (spec.allow_overlap is None or
                        param_name not in spec.allow_overlap):
                        raise ValueError(
                            f"Parameter '{param_name}' set by multiple scenarios: "
                            f"{param_sources[param_name]} and {spec.name}. "
                            f"Use conflict_policy='lww' or add to allow_overlap."
                        )

        # Track parameter sources BEFORE applying
        # (so we can detect conflicts for the current scenario)
        for param_name in spec.param_patches:
            param_sources[param_name] = spec.name

        # Apply the scenario
        current_params, current_config = spec.apply(current_params, current_config)

    return current_params, current_config, param_sources


def _freeze_jsonable(obj):
    """Convert arbitrary objects to deterministically serializable form."""
    if isinstance(obj, dict):
        return {k: _freeze_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_freeze_jsonable(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime-like objects
        return obj.isoformat()
    else:
        # For other objects, convert to string but ensure consistency
        return str(obj)


def scenario_hash(spec: ScenarioSpec) -> str:
    """Generate deterministic hash for a scenario.

    Used for provenance tracking and caching.

    Args:
        spec: Scenario specification

    Returns:
        Hex string hash of scenario content
    """
    from .cli.hashing import content_hash

    # Create deterministic representation
    content = {
        "name": spec.name,
        "param_patches": {k: v for k, v in sorted(spec.param_patches.items())},
        "config_patches": _freeze_jsonable({k: v for k, v in sorted(spec.config_patches.items())}),
        "conflict_policy": spec.conflict_policy,
        "allow_overlap": sorted(spec.allow_overlap) if spec.allow_overlap else None,
    }

    # Use canonical JSON hashing
    full_hash = content_hash(content)
    # Extract hex part after "sha256:" prefix and take first 16 characters
    return full_hash.split(":")[-1][:16]