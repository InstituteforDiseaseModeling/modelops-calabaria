"""Decorators for model discovery and registration.

These decorators mark methods for discovery during model initialization.
They are just markers - actual registration happens during instance
construction before the model is sealed.
"""

from typing import Optional, Dict, Any, Callable
import inspect

from .scenarios import ScenarioSpec
from .constants import SEED_COL


def model_output(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Mark a method as a data extractor.

    Decorators are markers only; registration happens on instance
    construction before seal. Extractors must be pure functions
    of (raw, seed) with no global state dependencies.

    IMPORTANT: Framework adds SEED_COL column automatically.
    Extractors must NOT add this column themselves.

    Args:
        name: Output name for this extractor
        metadata: Optional metadata for the output

    Returns:
        Decorator function

    Example:
        @model_output("prevalence")
        def extract_prevalence(self, raw, seed):
            # Extract prevalence time series
            return prevalence_df  # DataFrame without seed column
    """
    def decorator(func: Callable) -> Callable:
        # Validate it's a regular method (not static/class)
        if isinstance(func, (staticmethod, classmethod)):
            raise TypeError(
                f"@model_output '{name}' must be an instance method, "
                f"not static/classmethod"
            )

        # Mark the function
        func._is_model_output = True
        func._output_name = name
        func._output_metadata = metadata or {}
        return func
    return decorator


def model_scenario(name: str):
    """Mark a method as returning a ScenarioSpec.

    Decorators are markers only; registration happens on instance
    construction before seal. The decorated method must return
    a ScenarioSpec when called.

    If 'baseline' is defined via decorator, conflict_policy='lww'
    causes it to replace the default; 'strict' raises an error.

    Args:
        name: Scenario name for registration

    Returns:
        Decorator function

    Example:
        @model_scenario("lockdown")
        def lockdown_scenario(self):
            return ScenarioSpec(
                name="lockdown",
                doc="Reduced contact during lockdown",
                param_patches={"contact_rate": 2.0}
            )
    """
    def decorator(func: Callable) -> Callable:
        # Validate it's a regular method
        if isinstance(func, (staticmethod, classmethod)):
            raise TypeError(
                f"@model_scenario '{name}' must be an instance method, "
                f"not static/classmethod"
            )

        def wrapper(self):
            # Call the original function
            result = func(self)

            # Validate return type
            if not isinstance(result, ScenarioSpec):
                raise TypeError(
                    f"@model_scenario '{name}' must return ScenarioSpec, "
                    f"got {type(result).__name__}"
                )

            # Ensure name matches decorator
            if result.name != name:
                raise ValueError(
                    f"@model_scenario '{name}' returned ScenarioSpec "
                    f"with different name: '{result.name}'"
                )

            return result

        # Preserve metadata
        wrapper._is_model_scenario = True
        wrapper._scenario_name = name
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__

        return wrapper
    return decorator


def discover_decorated_methods(cls) -> tuple[Dict[str, str], Dict[str, str]]:
    """Discover decorated methods in a class.

    Scans the class for methods decorated with @model_output and
    @model_scenario, returning mappings from names to method names.

    This is called during __init_subclass__ to build discovery mappings
    that are used during instance construction.

    Args:
        cls: The class to scan

    Returns:
        Tuple of (output_methods, scenario_methods) where each is
        a dict mapping registered names to method names
    """
    outputs = {}
    scenarios = {}

    # Scan all attributes
    for attr_name in dir(cls):
        # Skip private attributes
        if attr_name.startswith('_'):
            continue

        try:
            attr = getattr(cls, attr_name)
        except AttributeError:
            continue

        # Check for decorator markers
        if hasattr(attr, '_is_model_output'):
            outputs[attr._output_name] = attr_name

        if hasattr(attr, '_is_model_scenario'):
            scenarios[attr._scenario_name] = attr_name

    return outputs, scenarios