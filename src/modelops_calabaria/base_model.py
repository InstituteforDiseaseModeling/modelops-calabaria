"""BaseModel abstract class for Calabaria models.

This module provides the core model interface that enforces immutability,
complete parameter specification, and proper separation of concerns.

Key invariants:
- simulate() ONLY accepts ParameterSet (complete M-space)
- All state is immutable after sealing
- Scenarios are patches, not transforms
- No hidden defaults - complete specification required
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Mapping, Optional, Callable, List, Union, final, Type
from types import MappingProxyType
import inspect

import polars as pl

from .parameters import ParameterSpace, ParameterSet, ConfigurationSpace, ConfigurationSet
from .scenarios import ScenarioSpec
from .constants import SEED_COL


class BaseModel(ABC):
    """Pure functional model with sealed registries.

    Uses decorators as markers; registration happens on instance construction
    before seal. If 'baseline' is defined via decorator, conflict_policy='lww'
    (last write wins) causes it to replace default; 'strict' raises.

    Invariants enforced:
    - simulate() ONLY accepts ParameterSet (complete M)
    - All state is immutable (frozen dataclasses, immutable collections)
    - Scenarios are patches, not transforms
    - Seal-on-first-use prevents configuration mutations
    - No hidden defaults - complete specification required
    """

    DEFAULT_SCENARIO: str = "baseline"  # Subclasses can override

    def __init_subclass__(cls, **kwargs):
        """Discover decorated methods during class definition."""
        super().__init_subclass__(**kwargs)

        # Inherit from parent discoveries (don't clobber!)
        parent_outputs = getattr(cls, "_discovered_outputs", {})
        parent_scenarios = getattr(cls, "_discovered_scenarios", {})

        # Start with copies of parent dicts
        cls._discovered_outputs = dict(parent_outputs)
        cls._discovered_scenarios = dict(parent_scenarios)

        # Add this class's discoveries
        for name, attr in cls.__dict__.items():
            if callable(attr):
                # Check for output decorators
                if getattr(attr, "_is_model_output", False):
                    # Validation already done in decorator
                    cls._discovered_outputs[attr._output_name] = name

                # Check for scenario decorators
                if getattr(attr, "_is_model_scenario", False):
                    # Validation already done in decorator
                    cls._discovered_scenarios[attr._scenario_name] = name

    def __init__(self, space: ParameterSpace, config_space: ConfigurationSpace, base_config: ConfigurationSet):
        """Initialize model with parameter space and configuration.

        Args:
            space: The complete parameter space (M-space)
            config_space: The configuration space (C-space)
            base_config: Base configuration set (must be complete for config_space)
        """
        self.space = space  # Immutable ParameterSpace
        self.config_space = config_space  # Immutable ConfigurationSpace
        self.base_config = base_config  # Immutable ConfigurationSet

        # Mutable until sealed
        self._scenarios: Dict[str, ScenarioSpec] = {
            "baseline": ScenarioSpec("baseline", doc="Default scenario")
        }
        self._outputs: Dict[str, Callable] = {}
        self._sealed = False

        # Register discovered items BEFORE setup_scenarios
        self._register_discovered()

        # Hook for subclasses to add more
        self.setup_scenarios()

    def _register_discovered(self, conflict_policy: str = "lww"):
        """Register discovered items in deterministic order.

        Sorts by name for reproducible registration and stable hashing.

        Args:
            conflict_policy: How to handle duplicate scenarios ("lww" or "strict")
        """
        # Register outputs in sorted order
        for output_name in sorted(self._discovered_outputs):
            method_name = self._discovered_outputs[output_name]
            method = getattr(self, method_name)

            # Validate it's a bound instance method
            if not callable(method):
                raise TypeError(f"Output '{output_name}' did not bind to callable")

            self._outputs[output_name] = method

        # Register scenarios in sorted order
        for scenario_name in sorted(self._discovered_scenarios):
            method_name = self._discovered_scenarios[scenario_name]
            method = getattr(self, method_name)
            spec = method()  # Call to get ScenarioSpec

            # Conflict policy check
            if scenario_name in self._scenarios:
                if conflict_policy == "strict":
                    raise ValueError(
                        f"Duplicate scenario '{scenario_name}'. "
                        f"Use conflict_policy='lww' to override default."
                    )
                # lww: decorator overrides default (including baseline)

            self._scenarios[scenario_name] = spec

    def setup_scenarios(self) -> None:
        """Hook for subclasses to add scenarios before sealing.

        Override this method to programmatically add scenarios that
        aren't defined via decorators.
        """
        pass

    def _seal(self) -> None:
        """Freeze registries on first use.

        After sealing, no modifications to scenarios or outputs are allowed.
        This ensures deterministic behavior and prevents state mutations.
        """
        if not self._sealed:
            self._scenarios = MappingProxyType(dict(self._scenarios))
            self._outputs = MappingProxyType(dict(self._outputs))
            self._sealed = True

    def _validate(self, params: ParameterSet) -> None:
        """Validate ParameterSet completeness.

        Args:
            params: The parameter set to validate

        Raises:
            TypeError: If params is not a ParameterSet
            ValueError: If params is incomplete or has wrong space
        """
        if not isinstance(params, ParameterSet):
            raise TypeError(
                f"simulate() requires ParameterSet, got {type(params).__name__}"
            )

        # Check it's for the right space
        if params.space is not self.space:
            raise ValueError("ParameterSet is for a different parameter space")

    @abstractmethod
    def build_sim(self, params: ParameterSet, config: ConfigurationSet) -> Any:
        """Build simulation state with COMPLETE ParameterSet M.

        This method should be pure and deterministic - no RNG or randomness.
        All stochasticity should be in run_sim.

        Args:
            params: Complete ParameterSet containing ALL model parameters
            config: Configuration set (may be patched by scenario)

        Returns:
            Simulation state object (backend-specific)
        """
        pass

    @abstractmethod
    def run_sim(self, state: Any, seed: int) -> Any:
        """Run simulation with single RNG touchpoint.

        ALL randomness flows through this single seed parameter.
        This ensures reproducibility.

        Args:
            state: Pre-built simulation state from build_sim
            seed: Random seed for all stochasticity

        Returns:
            Raw simulation output (backend-specific)
        """
        pass

    def extract_outputs(self, raw: Any, seed: int) -> Dict[str, pl.DataFrame]:
        """Extract outputs from raw simulation results.

        Automatically adds SEED_COL to all outputs for replicate tracking.

        IMPORTANT: Output extractors must NOT add SEED_COL themselves.
        The framework handles this automatically to ensure consistency.

        Args:
            raw: Raw simulation output from run_sim
            seed: Seed used for this simulation

        Returns:
            Dictionary mapping output names to DataFrames
        """
        self._seal()  # Ensure outputs are registered

        results = {}
        for name, fn in self._outputs.items():
            df = fn(raw, seed)

            # Validate extractor output
            if not isinstance(df, pl.DataFrame):
                raise TypeError(
                    f"Output extractor '{name}' must return pl.DataFrame, "
                    f"got {type(df).__name__}"
                )

            # Check extractor didn't add SEED_COL
            if SEED_COL in df.columns:
                raise ValueError(
                    f"Output extractor '{name}' must not add {SEED_COL} column. "
                    f"Framework adds this automatically."
                )

            # Add seed column (explicitly use Int64 to match numpy-based columns)
            df = df.with_columns(pl.lit(seed, dtype=pl.Int64).alias(SEED_COL))
            results[name] = df

        return results

    @final
    def simulate(self, params: ParameterSet, seed: int) -> Dict[str, pl.DataFrame]:
        """THE ONLY ENTRY POINT - enforces complete ParameterSet.

        This is the engineering constraint that prevents state bugs.
        No dicts, no kwargs, no partial parameters allowed.

        Args:
            params: Complete ParameterSet for space M
            seed: Random seed

        Returns:
            Dictionary of output DataFrames

        Raises:
            TypeError: If params is not a ParameterSet
            ValueError: If params is incomplete
        """
        # Use default scenario
        return self.simulate_scenario(self.DEFAULT_SCENARIO, params, seed)

    @final
    def simulate_scenario(self, scenario: str, params: ParameterSet, seed: int) -> Dict[str, pl.DataFrame]:
        """Run specific scenario with complete ParameterSet.

        Args:
            scenario: Name of scenario to run
            params: Complete ParameterSet for space M
            seed: Random seed

        Returns:
            Dictionary of output DataFrames

        Raises:
            ValueError: If scenario is unknown
        """
        self._seal()
        self._validate(params)

        if scenario not in self._scenarios:
            available = sorted(self._scenarios.keys())
            raise ValueError(
                f"Unknown scenario: {scenario}. Available: {available}"
            )

        spec = self._scenarios[scenario]

        # Apply scenario patches (can FIX but not TRANSFORM)
        params_patched, config_patched = spec.apply(params, self.base_config)

        # Run simulation pipeline
        state = self.build_sim(params_patched, config_patched)
        raw = self.run_sim(state, seed)

        # Extract outputs (includes SEED_COL addition)
        return self.extract_outputs(raw, seed)

    def list_scenarios(self) -> List[str]:
        """List available scenario names.

        Returns:
            Sorted list of scenario names
        """
        self._seal()
        return sorted(self._scenarios.keys())

    def list_outputs(self) -> List[str]:
        """List available output names.

        Returns:
            Sorted list of output names
        """
        self._seal()
        return sorted(self._outputs.keys())

    def get_scenario(self, name: str) -> ScenarioSpec:
        """Get a scenario specification by name.

        Args:
            name: Scenario name

        Returns:
            The ScenarioSpec

        Raises:
            KeyError: If scenario not found
        """
        self._seal()
        if name not in self._scenarios:
            available = sorted(self._scenarios.keys())
            raise KeyError(f"Scenario '{name}' not found. Available: {available}")
        return self._scenarios[name]

    def as_sim(self, scenario: str = "baseline") -> 'SimulatorBuilder':
        """Create a SimulatorBuilder for fluent API construction.

        This is the entry point for the fluent builder API. It returns a
        SimulatorBuilder that can be used to configure parameter fixing,
        transforms, and build a ModelSimulator.

        Args:
            scenario: Scenario name to use (default: "baseline")

        Returns:
            SimulatorBuilder for fluent configuration

        Example:
            >>> model = StochasticSEIR(...)
            >>> sim = (model
            ...        .as_sim("lockdown")
            ...        .fix(gamma=0.1)
            ...        .with_transforms(beta="log")
            ...        .build())
            >>> outputs = sim(z, seed=42)
        """
        from .builder import SimulatorBuilder
        return SimulatorBuilder(_model=self, _scenario=scenario)

