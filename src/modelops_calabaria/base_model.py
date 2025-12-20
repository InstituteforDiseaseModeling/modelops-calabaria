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

    Subclasses MUST set:
    - PARAMS: ParameterSpace defining the model's parameter space

    Subclasses MAY set:
    - CONFIG: ConfigurationSpace defining runtime configuration (optional, defaults to empty)
    - DEFAULT_SCENARIO: Name of default scenario (defaults to "baseline")
    """

    # Subclasses MUST set PARAMS
    PARAMS: Optional[ParameterSpace] = None

    # Subclasses MAY set CONFIG (defaults to empty)
    CONFIG: Optional[ConfigurationSpace] = None

    # Default scenario name
    DEFAULT_SCENARIO: str = "baseline"

    def __init_subclass__(cls, **kwargs):
        """Discover decorated methods and validate class attributes during class definition."""
        super().__init_subclass__(**kwargs)

        # Skip validation for BaseModel itself
        if cls.__name__ == 'BaseModel':
            return

        # Validate PARAMS (required)
        if not hasattr(cls, 'PARAMS') or cls.PARAMS is None:
            raise TypeError(
                f"{cls.__name__}.PARAMS must be set to a ParameterSpace. "
                f"Example: PARAMS = ParameterSpace((ParameterSpec(...), ...))"
            )
        if not isinstance(cls.PARAMS, ParameterSpace):
            raise TypeError(
                f"{cls.__name__}.PARAMS must be a ParameterSpace instance, "
                f"got {type(cls.PARAMS).__name__}"
            )

        # Validate CONFIG if provided (optional)
        if hasattr(cls, 'CONFIG') and cls.CONFIG is not None:
            if not isinstance(cls.CONFIG, ConfigurationSpace):
                raise TypeError(
                    f"{cls.__name__}.CONFIG must be a ConfigurationSpace instance, "
                    f"got {type(cls.CONFIG).__name__}"
                )

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

    def __init__(self, *, base_config: Optional[ConfigurationSet] = None):
        """Initialize model with optional base configuration override.

        The parameter space and configuration space are taken from class attributes
        PARAMS and CONFIG. If base_config is not provided, it defaults to the
        defaults from CONFIG.

        Args:
            base_config: Optional ConfigurationSet to override defaults from CONFIG.
                        If None, uses ConfigurationSet.from_defaults(cls.CONFIG).

        Raises:
            ValueError: If provided base_config space doesn't match cls.CONFIG
        """
        # Get spaces from class attributes
        self.space = type(self).PARAMS
        self.config_space = type(self).CONFIG or ConfigurationSpace(tuple())

        # Set base_config: use provided or create from defaults
        if base_config is None:
            self.base_config = ConfigurationSet.from_defaults(self.config_space)
        else:
            # Validate provided config matches our config_space
            if base_config.space != self.config_space:
                raise ValueError(
                    f"Provided base_config space doesn't match {type(self).__name__}.CONFIG. "
                    f"Expected space with {len(self.config_space.specs)} configs, "
                    f"got space with {len(base_config.space.specs)} configs."
                )
            self.base_config = base_config

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
    def simulate(
        self,
        params: Union[ParameterSet, Dict[str, Any]],
        seed: int,
        config: Optional[Union[ConfigurationSet, Dict[str, Any]]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pl.DataFrame]:
        """Run simulation with optional config replacement or overrides.

        Convenience method that accepts dicts for params and supports flexible
        configuration modification.

        Args:
            params: Complete ParameterSet or dict of parameter values
            seed: Random seed
            config: Optional config modification:
                - Dict: Patches base_config with specified overrides
                - ConfigurationSet: Complete replacement of base_config
            config_overrides: Optional dict to patch specific config values on base_config
                (alternative to passing dict to config)

        Returns:
            Dictionary of output DataFrames

        Raises:
            TypeError: If params is not ParameterSet or dict
            ValueError: If params is incomplete, or both config and config_overrides provided

        Example:
            >>> # Quick run with dict params
            >>> outputs = model.simulate({"beta": 0.08, "gamma": 0.1}, seed=42)
            >>>
            >>> # Override specific config values (two equivalent ways)
            >>> outputs = model.simulate(params, seed=42, config={"population": 5000})
            >>> outputs = model.simulate(params, seed=42, config_overrides={"population": 5000})
            >>>
            >>> # Complete config replacement
            >>> outputs = model.simulate(params, seed=42, config=my_config_set)
        """
        # Convert dict to ParameterSet if needed
        if isinstance(params, dict):
            params = ParameterSet.from_dict(self.space, params)

        return self.simulate_scenario(
            self.DEFAULT_SCENARIO, params, seed,
            config=config, config_overrides=config_overrides
        )

    @final
    def simulate_scenario(
        self,
        scenario: str,
        params: Union[ParameterSet, Dict[str, Any]],
        seed: int,
        config: Optional[Union[ConfigurationSet, Dict[str, Any]]] = None,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pl.DataFrame]:
        """Run specific scenario with optional config replacement or overrides.

        Order of operations:
        1. Start with base_config
        2. If config is dict or config_overrides provided: merge with base_config defaults
        3. If config is ConfigurationSet: use as complete replacement
        4. Apply scenario patches to the effective config
        5. Run simulation

        Args:
            scenario: Name of scenario to run
            params: Complete ParameterSet or dict of parameter values
            seed: Random seed
            config: Optional config modification:
                - Dict: Patches base_config with specified overrides
                - ConfigurationSet: Complete replacement of base_config
            config_overrides: Optional dict to patch specific config values on base_config

        Returns:
            Dictionary of output DataFrames

        Raises:
            ValueError: If scenario is unknown, or both config and config_overrides provided

        Example:
            >>> # Scenario with config overrides (two equivalent ways)
            >>> outputs = model.simulate_scenario(
            ...     "lockdown",
            ...     {"beta": 0.08, "gamma": 0.1},
            ...     seed=42,
            ...     config={"population": 5000}
            ... )
            >>> outputs = model.simulate_scenario(
            ...     "lockdown",
            ...     params,
            ...     seed=42,
            ...     config_overrides={"population": 5000}
            ... )
            >>>
            >>> # Scenario with complete config replacement
            >>> outputs = model.simulate_scenario(
            ...     "lockdown",
            ...     params,
            ...     seed=42,
            ...     config=my_config_set
            ... )
        """
        self._seal()

        # Convert dict to ParameterSet if needed
        if isinstance(params, dict):
            params = ParameterSet.from_dict(self.space, params)

        self._validate(params)

        if scenario not in self._scenarios:
            available = sorted(self._scenarios.keys())
            raise ValueError(
                f"Unknown scenario: {scenario}. Available: {available}"
            )

        # Validate config/config_overrides usage
        if config is not None and config_overrides is not None:
            raise ValueError(
                "Cannot specify both 'config' (complete replacement) and "
                "'config_overrides' (partial patch). Use one or the other."
            )

        spec = self._scenarios[scenario]

        # Determine effective config
        if config is not None:
            # If dict provided, treat as overrides (merge with base_config)
            # If ConfigurationSet provided, use as complete replacement
            if isinstance(config, dict):
                base = ConfigurationSet.from_defaults(self.config_space, **config)
            else:
                # Complete replacement
                base = config
        elif config_overrides is not None:
            # Patch base_config with overrides
            base = ConfigurationSet.from_defaults(self.config_space, **config_overrides)
        else:
            # Use base_config as-is
            base = self.base_config

        # Apply scenario patches to the effective config
        params_patched, config_patched = spec.apply(params, base)

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

    def builder(self, scenario: str = "baseline") -> 'SimulatorBuilder':
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
            ...        .builder("lockdown")
            ...        .fix(gamma=0.1)
            ...        .with_transforms(beta="log")
            ...        .build())
            >>> outputs = sim(z, seed=42)
        """
        from .builder import SimulatorBuilder
        return SimulatorBuilder(_model=self, _scenario=scenario)

    def __repr__(self) -> str:
        """Rich representation showing model structure."""
        lines = [
            f"{self.__class__.__name__}",
            f"Dimension: {len(self.PARAMS.specs)}",
            "",
            "Parameters:"
        ]

        for spec in self.PARAMS.specs:
            # Mathematical interval notation with unicode
            range_str = f"∈ [{spec.lower:.3g}, {spec.upper:.3g}]"
            type_badge = spec.kind[0].upper()  # F for float, I for int

            lines.append(
                f"  • {spec.name:15s} {range_str:20s} ({type_badge})  {spec.doc}"
            )

        # Show config if present
        if self.CONFIG and len(self.CONFIG.specs) > 0:
            lines.append("")
            lines.append(f"Configuration: {len(self.CONFIG.specs)} settings")
            for spec in self.CONFIG.specs[:5]:  # Show first 5
                lines.append(
                    f"  • {spec.name:20s} = {spec.default!r:10}  {spec.doc}"
                )
            if len(self.CONFIG.specs) > 5:
                lines.append(f"  ... and {len(self.CONFIG.specs) - 5} more")

        return "\n".join(lines)

    def __str__(self) -> str:
        """Same as repr for clean display."""
        return self.__repr__()

