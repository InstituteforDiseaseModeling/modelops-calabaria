# Calabaria MVP - Pure Functional Science, Clean Contracts
**Engineering Plan & Design Document**

> **tl;dr**: Pure functional core with pragmatic facades. Scenarios are model
> variants, not parameters. Clean contracts via SimTask.
> Content-addressed caching via provenance. 6-week MVP timeline.

---

## Executive Summary

### Vision

Build Calabaria as a **pure functional simulation framework** with an ergonomic
object-oriented facade. This guarantees reproducibility, enables
content-addressed caching, and scales from laptop to cloud—while keeping the
scientist experience delightful.

### Key Principles
1. **Pure functional core**: All computations depend only on explicit inputs
2. **Immutability by default**: Data flows through transformations, never mutates
3. **Explicit effects**: I/O, RNG, and side effects only at boundaries
4. **Content-addressed provenance**: Every computation has a deterministic hash
5. **Clean contracts**: Minimal, focused interfaces between systems

### What We're Building
- A parameter system with transforms and views
- A model interface with scenario variants (not parameter mutations)
- A bridge to ModelOps via clean SimTask contracts
- Content-addressed caching via provenance hashes
- A 6-week MVP implementation

### What We're NOT Building (Yet)
- Streaming/chunked outputs (future TableHandle extension)
- Coupled parameter transforms (separable only for MVP)
- Constraint systems or units (future layers)
- Multi-language support (Python-only MVP)

---

## Part I: Architecture & Design

### 1. Pure Functional Foundation

#### Why the Pure Functional Contract Matters (Even in Python)

**1. Reproducibility You Can Bet On**  

`simulate_raw(pset, seed, scenario)` and `extract(raw, seed, outputs)` are
deterministic. Same inputs ⇒ same outputs. No hidden globals, no accidental
dependence on wall-clock, env vars, or mutable singletons.

**2. Perfect Caching & Dedup on Any Executor**  

Determinism lets us content-address results (CAS) by their provenance hash.
Retries are idempotent; duplicates collapse; distributed runs won't "double
count". This saves real money at scale.

**3. Clean Provenance Trees**  

We can hash params, config, code identity, seed, and scenario into a stable
sim_root. Since `extract(...)` is pure, output selection does not pollute
simulation provenance (it's part of the entrypoint identity, not the sim).

**4. Safer Distributed Orchestration**  

Pure tasks are trivially parallelizable, resumable, and replayable. Crashes or
pre-emption do not corrupt state because there is no shared state.

**5. Minimal Interfaces, Maximal Composability**  

A small set of referentially transparent functions composes reliably:
`build_sim ∘ run_sim ∘ extract`. Currying scenario and outputs yields tailored,
cost-aware entrypoints without growing the SimTask surface.

**6. Debuggability & Auditability**  

If outputs look odd, re-run with the same `(pset, seed, scenario)` and confirm.
Provenance + purity gives an audit trail you can actually trust.

**Bottom line**: Purity turns "works on my machine" into "works anywhere,
forever". It enables hashing, caching, retries, and distribution with near-zero
foot-guns, while keeping the scientist's API ergonomic.

#### The Pattern

```python
# Pure functional core
def build_state(params: dict, config: dict) -> Any:
    """Deterministic assembly - no RNG, no seed"""
    # Build matrices, networks, initial conditions
    # ALL deterministic setup happens here
    return state

def run_sim(state: Any, seed: int) -> Any:
    """Single RNG touchpoint - all randomness flows through seed"""
    rng = np.random.default_rng(seed)
    # Run simulation using pre-built state
    return raw_output

# Pragmatic OO facade
class Model(BaseModel):
    DEFAULT_SCENARIO = "conservative"  # Configurable default

    def build_state(self, params: ParameterSet, config: Mapping[str, Any]) -> Any:
        """Implement deterministic assembly
        Args:
            params: Complete parameter set M (ALL model parameters)
        """
        return {...}  # No seed parameter!

    def run_sim(self, state: Any, seed: int) -> RawSimOutput:
        """Implement stochastic execution"""
        return {...}  # Single seed touchpoint

    @final
    def simulate(self, params: ParameterSet, seed: int):
        """Run simulation with complete parameters M
        Args:
            params: Complete ParameterSet containing ALL model parameters
            seed: Random seed
        """
        return super().simulate(params, seed)

# Parameter Space Distinction Example
"""
Model Space M: {beta, gamma, population, contact_rate, recovery_days}
Calibration Space P: {beta, gamma}  # Only these are optimized
Fixed Parameters: {population=10000, contact_rate=4.0, recovery_days=14}

During calibration:
- Optimizer works with P = {beta, gamma}
- CalibrationAdapter converts P → M before calling model.simulate()
- Model always receives complete M
"""
```

### 2. Core Types & Parameter System

The main goal of the parameter system is to **make the most common
researcher-user operations the easiest to do**. These operations are:

- Defining the model parameter space (M), and setting the bounds *once* and
  having everything propagate through, all the way to
  calibration, automatically.

- Fixing parameters (conditioning on) scalar values, an essential operation of
  pre-calibration checks, post-calibration sensitivity analyses, etc.

- These parameter operations are used by both downstream calibrations (e.g. a
  user fixes some parameters and runs a calibration on a reduced parameter
  space P ⊂ M from the model M), as well as exploratory sim work (e.g. fixing
  all parameters to their optima except one, and seeing how it impacts loss).

All of these should be easy to do, in a consistent, clean user-interface
designed to be expressive and powerful.

#### Type Aliases
```python
from typing import Union, Any

# Core type aliases for clarity
Scalar = Union[float, int, bool, str]  # Wire format for parameters
RawSimOutput = Any  # Backend-specific simulation state (e.g., numpy arrays, dicts)
OutputName = str    # Name of an output extractor
```

#### Parameter Specifications
```python
from dataclasses import dataclass
from typing import Literal, Tuple, Dict, Optional, Callable

@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float
    upper: float
    kind: Literal["real", "int", "cat"] = "real"
    doc: str = ""
    validator: Optional[Callable[[Scalar], bool]] = None
    
    def validate(self, value: Scalar) -> None:
        """Validate bounds and custom constraints"""
        if self.kind in ("real", "int"):
            if not (float(self.lower) <= float(value) <= float(self.upper)):
                raise ValueError(f"{self.name}={value} outside [{self.lower},{self.upper}]")
        if self.validator and not self.validator(value):
            raise ValueError(f"{self.name}={value} failed validator")

@dataclass(frozen=True)
class ParameterSpace:
    specs: Tuple[ParameterSpec, ...]
    version: str = "1.0"
    
    def by_name(self) -> Dict[str, ParameterSpec]:
        return {s.name: s for s in self.specs}
```

#### Parameter Spaces: Model (M) vs Calibration (P)

**Key Principle**: Models always work with complete parameter space M.
Calibration algorithms work with subset P ⊆ M. The CalibrationAdapter handles P
→ M conversion.

```python
# Two distinct parameter spaces:
# M = Model Parameter Space (complete, what model.simulate() expects)
# P = Calibration Parameter Space (subset being optimized, P ⊆ M)

@dataclass(frozen=True)
class ParameterSet:
    """Complete parameter assignment for space M."""
    values: Dict[str, Scalar]  # Immutable parameter assignment

    def __getitem__(self, k: str) -> Scalar:
        """Allow dict-style access to parameter values."""
        return self.values[k]

    def with_updates(self, **updates: Scalar) -> "ParameterSet":
        """Create new ParameterSet with updated values (immutable update)."""
        d = dict(self.values)
        d.update(updates)
        return ParameterSet(d)

    def reunite_with_free(self, fixed: Dict[str, Scalar],
                          free_values: Dict[str, Scalar]) -> "ParameterSet":
        """Combine fixed + free values into complete set M.

        This is the KEY method for P → M conversion:
        - fixed: Parameters held constant during calibration
        - free_values: Parameters being optimized (from space P)
        Returns: Complete parameter set for space M
        """
        total = {**fixed, **free_values}
        return ParameterSet(values=total)
```

#### Transform System
```python
from typing import Protocol

class Transform(Protocol):
    def forward(self, x: float) -> float: ...
    def backward(self, y: float) -> float: ...

@dataclass(frozen=True)
class Identity:
    """Identity transform for untransformed parameters."""
    def forward(self, x: float) -> float: 
        return float(x)
    def backward(self, y: float) -> float: 
        return float(y)

@dataclass(frozen=True)
class AffineSqueezedLogit:
    """
    Smooth logit for [0,1] parameters.
    Maps [0,1] → ℝ by squeezing to (ε, 1-ε) then applying logit.
    
    Forward:  y = log((ε + (1-2ε)x) / (1 - ε - (1-2ε)x))
    Backward: x = (σ(y) - ε) / (1 - 2ε), where σ(y) = 1/(1+e^(-y))
    """
    eps: float = 1e-6
    
    def forward(self, x: float) -> float:
        import math
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"AffineSqueezedLogit requires 0≤x≤1, got {x}")
        p = self.eps + (1.0 - 2.0*self.eps) * x
        return math.log(p / (1.0 - p))
    
    def backward(self, y: float) -> float:
        import math
        s = 1.0 / (1.0 + math.exp(-y))
        return (s - self.eps) / (1.0 - 2.0*self.eps)

@dataclass(frozen=True)
class TransformedView:
    """View with optimizer-friendly coordinates"""
    view: ParameterView
    transforms: Dict[str, Transform]
    
    def to_transformed(self, pset: ParameterSet) -> Dict[str, float]:
        """Natural → optimizer coordinates"""
        out = {}
        for name in self.view.free:
            val = pset.values[name]
            t = self.transforms.get(name, Identity())
            out[name] = t.forward(float(val))
        return out
    
    def from_transformed(self, coords: Dict[str, float]) -> ParameterSet:
        """Optimizer → natural coordinates"""
        values = dict(self.view.fixed)
        for name in self.view.free:
            t = self.transforms.get(name, Identity())
            values[name] = t.backward(coords[name])
        return ParameterSet(values=values)
```

### 3. Model Interface & Scenarios

#### Scenarios as Model Variants

A scenario is just a *pure transformation* on inputs before the core
simulation:

```
Scenario: (ParameterSet, Config) -> (ParameterSet', Config')

```

- A "lockdown" scenario might change contact patterns
- A "vaccination" scenario might add intervention logic
- Each scenario is a distinct simulation variant
- A scenario can "flip an engine switch" by patching Config.
- A scenario can override a parameter by patching ParameterSet.
- **A scenario is side-effect free and deterministic** (hence why *no seed is
  passed in*.

**Why Callable Transforms?**

Scenarios use `Callable` transforms instead of static dicts for maximum expressiveness:
- **Relative changes**: `lambda p: {**p, "beta": p["beta"] * 0.3}` (70% reduction)
- **Conditional logic**: `lambda p: {**p, "beta": 0.1 if p["gamma"] > 0.5 else p["beta"]}`
- **Complex relationships**: Any parameter interdependencies can be encoded
- **External data**: `lambda c: {**c, "contact_matrix": load_matrix()}`

At compile time, these transforms are "baked in" to create distinct functions.

```python
@dataclass(frozen=True)
class ScenarioSpec:
    """Pure transform specification for a scenario.
    
    A scenario is a pure transformation: (ParameterSet, Config) → (ParameterSet', Config')
    This covers engine switches (config patches) and parameter overrides.
    """
    name: str
    patch_params: Callable[[ParameterSet], ParameterSet] = lambda x: x  # identity
    patch_config: Callable[[Mapping[str, Any]], Mapping[str, Any]] = lambda x: x  # identity
    doc: str = ""
```

#### Decorators

```python
from typing import Callable, Any, Optional, Dict
import polars as pl

def model_output(name: str, metadata: Optional[Dict[str, Any]] = None):
    """Marks a method that converts raw sim output → pl.DataFrame."""
    def decorator(fn: Callable[..., pl.DataFrame]):
        setattr(fn, "_is_model_output", True)
        setattr(fn, "_output_name", name)
        setattr(fn, "_output_metadata", metadata or {})
        return fn
    return decorator

def model_scenario(name: str):
    """Declares a pure scenario that returns ScenarioSpec when called."""
    def decorator(fn: Callable[..., "ScenarioSpec"]):
        setattr(fn, "_is_model_scenario", True)
        setattr(fn, "_scenario_name", name)
        return fn
    return decorator
```

#### Base Model Interface

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union, final
from types import MappingProxyType
import io
import polars as pl
import numpy as np

# Type aliases
Scalar = Union[bool, int, float, str]
RawSimOutput = Any  # Backend-specific raw output
OutputTable = pl.DataFrame

class BaseModel(ABC):
    """Pure functional model with pragmatic facade.
    
    Key design principles:
    • build_state(params, config) → state: Deterministic assembly (NO seed)
    • run_sim(state, seed) → raw: Single RNG touchpoint, all randomness here
    • Scenarios compile to entrypoints, not runtime parameters
    • simulate() uses DEFAULT_SCENARIO via compiled closure for dev/prod parity
    • Registry freezes (seals) on first use to protect provenance
    
    Engineering Choices:
    - Seal-on-first-use: More flexible than immediate freezing
    - DEFAULT_SCENARIO: Configurable class attribute, no "baseline" magic
    - Stable ordering: Sort by name for deterministic behavior
    - Uniform compilation: No special cases in compile logic
    """
    
    # Adjustable default scenario name (subclasses can override)
    DEFAULT_SCENARIO: str = "baseline"
    
    def __init__(self, space: ParameterSpace, base_config: Mapping[str, Any]):
        # Immutable inputs
        self.space: ParameterSpace = space
        self.base_config: Mapping[str, Any] = MappingProxyType(dict(base_config))
        
        # Registries (mutable until sealed)
        self._scenarios: Dict[str, ScenarioSpec] = {}
        self._outputs: Dict[str, Callable[[RawSimOutput, int], pl.DataFrame]] = {}
        self._sealed: bool = False
        
        # Register an identity scenario under DEFAULT_SCENARIO
        self._scenarios[self.DEFAULT_SCENARIO] = ScenarioSpec(
            name=self.DEFAULT_SCENARIO,
            patch_params=lambda x: x,  # identity
            patch_config=lambda x: x,  # identity
            doc=f"Default {self.DEFAULT_SCENARIO} scenario with no modifications"
        )
        
        # Subclass hook to tweak scenarios before discovery
        self.setup_scenarios()
        
        # Auto-discover decorated scenarios/outputs with stable ordering by name
        for attr in sorted(dir(self)):
            fn = getattr(self, attr)
            if callable(fn):
                if getattr(fn, "_is_model_scenario", False):
                    spec = fn()  # Returns ScenarioSpec
                    self._scenarios[spec.name] = spec
                elif getattr(fn, "_is_model_output", False):
                    name = getattr(fn, "_output_name")
                    self._outputs[name] = fn
    
    # ----- Subclass responsibilities -----
    @abstractmethod
    def build_state(self, params: ParameterSet, config: Mapping[str, Any]) -> Any:
        """Deterministic assembly. No RNG. No seed.
        
        This is pure assembly - matrices, schedules, initial conditions.
        All randomness happens in run_sim, not here.
        """
        ...
    
    @abstractmethod
    def run_sim(self, state: Any, seed: int) -> RawSimOutput:
        """Execute stochastic/deterministic sim. RNG boundary lives here.
        
        All randomness derives from this seed via SeedSequence if needed.
        Returns backend-specific raw output.
        """
        ...
    
    @property
    @abstractmethod
    def __version__(self) -> str:
        """Human semantic version for the model (bundle digest is still source of truth)."""
        ...
    
    # Optional: subclasses may mutate _scenarios here only (pre-seal)
    def setup_scenarios(self) -> None:
        """Hook for subclasses to customize scenarios during init.
        
        Called exactly once during __init__ after DEFAULT_SCENARIO is registered
        but before scenarios are frozen. This is the ONLY place to programmatically
        modify scenarios (decorated scenarios are added after this).
        """
        pass
    
    
    # ----- Registry management (mutable until first use) -----
    def add_scenario(self, spec: ScenarioSpec) -> "BaseModel":
        """Register a scenario programmatically (before sealing only)."""
        self._ensure_not_sealed("add_scenario")
        self._scenarios[spec.name] = spec
        return self
    
    def add_model_output(self, name: str,
                         fn: Callable[[RawSimOutput, int], pl.DataFrame]) -> "BaseModel":
        """Register an output extractor programmatically (before sealing only)."""
        self._ensure_not_sealed("add_model_output")
        self._outputs[name] = fn
        return self
    
    def scenarios(self) -> Sequence[str]:
        """All scenario names (includes DEFAULT_SCENARIO)."""
        return sorted(self._scenarios.keys())
    
    def outputs(self) -> Sequence[str]:
        """All output names (stable order by name)."""
        return sorted(self._outputs.keys())
    
    @property
    def is_sealed(self) -> bool:
        """Check if model registries are sealed."""
        return self._sealed
    
    # ----- Dev convenience (uses compiled closure → prod parity) -----
    @final
    def simulate(self, params: ParameterSet, seed: int) -> Dict[str, pl.DataFrame]:
        """Run DEFAULT_SCENARIO and return DataFrames."""
        return self.simulate_scenario(self.DEFAULT_SCENARIO, params, seed)
    
    @final
    def simulate_scenario(self, scenario: str,
                          params: ParameterSet, seed: int) -> Dict[str, pl.DataFrame]:
        """Run named scenario and return DataFrames."""
        fn = self.compile_scenario(scenario)  # (dict, seed) -> {name: bytes}
        ipc = fn(dict(params.values), seed)
        return {k: pl.read_ipc(io.BytesIO(v)) for k, v in ipc.items()}
    
    # ----- Compilation to wire functions (entrypoints) -----
    @final
    def compile_scenario(self, scenario: Optional[str]) \
            -> Callable[[Dict[str, Scalar], int], Dict[str, bytes]]:
        """Compile a scenario to (params_dict, seed) -> {name: Arrow IPC bytes}."""
        self._seal()
        sc = scenario or self.DEFAULT_SCENARIO
        if sc not in self._scenarios:
            raise ValueError(f"Unknown scenario '{sc}'. Available: {self.scenarios()}")
        
        spec = self._scenarios[sc]
        output_names = tuple(self.outputs())  # Stable order snapshot
        
        def fn(params_dict: Dict[str, Scalar], seed: int) -> Dict[str, bytes]:
            pset = ParameterSet(values=dict(params_dict))
            pset2 = spec.patch_params(pset)
            cfg2 = spec.patch_config(self.base_config)
            
            state = self.build_state(pset2, cfg2)     # Deterministic
            raw = self.run_sim(state, seed)           # RNG touchpoint
            
            out: Dict[str, bytes] = {}
            for name in output_names:
                df = self._outputs[name](raw, seed)   # Should be cheap
                bio = io.BytesIO()
                df.write_ipc(bio)
                out[name] = bio.getvalue()
            return out
        
        return fn
    
    @final
    def compile_all(self) -> Dict[str, Callable[[Dict[str, Scalar], int], Dict[str, bytes]]]:
        """One entrypoint per scenario."""
        self._seal()
        return {sc: self.compile_scenario(sc) for sc in self.scenarios()}
    
    # ----- Provenance helpers -----
    def model_identity(self) -> str:
        """Stable-ish human identity for logs & provenance text.
        
        Includes module, class, version, and default scenario.
        Bundle digest remains the source of truth for caching.
        """
        return f"{self.__module__}.{type(self).__name__}:{self.__version__}:default={self.DEFAULT_SCENARIO}"
    
    # ----- Internals -----
    def _seal(self) -> None:
        """Freeze registries on first use to protect provenance."""
        if not self._sealed:
            self._scenarios = MappingProxyType(dict(self._scenarios))
            self._outputs = MappingProxyType(dict(self._outputs))
            self._sealed = True
    
    def _ensure_not_sealed(self, where: str) -> None:
        """Check that registries haven't been sealed."""
        if self._sealed:
            raise RuntimeError(
                f"{where}() not allowed after first compile/simulate; "
                "create a new model instance to change scenarios/outputs."
            )
    
    # Optional: helper for extractors that MUST subsample (discouraged)
    @staticmethod
    def _rng_for_output(seed: int, output_name: str):
        """Namespaced RNG: stable per (seed, output_name). Use rarely.
        
        If you must subsample in an extractor, use this for determinism.
        But consider this a code smell - heavy computation belongs in run_sim.
        
        Example:
            rng = BaseModel._rng_for_output(seed, "prevalence_subsample")
            idx = rng.choice(n, size=100, replace=False)
        """
        # Derive a deterministic child seed from (seed, name)
        s = int(np.uint64(hash(output_name)) ^ np.uint64(seed))
        return np.random.default_rng(s)
```

#### CalibrationAdapter: P → M Bridge for Optimization

```python
@dataclass
class CalibrationAdapter:
    """Adapter that bridges calibration space P to model space M.

    During calibration, optimization algorithms work with a subset P of
    parameters while others remain fixed. This adapter handles the P → M
    conversion, allowing models to remain pure and always expect complete
    parameter sets.

    Key responsibilities:
    • Store fixed parameters (M - P)
    • Convert optimizer's P values to complete M for model.simulate()
    • Maintain parameter metadata (bounds, transforms) for P only
    • Provide clean interface to optimization algorithms
    """

    model: BaseModel                    # The model (expects complete M)
    fixed_params: Dict[str, Scalar]     # Parameters held constant
    free_params: List[str]               # Parameters being optimized (P)
    transforms: Dict[str, Transform]     # Transforms for free params only

    @classmethod
    def from_model(cls,
                   model: BaseModel,
                   free_params: List[str],
                   fixed_values: Dict[str, Scalar],
                   transforms: Optional[Dict[str, Transform]] = None) -> "CalibrationAdapter":
        """Create adapter from model with specified free/fixed split.

        Args:
            model: BaseModel instance expecting complete parameters M
            free_params: Names of parameters to optimize (space P)
            fixed_values: Values for parameters not being optimized (M - P)
            transforms: Optional transforms for free parameters
        """
        # Validate all parameters are accounted for
        all_params = set(p.name for p in model.space.specs)
        free_set = set(free_params)
        fixed_set = set(fixed_values.keys())

        if free_set & fixed_set:
            overlap = free_set & fixed_set
            raise ValueError(f"Parameters cannot be both free and fixed: {overlap}")

        if (free_set | fixed_set) != all_params:
            missing = all_params - (free_set | fixed_set)
            raise ValueError(f"Missing parameter values: {missing}")

        return cls(
            model=model,
            fixed_params=fixed_values,
            free_params=free_params,
            transforms=transforms or {}
        )

    def simulate(self, free_values: Dict[str, Scalar], seed: int) -> Dict[str, pl.DataFrame]:
        """Run simulation with P parameters, converting to M internally.

        Args:
            free_values: Values for parameters in space P (being optimized)
            seed: Random seed

        Returns:
            Simulation outputs as DataFrames

        This is the KEY method: Takes P, creates M, runs model.
        """
        # Reunite fixed and free into complete parameter set M
        complete_params = ParameterSet(values={**self.fixed_params, **free_values})

        # Run model with complete parameters
        return self.model.simulate(complete_params, seed)

    def compile_for_optimization(self) -> Callable[[Dict[str, float], int], Dict[str, bytes]]:
        """Compile to wire function expecting P parameters only.

        Returns function with signature:
            (free_params: Dict[str, float], seed: int) -> Dict[str, bytes]

        This compiled function:
        1. Takes only free parameters (space P)
        2. Internally converts P → M using fixed values
        3. Runs model with complete parameters
        4. Returns Arrow IPC bytes
        """
        # Compile model's default scenario
        model_fn = self.model.compile_scenario(self.model.DEFAULT_SCENARIO)
        fixed = self.fixed_params  # Capture in closure

        def optimized_fn(free_params: Dict[str, float], seed: int) -> Dict[str, bytes]:
            # Convert P → M
            complete = {**fixed, **free_params}
            # Run with complete parameters
            return model_fn(complete, seed)

        return optimized_fn

    def get_bounds(self, transformed: bool = False) -> Dict[str, Tuple[float, float]]:
        """Get bounds for free parameters only (space P).

        Args:
            transformed: If True, return bounds in transformed space

        Returns:
            Dict mapping parameter names to (lower, upper) bounds
        """
        bounds = {}
        for name in self.free_params:
            spec = next(s for s in self.model.space.specs if s.name == name)
            if transformed and name in self.transforms:
                t = self.transforms[name]
                lower_t = t.forward(float(spec.lower))
                upper_t = t.forward(float(spec.upper))
                bounds[name] = (lower_t, upper_t)
            else:
                bounds[name] = (float(spec.lower), float(spec.upper))
        return bounds

    def from_transformed(self, coords: Dict[str, float]) -> Dict[str, Scalar]:
        """Convert from optimizer's transformed coordinates to natural P values.

        Args:
            coords: Transformed coordinates from optimizer

        Returns:
            Natural parameter values for space P
        """
        natural = {}
        for name, value in coords.items():
            if name in self.transforms:
                natural[name] = self.transforms[name].backward(value)
            else:
                natural[name] = value
        return natural

    def to_transformed(self, params: Dict[str, Scalar]) -> Dict[str, float]:
        """Convert natural P values to optimizer's transformed coordinates.

        Args:
            params: Natural parameter values for space P

        Returns:
            Transformed coordinates for optimizer
        """
        transformed = {}
        for name, value in params.items():
            if name in self.transforms:
                transformed[name] = self.transforms[name].forward(float(value))
            else:
                transformed[name] = float(value)
        return transformed
```

#### Engineering Rationale: Why This Design?

**1. Seal-on-First-Use Pattern**

Instead of freezing registries immediately in `__init__`, we seal on first `compile()` or `simulate()`:

```python
def _seal(self) -> None:
    if not self._sealed:
        self._scenarios = MappingProxyType(dict(self._scenarios))
        self._outputs = MappingProxyType(dict(self._outputs))
        self._sealed = True
```

Benefits:
- **Flexibility**: Can add scenarios programmatically after construction
- **Safety**: Once simulation starts, configuration is immutable
- **Clear contract**: "Mutate before use, immutable after"
- **Better error messages**: Explicit RuntimeError explains why mutations fail

**2. Configurable DEFAULT_SCENARIO**

No hardcoded "baseline" string - it's a class attribute:

```python
DEFAULT_SCENARIO: str = "baseline"  # Subclasses can override
```

This enables:
- Models to define what "default" means (e.g., "conservative", "status_quo")
- Uniform treatment - no special cases in compilation
- Clear semantics - `simulate()` just calls `simulate_scenario(DEFAULT_SCENARIO)`

**3. Stable Ordering Guarantees**

All iteration uses `sorted()` for deterministic behavior:
- `sorted(dir(self))` - Discovery order is alphabetical, not filesystem-dependent
- `sorted(self._scenarios.keys())` - Scenario listing is predictable
- `tuple(self.outputs())` - Output order captured at compile time

This ensures:
- **Reproducible hashes**: Same model → same discovery → same provenance
- **No platform drift**: Linux/Mac/Windows all produce same order
- **Cache stability**: Order changes don't invalidate caches unnecessarily

**4. Scenarios as Entrypoints, Not Runtime Flags**

Each scenario compiles to a distinct function with uniform signature:
```python
(params_dict: Dict[str, Scalar], seed: int) -> Dict[str, bytes]
```

Why not runtime parameters?
- **Provenance clarity**: Different scenarios → different sim_root hashes
- **Cache efficiency**: Can cache per-scenario, not per-parameter-combo
- **Infrastructure simplicity**: One entrypoint = one simulation type
- **Type safety**: No runtime string switching, all paths known at compile

**5. Deterministic Assembly, Single RNG Touchpoint**

The RNG boundary is crystal clear:
- `build_state()`: NO seed parameter, pure deterministic assembly
- `run_sim()`: Single seed parameter, ALL randomness flows through here

Benefits:
- **Cache reuse**: Same params → same state across all seeds
- **Debugging**: Can test assembly separately from stochastic execution
- **Parallelism**: Can build state once, run many seeds in parallel
- **Clarity**: No confusion about where randomness happens

**6. Extractors Should Be Cheap**

Output extractors are projections/aggregations on raw output:
```python
@model_output("prevalence")
def extract_prevalence(self, raw: RawSimOutput, seed: int) -> pl.DataFrame:
    return pl.DataFrame({
        "t": raw["t"],
        "prevalence": raw["I"] / (raw["S"] + raw["I"] + raw["R"])
    })
```

If subsampling is needed (discouraged), use namespaced RNG:
```python
rng = self._rng_for_output(seed, "prevalence_subsample")
```

This maintains determinism while marking the pattern as a code smell.

**7. __version__ vs Bundle Digest**

- `__version__`: Human-readable semantic version for logs and debugging
- Bundle digest: Cryptographic truth for caching and provenance

The model identity includes both:
```python
f"{module}.{class}:{version}:default={DEFAULT_SCENARIO}"
```

This gives humans context while infrastructure uses the digest.

#### RNG Boundary Patterns

The clean separation between `build_state` (deterministic) and `run_sim` (stochastic) enables different simulation engines:

**Pattern A: Agent-Based Model with NetworkX**
```python
def build_state(self, params: ParameterSet, config: Mapping[str, Any]) -> nx.Graph:
    """Build network deterministically from parameters."""
    G = nx.erdos_renyi_graph(
        n=int(params["n_agents"]), 
        p=params["connectivity"],
        seed=42  # FIXED seed for deterministic structure
    )
    for node in G.nodes():
        G.nodes[node]["infected"] = False
    return G

def run_sim(self, state: nx.Graph, seed: int) -> RawSimOutput:
    """Run stochastic simulation on fixed network."""
    rng = np.random.default_rng(seed)
    G = state.copy()  # Don't mutate input
    # Seed initial infections stochastically
    initial = rng.choice(list(G.nodes), size=5, replace=False)
    for node in initial:
        G.nodes[node]["infected"] = True
    # Run simulation...
```

**Pattern B: Compartmental Model with Pre-computed Matrices**
```python
def build_state(self, params: ParameterSet, config: Mapping[str, Any]) -> Dict:
    """Pre-compute all transition matrices deterministically."""
    return {
        "transition_matrix": compute_transitions(params),
        "contact_matrix": compute_contacts(params),
        "initial_state": np.array([params["S0"], params["I0"], params["R0"]])
    }

def run_sim(self, state: Dict, seed: int) -> RawSimOutput:
    """Run stochastic transitions using pre-computed matrices."""
    rng = np.random.default_rng(seed)
    # Use state["transition_matrix"] with rng for stochastic draws
```

**Pattern C: Hybrid Deterministic-Stochastic**
```python
def build_state(self, params: ParameterSet, config: Mapping[str, Any]) -> ODESystem:
    """Build ODE system deterministically."""
    return ODESystem(params)  # Pure math, no randomness

def run_sim(self, state: ODESystem, seed: int) -> RawSimOutput:
    """Add stochastic forcing to deterministic system."""
    rng = np.random.default_rng(seed)
    # Run ODE with stochastic forcing terms
    noise = rng.normal(0, state.noise_scale, size=state.n_steps)
    return state.solve_with_noise(noise)
```

The key insight: **ALL randomness flows through `run_sim`'s seed parameter**. This ensures:
1. Identical parameters → identical state (caching friendly)
2. Identical state + seed → identical outcomes (reproducible)
3. State building can be expensive (it's cached via params hash)
4. Multiple seeds can reuse the same built state efficiently

#### Bundle Compilation Helper
```python
def compile_bundle(*, bundle_ref: str, model_cls_path: str) -> dict[str, Callable[[dict, int], dict[str, bytes]]]:
    """
    Compile all scenarios from a model into wire-format functions.
    
    Returns: {entrypoint_id: (params_dict, seed) -> {name: Arrow IPC bytes}}
    """
    import importlib
    from modelops_contracts.entrypoint import format_entrypoint
    
    # Load model class
    mod, cls = model_cls_path.split(":")
    Model = getattr(importlib.import_module(mod), cls)
    model: BaseModel = Model()
    
    # Get all compiled scenarios
    fns = model.compile_all()  # {scenario: (params, seed) -> bytes}
    
    # Create entrypoint IDs for each scenario
    out = {}
    for sc, fn in fns.items():
        eid = format_entrypoint(
            import_path=model_cls_path.replace(":", "."),
            scenario=sc,
            oci_digest=bundle_ref
        )
        out[eid] = fn  # Already returns Arrow IPC bytes
    
    return out
```

#### Entrypoint Format

Entrypoints follow the grammar defined in `modelops-contracts.entrypoint`:

```
entrypoint   := <import-path> "/" <scenario> "@" <digest12>
import-path  := <module> "." <ClassName>              # e.g., "sir_model.SIRModel"
scenario     := [a-z0-9]([a-z0-9-_.]{0,62}[a-z0-9])? # lowercase slug
digest12     := first 12 hex chars of OCI digest      # e.g., "abc123def456"
```

Use the provided functions from `modelops_contracts.entrypoint`:
- `format_entrypoint()` to create IDs
- `parse_entrypoint()` to decompose IDs  
- `validate_entrypoint_matches_bundle()` for validation

No ad-hoc string splitting!

#### Local Development Without ModelOps
```python
# Local testing: run without any ModelOps dependency
from sir_model import SIRModel

# Create model and explore scenarios
model = SIRModel()

# Already includes the default scenario (whatever it's named)
scenarios = list(model.scenarios())
print(f"Available scenarios: {scenarios}")
# Output: ['aggressive', 'conservative', 'lockdown', 'vaccination']

# Compile all scenarios to wire functions
from calabaria.compile import compile_bundle
idx = compile_bundle(
    bundle_ref="sha256:abc123...",  # OCI digest
    model_cls_path="sir_model:SIRModel"
)

# List available entrypoints
for entrypoint_id in idx:
    print(entrypoint_id)
# Output (note: includes model's DEFAULT_SCENARIO, not hardcoded 'baseline'):
# sir_model.SIRModel/conservative@abc123def456
# sir_model.SIRModel/aggressive@abc123def456
# sir_model.SIRModel/lockdown@abc123def456
# sir_model.SIRModel/vaccination@abc123def456

# Execute a simulation function directly
# Using the model's DEFAULT_SCENARIO (conservative in this example)
sim_fn = idx["sir_model.SIRModel/conservative@abc123def456"]
ipc_outputs = sim_fn(
    params={"beta": 0.4, "gamma": 0.1},
    seed=7
)

# Read Arrow IPC data
import polars as pl
from io import BytesIO
df = pl.read_ipc(BytesIO(ipc_outputs["prevalence"]))
```

### 4. Provenance & Caching

#### Provenance Identity Separation

**sim_root**: Pure simulation identity
- Components: `hash(code_id, params, seed, scenario, config)`
- Does NOT include outputs selection
- Used for simulation-level caching
- Shared by tasks with different output selections

**task_id**: Materialization identity  
- Components: `hash(sim_root, entrypoint, outputs)`
- Includes outputs selection
- Used for scheduler deduplication
- Unique per output selection

**Key Insight**: Two tasks requesting different outputs from the same simulation 
share the same `sim_root` but have different `task_id` values. This enables 
cache reuse at the simulation level while preventing scheduler collisions.

#### Leveraging Existing Provenance System

The modelops-contracts already provides a complete provenance system:

```python
# From modelops_contracts/provenance.py
from modelops_contracts.provenance import (
    ProvenanceLeaf,
    canonical_json,
    sim_root,
    calib_root,
    LeafKind  # "params", "config", "code", "scenario", "seed", "env", "targets", "optimizer"
)

# Example usage in Calabaria
def compute_simulation_provenance(
    model: BaseModel,
    params: ParameterSet,
    seed: int,
    scenario: str
) -> str:
    """Compute provenance hash for simulation"""
    leaves = [
        ProvenanceLeaf(
            kind="params",
            name="parameters",
            digest=hash_params(params)
        ),
        ProvenanceLeaf(
            kind="scenario",
            name=scenario,
            digest=hash_scenario(model._scenarios[scenario])
        ),
        ProvenanceLeaf(
            kind="seed",
            name="seed",
            digest=hash_seed(seed)
        ),
        ProvenanceLeaf(
            kind="code",
            name=model.__class__.__name__,
            digest=hash_model_code(model)
        ),
    ]
    
    # Use the contracts function
    return sim_root(leaves)
```

#### Content-Addressed Storage

The provenance hash becomes the key for content-addressed storage:

```python
def store_simulation_result(
    sim_hash: str,
    result: SimReturn,
    storage: StorageBackend
) -> None:
    """Store result under provenance hash"""
    # Shard the path to avoid hot directories
    path = shard_path(sim_hash)
    
    # Store metadata and artifacts
    storage.put(path / "metadata.json", result.to_json())
    for name, artifact in result.outputs.items():
        if artifact.ref:
            # Already in CAS, just record reference
            storage.put(path / f"outputs/{name}.ref", artifact.ref)
        elif artifact.inline:
            # Store inline data
            storage.put(path / f"outputs/{name}.arrow", artifact.inline)

def shard_path(hash: str, depth: int = 2, width: int = 2) -> Path:
    """Convert hash to sharded path: 'abcdef...' → 'ab/cd/abcdef.../'"""
    parts = [hash[i*width:(i+1)*width] for i in range(depth)]
    parts.append(hash)
    return Path(*parts)
```

---

## Part II: Experiment Architecture & Provenance Model

### Critical Engineering Decision: Experiments Are NOT in Bundles

This is a fundamental design choice that enables clean provenance and efficient caching:

#### Bundle Digest Represents Code + Data ONLY

```
Bundle Digest = hash(code + model_definition + input_data + dependencies)
```

The bundle digest should change ONLY when:
- Model code changes
- Input data changes
- Dependencies change (uv.lock)

It should NOT change when:
- Different parameter sweeps are run
- Different scenarios are selected
- Different seed/replicate counts are used
- New experiments are designed

#### Why This Matters

1. **Cache Efficiency**: Same bundle + same parameters = same simulation result, regardless of how many experiments reference it
2. **Scientific Iteration**: Scientists can design new experiments without invalidating cached results
3. **Clear Provenance**: Bundle identity is stable for given code/data, experiment identity is separate

### Project Structure

```
project/
├── src/                          # Model code (IN BUNDLE)
│   └── my_model.py              
├── data/                         # Input data (IN BUNDLE)
│   ├── observations.csv         
│   └── targets/                 # Target data for calibration
│       └── prevalence.parquet
├── experiments/                  # Experiment specs (NOT IN BUNDLE!)
│   ├── sweeps/                  # Parameter exploration
│   │   ├── 001_initial_grid.yaml
│   │   ├── 002_sobol_exploration.yaml
│   │   └── 003_sensitivity.yaml
│   ├── calibrations/            # Optimization runs
│   │   ├── 001_baseline_fit.yaml
│   │   ├── 002_with_new_data.yaml
│   │   └── 003_final_optuna.yaml
│   └── README.md                # Experiment log (human-readable history)
├── bundle.yaml                  # Bundle manifest (NO experiments!)
└── .gitignore                   # Ignore outputs/, but track experiments/
```

### Output Path Organization

Output paths incorporate BOTH bundle and experiment identity for complete provenance:

```python
def compute_output_path(bundle_ref: str, experiment_spec: Any, 
                       param_id: str, replicate: int) -> str:
    """Hierarchical output structure preserving all provenance.
    
    Key insight: Output path encodes complete lineage
    - Which code/data (bundle)
    - Which experiment (sweep/calibration)
    - Which parameters (param_id)
    - Which replicate (seed derivation)
    """
    from modelops_contracts import shard
    
    # Bundle identity (code + data)
    bundle_digest = bundle_ref.split(":")[-1][:12]
    bundle_path = shard(bundle_digest, depth=2, width=2)
    
    # Experiment identity (sweep or calibration config)
    if isinstance(experiment_spec, ParameterSweepSpec):
        exp_type = "sweeps"
        exp_id = experiment_spec.sweep_id()
    else:  # CalibrationSpec
        exp_type = "calibrations"
        exp_id = experiment_spec.calibration_id()
    exp_path = shard(exp_id, depth=2, width=2)
    
    # Full path
    return f"outputs/bundles/{bundle_path}/{exp_type}/{exp_path}/params/{param_id}/rep_{replicate}/"
```

Example output structure:

```
outputs/
└── bundles/
    └── ab/cd/abcdef123456/          # Bundle identity (code+data)
        ├── sweeps/
        │   └── 12/34/123456789abc/  # Sweep experiment identity
        │       └── params/
        │           ├── param_001/    # Parameter set identity
        │           │   ├── rep_0/    # Replicate 0
        │           │   ├── rep_1/    # Replicate 1
        │           │   └── rep_2/    # Replicate 2
        │           └── param_002/
        │               └── ...
        └── calibrations/
            └── 56/78/567890def123/  # Calibration experiment identity
                └── trials/
                    └── trial_042/    # Optimization iteration
                        └── rep_0/
```

### Experiment Workflow

#### 1. Scientist Creates Experiment

```python
# In Calabaria (science side)
from calabaria.generators import SobolSequence
from calabaria.parameters import ParameterSpecs

# Define parameter space
param_specs = ParameterSpecs({
    "beta": {"min": 0.1, "max": 0.5, "transform": "log"},
    "gamma": {"min": 0.01, "max": 0.2, "transform": "log"}
})

# Generate parameter sets
sequence = SobolSequence(param_specs, n_samples=100)

# Export to wire format
sweep_spec = sequence.to_sweep_spec(
    entrypoint="sir_model.SIRModel/baseline@abc123def456",
    bundle_ref="sha256:abc123...",
    base_seed=42,
    n_replicates=10,
    outputs=["prevalence", "peak"]
)

# Save to experiments directory (NOT bundle!)
with open("experiments/sweeps/002_sobol_exploration.yaml", "w") as f:
    f.write(sweep_spec.to_yaml())
```

#### 2. Infrastructure Executes Experiment

```python
# In ModelOps (infrastructure side)
from modelops_contracts import ParameterSweepSpec
from modelops.sweep import ParameterSweepExecution

# Load experiment spec
with open("experiments/sweeps/002_sobol_exploration.yaml") as f:
    spec = ParameterSweepSpec.from_yaml(f.read())

# Execute using infrastructure
execution = ParameterSweepExecution(spec, simulation_service)
futures_by_param = execution.submit_all()

# Results organized by parameter ID
results = execution.gather_by_parameter(futures_by_param)
```

#### 3. Scientist Updates Model

```python
# Modify src/my_model.py
# This creates NEW bundle digest
# But experiments/sweeps/002_sobol_exploration.yaml unchanged!

# Re-run same experiment on new code
spec.bundle_ref = "sha256:newdigest..."  # Update to new bundle
execution = ParameterSweepExecution(spec, simulation_service)

# Output goes to DIFFERENT path due to different bundle digest
# outputs/bundles/newdigest/sweeps/same_sweep_id/...
# Can now compare results across code versions!
```

---

## Part III: Contract Integration

### 5. ModelOps Contracts (The Seam)

#### Core Philosophy
The contract layer is the **minimal seam** between systems:
- **UniqueParameterSet**: Just parameters + ID (no scenarios, transforms, or config)
- **SimTask**: Complete execution specification with caching control
- **SimulationService**: Submit tasks, gather results
- **SimReturn**: Results with inline/CAS storage via TableArtifact

#### Actual Contracts (from modelops-contracts)
```python
# modelops_contracts/types.py
@dataclass(frozen=True)
class UniqueParameterSet:
    """Minimal parameter representation for wire protocol"""
    params: Mapping[str, Scalar]  # Natural parameters only
    param_id: str                  # Stable hash of params
    
    @classmethod
    def from_dict(cls, params: dict) -> 'UniqueParameterSet':
        """Create with auto-generated param_id"""
        return cls(params=params, param_id=make_param_id(params))

# modelops_contracts/simulation.py
@dataclass(frozen=True)
class SimTask:
    """Specification for a single deterministic simulation task.
    
    CRITICAL: Scenarios are NOT parameters - they're compiled into entrypoints!
    """
    bundle_ref: str                          # OCI digest or bundle identifier
    entrypoint: EntryPointId                 # e.g. "pkg.Class/scenario@digest12"
    params: UniqueParameterSet               # Parameters with ID
    seed: int                                # Random seed (uint64)
    outputs: Optional[Sequence[str]] = None  # Normalized to sorted tuple internally
    config: Optional[Mapping[str, Any]] = None  # Runtime configuration
    env: Optional[Mapping[str, str]] = None    # Environment variables
    
    @classmethod
    def from_components(cls, *, import_path: str, scenario: str,
                       bundle_ref: str, params: dict[str, Any], seed: int,
                       outputs: Optional[Sequence[str]] = None,
                       config: Optional[dict] = None,
                       env: Optional[dict] = None) -> "SimTask":
        """Create SimTask from components - preferred factory method"""
        # Formats entrypoint properly from import_path and scenario
        
    @classmethod
    def from_entrypoint(cls, *, entrypoint: EntryPointId,
                       bundle_ref: str, params: Mapping[str, Any], seed: int,
                       outputs: Optional[Sequence[str]] = None,
                       config: Optional[Mapping] = None,
                       env: Optional[Mapping] = None) -> "SimTask":
        """Create SimTask when you already have a formatted entrypoint"""
    
    def sim_root(self) -> str:
        """Simulation identity (excludes outputs for cache reuse)"""
        # Uses sim_root_from_parts internally
    
    def task_id(self) -> str:
        """Task identity (includes outputs for materialization)"""
        # Uses task_id_from_parts internally

# modelops_contracts/artifacts.py
@dataclass(frozen=True)
class TableArtifact:
    """Table output with inline/CAS storage strategy"""
    content_type: str = "application/vnd.apache.arrow.stream"
    size: int = 0
    inline: Optional[bytes] = None   # If size <= 512KB
    ref: Optional[str] = None        # CAS path if size > 512KB
    checksum: str = ""               # BLAKE2b-256 hash

@dataclass(frozen=True)
class SimReturn:
    """Results from completed simulation task"""
    task_id: str
    sim_root: str                           # Provenance hash
    outputs: Mapping[str, TableArtifact]    # Named table outputs
    logs_ref: Optional[str] = None          # CAS path to logs
    metrics: Optional[Mapping[str, float]] = None  # Execution metrics
    cached: bool = False                    # Whether from cache

# modelops_contracts/simulation.py
class SimulationService(Protocol):
    def submit(self, task: SimTask) -> FutureLike:
        """Submit single simulation task"""
        ...
    
    def submit_batch(self, tasks: List[SimTask]) -> List[FutureLike]:
        """Submit multiple simulation tasks"""
        ...
    
    def submit_replicates(self, base_task: SimTask, n_replicates: int) -> List[FutureLike]:
        """Submit replicates with deterministic seed derivation.
        
        CRITICAL: This is how we handle replicates! The service uses
        numpy.random.SeedSequence internally for statistically independent seeds.
        """
        ...
    
    def gather(self, futures: List[FutureLike]) -> List[SimReturn]:
        """Collect results in order"""
        ...
    
    def gather_and_aggregate(self, futures: List[FutureLike], 
                            aggregator: Union[str, AggregatorFunction]) -> SimReturn:
        """Gather and aggregate (can run on workers for efficiency)"""
        ...
```

### 6. Experiment Contract Specifications

#### ParameterSweepSpec - The Wire Format for Exploration

```python
# modelops_contracts/sweep.py (NEW FILE)

@dataclass(frozen=True)
class ParameterSweepSpec:
    """Wire-friendly specification for parameter exploration.
    
    This is THE SEAM between science (Calabaria) and infrastructure (ModelOps).
    
    Key characteristics:
    - Predetermined parameter sets (not search space)
    - Can be serialized to/from YAML/JSON
    - Minimal - just execution info, no metadata
    - Immutable and hashable
    """
    
    # Execution target
    entrypoint: str  # e.g., "sir_model.SIRModel/baseline@abc123def456"
    bundle_ref: str  # e.g., "sha256:abc123..." or "local://dev"
    
    # Parameter sets to evaluate (explicit, not bounds)
    parameter_sets: List[Dict[str, Any]]  # List of parameter dicts
    
    # Execution configuration
    base_seed: int                        # For reproducible seed derivation
    n_replicates: int = 1                 # Replicates per parameter set
    outputs: Optional[List[str]] = None   # Which outputs to extract
    
    def sweep_id(self) -> str:
        """Compute stable identity for this sweep configuration.
        
        CRITICAL: Excludes bundle_ref so same sweep can run on different bundles!
        This enables comparing results across code versions.
        """
        from modelops_contracts import canonical_json, digest_bytes
        
        identity = {
            "type": "sweep",
            "entrypoint": self.entrypoint.split("@")[0],  # Exclude digest
            "parameter_sets": self.parameter_sets,
            "base_seed": self.base_seed,
            "n_replicates": self.n_replicates,
            "outputs": sorted(self.outputs) if self.outputs else None
        }
        
        return digest_bytes(canonical_json(identity))[:12]
    
    def execution_id(self) -> str:
        """Identity for this specific execution (bundle + sweep)."""
        from modelops_contracts import canonical_json, digest_bytes
        
        execution_identity = {
            "bundle_ref": self.bundle_ref,
            "sweep_id": self.sweep_id()
        }
        
        return digest_bytes(canonical_json(execution_identity))[:12]
    
    def to_yaml(self) -> str:
        """Serialize to YAML for storage/transmission."""
        import yaml
        return yaml.dump({
            "entrypoint": self.entrypoint,
            "bundle_ref": self.bundle_ref,
            "parameter_sets": self.parameter_sets,
            "base_seed": self.base_seed,
            "n_replicates": self.n_replicates,
            "outputs": self.outputs
        })
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ParameterSweepSpec":
        """Deserialize from YAML."""
        import yaml
        data = yaml.safe_load(yaml_str)
        return cls(**data)
```

#### CalibrationSpec - The Wire Format for Optimization

```python
# modelops_contracts/calibration.py (NEW FILE)

@dataclass(frozen=True)
class CalibrationSpec:
    """Wire-friendly specification for calibration/optimization.
    
    Key differences from ParameterSweepSpec:
    - Algorithm-driven parameter selection (not predetermined)
    - Includes target references and loss function
    - Supports adaptive sampling
    """
    
    # Execution target
    entrypoint: str
    bundle_ref: str
    
    # Algorithm configuration
    algorithm: str  # e.g., "optuna.TPESampler", "grid", "sobol"
    algorithm_config: Dict[str, Any]  # Algorithm-specific settings
    
    # Search space (bounds, not explicit sets)
    parameter_bounds: Dict[str, Dict[str, Any]]  # {param: {min, max, transform}}
    
    # Targets for loss computation
    target_refs: List[str]  # Paths to target data in bundle
    loss_function: str      # e.g., "mse", "nll", "weighted_mse"
    
    # Execution configuration
    n_trials: int           # Number of optimization iterations
    n_replicates: int = 1   # Replicates per trial
    base_seed: int = 42
    outputs: Optional[List[str]] = None  # Must include target-relevant outputs
    
    def calibration_id(self) -> str:
        """Compute stable identity for this calibration configuration."""
        from modelops_contracts import canonical_json, digest_bytes
        
        identity = {
            "type": "calibration",
            "entrypoint": self.entrypoint.split("@")[0],
            "algorithm": self.algorithm,
            "algorithm_config": self.algorithm_config,
            "parameter_bounds": self.parameter_bounds,
            "target_refs": sorted(self.target_refs),
            "loss_function": self.loss_function,
            "n_trials": self.n_trials,
            "n_replicates": self.n_replicates,
            "base_seed": self.base_seed
        }
        
        return digest_bytes(canonical_json(identity))[:12]
```

#### Why These Are The Right Seams

1. **Minimal Surface Area**: Only what's needed for execution, no implementation details
2. **Wire-Friendly**: Can serialize to YAML/JSON for storage, transmission, or human editing
3. **Version-Independent**: Same spec can run on different bundle versions
4. **Type-Safe**: Can use Pydantic for validation if needed
5. **Extensible**: Can add optional fields without breaking compatibility

### 7. Scenario Compilation & Bundle Strategy

#### Compilation Strategies

**Default (MVP): Compile-Time Resolution**
```python
# Each scenario compiles to a distinct entrypoint
# No runtime scenario parameter - cleaner provenance

class ScenarioCompiler:
    def compile_all(self, model: BaseModel) -> Dict[str, Callable]:
        """Generate standalone functions for each scenario"""
        return model.compile_all()  # Returns {scenario: (params, seed) -> bytes}

# Each entrypoint has signature: (params_dict, seed) -> {name: arrow_bytes}
# E.g., "model.Class/conservative@abc123" is a distinct function
```

**Alternative (not MVP): Runtime Resolution**
```python
# Single entry point that dispatches to scenario at runtime
# More complex provenance tracking, not recommended

def simulation_entry_point(params: Dict[str, Scalar], seed: int, 
                          scenario: str) -> SimReturn:
    """Generic entry point that dispatches to scenario"""
    model = load_model_from_bundle()
    # Would need runtime scenario switching - more complex
    # Not the recommended approach

# Results in bundle with multiple entry points:
# - my_model:baseline
# - my_model:lockdown
# - my_model:vaccination
```

#### Bundle Organization

Bundles are OCI artifacts containing Python packages with:
- `pyproject.toml` and `uv.lock` for reproducible dependencies
- Model code following Calabaria patterns
- Data files (Parquet, Arrow, etc.)
- No YAML manifest - entrypoints discovered via `compile_bundle()`

### 7. Bridge Implementation

#### CalabariaBridge Execution
```python
class CalabariaBridge:
    """Executes compiled simulation functions in ModelOps infrastructure.
    
    Handles function caching, output filtering, and CAS storage.
    """
    
    def __init__(self, cas_client: CasClient, threshold_bytes: int = 512_000):
        self.cas = cas_client
        self.threshold = threshold_bytes
        self._function_cache: Dict[str, Callable] = {}
    
    def _get_or_compile_function(self, task: SimTask) -> Callable:
        """Get cached or compile new function."""
        # Cache by entrypoint only (filtering happens after execution)
        cache_key = str(task.entrypoint)
        
        if cache_key not in self._function_cache:
            # Parse entrypoint and load model
            from modelops_contracts import parse_entrypoint
            import_path, scenario, _ = parse_entrypoint(task.entrypoint)
            
            # Load model and get compiled function for scenario
            mod_path, cls_name = import_path.rsplit(".", 1)
            import importlib
            Model = getattr(importlib.import_module(mod_path), cls_name)
            model = Model()
            
            # Get the compiled function for this scenario
            fns = model.compile_all()
            self._function_cache[cache_key] = fns[scenario]
        
        return self._function_cache[cache_key]
    
    def execute_task(self, task: SimTask) -> SimReturn:
        """Execute a SimTask using compiled simulation function."""
        # Get compiled function
        fn = self._get_or_compile_function(task)
        
        # Execute simulation
        ipc_outputs = fn(
            params=dict(task.params.params),
            seed=task.seed
        )
        
        # Filter outputs if specified in task
        if task.outputs:
            wanted = set(task.outputs)
            ipc_outputs = {k: v for k, v in ipc_outputs.items() if k in wanted}
        
        # Store outputs (inline vs CAS based on size)
        artifacts = {}
        for name, ipc_bytes in ipc_outputs.items():
            if len(ipc_bytes) <= self.threshold:
                # Small output - store inline
                artifacts[name] = TableArtifact(
                    content_type="application/vnd.apache.arrow.stream",
                    size=len(ipc_bytes),
                    inline=ipc_bytes,
                    checksum=self._hash(ipc_bytes)
                )
            else:
                # Large output - store in CAS
                ref = self.cas.put(ipc_bytes)
                artifacts[name] = TableArtifact(
                    content_type="application/vnd.apache.arrow.stream",
                    size=len(ipc_bytes),
                    ref=ref,
                    checksum=self._hash(ipc_bytes)
                )
        
        return SimReturn(
            task_id=task.task_id(),  # Uses task_id_from_parts internally
            sim_root=task.sim_root(),  # Uses sim_root_from_parts internally
            outputs=artifacts,
            cached=False
        )
```

#### Calibration Integration

By convention, calibrations run against the model's DEFAULT_SCENARIO. Use whichever name the model sets (e.g., "conservative", "status_quo", "baseline").

```python
# During calibration - use model's DEFAULT_SCENARIO entrypoint
def run_calibration(algo: AdaptiveAlgorithm, bridge: CalabariaBridge, model_class):
    """Run calibration using model's default scenario"""
    bundle_ref = "sha256:abc123..."
    # Get the actual default scenario name from the model
    default_scenario = model_class.DEFAULT_SCENARIO
    default_entrypoint = f"sir_model.SIRModel/{default_scenario}@abc123def456"
    
    for _ in range(max_iters):
        # Ask for proposals
        proposals = algo.ask(n=10)
        
        # Submit simulations using default scenario
        tasks = [
            SimTask(
                bundle_ref=bundle_ref,
                entrypoint=default_entrypoint,  # Use model's default scenario
                params=prop,
                seed=42
            )
            for prop in proposals
        ]
        
        futures = [bridge.execute_task(t) for t in tasks]
        results = compute_losses(futures, targets)
        
        # Tell results
        algo.tell(results)

# After calibration - run predictions across scenarios
def run_predictions(best_params: UniqueParameterSet, bridge: CalabariaBridge):
    """Run best parameters across all scenarios"""
    bundle_ref = "sha256:abc123..."
    
    # Get all entrypoints for this bundle
    idx = compile_bundle(
        bundle_ref=bundle_ref,
        model_cls_path="sir_model:SIRModel"
    )
    
    # Submit task for each scenario
    tasks = [
        SimTask(
            bundle_ref=bundle_ref,
            entrypoint=entrypoint_id,
            params=best_params,
            seed=42
        )
        for entrypoint_id in idx.keys()
    ]
    
    # Execute all scenarios
    results = {}
    for task in tasks:
        scenario = task.entrypoint.split("/")[1].split("@")[0]
        results[scenario] = bridge.execute_task(task)
    
    return results
```

#### Minimal SimTask (Simplified)
```python
@dataclass(frozen=True)
class SimTask:
    """Task specification with scenario compiled into entrypoint.
    
    Note: Scenario is NOT a field - it's part of the entrypoint ID.
    Outputs are automatically normalized to sorted tuple for deterministic task_id.
    """
    bundle_ref: str                          # OCI digest
    entrypoint: str                          # e.g. "sir_model.SIRModel/lockdown@abc123def456"
    params: UniqueParameterSet               # Parameters with ID
    seed: int                                # Random seed (uint64)
    outputs: Optional[Sequence[str]] = None  # Normalized to tuple(sorted(...))
    
    def sim_root(self) -> str:
        """Compute sim_root using sim_root_from_parts (excludes outputs)."""
        ...
    
    def task_id(self) -> str:
        """Compute task_id using task_id_from_parts (includes outputs)."""
        ...
```

Note: SimTask is a pure data carrier with provenance methods. Adapter logic belongs in CalabariaBridge.

---

## Part III: Parameter Generators & Execution

### 8. Parameter Generators (Science → Wire Bridge)

Parameter generators live in **modelops-calabaria** because they bridge Calabaria's science types to contract types:

```python
# modelops_calabaria/generators/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from calabaria.parameters import ParameterSpecs  # Science dependency

class ParameterGenerator(ABC):
    """Base class for parameter generators.
    
    Lives in modelops-calabaria because it bridges:
    - INPUT: Calabaria's ParameterSpecs (science types)
    - OUTPUT: Plain dicts for ParameterSweepSpec (contract types)
    """
    
    def __init__(self, param_specs: ParameterSpecs):
        self.param_specs = param_specs
    
    @abstractmethod
    def generate_parameter_sets(self) -> List[Dict[str, Any]]:
        """Generate parameter dicts ready for SimTask.from_components()."""
        ...
    
    def to_sweep_spec(self, 
                     entrypoint: str,
                     bundle_ref: str,
                     base_seed: int = 42,
                     n_replicates: int = 1,
                     outputs: Optional[List[str]] = None) -> "ParameterSweepSpec":
        """Export to wire format for ModelOps execution."""
        from modelops_contracts import ParameterSweepSpec
        
        return ParameterSweepSpec(
            entrypoint=entrypoint,
            bundle_ref=bundle_ref,
            parameter_sets=self.generate_parameter_sets(),
            base_seed=base_seed,
            n_replicates=n_replicates,
            outputs=outputs
        )
```

#### Grid Generator

```python
# modelops_calabaria/generators/grid.py

class GridGenerator(ParameterGenerator):
    """Grid search parameter generation with flexible specifications."""
    
    def __init__(self, 
                 param_specs: ParameterSpecs,
                 grid_specs: Dict[str, Dict[str, Any]]):
        """
        Grid spec examples:
        {
            "beta": {"type": "linspace", "min": 0.1, "max": 0.5, "n": 10},
            "gamma": {"type": "logspace", "min": 0.01, "max": 0.2, "n": 5},
            "population": {"type": "list", "values": [1000, 5000, 10000]}
        }
        """
        super().__init__(param_specs)
        self.grid_specs = grid_specs
    
    def generate_parameter_sets(self) -> List[Dict[str, Any]]:
        # Generate cartesian product of all grid dimensions
        # Respects parameter bounds and transforms from ParameterSpecs
        ...
```

#### Sobol Generator

```python
# modelops_calabaria/generators/sobol.py

from scipy.stats import qmc

class SobolGenerator(ParameterGenerator):
    """Quasi-random parameter generation for better space coverage."""
    
    def __init__(self,
                 param_specs: ParameterSpecs,
                 n_samples: int,
                 scramble: bool = True,
                 seed: Optional[int] = None):
        super().__init__(param_specs)
        self.n_samples = n_samples
        self.scramble = scramble
        self.seed = seed
    
    def generate_parameter_sets(self) -> List[Dict[str, Any]]:
        # Use Sobol sequence for quasi-random sampling
        # Automatically handles log-scale parameters from specs
        sampler = qmc.Sobol(d=len(self.param_specs), scramble=self.scramble, seed=self.seed)
        unit_samples = sampler.random(self.n_samples)
        
        # Scale to parameter bounds considering transforms
        ...
```

### 9. Execution Infrastructure (ModelOps Side)

#### ParameterSweepExecution

```python
# modelops/sweep.py

@dataclass
class ParameterSweepExecution:
    """Executes a ParameterSweepSpec using infrastructure.
    
    This is pure infrastructure - lives in modelops, uses only contracts.
    Single entrypoint, predetermined parameter sets.
    """
    
    spec: ParameterSweepSpec
    service: SimulationService
    
    def submit_all(self) -> Dict[str, List[FutureLike]]:
        """Submit all parameter sets with replicates.
        
        Returns:
            Dict[param_id] -> List[replicate_futures]
            
        This structure enables:
        - Easy aggregation across replicates
        - Parallel submission of all parameters
        - Clear provenance tracking
        """
        # Seed derivation using SeedSequence
        ss = np.random.SeedSequence(self.spec.base_seed)
        param_seeds = ss.spawn(len(self.spec.parameter_sets))
        
        results = {}
        for params, seed_seq in zip(self.spec.parameter_sets, param_seeds):
            seed = int(seed_seq.generate_state(1)[0])
            
            # Create base task for this parameter set
            task = SimTask.from_entrypoint(
                entrypoint=self.spec.entrypoint,
                bundle_ref=self.spec.bundle_ref,
                params=params,
                seed=seed,
                outputs=self.spec.outputs
            )
            
            # Submit replicates using service
            replicate_futures = self.service.submit_replicates(
                task, 
                self.spec.n_replicates
            )
            
            # Store by param_id for easy access
            param_id = UniqueParameterSet.from_dict(params).param_id
            results[param_id] = replicate_futures
        
        return results
    
    def gather_by_parameter(self, 
                           futures_dict: Dict[str, List[FutureLike]]) -> Dict[str, List[SimReturn]]:
        """Gather results grouped by parameter set."""
        results = {}
        for param_id, replicate_futures in futures_dict.items():
            results[param_id] = self.service.gather(replicate_futures)
        return results
    
    def aggregate_replicates(self,
                            futures_dict: Dict[str, List[FutureLike]],
                            aggregator: str = "numpy:mean") -> Dict[str, SimReturn]:
        """Aggregate replicates for each parameter set."""
        results = {}
        for param_id, replicate_futures in futures_dict.items():
            # Service handles aggregation (can run on workers!)
            results[param_id] = self.service.gather_and_aggregate(
                replicate_futures,
                aggregator
            )
        return results
```

---

## Part IV: Implementation Plan (Staged Approach)

### Stage 1: ParameterSweepSpec + Infrastructure (Weeks 1-2)

**Goal**: Get parameter sweeps working end-to-end with ModelOps

#### Deliverables

**modelops-contracts:**
- [ ] `sweep.py`: ParameterSweepSpec with serialization
- [ ] Update `__init__.py` to export ParameterSweepSpec

**modelops:**
- [ ] `sweep.py`: ParameterSweepExecution class
- [ ] Integration tests with DaskSimulationService

**modelops-calabaria:**
- [ ] `generators/base.py`: ParameterGenerator base class
- [ ] `generators/grid.py`: GridGenerator implementation
- [ ] `generators/sobol.py`: SobolGenerator implementation
- [ ] Tests for generator → sweep spec conversion

#### Success Criteria
- Can generate parameter sets from Calabaria ParameterSpecs
- Can export to ParameterSweepSpec YAML
- Can execute sweep using ModelOps infrastructure
- Output paths correctly organized by bundle/sweep/param/replicate

### Stage 2: Targets Integration (Weeks 3-4)

**Goal**: Port Calabaria targets for use with simulation results

#### Context: What Calabaria Already Has

Calabaria has a sophisticated target system:
- **Target**: Encapsulates observed data + alignment strategy + evaluation strategy
- **AlignmentStrategy**: How to join observed and simulated data (exact, asof, etc.)
- **EvaluationStrategy**: How to compute loss from aligned data
- **TargetEvaluation**: Result of evaluating a target

#### Deliverables

**modelops-calabaria/targets:**
- [ ] Port `target.py` from Calabaria
- [ ] Port `alignment.py` strategies
- [ ] Port `evaluation/` strategies
- [ ] Create `bundled.py` for target bundling support
- [ ] Create `adapter.py` to work with SimReturn

#### Key Adaptations Needed

```python
# modelops_calabaria/targets/adapter.py

class TargetAdapter:
    """Adapts Calabaria targets to work with SimReturn."""
    
    def __init__(self, targets: Targets):
        self.targets = targets
    
    def evaluate_sim_return(self, sim_return: SimReturn) -> Dict[str, float]:
        """Evaluate targets against SimReturn from simulation.
        
        Converts Arrow IPC artifacts to Polars DataFrames
        and runs target evaluation.
        """
        # Convert SimReturn.outputs (TableArtifacts) to Polars
        dfs = self._extract_dataframes(sim_return)
        
        # Run Calabaria's target evaluation
        sim_output = SimOutput(dfs)  # Wrap for Calabaria
        evaluations = self.targets.evaluate_all([sim_output])
        
        # Convert to loss dict
        return {eval.name: eval.loss for eval in evaluations}
    
    def _extract_dataframes(self, sim_return: SimReturn) -> Dict[str, pl.DataFrame]:
        """Convert TableArtifacts to Polars DataFrames."""
        import io
        dfs = {}
        
        for name, artifact in sim_return.outputs.items():
            if artifact.inline:
                # Small table, data is inline
                bio = io.BytesIO(artifact.inline)
                df = pl.read_ipc(bio)
            else:
                # Large table in CAS
                df = self._fetch_from_cas(artifact.ref)
            
            dfs[name] = df
        
        return dfs
```

#### Success Criteria
- Can load targets from bundle
- Can evaluate targets against SimReturn
- Alignment strategies work with simulation outputs
- Loss computation produces correct values

### Stage 3: CalibrationSpec + AdaptiveAlgorithm (Weeks 5-6)

**Goal**: Enable optimization workflows with adaptive algorithms

#### Deliverables

**modelops-contracts:**
- [ ] `calibration.py`: CalibrationSpec with serialization
- [ ] Ensure AdaptiveAlgorithm protocol is complete

**modelops:**
- [ ] `calibration.py`: CalibrationExecution class
- [ ] Integration with ask-tell loop

**modelops-calabaria:**
- [ ] `adapters/algorithms.py`: Algorithm adapters
  - [ ] GridSearchAdapter (GridSearchEngine → AdaptiveAlgorithm)
  - [ ] OptunaAdapter (OptunaEngine → AdaptiveAlgorithm)
- [ ] Tests for algorithm adapters

#### Example CalibrationExecution

```python
# modelops/calibration.py

@dataclass
class CalibrationExecution:
    """Executes a CalibrationSpec using infrastructure."""
    
    spec: CalibrationSpec
    service: SimulationService
    algorithm: AdaptiveAlgorithm  # From adapter
    target_adapter: TargetAdapter  # From Stage 2
    
    def run(self) -> CalibrationResult:
        """Run calibration using ask-tell loop."""
        history = []
        
        while not self.algorithm.finished():
            # Ask for proposals
            proposals = self.algorithm.ask(self.spec.n_trials)
            if not proposals:
                time.sleep(0.1)
                continue
            
            # Create tasks
            tasks = []
            for prop in proposals:
                task = SimTask.from_entrypoint(
                    entrypoint=self.spec.entrypoint,
                    bundle_ref=self.spec.bundle_ref,
                    params=prop.params,
                    seed=self.spec.base_seed,
                    outputs=self.spec.outputs
                )
                tasks.append(task)
            
            # Submit with replicates
            futures_by_param = {}
            for task, prop in zip(tasks, proposals):
                futures = self.service.submit_replicates(task, self.spec.n_replicates)
                futures_by_param[prop.param_id] = futures
            
            # Aggregate replicates
            aggregated = {}
            for param_id, futures in futures_by_param.items():
                aggregated[param_id] = self.service.gather_and_aggregate(futures, "mean")
            
            # Compute losses
            trial_results = []
            for prop in proposals:
                sim_return = aggregated[prop.param_id]
                losses = self.target_adapter.evaluate_sim_return(sim_return)
                total_loss = sum(losses.values())  # Or weighted
                
                trial_results.append(TrialResult(
                    param_id=prop.param_id,
                    loss=total_loss,
                    diagnostics=losses,
                    status=TrialStatus.COMPLETED
                ))
            
            # Tell algorithm
            self.algorithm.tell(trial_results)
            history.extend(trial_results)
        
        return CalibrationResult(history=history)
```

---

## Part V: Complete Example Workflows

### Example 1: Parameter Sweep Workflow

```python
# Step 1: Scientist creates sweep in Calabaria
from calabaria.generators import SobolGenerator
from calabaria.parameters import ParameterSpecs

# Define parameter space
param_specs = ParameterSpecs({
    "beta": {"min": 0.1, "max": 0.5, "transform": "log"},
    "gamma": {"min": 0.01, "max": 0.2, "transform": "log"},
    "population": {"min": 10000, "max": 1000000}
})

# Generate Sobol sequence
generator = SobolGenerator(param_specs, n_samples=1000, scramble=True)

# Export to wire format
sweep_spec = generator.to_sweep_spec(
    entrypoint="sir_model.SIRModel/baseline@abc123def456",
    bundle_ref="sha256:abc123...",
    base_seed=42,
    n_replicates=10,
    outputs=["prevalence", "peak"]
)

# Save to experiments directory (NOT in bundle!)
with open("experiments/sweeps/001_sobol_exploration.yaml", "w") as f:
    f.write(sweep_spec.to_yaml())

# Step 2: Infrastructure executes sweep
from modelops_contracts import ParameterSweepSpec
from modelops.sweep import ParameterSweepExecution
from modelops.services import get_simulation_service

# Load sweep spec
with open("experiments/sweeps/001_sobol_exploration.yaml") as f:
    spec = ParameterSweepSpec.from_yaml(f.read())

# Execute using infrastructure
service = get_simulation_service()  # Gets DaskSimulationService
execution = ParameterSweepExecution(spec, service)

# Submit all parameters with replicates
futures_by_param = execution.submit_all()
# Returns Dict[param_id] -> List[replicate_futures]

# Aggregate replicates for each parameter
aggregated = execution.aggregate_replicates(futures_by_param, "mean")
# Returns Dict[param_id] -> SimReturn (aggregated)

# Save results
for param_id, result in aggregated.items():
    path = compute_output_path(spec.bundle_ref, spec, param_id, "aggregated")
    save_sim_return(result, path)
```

### Example 2: Model Definition
```python
class SIRModel(BaseModel):
    __version__ = "1.0.0"
    
    def __init__(self):
        space = ParameterSpace(specs=(
            ParameterSpec("beta", 0.0, 1.0, doc="Transmission rate"),
            ParameterSpec("gamma", 0.0, 1.0, doc="Recovery rate"),
        ))
        config = {"population": 100000, "initial_infected": 10, "days": 365}
        super().__init__(space, config)
    
    @model_scenario("lockdown")
    def _scn_lockdown(self) -> ScenarioSpec:
        return ScenarioSpec(
            name="lockdown",
            patch_params=lambda p: p.with_updates(beta=float(p["beta"]) * 0.35),
            patch_config=lambda c: {**c, "contact_reduction": 0.65},
            doc="Reduce beta by 65%"
        )
    
    def build_state(self, params, config):
        # Build initial simulation state (NO SEED!)
        ...
    
    def run_sim(self, sim, seed):
        # Run simulation, return raw results
        ...
    
    @model_output("prevalence")
    def extract_prevalence(self, raw, seed) -> pl.DataFrame:
        return pl.DataFrame({"t": raw["t"], "I": raw["I"]})
    
    @model_output("peak")
    def extract_peak(self, raw, seed) -> pl.DataFrame:
        peak_idx = np.argmax(raw["I"])
        return pl.DataFrame({
            "peak_time": [raw["t"][peak_idx]],
            "peak_infected": [raw["I"][peak_idx]]
        })
```

### 10. Phase 3: Contracts & Bridge (Week 3-4)

#### Deliverables
- [ ] **No contract updates needed** - contracts already have SimTask, TableArtifact, provenance
- [ ] `bridge/adapter.py`: Calabaria ↔ ModelOps adapter using SimTask
- [ ] `bridge/compiler.py`: Scenario compilation
- [ ] `bridge/artifacts.py`: TableArtifact creation and handling

#### Acceptance Criteria
- SimTask.task_id() is deterministic
- Bridge correctly translates between type systems
- TableArtifact inline/CAS logic works correctly
- Compiled scenarios work with SimulationService
- Arrow IPC serialization works

### 11. Phase 4: Provenance & Caching Integration (Week 4-5)

#### Deliverables
- [ ] `provenance/adapter.py`: Wrap contracts provenance functions
- [ ] `storage/sharded.py`: Sharded path generation
- [ ] `storage/cache.py`: Content-addressed cache using sim_root
- [ ] `storage/cas.py`: CAS integration for TableArtifact refs

#### Acceptance Criteria
- Use contracts' canonical_json and sim_root functions
- Same inputs produce same root hash via sim_root()
- Sharded paths distribute evenly
- Cache lookups work with SimTask.task_id()

#### Hash Stability Test
```python
def test_provenance_deterministic():
    tree1 = create_provenance_tree(params, config, seed)
    tree2 = create_provenance_tree(params, config, seed)
    assert tree1.root_hash == tree2.root_hash
```

### 12. Essential Test Specifications

#### Extractor Order Stability
```python
def test_output_order_stable():
    """Output order is deterministic across instances."""
    m1, m2 = SIRModel(), SIRModel()
    assert list(m1.outputs()) == list(m2.outputs())
```

#### Selective Extraction Equivalence
```python
def test_filtered_outputs_equivalence():
    """Infrastructure-level filtering should match direct filtering"""
    m = SIRModel()
    fn = m.compile_scenario(m.DEFAULT_SCENARIO)
    params = {"beta": 0.5, "gamma": 0.1}
    seed = 42
    
    # Get all outputs
    all_ipc = fn(params, seed)
    
    # Filter to subset (simulating what infrastructure does with task.outputs)
    subset_names = {"prevalence", "peak"}
    filtered_ipc = {k: v for k, v in all_ipc.items() if k in subset_names}
    
    # SimTask.outputs causes infra to filter; ensure bytes match direct filtering
    assert set(filtered_ipc.keys()) == subset_names
    assert filtered_ipc["prevalence"] == all_ipc["prevalence"]
    assert filtered_ipc["peak"] == all_ipc["peak"]
```

#### Task ID Semantics
```python
def test_task_id_semantics():
    """Different outputs -> same sim_root, different task_id"""
    from modelops_contracts.provenance import sim_root
    
    # Two tasks with different outputs
    task1 = SimTask(
        bundle_ref="sha256:abc123...",
        entrypoint="sir_model.SIRModel/baseline@abc123def456",
        params=UniqueParameterSet.from_dict({"beta": 0.5}),
        seed=42,
        outputs=["prevalence"]
    )
    task2 = SimTask(
        bundle_ref="sha256:abc123...",
        entrypoint="sir_model.SIRModel/baseline@abc123def456",
        params=UniqueParameterSet.from_dict({"beta": 0.5}),
        seed=42,
        outputs=["peak"]
    )
    
    # Same simulation (sim_root should be same)
    assert task1.sim_root() == task2.sim_root()
    
    # Different materializations (task_id should differ)
    assert task1.task_id() != task2.task_id()
```

#### Determinism Property Test
```python
def test_simulate_deterministic():
    """Same inputs always produce same outputs."""
    m = SIRModel()
    fn = m.compile_scenario(m.DEFAULT_SCENARIO)
    params = {"beta": 0.5, "gamma": 0.1}
    seed = 123
    
    # Run twice with same inputs
    a, b = fn(params, seed), fn(params, seed)
    
    # Should be identical
    assert a.keys() == b.keys()
    for k in a:
        assert a[k] == b[k]  # Byte-for-byte identical
```

### 13. Phase 5: Integration & Testing (Week 5-6)

#### Deliverables
- [ ] End-to-end SIR model example
- [ ] Distributed execution example
- [ ] Performance benchmarks
- [ ] User documentation

#### Integration Tests
```python
def test_end_to_end_execution():
    # Create model
    model = SIRModel(space, config)
    
    # Compile scenario
    sim = model.compile_scenario("lockdown")
    
    # Create bundle
    bundle = create_bundle(model)
    bundle_ref = push_bundle(bundle)
    
    # Submit via SimulationService
    from modelops_contracts.types import UniqueParameterSet
    from modelops_contracts.simulation import SimTask
    
    task = SimTask(
        bundle_ref="sha256:abc123...",  # OCI digest of the bundle
        entrypoint="sir_model.SIRModel/lockdown@abc123def456",  # Compiled entrypoint
        params=UniqueParameterSet.from_dict({"beta": 0.5, "gamma": 0.1}),
        seed=42,
        outputs=["prevalence", "peak"],  # Specific extractors
    )
    
    # Execute and verify
    future = sim_service.submit(task)
    result = sim_service.gather([future])[0]
    
    # Check results structure
    assert result.task_id == task.task_id()
    assert "prevalence" in result.outputs
    assert "peak" in result.outputs
    
    # Access table data
    prevalence_artifact = result.outputs["prevalence"]
    if prevalence_artifact.inline:
        # Small table, data is inline
        df = pl.read_ipc(BytesIO(prevalence_artifact.inline))
    else:
        # Large table, need to fetch from CAS
        df = fetch_and_read(prevalence_artifact.ref)
```

#### Performance Targets
- Parameter round-trip: < 1μs
- Transform application: < 10μs
- Provenance hash: < 100μs
- Cache lookup: < 1ms
- Simulation overhead: < 10ms

---

## Part IV: Open Decisions (MVP)

### Output Selection Strategy
**Decision**: Runtime via SimTask.outputs (not pre-compiled)
- Avoids combinatorial explosion of entrypoints
- Bridge can curry outputs on demand if needed
- Simpler for MVP

### Import Isolation
**Decision**: TODO for production
- MVP relies on correct environment setup
- Add proper bundle path isolation later
- Document requirement for clean environments

### Local Development
**Decision**: Use real OCI artifacts for MVP
- Skip git-based fallback for simplicity
- Require proper bundle push/pull workflow
- Can add local dev shortcuts in v2

### Cache Strategy
**Decision**: Simple dict cache in CalabariaBridge
- No LRU for MVP (add if memory becomes issue)
- Focus on correctness over optimization
- Profile before adding complexity

### Dependencies
**Decision**: ModelOps workers need Calabaria runtime
- Pinned via bundle's uv.lock
- Installed in isolated venv
- Bundle must include all dependencies

### Seed Width
**Decision**: uint64 (0 to 2^64-1)
- Matches numpy's requirements
- Validated in SimTask.__post_init__
- Document in all interfaces

---

## Part V: Appendices

### A. Complete SIR Model Example

```python
from calabaria import BaseModel, ParameterSpace, ParameterSpec, ParameterSet
from calabaria.decorators import model_scenario, model_output
from typing import Mapping, Any
import numpy as np
import polars as pl

class SIRModel(BaseModel):
    __version__ = "1.2.0"
    DEFAULT_SCENARIO = "conservative"  # Override default scenario name
    
    def __init__(self):
        space = ParameterSpace(specs=(
            ParameterSpec("beta", 0.0, 1.0, doc="Transmission rate"),
            ParameterSpec("gamma", 0.0, 1.0, doc="Recovery rate"),
        ))
        config = {
            "population": 100_000,
            "initial_infected": 10,
            "days": 200,
            "dt": 0.25
        }
        super().__init__(space, config)
        
        # Programmatic scenario addition (before sealing)
        self.add_scenario(ScenarioSpec(
            name="aggressive",
            patch_params=lambda p: p.with_updates(beta=float(p["beta"]) * 1.2),
            patch_config=lambda c: {**c, "days": c["days"] + 50},
            doc="20% higher transmission, longer horizon"
        ))
    
    def setup_scenarios(self):
        """Customize the default scenario (called during __init__)."""
        # Replace the "conservative" default with custom transforms
        self._scenarios[self.DEFAULT_SCENARIO] = ScenarioSpec(
            name=self.DEFAULT_SCENARIO,
            patch_params=lambda p: p.with_updates(beta=float(p["beta"]) * 0.9),
            patch_config=lambda c: c,  # No config changes
            doc="Conservative scenario with 10% lower transmission"
        )
    
    
    @model_scenario("vaccination")
    def _scn_vaccination(self) -> ScenarioSpec:
        return ScenarioSpec(
            name="vaccination",
            patch_params=lambda p: p,  # No param changes
            patch_config=lambda c: {**c, "vaccination_rate": 0.01},  # 1% per day
            doc="Daily vaccination at 1% of susceptible population"
        )
    
    def build_state(self, params: ParameterSet, config: Mapping[str, Any]) -> Mapping[str, Any]:
        """Deterministic assembly. No RNG. No seed."""
        N = int(config["population"])
        I0 = int(config["initial_infected"])
        
        return {
            "N": N,
            "S": float(N - I0),
            "I": float(I0),
            "R": 0.0,
            "beta": float(params["beta"]),
            "gamma": float(params["gamma"]),
            "dt": float(config["dt"]),
            "steps": int(config["days"] / config["dt"]),
            "vacc_rate": float(config.get("vaccination_rate", 0.0))
        }
    
    def run_sim(self, state: Mapping[str, Any], seed: int) -> RawSimOutput:
        """Execute stochastic sim. RNG boundary lives here."""
        rng = np.random.default_rng(seed)
        
        # Extract from immutable state
        S, I, R = state["S"], state["I"], state["R"]
        beta, gamma = state["beta"], state["gamma"]
        dt, steps, N = state["dt"], state["steps"], state["N"]
        vacc = state["vacc_rate"]
        
        # Storage arrays
        S_arr = np.empty(steps)
        I_arr = np.empty(steps)
        R_arr = np.empty(steps)
        
        # Simulation loop with stochastic elements
        for t in range(steps):
            # Optional stochastic modulation (bounded)
            beta_t = beta * np.clip(rng.normal(1.0, 0.1), 0.5, 1.5)
            
            # Vaccination (if configured)
            if vacc > 0 and S > 1:
                vacced = rng.binomial(int(S), min(vacc * dt, 1.0))
                S -= vacced
                R += vacced
            
            # SIR dynamics
            dS = -beta_t * S * I / N
            dI = beta_t * S * I / N - gamma * I
            dR = gamma * I
            
            S += dS * dt
            I += dI * dt
            R += dR * dt
            
            S_arr[t], I_arr[t], R_arr[t] = S, I, R
        
        return {"S": S_arr, "I": I_arr, "R": R_arr, "dt": dt}
    
    @model_output("prevalence")
    def out_prevalence(self, raw: RawSimOutput, seed: int) -> pl.DataFrame:
        """Extract infection prevalence over time."""
        n = raw["S"] + raw["I"] + raw["R"]
        t = pl.Series("t", [i * raw["dt"] for i in range(len(raw["I"]))])
        prev = pl.Series("prevalence", raw["I"] / n)
        return pl.DataFrame([t, prev])
    
    @model_output("compartments")
    def out_compartments(self, raw: RawSimOutput, seed: int) -> pl.DataFrame:
        """Extract all compartment time series."""
        dt = raw["dt"]
        return pl.DataFrame({
            "t": [i * dt for i in range(len(raw["S"]))],
            "S": raw["S"],
            "I": raw["I"],
            "R": raw["R"]
        })
    
    @model_output("peak")
    def out_peak(self, raw: RawSimOutput, seed: int) -> pl.DataFrame:
        """Extract peak infection metrics."""
        idx = int(np.argmax(raw["I"]))
        return pl.DataFrame({
            "t_peak": [idx * raw["dt"]],
            "I_peak": [float(raw["I"][idx])]
        })
    
    @model_output("prevalence_subsampled")
    def out_prev_subsampled(self, raw: RawSimOutput, seed: int) -> pl.DataFrame:
        """Example of extractor needing reproducible subsampling (discouraged)."""
        # If you must subsample in an extractor, use namespaced RNG for determinism
        rng = self._rng_for_output(seed, "prevalence_subsampled")
        n = len(raw["I"])
        keep = np.sort(rng.choice(n, size=min(100, n), replace=False))
        dt = raw["dt"]
        return pl.DataFrame({
            "t": [k * dt for k in keep],
            "I": [float(raw["I"][k]) for k in keep]
        })
```

### B. Complete Workflow Example

```python
# Development workflow using the refined API
from sir_model import SIRModel, ParameterSet, ScenarioSpec
import polars as pl

# 1. Create and explore model
m = SIRModel()
print(f"Scenarios: {m.scenarios()}")  # ['aggressive', 'conservative', 'lockdown', 'vaccination']
print(f"Default: {m.DEFAULT_SCENARIO}")  # 'conservative'
print(f"Sealed: {m.is_sealed}")  # False (not sealed yet)

# 2. Can still add scenarios before first use
m.add_scenario(ScenarioSpec(
    name="extreme",
    patch_params=lambda p: p.with_updates(beta=float(p["beta"]) * 2.0),
    doc="Double transmission for worst-case analysis"
))

# 3. Run DEFAULT_SCENARIO ("conservative") - this seals the model
ps = ParameterSet({"beta": 0.35, "gamma": 0.1})
dfs_default = m.simulate(ps, seed=7)  # Runs "conservative" scenario
print(f"Sealed: {m.is_sealed}")  # True (sealed after first simulate)

# 4. Try to add scenario after sealing (will fail)
try:
    m.add_scenario(ScenarioSpec(name="too_late", doc="Won't work"))
except RuntimeError as e:
    print(f"Error: {e}")  # "add_scenario() not allowed after first compile/simulate..."

# 5. Run specific scenarios
dfs_lock = m.simulate_scenario("lockdown", ps, seed=7)
dfs_vacc = m.simulate_scenario("vaccination", ps, seed=7)

# 6. Compile for production (returns Arrow IPC bytes)
fns = m.compile_all()  # All scenarios as wire functions
print(fns.keys())  # ['aggressive', 'conservative', 'extreme', 'lockdown', 'vaccination']

# 7. Use compiled functions with wire protocol
ipc_default = fns["conservative"](dict(ps.values), 7)  # params dict + seed
ipc_lock = fns["lockdown"](dict(ps.values), 7)

# 8. Reading Arrow IPC results back to DataFrames
from io import BytesIO
for name, arrow_bytes in ipc_lock.items():
    df = pl.read_ipc(BytesIO(arrow_bytes))
    print(f"{name}: {len(df)} rows")

# 9. Model identity for provenance
print(m.model_identity())
# 'sir_model.SIRModel:1.2.0:default=conservative'
```

### C. Testing Strategy

#### Unit Tests (Phase 1-2)
- Parameter validation and bounds
- Transform invertibility and monotonicity
- View fix/bind operations
- Scenario patching
- RNG boundary: build_state has no randomness
- RNG boundary: run_sim uses seed correctly

#### Integration Tests (Phase 3-4)
- Model compilation
- Bridge translation
- Wire protocol serialization
- Cache key stability
- Scenario compilation produces correct closures
- Arrow IPC round-trip

#### System Tests (Phase 5-6)
- End-to-end execution
- Distributed scenarios
- Performance benchmarks
- Provenance verification
- Replicate seed derivation
- Cache hit rates

#### Property-Based Tests
```python
from hypothesis import given, strategies as st

@given(
    params=st.dictionaries(
        st.sampled_from(["beta", "gamma"]),
        st.floats(min_value=0.0, max_value=1.0)
    ),
    seed=st.integers(min_value=0, max_value=2**31-1)
)
def test_simulation_deterministic(params, seed):
    model = SIRModel()
    pset = ParameterSet(values=params)
    
    result1 = model.simulate(pset, seed)
    result2 = model.simulate(pset, seed)
    
    for key in result1:
        assert result1[key].equals(result2[key])
```

### C. Migration Path

#### From Current Calabaria
1. **Week 1-2**: Implement new parameter system alongside old
2. **Week 3-4**: Port models to new BaseModel interface
3. **Week 5-6**: Update optimizers to use new contracts
4. **Post-MVP**: Deprecate old interfaces

#### Compatibility Shims
```python
# Temporary adapter for old-style models
class LegacyModelAdapter(BaseModel):
    def __init__(self, old_model):
        self.old_model = old_model
        super().__init__(
            space=self._extract_space(old_model),
            base_config=self._extract_config(old_model)
        )
```

### D. Future Extensions

#### Post-MVP Roadmap
1. **Streaming Outputs**: TableHandle for large results
2. **Coupled Transforms**: Non-separable parameter transforms
3. **Constraint System**: Parameter constraints beyond bounds
4. **Units & Dimensions**: Dimensional analysis
5. **Multi-language**: Support for R, Julia models

#### Extension Points
- Transform protocol can add `jacobian()` method
- OutputTable alias can become union type with handles
- Scenarios can support code patches (not just config/params)
- Provenance can include data lineage

---

## Part VI: Key Engineering Decisions

### 1. Why Experiments Are NOT in Bundles

**Decision**: Experiment specifications (sweeps, calibrations) are stored in `experiments/` directory, tracked in git, but NOT included in bundles.

**Rationale**:
- **Bundle Purity**: Bundle digest = hash(code + data) only. Adding experiments would change digest even when code/data unchanged.
- **Cache Efficiency**: Same bundle + same parameters = same result, regardless of experiment design.
- **Scientific Workflow**: Scientists iterate on experiments without invalidating caches.
- **Version Comparison**: Can run same experiment on different bundle versions.

**Implications**:
- Experiments are scientific history (git-tracked)
- Bundles are computational artifacts (content-addressed)
- Output paths incorporate both identities

### 2. Why Separate Sweeps from Calibrations

**Decision**: Two distinct contract types: ParameterSweepSpec and CalibrationSpec.

**Rationale**:
- **Different Execution**: Sweeps are batch/parallel, calibrations are adaptive/sequential
- **Different Data**: Sweeps have predetermined parameters, calibrations have search spaces
- **Different Outputs**: Sweeps organize by parameter, calibrations by trial/iteration
- **Clear Intent**: Scientists know if they're exploring or optimizing

**Implications**:
- Cleaner implementations (no if/else for mode)
- Better type safety
- More appropriate APIs for each use case

### 3. Why ParameterSweepSpec Is Minimal

**Decision**: The spec contains only execution information, no metadata about generation.

**Rationale**:
- **Separation of Concerns**: How parameters were generated is not needed for execution
- **Wire Protocol**: Smaller, cleaner data to serialize/transmit
- **Flexibility**: Science and infrastructure can evolve independently
- **No Coupling**: ModelOps doesn't need to know about Sobol vs Grid

**Implications**:
- Generation metadata stored separately (in bundle or git)
- Contracts stay stable even as generators evolve
- Maximum flexibility for both sides

### 4. Why Scenarios Are Entrypoints, Not Parameters

**Decision**: Each scenario compiles to a distinct entrypoint like `model.Class/scenario@digest`.

**Rationale**:
- **First-Class Compilation**: Scenarios are compiled artifacts, not runtime switches
- **Better Caching**: Different scenarios have different sim_roots
- **Cleaner Execution**: All entrypoints have same signature (params, seed) → output
- **No Special Cases**: Infrastructure doesn't need scenario logic

**Implications**:
- Bundle compilation creates multiple entrypoints
- SimTask uses entrypoint, not scenario parameter
- Scenarios are immutable once compiled

### 5. Why Replicates Use submit_replicates()

**Decision**: SimulationService has explicit `submit_replicates()` method rather than handling replicates at higher level.

**Rationale**:
- **Efficiency**: Service can optimize replicate execution (e.g., share data)
- **Seed Management**: Service handles SeedSequence derivation correctly
- **Provenance**: Replicates grouped naturally in output paths
- **Simplicity**: Higher-level code doesn't manage seed derivation

**Implications**:
- Consistent seed derivation across all services
- Easy aggregation of replicates
- Clear provenance tracking

### 6. Dependency Structure

**Decision**: Strict layering with no circular dependencies.

```
calabaria (pure science)
    ↓
modelops-contracts (protocols)
    ↑
modelops (infrastructure)
    ↑
modelops-calabaria (bridge/glue)
```

**Rationale**:
- **Clean Architecture**: Each layer has single responsibility
- **Independent Evolution**: Can update layers independently
- **No Lock-in**: Could replace any layer without affecting others
- **Clear Ownership**: Each team owns their layer

**Implications**:
- Calabaria has NO dependency on ModelOps or contracts
- ModelOps has NO dependency on Calabaria
- All coupling happens in modelops-calabaria bridge

---

## Conclusion

This comprehensive MVP plan delivers:

### Technical Achievements
1. **Clean Provenance Model**: Bundle digest (code+data) separate from experiment identity
2. **Wire-Friendly Contracts**: ParameterSweepSpec and CalibrationSpec as the seams
3. **Proper Replicate Handling**: Via submit_replicates() with SeedSequence
4. **Scenario Compilation**: Scenarios as entrypoints, not parameters
5. **Content-Addressed Caching**: Using sim_root() and task_id() for deterministic hashing

### Scientific Benefits
1. **Experiment Tracking**: Git-tracked experiment history in `experiments/`
2. **Cache Efficiency**: Experiments don't pollute bundle digest
3. **Version Comparison**: Run same experiment on different code versions
4. **Clear Workflows**: Separate sweeps (exploration) from calibrations (optimization)
5. **Reproducibility**: Every computation has deterministic provenance

### Architectural Wins
1. **Zero Coupling**: Calabaria and ModelOps remain independent
2. **Clean Layers**: Each package has single responsibility
3. **Extensible Contracts**: Can add fields without breaking
4. **Type Safety**: Immutable types with validation throughout
5. **Bridge Pattern**: All adaptation in modelops-calabaria

### Implementation Timeline
- **Weeks 1-2**: ParameterSweepSpec + Infrastructure (Stage 1)
- **Weeks 3-4**: Targets Integration (Stage 2)  
- **Weeks 5-6**: CalibrationSpec + AdaptiveAlgorithm (Stage 3)

The staged approach ensures we have working end-to-end functionality early, with each stage building on proven foundations. The experiment architecture particularly stands out as enabling the scientific workflow while maintaining computational efficiency through proper separation of concerns.
