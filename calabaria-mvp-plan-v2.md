# Calabaria MVP - Pure Functional Science, Clean Contracts
**Engineering Plan & Design Document v2**

> **tl;dr**: Pure functional core with immutable types throughout. ParameterSet is the
> only gateway to simulation. Scenarios are patches, not transforms. Clean separation
> between M-space (models) and P-space (research). Zero state leakage via frozen dataclasses.

---

## Executive Summary

### Vision

Build Calabaria as a **pure functional simulation framework** with immutable
data structures throughout. This guarantees reproducibility, eliminates
parameter state bugs, enables content-addressed caching, and scales from laptop
to cloud—while keeping the scientist experience delightful.

### Core Engineering Principles

1. **Immutability prevents state bugs**: ALL parameter types use
   @dataclass(frozen=True)
2. **ParameterSet is the ONLY gateway**: BaseModel.simulate() accepts only
   complete ParameterSet
3. **Scenarios are patches, not transforms**: Dictionary updates that can fix
   but not transform
4. **Transforms are for optimization only**: Change coordinates, not
   dimensionality
5. **Reparameterizations restructure parameter space**: Not in MVP, future extension
6. **ParameterView.bind() is the P→M bridge**: The only way to create complete
   ParameterSets
7. **Pure functional core**: All computations depend only on explicit inputs
8. **Content-addressed provenance**: Every computation has a deterministic hash

### Key Design Distinctions

**Scenarios vs Transforms vs Reparameterizations** - Three concepts that must not be conflated:

- **Scenarios**: User-facing convenience for model variants
  - Created by patches (dictionary updates) to parameters OR configs
  - Patches compose deterministically (last write wins, not order-invariant)
  - Can FIX parameters but NOT transform them
  - Example: "lockdown" scenario sets contact_rate=2, overriding base value

- **Transforms**: Mathematical mappings from the model parameter space to the
  transformed optimization (or sampling, for Bayesian methods) parameter space.
  - Only applies to free parameters in optimization/sampling contexts; there is
    no way to transform a fixed parameter; it is a point, not a space.
  - Change in coordinate system, not dimensionality
  - Example: logit transform for [0, 1] bounded parameters

- **Reparameterizations**: Unlike parameter transforms, these can reduce the
  dimensions of the parameter space P, e.g. φ := f(θ, α) = θ α (compound
  parameters). (NOT in MVP)
  - Map multiple parameters to different (possibly new) parameters or fewer
    through user-specified functions 
  - Can reduce dimensionality (many → few)
  - Unlike transforms, these do operations for parameters f(θ, μ) -> g(γ, α) or
    f(θ, μ) -> h(κ).
  - Example: Using R0 instead of beta, where R0 = beta/gamma

### What We're Building

- Immutable parameter system with ParameterView for fix/free partition
- DerivedModel facade for ergonomic P-space operations
- BaseModel with strict ParameterSet-only interface
- Scenarios as patches (not parameter mutations, patches of configs or fixing
  of parameters)
- Clean SimTask contracts for ModelOps integration
- Content-addressed caching via provenance hashes

### What We're NOT Building (Yet)

- Reparameterizations
- Constraint systems or units
- Multi-language support

---

## Grammar of Parameters

### Intent

Like the Grammar of Graphics, this "Grammar of Parameters" provides a small set of primitives, operators, and laws that compose into all common parameter workflows—exploration, scenarios, calibration—while staying pure, immutable, and predictable.

### Notation Guide

- `M` = Complete model parameter space (what BaseModel requires)
- `P` = Free parameter subspace (what researchers vary)
- `m ∈ M` = A point in M (a complete ParameterSet)
- `view` = Defines the P ⊆ M partition (fixed vs free)
- `∘` = Function composition, read right-to-left: `f ∘ g ∘ h` means `lambda x: f(g(h(x)))`

### 1. Primitives (Objects)

**ParameterSpace (M)**
- The model's complete set of named parameters with domains
- Intuition: "The full list of knobs the model understands"
- Names: `Names(M)` - finite set of parameter names
- Domain per name: `Domain(name)` - interval, integer range, or categorical
- Cartesian space: M is the product of all domains

**ParameterSet (m ∈ M)**
- A complete, immutable assignment for every name in M
- Intuition: "One concrete knob setting—what the model actually runs with"
- Enforced at BaseModel.simulate() boundary

**ParameterView**
- A partition of Names(M) into fixed and free
- Intuition: "Hold these steady, experiment with those"
- `fixed`: {name → value} - parameters held constant
- `free`: (name₁, ..., nameₖ) - parameters to vary (ordered)
- Induced free subspace: P(view) = ∏_{name ∈ free} Domain(name)

**Transform Set (T)**
- Per-free-parameter bijections for optimization coordinates
- Intuition: "Change units for the optimizer without changing the model"
- For each free name: T[name]: Domain(name) ↔ Coord(name)
- MVP: Independent transforms only, no coupling

**ScenarioPatch**
- Two partial updates applied at execution:
  - `param_patches`: {name → value} - fix/override parameters
  - `config_patches`: {key → value} - modify configuration
- Intuition: "Preset some knobs and options to embody a scenario"

**DerivedModel**
- An immutable research-facing object: (model, view, optional transforms, optional scenario)
- Intuition: "A base model specialized for the analysis you want"

### 2. Operators (Morphisms)

**Bind (P → M)**
```
bind(view, free_values) → ParameterSet
```
- Defined iff keys(free_values) == set(view.free)
- Returns ParameterSet(fixed ∪ free_values)
- Intuition: "Fill in the free knobs to get a full setting"

**ApplyPatch**
```
apply_patch(patch, m, cfg) → (m', cfg')
```
- Last-write-wins (LWW) for overlapping keys
- Intuition: "Bake scenario overrides into params/config"

**Transform/Inverse**
```
to_transformed(T, natural_P) → transformed_Z
from_transformed(T, transformed_Z) → natural_P
```
- Applied only to free parameters
- No dimensionality change
- Intuition: "Switch to optimizer space and back"

**ComposePatches (Monoid)**
```
compose(p₂, p₁) = "apply p₁ then p₂"
```
- Identity is the empty patch
- Associative: compose(p₃, compose(p₂, p₁)) = compose(compose(p₃, p₂), p₁)
- Intuition: "Stack scenarios deterministically"

**DerivedView (Reconciliation)**
```
derived_view(view, patch) → view'
```
- Moves any name in patch.param_patches from free → fixed with that value
- Intuition: "If a scenario fixes a free knob, it stops being free"
- This prevents double-specification bugs

### 3. Laws (Predictability & Safety)

**Views & Binding**
1. Partition: fixed ∩ free = ∅ and fixed ∪ free = Names(M)
2. Bind domain: bind(view, x) defined ⟺ keys(x) = set(free)
3. Fix idempotence: Adding same fixed kvs twice yields same view

**Transforms**
1. Per-coordinate bijection: to_transformed ∘ from_transformed = id
2. Domain safety: Transforms validate their natural domain (e.g., [0,1] for logit)

**Patches (Monoid)**
1. Associativity: compose(p₃, compose(p₂, p₁)) = compose(compose(p₃, p₂), p₁)
2. Identity: compose(p, empty) = compose(empty, p) = p
3. Right precedence (LWW): If both set a name, right operand wins

**Reconciliation Law**

For any view, patch, and free assignment x over view.free:
```
apply_patch(patch, bind(view, x))
  ==
bind(derived_view(view, patch), x \ keys(patch.param_patches))
```
Where `x \ keys` means "x with those keys removed"

Intuition: "Fixing a free parameter via scenario = moving it to fixed"

### 4. Execution Pipeline

**Natural P-space (no transforms):**
```
tables = extract ∘ run_sim(seed) ∘ build_sim ∘ ApplyPatch ∘ bind(view)
```
Read right-to-left: "Bind free values, apply scenario, build, run, extract"

**Transformed coordinates (optimization):**
```
tables = extract ∘ run_sim(seed) ∘ build_sim ∘ ApplyPatch ∘ bind(view) ∘ from_transformed(T)
```
Adds: "First map optimizer coords to natural, then proceed"

**Type signatures (precise flow):**
- `bind(view)`: FreeVals(view) → M
- `ApplyPatch(patch)`: M → (M, Cfg)
- `build_sim`: (M, Cfg) → State
- `run_sim(seed)`: State → Raw
- `extract`: Raw → Tables

### 5. Canonical Error Messages

```
# Bind mismatch
bind() error: missing={beta, gamma}; unexpected={alpha}

# Transform domain
transform 'beta:Logit01' requires 0≤x≤1, got 1.23

# Scenario overlap (strict mode)
scenario 'lockdown' overlaps keys {contact_rate};
set allow_overlap or use conflict_policy='lww'

# Unknown parameters
patch references unknown parameter(s): {foo, bar}
```

### Grammar in Action: Calibration Example

```python
# M has 6 parameters
view = ParameterView.from_fixed(
    M, population=10000, contact_rate=4.0,
    recovery_days=14, initial_infected=10
)
# ⇒ free = (beta, gamma)

# Scenario patches contact_rate
lockdown = ScenarioPatch(param_patches={"contact_rate": 2.0})

# Reconciliation: contact_rate moves from fixed=4.0 to fixed=2.0
view' = derived_view(view, lockdown)
# Still free = (beta, gamma), but fixed contact_rate changed

# Transforms for optimization
T = {"beta": Logit01(), "gamma": Logit01()}

# Pipeline execution:
Z_coords → from_transformed(T) → free_values
free_values → bind(view') → m
m → ApplyPatch(lockdown) → (m', cfg')
(m', cfg') → build_sim → state
state → run_sim(seed) → raw
raw → extract → tables
```

Note: Reconciliation makes the patch a no-op for parameters (contact_rate already in fixed), preventing double-specification.

---

## Part I: Architecture & Design

### 1. Pure Functional Foundation with Immutability

#### Why Immutability Matters

Parameter state bugs are a major source of irreproducibility in scientific computing.
A parameter dictionary that gets mutated during execution can lead to results that
depend on execution order, hidden state, and subtle bugs. We eliminate this entire
class of bugs through immutability:

- All parameter types are frozen dataclasses
- Methods return NEW objects, never mutate in place
- Collections use immutable types (Tuple not List, MappingProxyType not dict)

#### The Core Constraint

```python
# ENGINEERING CONSTRAINT: ParameterSet is the ONLY interface into simulate()
# This eliminates parameter state bugs and enforces complete specification

# Pure functional core
def build_sim(params: dict, config: dict) -> Any:
    """Deterministic assembly - no RNG, no seed"""
    return state

def run_sim(state: Any, seed: int) -> Any:
    """Single RNG touchpoint - all randomness flows through seed"""
    rng = np.random.default_rng(seed)
    return raw_output

# BaseModel with STRICT parameter interface
class Model(BaseModel):
    DEFAULT_SCENARIO = "baseline"

    def build_sim(self, params: ParameterSet, config: Mapping[str, Any]) -> Any:
        """ONLY accepts complete ParameterSet for space M

        Args:
            params: Complete ParameterSet with ALL model parameters

        No defaults, no partials, no dicts - complete specification required.
        """
        return {...}

    def run_sim(self, state: Any, seed: int) -> RawSimOutput:
        """Single RNG touchpoint"""
        return {...}

    @final
    def simulate(self, params: ParameterSet, seed: int):
        """THE gateway - ONLY accepts complete ParameterSet

        This constraint prevents parameter state bugs.
        Use ParameterView.bind() or DerivedModel for ergonomic P-space operations.
        """
        # Type enforcement - no dicts, no kwargs
        if not isinstance(params, ParameterSet):
            raise TypeError("simulate() requires ParameterSet")
        return super().simulate(params, seed)
```

### 2. Core Types & Parameter System

#### Design Goals

The parameter system must:
- Make common research operations easy (fixing parameters, exploration, calibration)
- Prevent state bugs through immutability
- Enforce complete specification (no hidden defaults)
- Provide clean separation between M-space (models) and P-space (research)

#### Type Aliases

```python
from typing import Union, Any, Tuple, Dict, Optional, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

Scalar = Union[float, int, bool, str]  # Wire format for parameters
RawSimOutput = Any  # Backend-specific simulation state
```

#### Parameter Specifications (Immutable)

```python
@dataclass(frozen=True)
class ParameterSpec:
    """Immutable parameter metadata"""
    name: str
    lower: float
    upper: float
    kind: str = "real"  # "real" | "int" | "cat"
    doc: str = ""

@dataclass(frozen=True)
class ParameterSpace:
    """Immutable space definition - foundation of type safety"""
    specs: Tuple[ParameterSpec, ...]  # Tuple for immutability
    version: str = "1.0"

    def __post_init__(self):
        # Validate uniqueness
        names = [s.name for s in self.specs]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate parameter names: {names}")

    def names(self) -> Tuple[str, ...]:
        return tuple(s.name for s in self.specs)

    def by_name(self) -> Dict[str, ParameterSpec]:
        return {s.name: s for s in self.specs}
```

#### ParameterSet - The ONLY Gateway

```python
@dataclass(frozen=True)
class ParameterSet:
    """Complete immutable parameter assignment for space M.

    THE ONLY TYPE accepted by BaseModel.simulate().
    Enforces complete specification - no defaults allowed.
    Immutability prevents parameter mutation bugs.
    """
    values: Dict[str, Scalar]

    def __post_init__(self):
        # Make values immutable
        object.__setattr__(self, 'values', MappingProxyType(self.values))

    def __getitem__(self, k: str) -> Scalar:
        return self.values[k]

    def with_updates(self, **updates: Scalar) -> "ParameterSet":
        """Return NEW ParameterSet with updates (immutable pattern)"""
        d = dict(self.values)
        d.update(updates)
        return ParameterSet(d)
```

#### ParameterView - Immutable Fix/Free Partition

```python
@dataclass(frozen=True)
class ParameterView:
    """Immutable lens defining fix/free partition over space M.

    Single source of truth for what's fixed vs free.
    All operations return NEW views (immutability).
    The bind() method is THE bridge from P-space to M-space.
    """
    space: ParameterSpace
    fixed: Dict[str, Scalar] = field(default_factory=dict)
    free: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Make collections immutable
        object.__setattr__(self, 'fixed', MappingProxyType(self.fixed))

        # Validate complete partition
        all_names = set(self.space.names())
        fixed_names = set(self.fixed.keys())
        free_names = set(self.free)

        if not fixed_names.issubset(all_names):
            raise ValueError(f"Unknown fixed params: {fixed_names - all_names}")
        if not free_names.issubset(all_names):
            raise ValueError(f"Unknown free params: {free_names - all_names}")
        if fixed_names & free_names:
            raise ValueError(f"Params both fixed and free: {fixed_names & free_names}")
        if (fixed_names | free_names) != all_names:
            missing = all_names - (fixed_names | free_names)
            raise ValueError(f"Unaccounted params: {missing}")

    @classmethod
    def all_free(cls, space: ParameterSpace) -> "ParameterView":
        """Create view with all parameters free"""
        return cls(space=space, fixed={}, free=space.names())

    @classmethod
    def from_fixed(cls, space: ParameterSpace, **fixed: Scalar) -> "ParameterView":
        """Create view with specified parameters fixed"""
        free = tuple(n for n in space.names() if n not in fixed)
        return cls(space=space, fixed=fixed, free=free)

    def fix(self, **kv: Scalar) -> "ParameterView":
        """Return NEW view with additional parameters fixed (immutable)"""
        merged = {**self.fixed, **kv}
        free = tuple(n for n in self.free if n not in kv)
        return ParameterView(self.space, merged, free)

    def bind(self, **free_values: Scalar) -> ParameterSet:
        """Create complete ParameterSet M from free values P.

        This is THE CRITICAL BRIDGE from P-space to M-space.
        Validates that exactly the free parameters are provided.
        """
        provided = set(free_values.keys())
        expected = set(self.free)

        if provided != expected:
            missing = expected - provided
            extra = provided - expected
            msgs = []
            if missing: msgs.append(f"missing: {missing}")
            if extra: msgs.append(f"unexpected: {extra}")
            raise ValueError(f"bind() error: {'; '.join(msgs)}")

        # Create complete M from fixed + free
        complete = {**self.fixed, **free_values}
        return ParameterSet(complete)
```

#### Transforms for Optimization (NOT Scenarios)

```python
from typing import Protocol

class Transform(Protocol):
    """Transform protocol for optimization coordinates.

    Transforms do NOT change dimensionality, only coordinate system.
    Applied ONLY to free parameters in optimization contexts.
    NOT used for scenarios (scenarios use patches).
    """
    def forward(self, x: float) -> float: ...
    def backward(self, y: float) -> float: ...

@dataclass(frozen=True)
class Identity:
    """Identity transform (no-op)"""
    def forward(self, x: float) -> float: return float(x)
    def backward(self, y: float) -> float: return float(y)

@dataclass(frozen=True)
class AffineSqueezedLogit:
    """Transform for [0,1] bounded parameters in optimization"""
    eps: float = 1e-6

    def forward(self, x: float) -> float:
        """Natural [0,1] → Unbounded for optimizer"""
        import math
        if not (0.0 <= x <= 1.0):
            raise ValueError(f"AffineSqueezedLogit requires 0≤x≤1, got {x}")
        p = self.eps + (1.0 - 2.0*self.eps) * x
        return math.log(p / (1.0 - p))

    def backward(self, y: float) -> float:
        """Unbounded → Natural [0,1]"""
        import math
        s = 1.0 / (1.0 + math.exp(-y))
        return (s - self.eps) / (1.0 - 2.0*self.eps)

@dataclass(frozen=True)
class TransformedView:
    """Immutable view with transforms for optimization.

    Transforms are ONLY for optimization coordinates.
    NOT for scenarios - scenarios FIX parameters via patches.
    """
    view: ParameterView
    transforms: Dict[str, Transform]  # Only for free parameters

    def __post_init__(self):
        # Make transforms immutable
        object.__setattr__(self, 'transforms', MappingProxyType(self.transforms))

        # Validate transforms only for free params
        transform_names = set(self.transforms.keys())
        free_names = set(self.view.free)
        if not transform_names.issubset(free_names):
            extra = transform_names - free_names
            raise ValueError(f"Transforms for non-free params: {extra}")

    def to_transformed(self, free_values: Dict[str, Scalar]) -> Dict[str, float]:
        """Convert natural P to transformed coordinates for optimizer"""
        out = {}
        for name in self.view.free:
            val = float(free_values[name])
            t = self.transforms.get(name, Identity())
            out[name] = t.forward(val)
        return out

    def from_transformed(self, coords: Dict[str, float]) -> Dict[str, Scalar]:
        """Convert optimizer coordinates back to natural P"""
        out = {}
        for name in self.view.free:
            t = self.transforms.get(name, Identity())
            out[name] = t.backward(coords[name])
        return out
```

### 3. Model Interface & Scenarios

#### Scenarios as Patches (NOT Transforms)

Scenarios are user-facing convenience for creating model variants through patches:

```python
@dataclass(frozen=True)
class ScenarioSpec:
    """Scenario as composition of patches.

    Key principles:
    - Scenarios can FIX parameters but NOT transform them
    - Patches compose deterministically (last write wins by default)
    - Not order-invariant for overlapping keys
    - Applied at simulation time, not at configuration time
    """
    name: str
    doc: str = ""
    param_patches: Dict[str, Scalar] = field(default_factory=dict)
    config_patches: Dict[str, Any] = field(default_factory=dict)
    conflict_policy: Literal["lww", "strict"] = "lww"  # last-write-wins or strict
    allow_overlap: Optional[Tuple[str, ...]] = None  # params allowed to overlap

    def __post_init__(self):
        # Make patches immutable
        object.__setattr__(self, 'param_patches', MappingProxyType(self.param_patches))
        object.__setattr__(self, 'config_patches', MappingProxyType(self.config_patches))

    def apply(self, params: ParameterSet, config: Mapping) -> Tuple[ParameterSet, Mapping]:
        """Apply patches to create scenario variant.

        Patches are dictionary updates - last write wins (or strict checking).
        Can FIX parameters but cannot TRANSFORM them.
        """
        # Check for conflicts in strict mode
        if self.conflict_policy == "strict":
            overlapping = set(self.param_patches.keys()) & set(params.values.keys())
            if overlapping and self.allow_overlap:
                overlapping -= set(self.allow_overlap)
            if overlapping:
                raise ValueError(
                    f"Scenario '{self.name}' overlaps parameters {overlapping}. "
                    f"Set allow_overlap or use conflict_policy='lww'"
                )

        # Apply parameter patches (fixing, not transforming)
        if self.param_patches:
            new_params = params.with_updates(**self.param_patches)
        else:
            new_params = params

        # Apply config patches
        if self.config_patches:
            new_config = {**config, **self.config_patches}
        else:
            new_config = config

        return new_params, MappingProxyType(new_config)
```

#### BaseModel - Pure M-Space Interface

```python
from abc import ABC, abstractmethod
import polars as pl

class BaseModel(ABC):
    """Pure functional model accepting ONLY complete ParameterSet.

    Invariants enforced:
    - simulate() ONLY accepts ParameterSet (complete M)
    - All state is immutable (frozen dataclasses, immutable collections)
    - Scenarios are patches, not transforms
    - Seal-on-first-use prevents configuration mutations
    - No hidden defaults - complete specification required
    """

    DEFAULT_SCENARIO: str = "baseline"  # Subclasses can override

    def __init__(self, space: ParameterSpace, base_config: Mapping[str, Any] = None):
        self.space = space  # Immutable ParameterSpace
        self.base_config = MappingProxyType(base_config or {})

        # Mutable until sealed
        self._scenarios: Dict[str, ScenarioSpec] = {
            "baseline": ScenarioSpec("baseline", doc="Default scenario")
        }
        self._outputs: Dict[str, Callable] = {}
        self._sealed = False

        # Hook for subclasses
        self.setup_scenarios()

    def setup_scenarios(self) -> None:
        """Hook for subclasses to add scenarios before sealing"""
        pass

    @abstractmethod
    def build_sim(self, params: ParameterSet, config: Mapping[str, Any]) -> Any:
        """Build simulation state with COMPLETE ParameterSet M.

        Args:
            params: Complete ParameterSet containing ALL model parameters
            config: Configuration dictionary (may be patched by scenario)

        No defaults, no partials - complete specification required.
        This is pure deterministic assembly - no RNG, no seed.
        """
        pass

    @abstractmethod
    def run_sim(self, state: Any, seed: int) -> RawSimOutput:
        """Run simulation with single RNG touchpoint.

        Args:
            state: Pre-built simulation state from build_sim
            seed: Random seed for all stochasticity

        ALL randomness flows through this single seed parameter.
        """
        pass

    @model_output("prevalence")
    def output_prevalence(self, raw: RawSimOutput, seed: int) -> pl.DataFrame:
        """Example output extractor"""
        pass

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
        """
        # Type enforcement
        if not isinstance(params, ParameterSet):
            raise TypeError(
                f"simulate() requires ParameterSet, got {type(params).__name__}. "
                "Use ParameterView.bind() to create complete parameter sets."
            )

        # Completeness check
        required = set(self.space.names())
        provided = set(params.values.keys())
        if provided != required:
            missing = required - provided
            extra = provided - required
            raise ValueError(
                f"ParameterSet incomplete. "
                f"Missing: {missing}, Extra: {extra}"
            )

        return self.simulate_scenario(self.DEFAULT_SCENARIO, params, seed)

    @final
    def simulate_scenario(self, scenario: str, params: ParameterSet, seed: int) -> Dict[str, pl.DataFrame]:
        """Run specific scenario with complete ParameterSet"""
        self._seal()

        if scenario not in self._scenarios:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self._scenarios.keys())}")

        spec = self._scenarios[scenario]

        # Apply scenario patches (can FIX but not TRANSFORM)
        params_patched, config_patched = spec.apply(params, self.base_config)

        # Run simulation pipeline
        state = self.build_sim(params_patched, config_patched)
        raw = self.run_sim(state, seed)

        # Extract outputs
        results = {}
        for name, extractor in self._outputs.items():
            results[name] = extractor(raw, seed)

        return results

    @final
    def compile_scenario(self, scenario: Optional[str] = None) -> Callable[[Dict[str, Scalar], int], Dict[str, bytes]]:
        """Compile scenario to wire function for infrastructure"""
        self._seal()
        sc = scenario or self.DEFAULT_SCENARIO

        if sc not in self._scenarios:
            raise ValueError(f"Unknown scenario: {sc}")

        spec = self._scenarios[sc]
        output_names = tuple(self._outputs.keys())

        def wire_fn(params_dict: Dict[str, Scalar], seed: int) -> Dict[str, bytes]:
            # Convert to ParameterSet
            pset = ParameterSet(values=params_dict)

            # Apply patches
            pset_patched, config_patched = spec.apply(pset, self.base_config)

            # Run simulation
            state = self.build_sim(pset_patched, config_patched)
            raw = self.run_sim(state, seed)

            # Extract and serialize outputs
            out: Dict[str, bytes] = {}
            for name in output_names:
                df = self._outputs[name](raw, seed)
                bio = io.BytesIO()
                df.write_ipc(bio)
                out[name] = bio.getvalue()

            return out

        return wire_fn

    def extract_outputs(self, raw: RawSimOutput, seed: int) -> Dict[str, pl.DataFrame]:
        """Extract outputs from raw simulation results.

        Helper method so DerivedModel can reuse extraction logic.
        """
        return {name: fn(raw, seed) for name, fn in self._outputs.items()}

    def derived_view(self, view: ParameterView, scenario: ScenarioSpec) -> ParameterView:
        """Reconcile view with scenario patches.

        Moves any parameter in scenario.param_patches from free→fixed.
        This prevents double-specification bugs where a scenario fixes
        a parameter that the user might try to vary.

        Args:
            view: Current parameter view
            scenario: Scenario with patches to reconcile

        Returns:
            New view with patched parameters moved to fixed
        """
        if not scenario.param_patches:
            return view  # No patches, no reconciliation needed

        # Move patched parameters from free to fixed
        patched_names = set(scenario.param_patches.keys())
        new_fixed = {**dict(view.fixed), **scenario.param_patches}
        new_free = tuple(n for n in view.free if n not in patched_names)

        return ParameterView(self.space, new_fixed, new_free)

    def _seal(self) -> None:
        """Freeze configuration on first use (seal-on-first-use pattern)"""
        if not self._sealed:
            self._scenarios = MappingProxyType(self._scenarios)
            self._outputs = MappingProxyType(self._outputs)
            self._sealed = True
```

#### DerivedModel - Ergonomic P-Space Facade with Reconciliation

```python
@dataclass(frozen=True)
class DerivedModel:
    """Model derived from base with fixed view and/or scenario patches.

    This is the ergonomic P-space interface for researchers.
    Immutable - all derivations create new objects.
    Scenarios are reconciled with the view to prevent double-specification.
    """
    model: BaseModel
    view: ParameterView
    tview: Optional[TransformedView] = None
    scenario_spec: Optional[ScenarioSpec] = None

    def __post_init__(self):
        """Reconcile view with scenario to prevent double-specification"""
        if self.scenario_spec and self.scenario_spec.param_patches:
            # Move any patched parameters from free to fixed
            reconciled_view = self.model.derived_view(self.view, self.scenario_spec)
            object.__setattr__(self, 'view', reconciled_view)

    def simulate(self, seed: int, **free_values: Scalar) -> Dict[str, pl.DataFrame]:
        """Clean API - no scenario parameter needed.

        Only accepts free parameters, converts to M internally via bind().
        Scenario patches are already reconciled with view.
        """
        # P → M conversion via bind()
        params_M = self.view.bind(**free_values)

        # Apply scenario if we have one
        if self.scenario_spec:
            params_M, config = self.scenario_spec.apply(params_M, self.model.base_config)
        else:
            config = self.model.base_config

        # Direct execution with patches applied
        state = self.model.build_sim(params_M, config)
        raw = self.model.run_sim(state, seed)
        return self.model.extract_outputs(raw, seed)

    def with_scenario(self, scenario_name: str) -> "DerivedModel":
        """Derive new model with scenario patches.

        Note: This may change which parameters are free!
        Scenario patches move from free→fixed via reconciliation.
        """
        spec = self.model._scenarios[scenario_name]
        return DerivedModel(self.model, self.view, self.tview, spec)

    def with_view(self, view: ParameterView) -> "DerivedModel":
        """Derive new model with different view"""
        return DerivedModel(self.model, view, self.tview, self.scenario_spec)

    def compile(self) -> Callable[[Dict[str, Scalar], int], Dict[str, bytes]]:
        """Compile P-space wire function for optimization.

        Returns function accepting ONLY free parameters (P-space).
        Handles P→M conversion and scenario patches internally.
        """
        fixed = dict(self.view.fixed)
        free_names = set(self.view.free)
        scenario_spec = self.scenario_spec
        base_config = self.model.base_config

        def p_wire_fn(free_params: Dict[str, Scalar], seed: int) -> Dict[str, bytes]:
            # Validate P-space parameters
            if set(free_params.keys()) != free_names:
                raise ValueError(
                    f"Expected free params {free_names}, "
                    f"got {set(free_params.keys())}"
                )

            # P → M conversion
            complete_M = {**fixed, **free_params}
            params = ParameterSet(complete_M)

            # Apply scenario patches if present
            if scenario_spec:
                params, config = scenario_spec.apply(params, base_config)
            else:
                config = base_config

            # Execute pipeline
            state = self.model.build_sim(params, config)
            raw = self.model.run_sim(state, seed)

            # Serialize outputs
            outputs = {}
            for name, fn in self.model._outputs.items():
                df = fn(raw, seed)
                bio = io.BytesIO()
                df.write_ipc(bio)
                outputs[name] = bio.getvalue()

            return outputs

        return p_wire_fn

    def bounds(self, transformed: bool = False) -> Dict[str, Tuple[float, float]]:
        """Get bounds for free parameters.

        Args:
            transformed: If True, return bounds in transformed space for optimization
        """
        bounds = {}
        meta = self.view.space.by_name()

        for name in self.view.free:
            spec = meta[name]
            lo, hi = float(spec.lower), float(spec.upper)

            # Apply transforms for optimization (NOT scenarios)
            if transformed and self.tview:
                t = self.tview.transforms.get(name)
                if t:
                    lo, hi = t.forward(lo), t.forward(hi)

            bounds[name] = (lo, hi)

        return bounds

    def to_transformed(self, natural_P: Dict[str, Scalar]) -> Dict[str, float]:
        """Convert natural P-space parameters to transformed optimization coordinates.

        Args:
            natural_P: Natural parameter values for free parameters

        Returns:
            transformed_Z: Transformed coordinates for optimizer
        """
        if not self.tview:
            return {k: float(v) for k, v in natural_P.items()}
        return self.tview.to_transformed(natural_P)

    def from_transformed(self, transformed_Z: Dict[str, float]) -> Dict[str, Scalar]:
        """Convert transformed optimization coordinates to natural P-space parameters.

        Args:
            transformed_Z: Transformed coordinates from optimizer

        Returns:
            natural_P: Natural parameter values for free parameters
        """
        if not self.tview:
            return dict(transformed_Z)
        return self.tview.from_transformed(transformed_Z)

    @classmethod
    def from_model(cls, model: BaseModel, **fixed_params) -> "DerivedModel":
        """Create derived model with parameters fixed"""
        view = ParameterView.from_fixed(model.space, **fixed_params)
        return cls(model, view)

    @classmethod
    def from_scenario(cls, model: BaseModel, scenario: str, **fixed_params) -> "DerivedModel":
        """Create derived model with scenario and fixed params"""
        view = ParameterView.from_fixed(model.space, **fixed_params)
        spec = model._scenarios[scenario]
        return cls(model, view, scenario_spec=spec)
```

## Part II: User Workflows

### Basic Research Workflow

```python
# 1. Define model with complete parameter space M
class SIRModel(BaseModel):
    """SIR epidemic model - requires complete parameter specification"""

    PARAM_SPACE = ParameterSpace(specs=(
        ParameterSpec("beta", 0.0, 1.0, doc="Transmission rate"),
        ParameterSpec("gamma", 0.0, 1.0, doc="Recovery rate"),
        ParameterSpec("population", 100, 1000000, kind="int"),
        ParameterSpec("contact_rate", 1.0, 20.0),
        ParameterSpec("recovery_days", 1, 30, kind="int"),
        ParameterSpec("initial_infected", 1, 100, kind="int"),
    ))

    def __init__(self):
        super().__init__(self.PARAM_SPACE)

    def setup_scenarios(self):
        """Define scenarios as PATCHES not TRANSFORMS"""
        self._scenarios["lockdown"] = ScenarioSpec(
            "lockdown",
            doc="Reduced contact scenario",
            param_patches={"contact_rate": 2.0}  # FIX parameter via patch
        )
        self._scenarios["vaccination"] = ScenarioSpec(
            "vaccination",
            doc="With vaccine rollout",
            config_patches={"vaccine_start": 100, "vaccine_efficacy": 0.9}
        )

    def build_sim(self, params: ParameterSet, config: Mapping) -> Any:
        """Build with complete M - no defaults!"""
        return {
            "N": params["population"],
            "beta": params["beta"] * params["contact_rate"],
            "gamma": 1.0 / params["recovery_days"],
            "I0": params["initial_infected"],
            "config": config
        }

    def run_sim(self, state: Any, seed: int) -> RawSimOutput:
        """Run simulation with single RNG"""
        rng = np.random.default_rng(seed)
        # ... simulation logic ...
        return results

# 2. Research exploration with ParameterView
model = SIRModel()

# Fix experimental constants
view = ParameterView.from_fixed(
    model.space,
    population=10000,
    contact_rate=4.0,
    recovery_days=14,
    initial_infected=10
)
# Now view.free = ("beta", "gamma")

# Quick exploration - must use bind() for P→M conversion
for beta in [0.2, 0.3, 0.4]:
    for gamma in [0.08, 0.1, 0.12]:
        params_M = view.bind(beta=beta, gamma=gamma)  # P → M
        results = model.simulate(params_M, seed=42)
        print(f"β={beta}, γ={gamma}: peak={results['compartments']['I'].max()}")

# 3. Use DerivedModel for cleaner syntax
derived = DerivedModel(model, view)

# Now can pass just P-space parameters
results = derived.simulate(seed=42, beta=0.3, gamma=0.1)

# 4. Compare scenarios (patches, not transforms)
baseline = DerivedModel(model, view)  # No scenario = baseline
lockdown = derived.with_scenario("lockdown")  # Derives new model with patches

# Same parameters, different scenarios
P = {"beta": 0.3, "gamma": 0.1}
results_baseline = baseline.simulate(seed=42, **P)
results_lockdown = lockdown.simulate(seed=42, **P)
# lockdown has contact_rate=2.0 due to patch
```

### Calibration Workflow

```python
# 1. Set up transforms for optimization (NOT scenarios)
tview = TransformedView(
    view=view,
    transforms={
        "beta": AffineSqueezedLogit(),  # [0,1] bounded
        "gamma": AffineSqueezedLogit()  # [0,1] bounded
    }
)

# 2. Create calibration model
calib_model = DerivedModel(model, view, tview)

# 3. Compile P-space function for optimization
wire_fn = calib_model.compile()  # Returns fn(P, seed) -> bytes

# 4. Optimizer works in transformed space
from scipy.optimize import differential_evolution

def loss(Z_coords):  # Z = transformed coordinates
    # Convert from transformed to natural P
    P_natural = calib_model.from_transformed(
        dict(zip(["beta", "gamma"], Z_coords))
    )

    # Run simulation
    results_bytes = wire_fn(P_natural, seed=42)

    # Compute loss
    results = deserialize(results_bytes)
    return compute_loss(results, targets)

# Get bounds in transformed space
bounds = calib_model.bounds(transformed=True)
bounds_list = [(bounds["beta"]), (bounds["gamma"])]

# Run optimization
opt_result = differential_evolution(loss, bounds_list)

# 5. Convert back to natural parameters
optimal_P = calib_model.from_transformed(
    dict(zip(["beta", "gamma"], opt_result.x))
)
print(f"Optimal: β={optimal_P['beta']:.3f}, γ={optimal_P['gamma']:.3f}")
```

### Post-Calibration Sensitivity

```python
# Fix optimal values, vary one at a time
optimal = {"beta": 0.35, "gamma": 0.095}

# Create new view with all optimal except beta
# Note: returns NEW view (immutable)
sens_view = view.fix(**optimal)  # Fix both
sens_view = ParameterView.from_fixed(
    model.space,
    **{**view.fixed, **optimal, "beta": view.fixed.get("beta", 0.35)}
)
# Remove beta from fixed to make it free
sens_view = ParameterView.from_fixed(
    model.space,
    population=10000,
    contact_rate=4.0,
    recovery_days=14,
    initial_infected=10,
    gamma=0.095  # Fixed at optimal
)
# Now only beta is free

# Sweep beta around optimal
sens_model = DerivedModel(model, sens_view)
beta_values = np.linspace(0.25, 0.45, 20)
sensitivity_results = []

for beta in beta_values:
    results = sens_model.simulate(seed=42, beta=beta)
    peak = results['compartments']['I'].max()
    sensitivity_results.append((beta, peak))

# Plot sensitivity...
```

## Part III: Execution & Infrastructure

### SimTask Contract

```python
@dataclass(frozen=True)
class UniqueParameterSet:
    """Parameters with deterministic ID for deduplication"""
    values: Dict[str, Scalar]

    @property
    def param_id(self) -> str:
        """Deterministic hash of parameters"""
        import hashlib
        import json
        canonical = json.dumps(self.values, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

@dataclass(frozen=True)
class SimTask:
    """Immutable simulation task for infrastructure.

    Complete specification for deterministic execution.
    """
    bundle_ref: str  # Code version
    entrypoint: str  # "module.Class/scenario"
    params: UniqueParameterSet  # Complete M-space parameters
    seed: int
    outputs: Optional[Tuple[str, ...]] = None  # Which outputs to extract

    def __post_init__(self):
        # Make outputs immutable
        if self.outputs is not None:
            object.__setattr__(self, 'outputs', tuple(sorted(self.outputs)))

    @property
    def sim_root(self) -> str:
        """Deterministic simulation identity (excludes outputs)"""
        # Hash of (code, params, seed, scenario)
        # NOT including outputs - enables cache reuse
        pass

    @property
    def task_id(self) -> str:
        """Complete task identity (includes outputs)"""
        # Hash of (sim_root, outputs)
        # Unique per output selection
        pass
```

### Direct Execution (No Bridge Needed)

```python
def execute_task(task: SimTask) -> SimReturn:
    """Execute simulation task directly"""

    # Parse entrypoint
    import_path, scenario = parse_entrypoint(task.entrypoint)

    # Load model class
    module_path, class_name = import_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    # Instantiate and compile
    model = model_class()
    wire_fn = model.compile_scenario(scenario)

    # Execute with complete M-space parameters
    result_bytes = wire_fn(task.params.values, task.seed)

    # Filter outputs if requested
    if task.outputs:
        result_bytes = {k: v for k, v in result_bytes.items()
                       if k in task.outputs}

    return SimReturn(outputs=result_bytes)
```

### Calibration Execution

```python
def setup_calibration(model_class, fixed_params, free_params, targets):
    """Set up calibration with proper P→M handling"""

    # Create model and view
    model = model_class()
    view = ParameterView.from_fixed(model.space, **fixed_params)

    # Validate free parameters
    if set(free_params) != set(view.free):
        raise ValueError(f"Free params mismatch: {free_params} vs {view.free}")

    # Set up transforms for bounded parameters
    transforms = {}
    meta = model.space.by_name()
    for name in free_params:
        spec = meta[name]
        # Use logit for [0,1] bounded parameters
        if spec.lower == 0.0 and spec.upper == 1.0:
            transforms[name] = AffineSqueezedLogit()

    tview = TransformedView(view=view, transforms=transforms)

    # Create bound model for calibration
    derived = DerivedModel(model, view, tview)

    # Compile P-space function
    sim_fn = derived.compile()

    return derived, sim_fn, targets

def run_calibration(derived: DerivedModel, sim_fn: Callable, targets: Any):
    """Run calibration in P-space"""

    from scipy.optimize import differential_evolution

    def loss(Z_coords):
        # Convert from transformed to natural P
        P = derived.from_transformed(
            dict(zip(derived.view.free, Z_coords))
        )

        # Run simulation
        results = sim_fn(P, seed=42)

        # Compute loss against targets
        return compute_loss(results, targets)

    # Get bounds in transformed space
    bounds = derived.bounds(transformed=True)
    bounds_list = [bounds[name] for name in derived.view.free]

    # Optimize
    result = differential_evolution(loss, bounds_list)

    # Return natural parameters
    return derived.from_transformed(
        dict(zip(derived.view.free, result.x))
    )
```

## Part IV: Key Engineering Decisions

### Why Immutability Everywhere

Parameter state bugs are insidious and hard to debug. A mutable parameter dictionary
that gets modified during execution leads to:
- Results that depend on execution order
- Hidden coupling between components
- Irreproducible results
- Difficult debugging

We eliminate this entire class of bugs through frozen dataclasses and immutable collections.

### Why ParameterSet is the Only Gateway

Having multiple ways to pass parameters (dicts, kwargs, partials) leads to:
- Ambiguity about what parameters are required
- Hidden defaults that change behavior
- Difficulty tracking parameter flow
- Versioning nightmares

By enforcing ParameterSet as the only interface, we get:
- Complete specification always required
- Type safety at the boundary
- Clear parameter flow
- Reproducible results

### Why Scenarios are Patches, Not Transforms

Scenarios and transforms solve different problems:
- **Scenarios**: Model variants for research (e.g., "lockdown" conditions)
- **Transforms**: Coordinate changes for optimization (e.g., logit transform)

Conflating them leads to confusion about when transforms apply and makes
scenarios dependent on optimization setup. Keeping them separate provides
clarity and composability.

### Why No Reparameterizations in MVP

Reparameterizations restructure the parameter space and add complexity:
- Can reduce dimensionality (e.g., R0 = beta/gamma maps two params to one)
- Can maintain dimensionality with different structure (e.g., coupling parameters φ = θα)
- Create compound or derived parameters requiring inverse mappings
- Complicate bounds, constraints, and Jacobian calculations
- Add another layer beyond simple coordinate transforms

We defer this to post-MVP to keep the initial system simple and correct.

## Part V: Implementation Timeline

### Week 1-2: Core Parameter System
- Implement frozen dataclasses for all types
- ParameterSet, ParameterSpace, ParameterSpec
- ParameterView with bind() for P→M
- Transform system for optimization
- Comprehensive unit tests

### Week 3-4: Model Interface
- BaseModel with simulate() enforcement
- Scenario system as patches
- DerivedModel facade with reconciliation
- Compilation to wire functions
- Integration tests

### Week 5-6: Infrastructure Integration
- SimTask contracts
- Direct execution without bridge
- Calibration workflow
- Provenance and caching
- End-to-end tests

## Conclusion

This design achieves:
- **Zero state leakage** through immutability
- **Type safety** at all boundaries
- **Clear separation** between M-space and P-space
- **Ergonomic research API** via DerivedModel
- **Reproducible science** through pure functions
- **Scalable execution** via clean contracts

The immutable parameter system with ParameterView/DerivedModel provides researchers
with intuitive workflows while maintaining architectural purity and preventing
entire classes of bugs.
