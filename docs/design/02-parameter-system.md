# Parameter System Design

## Overview

The Calabaria parameter system enforces immutability and completeness to
prevent an entire class of state bugs common in simulation frameworks. Every
parameter must be explicitly specified, validated, and immutable after
creation.

## Parameters vs Configuration

### Parameters (in M-space)

- **Type**: Numeric values only (float or int)
- **Purpose**: Can be calibration targets
- **Mutability**: Can be fixed or free in ParameterView
- **Transforms**: Support mathematical transforms for optimization
- **Examples**: `transmission_rate=0.3`, `population_size=10000`, `gamma=0.07`

### Configuration (separate from M-space)

- **Type**: Structural choices and categorical settings
- **Purpose**: Never calibration targets (always "fixed" in a sense)
- **Mutability**: Modified only through scenario patches
- **Transforms**: No transforms possible or needed
- **Examples**: `network_type="scale_free"`, `distribution_family="gamma"`, `output_format="parquet"`

This separation ensures the parameter grammar remains pure and mathematical
while still supporting rich model configuration through scenarios.

## State Bugs We Prevent

These are real failure modes encountered in production simulation systems when
parameters are mutable.

### 1. Silent Drift from In-Place Edits

**Buggy (mutable)**:
```python
params = {"beta": 0.3, "gamma": 0.1}
simulate(params)           # run A
params["beta"] *= 0.9      # tweak for a quick test
simulate(params)           # run B "with same inputs" (nope)
```

**Correct (immutable)**:
```python
base = ParameterSet(space, {"beta": 0.3, "gamma": 0.1})
run_a = model.simulate(base, seed=7)

test = base.with_updates(beta=base["beta"] * 0.9)  # new object
run_b = model.simulate(test, seed=7)
```

### 2. Scenario Patches That Leak

**Buggy (mutable)**:
```python
def apply_lockdown(p, cfg):
    p["contact_rate"] = 2.0   # mutates caller's dict
    cfg["mobility"] = "low"    # mutates global cfg
    return p, cfg
```

**Correct (immutable)**:
```python
spec = ScenarioSpec(
    name="lockdown",
    param_patches={"contact_rate": 2.0},
    config_patches={"mobility": "low"},
)
params2, cfg2 = spec.apply(pset, model.base_config)  # returns new values only
```

### 3. Cache Poisoning via Hash-Then-Mutate

**Buggy (mutable)**:
```python
key = hash_params(params)         # computed on current dict
params["beta"] = tune(beta)       # later mutation
cache.put(key, results)           # now key ≠ actual inputs used
```

**Correct (immutable)**:
```python
key = hash_params(pset.values)    # MappingProxyType; can't change
# any change yields a brand-new ParameterSet (and thus a new key)
```

### 4. Late-Bound Closure Over Mutable Dict

**Buggy (mutable)**:
```python
params = {"beta": 0.3}
def compiled(seed):
    return simulate(params, seed)    # closes over mutable dict

fn = compiled
params["beta"] = 0.1                 # retroactively changes fn()
```

**Correct (sealed & immutable)**:
```python
fn = model.compile_entrypoint()      # closes over sealed registries
result = fn(dict(pset.values), seed) # params provided per call
```

### 5. Typos Silently Use Wrong Defaults

**Buggy (defaults + typos)**:
```python
params = {}
params.setdefault("beta", 0.3)      # sets default
user_params["betta"] = 0.5          # typo ignored!
params.update(user_params)           # typo has no effect
simulate(params)                     # runs with beta=0.3, not 0.5
```

**Correct (no defaults, validation)**:
```python
# ParameterSet validates all keys at construction
pset = ParameterSet(space, {"betta": 0.5})  # KeyError: unknown parameter 'betta'
# Typos caught immediately, not silently ignored
```

## Core Types

### ParameterSpace
The complete specification of model parameter space (M-space).

```python
@dataclass(frozen=True)
class ParameterSpace:
    """Complete specification of model parameter space.

    Defines all parameters a model accepts, their types, and bounds.
    The space is immutable - parameters cannot be added or removed after creation.
    """
    specs: List[ParameterSpec]

    def __post_init__(self):
        # Check for duplicate names
        # Create immutable lookup structures
        # Freeze the specs list
```

**Key Properties**:
- Immutable after creation
- No duplicate parameter names
- Provides validation for all parameters
- Defines the complete M-space

### ParameterSpec

Specification for a single numeric parameter.

```python
@dataclass(frozen=True)
class ParameterSpec:
    """Specification for a single numeric parameter."""
    name: str
    min: Scalar
    max: Scalar
    kind: str = "float"  # "float" or "int"
    doc: str = ""

    def validate_value(self, value: Scalar) -> None:
        """Validate a value against this specification."""
        # Type checking (with special handling for integer-like floats)
        # Bounds checking
        # NaN/Inf rejection
```

**Validation Rules**:
- Integer parameters accept integer-like floats (3.0) but reject non-integer
  floats (3.9)
- Explicitly rejects boolean values (even though bool is int subclass in
  Python)
- Rejects NaN and infinite values
- Enforces bounds inclusively

### ParameterSet

An immutable, complete assignment of values in M-space.

```python
@dataclass(frozen=True)
class ParameterSet:
    """Immutable, complete assignment of values in M-space.

    This is THE ONLY way to pass parameters to BaseModel.simulate(),
    enforcing complete specification and preventing partial parameter bugs.
    """
    space: ParameterSpace
    values: Dict[str, Scalar]

    def __post_init__(self):
        # Check completeness - all parameters must be specified
        # Check for unknown parameters (catches typos)
        # Validate each value against its specification
        # Freeze the values dictionary
```

**Key Invariants**:
- Must specify EVERY parameter in the space
- Unknown parameters cause immediate failure (no typos)
- Values are frozen after creation (MappingProxyType)
- Only way to "modify" is creating new instance with `with_updates()`

### ParameterView

A partial valuation of M that pins some parameters to fixed values.

```python
@dataclass(frozen=True)
class ParameterView:
    """Partial valuation inducing calibration subspace.

    Pins some parameters to fixed values; the unpinned parameters
    form the calibration subspace P.
    """
    space: ParameterSpace
    fixed: Dict[str, Scalar]  # Pinned parameters

    @property
    def free(self) -> List[str]:
        """Parameters not in fixed (derived by complement)."""
        return [name for name in self.space.names()
                if name not in self.fixed]

    def bind(self, **free_values) -> ParameterSet:
        """Embedding operation: P → M."""
        complete = {**self.fixed, **free_values}
        return ParameterSet(self.space, complete)
```

**Mathematical Role**:
- Defines the calibration subspace Pᵥ (via free parameters)
- Provides embedding ι: Pᵥ → M (via bind)
- Provides projection π: M → Pᵥ (via project)

## Factory Methods and Convenience

### ParameterSet.new()

Factory method for creating ParameterSet with keyword arguments:

```python
# Instead of:
pset = ParameterSet(space, {"alpha": 0.5, "beta": 0.3})

# Can use:
pset = ParameterSet.new(space, alpha=0.5, beta=0.3)
```

### ParameterView factories
- `ParameterView.all_free(space)` - All parameters free
- `ParameterView.from_fixed(space, fixed_dict)` - Specify fixed, derive free

## Guardrails and Enforcement

### Immutability Enforcement
- `@dataclass(frozen=True)` on all parameter types
- `MappingProxyType` for all dictionary attributes
- No in-place modification methods

### Validation Points

1. **ParameterSpace creation**: Check for duplicates, validate bounds
2. **ParameterSet creation**: Check completeness, validate all values
3. **ParameterView operations**: Validate fixed parameters exist
4. **BaseModel.simulate()**: Type-check for ParameterSet (not dict)

### No Defaults Policy
- Every parameter must be explicitly specified
- Missing parameters fail immediately with clear error
- Typos in parameter names fail immediately
- No `setdefault()` or fallback values

## Property Tests

```python
from hypothesis import given, strategies as st

class TestImmutabilityInvariants:
    """Property tests ensuring our immutability guarantees hold."""

    @given(params=st.dictionaries(st.text(), st.floats()))
    def test_parameter_set_immutable(self, params):
        """ParameterSet values cannot be modified after creation."""
        pset = ParameterSet(space, params)
        original = dict(pset.values)

        # Attempt mutations (should all fail)
        with pytest.raises(TypeError):
            pset.values["new_key"] = 1.0

        assert dict(pset.values) == original

    def test_no_silent_defaults(self):
        """Missing parameters fail loudly, no silent defaults."""
        # Missing parameter fails
        with pytest.raises(ValueError, match="Missing required parameter"):
            ParameterSet(space, {"gamma": 0.1})  # no beta

        # Typo fails
        with pytest.raises(ValueError, match="Unknown parameter"):
            ParameterSet(space, {"betta": 0.3, "gamma": 0.1})
```

## Benefits

### For Researchers

- **No silent bugs**: Typos and missing parameters caught immediately
- **Perfect reproducibility**: Immutable parameters can't change
- **Clear semantics**: Complete vs partial parameters are different types

### For Infrastructure

- **Safe caching**: Immutable keys can't be corrupted
- **Parallel safety**: No shared mutable state
- **Clear contracts**: BaseModel only accepts complete ParameterSet

### For Science
 
- **Audit trail**: Every ParameterSet is a complete record
- **No hidden defaults**: Everything is explicit
- **Composable**: Operations return new objects, enabling functional composition
