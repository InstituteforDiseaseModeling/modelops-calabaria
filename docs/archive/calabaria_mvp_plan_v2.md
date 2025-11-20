# Calabaria MVP - Pure Functional Science, Clean Contracts
**Part 1: BaseModel Simulation Interface**
**Engineering Plan & Design Document v2**

> **tl;dr**: Pure functional core with immutable types throughout. ParameterSet is the
> only gateway to simulation. Scenarios are patches, not transforms. Clean separation
> between M-space (complete model parameter space) and P-space (free parameter subspace
> for exploration). Zero state leakage via frozen dataclasses.

---

## Executive Summary

### Scope

This document covers **Part 1 of the Calabaria framework: the BaseModel simulation interface**.
Calabaria consists of three core components:

1. **BaseModel interface** (this document): Pure functional simulation with
   immutable parameters
2. **Targets**: Data alignment and loss function specifications (future
   document)
3. **Calibration adapters**: Integration with optimization/sampling algorithms
   (future document)

While this document includes calibration examples to demonstrate how the
BaseModel interface supports these workflows, the actual calibration
implementations are beyond this MVP's scope.

### Vision

Build Calabaria's BaseModel as a **pure functional simulation framework** with
immutable data structures throughout. This guarantees reproducibility,
eliminates parameter state bugs, enables content-addressed caching, and scales
from laptop to cloud—while keeping the scientist experience delightful.

### Key Terms

- **Parameters**: any model input that *could be or is a knob in calibration*.
- **Config**: all model input that *could not be* a knob in calibration (e.g.
  runtime duration, simulated region, number of agents, etc.).
- **M-space**: The complete model parameter space containing all numeric
  parameters the model requires to run
- **P-space**: The free parameter subspace - the subset of parameters
  researchers vary during exploration/calibration
- **Parameters**: Numeric values (float/int) that define model behavior and can
  be calibration targets
- **Configuration**: Structural choices and categorical settings that control
  model behavior but are never calibrated
- **ParameterSet**: An immutable, complete assignment of values for every
  parameter in M-space
- **ParameterView**: A partial valuation of M that pins some parameters to
  fixed values; the unpinned parameters form the calibration subspace P
- **ModelVariant**: A facade over BaseModel that represents a model variant (in
  config or parameters) provides ergonomic P-space operations while maintaining
  immutability
- **Scenarios**: Named patches that modify parameter values or configuration
  settings for specific model variants
- **Transforms**: Mathematical coordinate changes applied to free parameters
  for optimization (e.g., logit for bounded params)

### Core Engineering Principles

1. **Immutability prevents state bugs**: ALL parameter types use
   @dataclass(frozen=True)
2. **ParameterSet is the ONLY gateway**: BaseModel.simulate() accepts only
   complete ParameterSet
3. **Scenarios are patches, not transforms**: Dictionary updates that can fix
   but not transform
4. **Transforms are for optimization only**: Change coordinates, not
   dimensionality
5. **Reparameterizations restructure parameter space**: Not in MVP, future
   extension
6. **ParameterView.bind() is the embedding ι: Pᵥ→M**: Embeds free parameter
   values into complete ParameterSets using fixed values
7. **Pure functional core**: All computations depend only on explicit inputs
8. **Content-addressed provenance**: Every computation has a deterministic hash
9. **No silent defaults**: All parameters must be explicit; missing parameters
   fail loudly at validation time, preventing typos and hidden defaults

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

### Parameters vs Configuration

**Parameters** (in M-space):
- Numeric values only (float or int)
- Can be calibration targets
- Can be fixed or free in ParameterView
- Support mathematical transforms for optimization
- Examples: `transmission_rate=0.3`, `population_size=10000`, `gamma=0.07`

**Configuration** (separate from M-space):
- Structural choices and categorical settings
- Never calibration targets (always "fixed" in a sense)
- Modified only through scenario patches
- No transforms possible or needed
- Examples: `network_type="scale_free"`, `distribution_family="gamma"`,
  `output_format="parquet"`

This separation ensures the parameter grammar remains pure and mathematical
while still supporting rich model configuration through scenarios.

### What We're Building (BaseModel Interface)

- Immutable parameter system with ParameterView defining which parameters are
  fixed vs free
- ModelVariant facade providing clean API for working with free parameters only
- BaseModel with strict ParameterSet-only interface
- Scenarios as patches (not parameter mutations, patches of configs or fixing
  of parameters)
- Clean SimTask contracts for ModelOps integration
- Content-addressed caching via provenance hashes
- Interface design that naturally supports calibration workflows (demonstrated
  via examples)

### What We're NOT Building (Yet)

- Reparameterizations
- Constraint systems or units
- Multi-language support
- Target specifications and loss functions (Part 2)
- Calibration algorithm implementations (Part 3)

---

## Motivation: State Bugs We Prevent

These are real failure modes we've encountered in production simulation systems
when parameters and configuration are mutable. Each example shows the buggy
pattern and our immutable fix using `ParameterSet`, `ParameterView.bind()`, and
`ScenarioSpec.apply()`.

### Parameter Mutation Bugs

#### 1. Silent drift from in-place edits between runs

**Symptom**: A notebook cell tweaks a shared params dict; subsequent runs
produce different results with "the same" inputs.

**Buggy (mutable)**:
```python
params = {"beta": 0.3, "gamma": 0.1}
simulate(params)           # run A
params["beta"] *= 0.9      # tweak for a quick test
simulate(params)           # run B "with same inputs" (nope)
```

**Correct (immutable)**:
```python
base = ParameterSet({"beta": 0.3, "gamma": 0.1})
run_a = model.simulate(base, seed=7)

test = base.with_updates(beta=base["beta"] * 0.9)  # new object
run_b = model.simulate(test, seed=7)
```

#### 2. Scenario patches that leak across experiments

**Symptom**: Applying a scenario mutates the "base" params/config, so later
runs (even baseline) inherit the patch.

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

#### 3. Cache poisoning via "hash-then-mutate"

**Symptom**: A caching layer computes a hash of params, then some code mutates
the dict before/after execution. Cache keys no longer reflect reality.

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

### Shared State Bugs

#### 4. Cross-replicate contamination

**Symptom**: A replicate modifies arrays in place (e.g.,
`params["init_state"]`) and the next replicate starts from the mutated state.

**Buggy (mutable)**:
```python
def run_sim(state, seed):
    rng = np.random.default_rng(seed)
    state["I0"] += rng.integers(0, 5)  # mutates shared state
    ...
```

**Correct (immutable boundary)**:
```python
def build_state(pset, cfg):
    return {
        "I0": int(pset["initial_infected"]),
        # immutable primitives or fresh arrays
    }

def run_sim(state, seed):
    rng = np.random.default_rng(seed)
    I0 = state["I0"] + int(rng.integers(0, 5))  # local var, not mutating input
    ...
```

#### 5. Late-bound closure over mutable dict

**Symptom**: You "compile" a function that closes over a mutable params dict;
later edits retroactively change compiled behavior.

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
fn = model.compile_entrypoint()  # closes over sealed registries
result = fn(dict(pset.values), seed)        # params provided per call
```

#### 6. Transform writes back "helpful" derived fields

**Symptom**: An optimizer transform stores `log_beta` back into the same dict;
downstream code starts reading the wrong field or double-transforms.

**Buggy (mutable)**:
```python
def logitize(p):
    p["beta"] = logit(p["beta"])   # overwrites natural value
```

**Correct (separated coordinates)**:
```python
# Using transforms with VariantSpec and ModelVariant
spec = VariantSpec(
    view=ParameterView.from_fixed(model.space, {"population": 10000}),
    transforms={"beta": AffineSqueezedLogit()}
)
variant = ModelVariant(model, spec)
z = variant.to_transformed(beta=0.3)     # returns NEW coords in transformed space
x = variant.from_transformed(**z)         # back to physical space
```

### Concurrency Bugs

#### 7. Thread/process races on shared dicts

**Symptom**: Parallel runs mutate a shared params dict or config, producing nondeterministic results.

**Buggy (mutable)**:
```python
shared = {"beta": 0.3}
def worker():
    shared["beta"] += np.random.rand() * 1e-6   # data race

pool.map(worker, range(8))
```

**Correct (share-nothing)**:
```python
def worker(pset: ParameterSet):
    return model.simulate(pset, seed=next_seed())

pool.map(worker, per_task_psets)  # each task gets its own immutable pset
```

### Default and Validation Bugs

#### 8. Hidden defaults sneak into provenance

**Symptom**: `build_state` silently fills missing keys and writes them back to
params for "convenience"; later runs start from those mutated defaults.

**Buggy (mutable)**:
```python
def build_state(params, cfg):
    params.setdefault("contact_rate", 4.0)  # silently mutates
    ...
```

**Correct (validation + explicit construction)**:
```python
def build_state(pset: ParameterSet, cfg):
    # Validation done when creating ParameterSet; all keys present.
    contact = float(pset["contact_rate"])
    ...
```

#### 9. Typos silently use wrong defaults

**Symptom**: User typo in parameter name combined with defaults means the wrong
model runs silently.

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
pset = ParameterSet({"betta": 0.5})  # KeyError: unknown parameter 'betta'
# Typos caught immediately, not silently ignored
```

### Guardrails We Add (and Why They Work)

- **`@dataclass(frozen=True)` + `MappingProxyType`**: Prevents any in-place
  param/config edits
- **`ParameterSet` as the only gateway**: You can't call `simulate()` with a
  dict, so all inputs are complete & immutable
- **`ParameterView.bind()`**: Forces explicit, exact P→M assembly;
  wrong/missing keys error out
- **`ScenarioSpec.apply()`**: Returns new params/config; patches can't leak
- **Seal-on-first-use**: Output/extractor/scenario registries freeze when first
  compiled or simulated; closures can't later change behavior
- **Canonicalization for provenance**: Keys are sorted before hashing; order
  can't wiggle IDs
- **No defaults allowed**: Every parameter must be explicitly specified;
  missing parameters or typos fail immediately

### Property Tests for Invariants

```python
from hypothesis import given, strategies as st
import pytest

class TestImmutabilityInvariants:
    """Property tests ensuring our immutability guarantees hold."""

    @given(params=st.dictionaries(st.text(), st.floats()))
    def test_parameter_set_immutable(self, params):
        """ParameterSet values cannot be modified after creation."""
        pset = ParameterSet(params)
        original = dict(pset.values)

        # Attempt mutations (should all fail or have no effect)
        with pytest.raises(TypeError):
            pset.values["new_key"] = 1.0

        assert dict(pset.values) == original

    @given(base=st.dictionaries(st.text(), st.floats()),
           updates=st.dictionaries(st.text(), st.floats()))
    def test_with_updates_creates_new(self, base, updates):
        """with_updates() creates a new object, doesn't mutate original."""
        pset1 = ParameterSet(base)
        pset2 = pset1.with_updates(**updates)

        # Original unchanged
        assert dict(pset1.values) == base
        # New object created
        assert pset1 is not pset2
        # Updates applied to new only
        for k, v in updates.items():
            assert pset2[k] == v

    def test_no_silent_defaults(self):
        """Missing parameters fail loudly, no silent defaults."""
        space = ParameterSpace(...)  # with beta, gamma required

        # Missing parameter fails
        with pytest.raises(ValueError, match="Missing required parameter: beta"):
            ParameterSet({"gamma": 0.1})  # no beta

        # Typo fails
        with pytest.raises(ValueError, match="Unknown parameter: betta"):
            ParameterSet({"betta": 0.3, "gamma": 0.1})
```

---

## Grammar of Parameters

### Intent

Like the Grammar of Graphics, this "Grammar of Parameters" provides a small set
of primitives, operators, and laws that compose into all common parameter
workflows—exploration, scenarios, calibration—while staying pure, immutable,
and predictable.

The design patterns below directly prevent the state bugs documented in the previous section.
Every primitive and operator enforces immutability, ensuring parameters can never be
accidentally mutated, leaked between experiments, or silently corrupted.

### Notation Guide

- `M` = Complete model parameter space (what BaseModel requires)
- `Pᵥ` = The calibration subspace induced by free parameters
- `m ∈ M` = A point in M (a complete ParameterSet)
- `p ∈ Pᵥ` = A point in the calibration subspace
- `view` = A partial valuation (fixed mapping); free derived by complement
- `domain(view)` = The fixed parameter names
- `free(view)` = names(M) \ domain(view) (the free parameters)
- `π` = Projection from M to Pᵥ (extracts free coordinates)
- `ι` = Embedding from Pᵥ to M (bind operation)
- `∘` = Function composition, read right-to-left: `f ∘ g ∘ h` means `lambda x: f(g(h(x)))`

### 1. Primitives (Objects)

**ParameterSpace (M)**
- The model's complete set of named numeric parameters with domains
- Intuition: "The full list of knobs the model understands"
- Names: `Names(M)` - finite set of parameter names
- Domain per name: `Domain(name)` - interval or integer range (no categorical)
- Cartesian space: M is the product of all numeric domains
- Note: Categorical choices belong in configuration, not parameters

**ParameterSet (m ∈ M)**
- A complete, immutable assignment for every name in M
- Intuition: "One concrete knob setting—what the model actually runs with"
- Enforced at BaseModel.simulate() boundary, which will only take `m` which
  specifies a point in `M` (all parameters must have values), since that's
  what's needed to run a simulation.

**ParameterView — Partial Valuation Inducing Calibration Subspace**
- A partial valuation: mapping from subset of names(M) to values
- Essential data: `fixed`: {name → value} mapping
- Derived data: `free` = names(M) \ domain(fixed) (the complement)
- The free names induce the calibration parameter subspace:
  Pᵥ = ∏_{n ∈ free} Domain(n)
- Provides two canonical operations:
  - **Projection π**: M → Pᵥ - extracts free coordinates
  - **Embedding ι (bind)**: Pᵥ → M - combines free values with fixed mapping
- Formally: view = fixed_mapping, which implicitly defines free by complement
- Intuition: "Pin some parameters, let others vary for calibration"

**Transform Set (T)**
- Per-free-parameter bijections for optimization coordinates
- Intuition: "Change units for the optimizer without changing the model"
- For each free name: T[name]: Domain(name) ↔ Coord(name)
- MVP: Independent transforms only, no coupling

**ScenarioPatch**
- Two partial updates applied at execution:
  - `param_patches`: {name → value} - fix/override numeric parameters
  - `config_patches`: {key → value} - modify categorical/structural configuration
- Intuition: "Preset numeric knobs and categorical choices to embody a scenario"
- Note: Config patches handle categorical values that can't be calibration targets

**ModelVariant**
- An immutable research-facing object: (model, view, optional transforms, optional scenario)
- Intuition: "A base model specialized for the analysis you want"

### Why ModelVariant.simulate(seed, **free) Not ParameterSet

**The Design Choice**: ModelVariant takes free parameters as kwargs, not a ParameterSet.

**Core Invariant We Preserve**:
- `ParameterSet` (the class) means "complete M-space point" - ONLY
- BaseModel.simulate accepts ONLY complete ParameterSets
- This invariant prevents parameter state bugs

**Why Not Match BaseModel's Signature?**

If ModelVariant also took a "ParameterSet" (but for P-space), we'd have:
- Two kinds of ParameterSet (complete vs partial)
- Conceptual leakage between M-space and P-space
- Runtime confusion about which ParameterSet goes where
- We'd blur the boundary and re-introduce the very ambiguity we're trying to remove

**What Goes Wrong with P-space ParameterSets:**
1. **Conceptual leakage**: People try to reuse objects between P and M calls ("it's a ParameterSet, right?")
2. **Runtime bugs**: A "P-ParameterSet" can't be validated against full schema without carrying view
3. **Transform confusion**: ModelVariant may carry transforms - are they applied? To which params?
4. **Cache/provenance muddle**: M-space has deterministic identity; P-space points are view-dependent

**Why simulate(seed, **free) Is Right:**
1. **Ergonomics for researchers**: `derived.simulate(seed=1, beta=0.3, gamma=0.1)` reads clean
2. **Clear semantics**: "I'm giving only the free parameters" - anything extra/missing is a tight error
3. **Preserves the barrier**: ModelVariant internally binds via ι to produce the ParameterSet
4. **Type safety at the right level**: Each API matches its abstraction
   - BaseModel (M-space): Complete specification required
   - ModelVariant (P-space): Only calibration parameters needed

**The Bind Happens Internally:**
```python
def simulate(self, seed: int, **free_values):
    params_M = self.view.bind(**free_values)  # P → M via embedding
    # ... passes complete ParameterSet to BaseModel
```

This gives researchers ergonomic P-space operations while maintaining the
architectural constraint that ParameterSet = complete M-space point only.

### 2. Operators (Morphisms)

**Bind (Embedding ι: P → M)**

```
bind(view, free_values) → ParameterSet
```

- The embedding operation ιᵥ: Pᵥ → M
- Takes values in calibration space P, merges with partial valuation, returns point in M
- bind(view, p) = ParameterSet(view.fixed ∪ {n → p[n] for n in free(view)})
- Defined iff keys(p) == free(view)
- Merges the partial valuation with free parameter values
- This is THE bridge from calibration subspace back to full M-space
- Satisfies: π ∘ ι = id_P (project after bind recovers free values)
- Intuition: "Complete the partial valuation with calibration parameters"

**Project (Projection π: M → P)** (optional helper)

```
project(view, params) → free_values
```

- The projection operation πᵥ: M → Pᵥ
- Extracts only the free coordinates from a complete ParameterSet
- Returns ordered dict matching view.free ordering
- Satisfies: ι ∘ π = clamp_fixed (bind after project clamps to fixed)
- Intuition: "Extract the free dimensions, forgetting fixed values"

**ApplyPatch**

```
apply_patch(patch, m, cfg) → (m', cfg')
```

- Applies ScenarioSpec patches to parameters and config
- Parameter patches: Direct value overrides in M-space (no transforms/reparams)
- Config patches: Key-value updates to configuration dictionary
- Last-write-wins (LWW) for overlapping keys in stacked scenarios
- Returns updated (m', cfg') without modifying originals
- Intuition: "Bake scenario overrides into params/config"

**Transform/Inverse**

```
to_transformed(T, natural_P) → transformed_Z
from_transformed(T, transformed_Z) → natural_P
```

- Applied only to free parameters
- No dimensionality change
- Intuition: "Switch to optimizer space and back"
- Guardrails: composition of `to_transformed` and `from_transformed` should
  equal the identity (up to floating point errors). Caution is needed in bounds
  handling, e.g. from log space and back

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
- `bind(view)` = ι: Pᵥ → M (embedding operation)
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

> **Note**: This example demonstrates how the BaseModel interface naturally supports
> calibration workflows. The actual calibration algorithms (optimization, sampling)
> are not part of this MVP - they will be covered in Part 3 of Calabaria.

```python
# M has 6 parameters
view = ParameterView.from_fixed(
    M, population=10000, contact_rate=4.0,
    gamma=0.07, initial_infected=10
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

### Grammar in Action: Categorical Choices Example

```python
# WRONG: Categorical as parameter (can't be optimized!)
params = ParameterSpace(
    transmission_mode=Categorical(["density", "frequency"]),  # ❌ Not supported
    transmission_rate=Float(0.0, 1.0)
)

# RIGHT: Categorical in config, numeric in parameters
params = ParameterSpace(
    transmission_rate=Float(0.0, 1.0)  # ✓ Numeric, can be calibrated
)

config = {
    "transmission_mode": "frequency",  # Categorical choice
    "network_type": "scale_free",      # Structural setting
    "output_format": "parquet"         # Implementation detail
}

# Scenarios can patch both
lockdown_scenario = ScenarioSpec(
    param_patches={"transmission_rate": 0.1},  # Reduce numeric parameter
    config_patches={"transmission_mode": "density"}  # Switch categorical
)
```

This separation keeps the parameter space purely numeric (required for optimization) while still supporting rich categorical choices through configuration.

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
        Use ParameterView.bind() or ModelVariant for ergonomic P-space operations.
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
from typing import Union, Any, Tuple, Dict, Optional, Callable, Mapping, Literal
from dataclasses import dataclass, field
from types import MappingProxyType

Scalar = Union[float, int]  # Numeric parameters only
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
    """Complete, validated, immutable parameter assignment for space M.

    THE ONLY TYPE accepted by BaseModel.simulate().
    Validates completeness, types, and bounds at construction.
    Immutability prevents parameter mutation bugs.
    """
    space: ParameterSpace
    values: Dict[str, Scalar]

    def __post_init__(self):
        """Validate and freeze parameter values."""
        meta = self.space.by_name()
        validated = {}

        # Validate each parameter
        for k, v in self.values.items():
            if k not in meta:
                raise ValueError(f"Unknown parameter: {k}")

            spec = meta[k]

            # Type validation and conversion
            if spec.kind == "int":
                if not isinstance(v, (int, float)):  # Allow float for int conversion
                    raise TypeError(f"{k} must be numeric, got {type(v).__name__}")
                val = int(v)
            else:  # "real"
                if not isinstance(v, (int, float)):
                    raise TypeError(f"{k} must be numeric, got {type(v).__name__}")
                val = float(v)

            # Bounds validation
            if not (spec.lower <= val <= spec.upper):
                raise ValueError(
                    f"{k}={val} outside bounds [{spec.lower}, {spec.upper}]"
                )

            validated[k] = val

        # Completeness check
        expected_names = set(self.space.names())
        provided_names = set(validated.keys())
        if provided_names != expected_names:
            missing = sorted(expected_names - provided_names)
            extra = sorted(provided_names - expected_names)
            msgs = []
            if missing:
                msgs.append(f"missing={missing}")
            if extra:
                msgs.append(f"extra={extra}")
            raise ValueError(f"ParameterSet incomplete: {', '.join(msgs)}")

        # Freeze validated values
        object.__setattr__(self, 'values', MappingProxyType(validated))

    def __getitem__(self, k: str) -> Scalar:
        return self.values[k]

    def with_updates(self, **updates: Scalar) -> "ParameterSet":
        """Return NEW ParameterSet with updates (immutable pattern)."""
        new_values = dict(self.values)
        new_values.update(updates)
        return ParameterSet(self.space, new_values)
```

#### ParameterView - Immutable Fix/Free Partition

```python
@dataclass(frozen=True)
class ParameterView:
    """ParameterView: A partial valuation of parameter space.

    The essential data is the 'fixed' mapping (partial valuation).
    The 'free' is derived as a property: names(M) \ domain(fixed).
    Together they induce the calibration subspace P.

    Single source of truth for what's fixed vs free.
    All operations return NEW views (immutability).
    The bind() method is THE embedding from P-space to M-space.
    """
    space: ParameterSpace
    fixed: Dict[str, Scalar] = field(default_factory=dict)

    def __post_init__(self):
        # Make fixed mapping immutable
        object.__setattr__(self, 'fixed', MappingProxyType(dict(self.fixed)))

        # Validate fixed keys exist in space
        all_names = set(self.space.names())
        fixed_names = set(self.fixed.keys())
        if not fixed_names.issubset(all_names):
            unknown = sorted(fixed_names - all_names)
            raise ValueError(f"Unknown fixed parameters: {unknown}")

        # Validate fixed values are in bounds (but allow type conversion at bind time)
        meta = self.space.by_name()
        for k, v in self.fixed.items():
            spec = meta[k]
            if not isinstance(v, (int, float)):
                raise TypeError(f"Fixed value for {k} must be numeric, got {type(v).__name__}")

    @property
    def free(self) -> Tuple[str, ...]:
        """Derive free parameters as complement of fixed."""
        return tuple(n for n in self.space.names() if n not in self.fixed)

    @classmethod
    def all_free(cls, space: ParameterSpace) -> "ParameterView":
        """Create view with all parameters free."""
        return cls(space=space, fixed={})

    @classmethod
    def from_fixed(cls, space: ParameterSpace, **fixed: Scalar) -> "ParameterView":
        """Create view with specified parameters fixed."""
        return cls(space=space, fixed=fixed)

    def fix(self, **kv: Scalar) -> "ParameterView":
        """Return NEW view with additional parameters fixed (immutable)."""
        merged = {**self.fixed, **kv}
        return ParameterView(self.space, merged)

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
        return ParameterSet(self.space, complete)
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

# Note: TransformedView removed - use transforms in VariantSpec instead
# VariantSpec provides transform functionality via:
#   - with_transforms(**name_to_transform) to add transforms
#   - to_transformed(free_values) to convert to optimizer space
#   - from_transformed(coords) to convert back to physical space
```

### 3. Model Decorators & Discovery

#### Decorator-Based Registration

Models use decorators to mark methods for discovery. These are just markers - actual registration happens during instance construction before sealing.

```python
from calabaria.core.constants import SEED_COL
import inspect

def model_output(name: str, metadata: Optional[dict] = None):
    """Mark a method as a data extractor.

    Decorators are markers only; registration happens on instance
    construction before seal. Extractors must be pure functions
    of (raw, seed) with no global state dependencies.

    IMPORTANT: Framework adds SEED_COL column automatically.
    Extractors must NOT add this column.

    Parameters
    ----------
    name : str
        Output name for this extractor
    metadata : dict, optional
        Optional metadata for the output
    """
    def decorator(func):
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

    Parameters
    ----------
    name : str
        Scenario name for registration
    """
    def decorator(fn):
        def wrapper(self):
            result = fn(self)
            if not isinstance(result, ScenarioSpec):
                raise TypeError(
                    f"@model_scenario '{name}' must return ScenarioSpec, "
                    f"got {type(result).__name__}"
                )
            return result

        wrapper._is_model_scenario = True
        wrapper._scenario_name = name
        wrapper.__name__ = fn.__name__
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return decorator
```

### 4. Model Interface & Scenarios

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

    def apply(self, params: ParameterSet, config: Mapping,
              space: ParameterSpace) -> Tuple[ParameterSet, Mapping]:
        """Apply patches with validation.

        Validates parameter names exist and values are in bounds.
        Patches are dictionary updates - last write wins (or strict checking).
        Can FIX parameters but cannot TRANSFORM them.
        """
        # Validate param_patches
        if self.param_patches:
            meta = space.by_name()

            for param_name, value in self.param_patches.items():
                # Check parameter exists
                if param_name not in meta:
                    raise ValueError(
                        f"Scenario '{self.name}' references unknown parameter: {param_name}"
                    )

                # Check bounds
                spec = meta[param_name]
                if spec.kind == "int":
                    val = int(value)
                    if not (spec.lower <= val <= spec.upper):
                        raise ValueError(
                            f"Scenario '{self.name}' sets {param_name}={val} "
                            f"outside bounds [{spec.lower}, {spec.upper}]"
                        )
                else:  # real
                    val = float(value)
                    if not (spec.lower <= val <= spec.upper):
                        raise ValueError(
                            f"Scenario '{self.name}' sets {param_name}={val} "
                            f"outside bounds [{spec.lower}, {spec.upper}]"
                        )

        # Note: Conflict checking happens at scenario composition time in ModelVariant.scenarios()
        # not here. This allows proper detection of conflicts between multiple scenarios.

        # Apply parameter patches (fixing, not transforming)
        if self.param_patches:
            new_params = params.with_updates(**self.param_patches)
        else:
            new_params = params

        # Apply config patches
        if self.config_patches:
            new_config = MappingProxyType({**config, **self.config_patches})
        else:
            new_config = config

        return new_params, new_config
```

#### Conceptual Distinction: Scenarios vs VariantSpec vs ModelVariant

**Mental Model in One Line**
- **Scenario**: "Lockdown sets contact_rate=2 and switches vaccination on." (data patches)
- **VariantSpec**: "Fix population=10k, vary (beta, gamma) with logit transforms, apply lockdown scenario." (complete configuration)
- **ModelVariant**: "Execute this model with this variant spec." (simple executor)

**What Each One Is**

**ScenarioSpec = Data Patches**
- A named, immutable patch: `{param_patches, config_patches, policy}`
- Fixes numeric params and tweaks config
- No transforms, no reparam, no logic, no RNG
- Think "preset conditions" - declarative and reusable

**VariantSpec = Complete Configuration**
- Defines a model variant with all settings:
  `(fixed/free split + scenario_list + transforms [+ reparam?])`
- Immutable configuration object
- Can be serialized to YAML/JSON
- Contains the complete identity of a variant

**ModelVariant = Simple Executor**
- Just pairs BaseModel with VariantSpec
- Provides execution methods: `simulate()`, `to_yaml_*()`, `bounds()`
- No configuration methods - all config lives in VariantSpec
- Thin convenience wrapper for execution

They operate at different layers: Scenarios are data patches; VariantSpec is complete configuration; ModelVariant just executes.

**Why Both? Separation of Concerns**

Having both is good - it keeps concerns separate:
- **Scenarios** stay declarative and reusable across variants and the cloud wire
- **Variants** capture the current research choice (which scenarios, which fixed values, transforms, etc.) and provide ergonomics + provenance identity

Where teams get into trouble is introducing a third thing (e.g., a "ScenarioBuilder" or letting scenarios do transforms). Don't do that.

**Guardrails for Clean Separation**

✅ **Do:**
- Keep ScenarioSpec tiny and pure (patches only)
- Keep all composition semantics (order, LWW/strict, allow_overlap) inside `VariantSpec`
- On variant build, reconcile param patches → move those params from free→fixed
- Include scenario_stack (names) in provenance/variant keys
- Let YAML carry both scenario_branches and any config_overrides so cloud runs match local

❌ **Don't:**
- Put transforms/reparameterizations inside scenarios
- Add a separate "scenario builder"
- Apply patches in multiple places with divergent rules (centralize in ModelVariant)
- Introduce a third abstraction layer

This separation ensures scenarios remain portable data while variants provide the ergonomic API for research workflows.

#### BaseModel - Pure M-Space Interface

```python
from abc import ABC, abstractmethod
import polars as pl

class BaseModel(ABC):
    """Pure functional model with sealed registries.

    Uses decorators as markers; registration happens on instance
    construction before seal. If 'baseline' is defined via decorator,
    conflict_policy='lww' causes it to replace default; 'strict' raises.

    Invariants enforced:
    - simulate() ONLY accepts ParameterSet (complete M)
    - All state is immutable (frozen dataclasses, immutable collections)
    - Scenarios are patches, not transforms
    - Seal-on-first-use prevents configuration mutations
    - No hidden defaults - complete specification required
    """

    DEFAULT_SCENARIO: str = "baseline"  # Subclasses can override

    def __init_subclass__(cls, **kwargs):
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
                    # Reject static/class methods
                    if isinstance(attr, (staticmethod, classmethod)):
                        output_name = attr._output_name
                        raise TypeError(
                            f"@model_output '{output_name}' must be an instance method, "
                            f"not static/classmethod"
                        )
                    cls._discovered_outputs[attr._output_name] = name

                # Check for scenario decorators
                if getattr(attr, "_is_model_scenario", False):
                    if isinstance(attr, (staticmethod, classmethod)):
                        scenario_name = attr._scenario_name
                        raise TypeError(
                            f"@model_scenario '{scenario_name}' must be an instance method"
                        )
                    cls._discovered_scenarios[attr._scenario_name] = name

    def __init__(self, space: ParameterSpace, base_config: Mapping[str, Any] = None):
        self.space = space  # Immutable ParameterSpace
        self.base_config = MappingProxyType(base_config or {})

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
        """
        # Register outputs in sorted order
        for output_name in sorted(self._discovered_outputs):
            method_name = self._discovered_outputs[output_name]
            method = getattr(self, method_name)

            # Validate it's a bound instance method
            if not inspect.ismethod(method):
                raise TypeError(f"Output '{output_name}' did not bind to instance")

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
        """Hook for subclasses to add scenarios before sealing"""
        pass

    def _seal(self) -> None:
        """Freeze registries on first use."""
        if not getattr(self, "_sealed", False):
            self._scenarios = MappingProxyType(dict(self._scenarios))
            self._outputs = MappingProxyType(dict(self._outputs))
            self._sealed = True

    def _validate(self, params: ParameterSet) -> None:
        """Validate ParameterSet completeness."""
        if not isinstance(params, ParameterSet):
            raise TypeError(
                f"simulate() requires ParameterSet, got {type(params).__name__}"
            )

        # Check completeness
        required = set(self.space.names())
        provided = set(params.values.keys())

        if provided != required:
            missing = required - provided
            extra = provided - required
            msgs = []
            if missing:
                msgs.append(f"Missing: {missing}")
            if extra:
                msgs.append(f"Extra: {extra}")
            raise ValueError(f"ParameterSet incomplete. {'; '.join(msgs)}")

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
        self._validate(params)

        if scenario not in self._scenarios:
            raise ValueError(f"Unknown scenario: {scenario}. Available: {list(self._scenarios.keys())}")

        spec = self._scenarios[scenario]

        # Apply scenario patches (can FIX but not TRANSFORM)
        params_patched, config_patched = spec.apply(params, self.base_config, self.space)

        # Run simulation pipeline
        state = self.build_sim(params_patched, config_patched)
        raw = self.run_sim(state, seed)

        # Extract outputs (includes SEED_COL addition)
        return self.extract_outputs(raw, seed)

    @classmethod
    def compile_entrypoint(cls, alias: Optional[str] = None) -> EntryRecord:
        """Compile model to cloud entrypoint with single canonical wire.

        The wire function is M-space only, with scenario stack as parameter.
        Returns EntryRecord registered globally for cloud lookup.
        """
        model_hash = hash_model_code_and_registry(cls)
        entry_id = compute_entry_id(cls)

        # Create temporary instance for introspection
        temp_model = cls()
        param_space = temp_model.space

        def run_wire(*,
                    params_M: Dict[str, Scalar],
                    seed: int,
                    scenario_stack: Union[List[str], Tuple[str, ...]] = ("baseline",),
                    outputs: Optional[List[str]] = None) -> WireResponse:
            """Single canonical M-space wire for cloud execution.

            Parameters:
            - params_M: Complete M-space parameters
            - seed: Random seed
            - scenario_stack: Ordered list of scenarios to compose
            - outputs: Optional subset of outputs

            Returns WireResponse with IPC outputs and provenance.
            """
            # Create model instance
            model = cls()

            # Normalize stack
            stack = tuple(scenario_stack)

            # Validate all scenarios exist
            for nm in stack:
                if nm not in model._scenarios:
                    available = sorted(model._scenarios.keys())
                    raise ValueError(
                        f"Scenario '{nm}' not found. Available: {available}"
                    )

            # Convert to ParameterSet
            pset = ParameterSet(params_M)

            # Compose scenario patches in order (LWW)
            config = dict(model.base_config)
            for nm in stack:
                spec = model._scenarios[nm]
                pset, config = spec.apply(pset, MappingProxyType(config), model.space)

            # Run simulation
            state = model.build_sim(pset, config)
            raw = model.run_sim(state, seed)
            outs = model.extract_outputs(raw, seed)

            # Filter outputs if requested
            if outputs:
                unknown = set(outputs) - set(outs.keys())
                if unknown:
                    raise ValueError(
                        f"Requested outputs not found: {sorted(unknown)}; "
                        f"available: {sorted(outs.keys())}"
                    )
                outs = {k: v for k, v in outs.items() if k in outputs}

            # Serialize to IPC
            serialized = serialize_ipc(outs)

            # Compute provenance with stack
            scenario_hashes = [scenario_spec_hash(model._scenarios[nm]) for nm in stack]
            sim_root, task_id = provenance_ids(
                model_hash=model_hash,
                scenario_stack=scenario_hashes,
                params_M=params_M,
                seed=seed,
                outputs=tuple(sorted(outs.keys()))
            )

            # Generate variant key for humans
            variant = variant_key(
                class_name=cls.__name__,
                stack=stack,
                fixed={}  # Wire doesn't know fixed/free split
            )

            return WireResponse(
                outputs=serialized,
                provenance={
                    "entry_id": entry_id,
                    "model_hash": model_hash,
                    "scenario_stack": list(stack),
                    "sim_root": sim_root,
                    "task_id": task_id,
                    "variant_key": variant,
                    "outputs": sorted(outs.keys())
                }
            )

        # Build entry record
        record = EntryRecord(
            id=entry_id,
            model_hash=model_hash,
            wire=run_wire,
            scenarios=sorted(temp_model._scenarios.keys()),
            outputs=sorted(temp_model._outputs.keys()),
            space=param_space,
            alias=alias
        )

        # Register globally
        ENTRY_REGISTRY[entry_id] = record

        return record

    def extract_outputs(self, raw: RawSimOutput, seed: int) -> Dict[str, pl.DataFrame]:
        """Extract outputs from raw simulation results.

        Helper method so ModelVariant can reuse extraction logic.
        Automatically adds SEED_COL to all outputs for replicate tracking.

        IMPORTANT: Output extractors must NOT add SEED_COL themselves.
        The framework handles this automatically to ensure consistency.
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

            # Add seed column for replicate tracking
            df_with_seed = df.with_columns(pl.lit(seed).alias(SEED_COL))
            results[name] = df_with_seed

        return results

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

```

#### ModelVariant - Simple Executor for Model + VariantSpec

```python
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any, Callable
import itertools

@dataclass(frozen=True)
class ModelVariant:
    """Simple pairing of a BaseModel with a VariantSpec for execution.

    This is NOT a fluent builder - all configuration lives in VariantSpec.
    ModelVariant just provides convenient execution methods for the paired
    model and spec.

    Key responsibilities:
    - Execute simulation in P-space (free parameters only)
    - Export to YAML for runs/sweeps
    - Get bounds for free parameters
    - Apply transforms to/from optimization space

    See "Conceptual Distinction: Scenarios vs ModelVariants" for architectural rationale.
    """
    model: BaseModel
    spec: VariantSpec

    def __post_init__(self):
        """Validate spec is compatible with model."""
        # Ensure spec's space matches model's space
        if self.spec.view.space != self.model.space:
            raise ValueError(
                "VariantSpec's parameter space doesn't match model's space"
            )

    def simulate(self, *, seed: int, **free_vals: Scalar) -> Dict[str, pl.DataFrame]:
        """Run simulation with free parameters only.

        This is the main execution method. Converts P→M internally using the
        VariantSpec's view and applies any scenarios.

        Args:
            seed: Random seed
            **free_vals: Values for free parameters only (P-space)

        Returns:
            Dictionary of output DataFrames
        """
        # P → M conversion using view
        params_M = self.spec.view.bind(**free_vals)

        # Apply scenarios if present
        config = dict(self.model.base_config)
        for scenario_name in self.spec.scenarios:
            if scenario_name not in self.model._scenarios:
                raise ValueError(f"Unknown scenario: {scenario_name}")
            scenario = self.model._scenarios[scenario_name]
            params_M, config = scenario.apply(params_M, config)

        # Execute simulation
        state = self.model.build_sim(params_M, MappingProxyType(config))
        raw = self.model.run_sim(state, seed)
        return self.model.extract_outputs(raw, seed)

    def to_transformed(self, **free_values: Scalar) -> Dict[str, float]:
        """Convert natural P-space values to transformed coordinates for optimizer.

        Applies forward transforms to each free parameter.
        """
        out = {}
        for name in self.spec.view.free:
            val = float(free_values[name])
            t = self.spec.transforms.get(name)
            out[name] = t.forward(val) if t else val
        return out

    def from_transformed(self, **coords: float) -> Dict[str, Scalar]:
        """Convert optimizer coordinates back to natural P-space.

        Applies inverse transforms to restore physical values.
        """
        out = {}
        for name in self.spec.view.free:
            t = self.spec.transforms.get(name)
            out[name] = t.backward(coords[name]) if t else coords[name]
        return out

    def bounds(self, transformed: bool = False) -> Dict[str, Tuple[float, float]]:
        """Get bounds for free parameters.

        Args:
            transformed: If True and transforms present, return transformed bounds

        Returns:
            Dictionary of parameter name to (min, max) tuples
        """
        bounds = {}
        meta = self.model.space.by_name()

        for name in self.spec.view.free:
            spec = meta[name]
            lo, hi = float(spec.min), float(spec.max)

            # Apply transform if requested
            if transformed and name in self.spec.transforms:
                t = self.spec.transforms[name]
                lo_t, hi_t = t.forward(lo), t.forward(hi)
                # Non-linear transforms can invert bounds
                if lo_t > hi_t:
                    lo_t, hi_t = hi_t, lo_t
                bounds[name] = (lo_t, hi_t)
            else:
                bounds[name] = (lo, hi)

        return bounds

    # ---- Cloud Export: Always M-space ----

    def to_yaml_runs(self,
                     runs: List[Tuple[Dict[str, Scalar], int]],
                     outputs: Optional[List[str]] = None) -> str:
        """Export specific P-space runs as M-space YAML.

        Args:
            runs: List of (free_params, seed) tuples
            outputs: Subset of outputs to include

        Returns:
            YAML string for cloud execution
        """
        import yaml

        # Get entry record
        rec = self.model.__class__.compile_entrypoint()

        # Use scenarios from spec
        branches = [list(self.spec.scenarios) if self.spec.scenarios else ["baseline"]]

        # Convert P-space runs to M-space
        m_runs = []
        for free_params, seed in runs:
            params_M = self.spec.view.bind(**free_params)
            # Apply scenario patches
            for scenario_name in self.spec.scenarios:
                scenario = self.model._scenarios[scenario_name]
                params_M, _ = scenario.apply(params_M, {})

            m_runs.append({
                "params_M": dict(params_M.values),
                "seed": int(seed)
            })

        # Build YAML document
        doc = {
            "schema": "calabaria.job.v1",
            "bundle": self.model.__class__.__module__.split('.')[0],
            "entry_id": rec.id,
            "scenario_branches": branches,
            "runs": m_runs,
            "variant_key": variant_key(
                class_name=self.model.__class__.__name__,
                stack=self.spec.scenarios,
                fixed=dict(self.spec.view.fixed)
            )
        }

        if outputs:
            doc["outputs"] = list(outputs)

        return yaml.dump(doc, default_flow_style=False)

    def to_yaml_sweep(self,
                      sweep: Dict[str, List[Scalar]],
                      seeds: List[int],
                      outputs: Optional[List[str]] = None) -> str:
        """Export P-space sweep as expanded M-space YAML.

        Builds Cartesian product locally, converts to M-space.

        Args:
            sweep: Parameter names to values (free params only)
            seeds: List of random seeds
            outputs: Subset of outputs

        Returns:
            YAML with fully expanded M-space runs
        """
        # Validate sweep params are free
        expected_free = set(self.spec.view.free)
        sweep_params = set(sweep.keys())

        if not sweep_params.issubset(expected_free):
            extra = sweep_params - expected_free
            raise ValueError(
                f"Sweep contains non-free params: {extra}. "
                f"Free params: {expected_free}"
            )

        # Build Cartesian product
        keys = sorted(sweep.keys())
        values = [sweep[k] for k in keys]

        runs = []
        for combo in itertools.product(*values):
            free_dict = dict(zip(keys, combo))

            # Ensure all free params are specified
            for param in expected_free:
                if param not in free_dict:
                    raise ValueError(
                        f"Sweep missing free param '{param}'"
                    )

            # Add all seeds
            for seed in seeds:
                runs.append((free_dict, seed))

        return self.to_yaml_runs(runs, outputs=outputs)

```

### Architectural Justification: Why Scenarios Live on the Variant

The design choice to have scenarios, transforms, and config all live on a single `ModelVariant` variant rather than separate builders is intentional and provides several key benefits:

#### Single Identity Surface
A single object (the variant) determines free/fixed parameters, transforms, scenario stack, and effective config. This means:
- **Clear provenance**: Your provenance hash and variant key come from (model code, fixed params, scenario stack, config)
- **No confusion**: All configuration is in one place - no "where did this override come from?" drift
- **Simplified debugging**: One object to inspect when understanding model behavior

#### Scenario and Config Handling
- **Scenarios live on the variant** as an ordered stack that patches parameters (fixes them) and config (categoricals)
- **Parameter patches**: Move parameter names from free→fixed with the scenario's value (scenario wins via LWW)
- **Config patches**: Are composed into a `composed_config` map that's applied at runtime
- **Ad-hoc overrides**: You can optionally add config overrides via `with_config()` for one-off studies; they apply after the scenario stack and win last

#### Why Not a Separate Scenario Builder?
- **Avoiding double LWW layers**: Splitting scenario composition into another builder creates two Last-Write-Wins layers and two places where conflicts can happen
- **Provenance clarity**: Keeping everything in VariantSpec ensures the complete configuration is captured in one place
- **Clear separation**: VariantSpec holds all configuration; ModelVariant just executes

```python
# Clear separation: configuration then execution
spec = VariantSpec(
    view=ParameterView.from_fixed(model.space, {
        "population": 10_000,
        "gamma": 0.07,
        "initial_infected": 10
    }),
    scenarios=("lockdown", "vaccination"),
    transforms={"beta": Logit01(), "gamma": Logit01()}
)

variant = ModelVariant(model, spec)

# Run simulation with just free parameters
tables = variant.simulate(seed=7, beta=0.32, gamma=0.11)
```

The VariantSpec pattern provides all the flexibility needed while maintaining clear separation between configuration and execution.

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
            "gamma": params["gamma"],
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
    gamma=0.07,
    initial_infected=10
)
# Now view.free = ("beta", "gamma")

# Quick exploration - must use bind() for P→M conversion
for beta in [0.2, 0.3, 0.4]:
    for gamma in [0.08, 0.1, 0.12]:
        params_M = view.bind(beta=beta, gamma=gamma)  # P → M
        results = model.simulate(params_M, seed=42)
        print(f"β={beta}, γ={gamma}: peak={results['compartments']['I'].max()}")

# 3. Use VariantSpec and ModelVariant for cleaner syntax
spec = VariantSpec(view=view)
variant = ModelVariant(model, spec)

# Now can pass just P-space parameters
results = variant.simulate(seed=42, beta=0.3, gamma=0.1)

# 4. Compare scenarios (patches, not transforms)
baseline_spec = VariantSpec(view=view)  # No scenario = baseline
lockdown_spec = VariantSpec(view=view, scenarios=("lockdown",))

baseline = ModelVariant(model, baseline_spec)
lockdown = ModelVariant(model, lockdown_spec)

# Same parameters, different scenarios
P = {"beta": 0.3, "gamma": 0.1}
results_baseline = baseline.simulate(seed=42, **P)
results_lockdown = lockdown.simulate(seed=42, **P)
# lockdown has contact_rate=2.0 due to patch
```

### Calibration Workflow

> **Interface Support Example**: The code below shows how the BaseModel's pure functional
> interface enables calibration. The optimizer (differential_evolution) is external -
> Calabaria just provides the clean parameter interface that calibration algorithms need.

```python
# 1. Set up transforms for optimization (NOT scenarios)
spec_opt = VariantSpec(
    view=view,
    transforms={
        "beta": AffineSqueezedLogit(),  # [0,1] bounded
        "gamma": AffineSqueezedLogit()   # [0,1] bounded
    }
)
variant_opt = ModelVariant(model, spec_opt)

# 2. Get bounds for optimization
bounds = variant_opt.bounds(transformed=True)

# 3. Optimizer works in transformed space
from scipy.optimize import differential_evolution

def loss(Z_coords):  # Z = transformed coordinates
    # Convert from transformed to natural P
    P_natural = variant_opt.from_transformed(
        **dict(zip(["beta", "gamma"], Z_coords))
    )

    # Run simulation
    results = variant_opt.simulate(seed=42, **P_natural)

    # Compute loss
    return compute_loss(results, targets)

# Get bounds in transformed space
bounds_list = [(bounds["beta"]), (bounds["gamma"])]

# Run optimization
opt_result = differential_evolution(loss, bounds_list)

# 4. Convert back to natural parameters
optimal_P = variant_opt.from_transformed(
    **dict(zip(["beta", "gamma"], opt_result.x))
)
print(f"Optimal: β={optimal_P['beta']:.3f}, γ={optimal_P['gamma']:.3f}")
```

### Post-Calibration Sensitivity

> **Note**: Sensitivity analysis naturally builds on the same BaseModel interface.
> The parameter exploration patterns shown here work with any external analysis tool.

```python
# Fix optimal values, vary one at a time
optimal = {"beta": 0.35, "gamma": 0.095}

# Create new view with optimal gamma fixed but beta free
sens_view = ParameterView.from_fixed(
    model.space,
    population=10000,
    contact_rate=4.0,
    initial_infected=10,
    gamma=0.095  # Fixed at optimal
)
# Now only beta is free

# Sweep beta around optimal
sens_spec = VariantSpec(view=sens_view)
sens_model = ModelVariant(model, sens_spec)
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
class SimTask:
    """Immutable simulation task for infrastructure.

    Complete specification for deterministic execution.
    """
    bundle_ref: str  # Code version
    entrypoint: str  # "module.Class/scenario"
    params: ParameterSet  # Complete M-space parameters
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
    wire_fn = model.compile_entrypoint()  # Uses scenario from stack

    # Execute with complete M-space parameters
    result_bytes = wire_fn(task.params.values, task.seed)

    # Filter outputs if requested
    if task.outputs:
        result_bytes = {k: v for k, v in result_bytes.items()
                       if k in task.outputs}

    return SimReturn(outputs=result_bytes)
```

### Calibration Execution

> **Implementation Pattern**: These helper functions show how to set up the BaseModel
> interface for use with external calibration libraries. The actual calibration
> algorithms (like scipy.optimize) remain outside Calabaria's scope for this MVP.

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

    # Create derived model with transforms for calibration
    derived = model.variant().fix(**fixed_params).with_transforms(**transforms)

    # Compile P-space function
    sim_fn = derived.compile_local()

    return derived, sim_fn, targets

def run_calibration(derived: ModelVariant, sim_fn: Callable, targets: Any):
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

### Why ModelVariant is Simple

ModelVariant is now just a simple pairing of BaseModel + VariantSpec:
- **Single responsibility**: Only executes the variant, doesn't configure it
- **Clear separation**: VariantSpec=configuration, ModelVariant=execution
- **No fluent API**: All configuration happens in VariantSpec
- **Immutable**: Both model and spec are frozen
- **Identity from spec**: VariantSpec carries all variant metadata

### Why Scenario Stacks (Composition)

**In flux**

Scenario stacks provide first-class composition:
- **Ordered composition**: Apply patches in sequence (LWW semantics)
- **Branching**: Run parallel branches with different compositions
- **Reconciliation**: Automatically move patched params to fixed
- **Deterministic**: Stack order determines final state
- **Flexible**: Single scenarios are just 1-element stacks

Examples: `["lockdown", "vaccination"]` vs `["vaccination", "lockdown"]` - order matters!

### Why Single M-Space Wire

Cloud infrastructure uses ONE wire function per model:
- **Identity**: Unambiguous cache keys from M-space parameters
- **Simplicity**: No P/M mode confusion on cloud
- **Provenance**: Clean content-addressed storage
- **Scenario stacks**: Passed as parameter, not compile-time
- **Local ergonomics**: P-space operations stay on workstation

### Why Variant Keys

Human-readable variant identifiers complement hash-based provenance:
- **Format**: `ClassName:scenario1+scenario2|param=val@hash8`
- **Dashboard friendly**: Obvious what's running at a glance
- **Collision resistant**: Hash suffix ensures uniqueness
- **Stable**: Same inputs generate same key
- **Separate from CAS**: Doesn't interfere with provenance

### YAML as Cloud Contract

YAML serves as the boundary between local and cloud:
- **Always M-space**: Cloud never sees P-space
- **Explicit runs**: Every parameter combination listed
- **Scenario branches**: Support both single and composed stacks
- **Deterministic**: Same YAML = same execution
- **P→M expansion**: Happens locally before export

## Part V: Complete Example with Decorators

### Full SIR Model Implementation

This example shows a complete model using decorators for outputs and scenarios:

```python
from calabaria.core import BaseModel, ParameterSpace, ParameterSpec, ParameterSet
from calabaria.core import model_output, model_scenario, ScenarioSpec
from calabaria.core.constants import SEED_COL
import polars as pl
import numpy as np
from typing import Any, Dict, Mapping

class SIRModel(BaseModel):
    """Complete SIR epidemic model with decorator-based registration."""

    # Define parameter space at class level
    PARAM_SPACE = ParameterSpace(specs=(
        ParameterSpec("beta", 0.0, 1.0, doc="Transmission rate"),
        ParameterSpec("gamma", 0.0, 1.0, doc="Recovery rate"),
        ParameterSpec("population", 100, 1000000, kind="int"),
        ParameterSpec("contact_rate", 1.0, 20.0),
        ParameterSpec("initial_infected", 1, 100, kind="int"),
    ))

    def __init__(self):
        super().__init__(self.PARAM_SPACE)

    # === SCENARIOS via decorators ===

    @model_scenario("lockdown")
    def lockdown_scenario(self) -> ScenarioSpec:
        """Lockdown reduces contact rate to 25% of baseline."""
        return ScenarioSpec(
            name="lockdown",
            doc="Reduced contact during lockdown",
            param_patches={"contact_rate": 2.0}  # FIX at 2.0
        )

    @model_scenario("vaccination")
    def vaccination_scenario(self) -> ScenarioSpec:
        """Vaccination rollout starting at day 100."""
        return ScenarioSpec(
            name="vaccination",
            doc="With vaccine rollout",
            config_patches={
                "vaccine_start": 100,
                "vaccine_efficacy": 0.9,
                "vaccine_rate": 0.01  # 1% per day
            }
        )

    @model_scenario("combined")
    def combined_scenario(self) -> ScenarioSpec:
        """Both lockdown and vaccination."""
        return ScenarioSpec(
            name="combined",
            doc="Lockdown plus vaccination",
            param_patches={"contact_rate": 2.0},
            config_patches={
                "vaccine_start": 50,
                "vaccine_efficacy": 0.9,
                "vaccine_rate": 0.02
            }
        )

    # === SIMULATION PIPELINE ===

    def build_sim(self, params: ParameterSet, config: Mapping) -> Any:
        """Build simulation state from complete M-space parameters."""
        # All parameters must be present - no defaults
        N = params["population"]
        beta = params["beta"] * params["contact_rate"]
        gamma = params["gamma"]
        I0 = params["initial_infected"]

        # Vaccination config from scenario (or empty)
        vaccine_start = config.get("vaccine_start", np.inf)
        vaccine_efficacy = config.get("vaccine_efficacy", 0.0)
        vaccine_rate = config.get("vaccine_rate", 0.0)

        return {
            "N": N,
            "beta": beta,
            "gamma": gamma,
            "S0": N - I0,
            "I0": I0,
            "R0": 0,
            "vaccine": {
                "start": vaccine_start,
                "efficacy": vaccine_efficacy,
                "rate": vaccine_rate
            }
        }

    def run_sim(self, state: Any, seed: int) -> Any:
        """Run SIR simulation with stochastic events."""
        rng = np.random.default_rng(seed)

        # Extract state
        N = state["N"]
        beta = state["beta"]
        gamma = state["gamma"]
        S, I, R = state["S0"], state["I0"], state["R0"]
        vaccine = state["vaccine"]

        # Time series storage
        days = 365
        S_ts = np.zeros(days)
        I_ts = np.zeros(days)
        R_ts = np.zeros(days)
        new_infections_ts = np.zeros(days)  # Track daily new infections

        # Run simulation
        for t in range(days):
            # Vaccination starts
            if t >= vaccine["start"] and S > 0:
                vaccinated = min(S, int(S * vaccine["rate"]))
                S -= vaccinated
                R += vaccinated

            # Disease dynamics with stochastic component
            force_of_infection = beta * I / N

            # Add stochastic noise
            noise = rng.normal(1.0, 0.1)
            force_of_infection *= max(0.0, noise)

            # New infections
            new_infected = min(S, int(S * force_of_infection))

            # Recoveries
            new_recovered = min(I, int(I * gamma))

            # Update compartments
            S = S - new_infected
            I = I + new_infected - new_recovered
            R = R + new_recovered

            # Store
            S_ts[t] = S
            I_ts[t] = I
            R_ts[t] = R
            new_infections_ts[t] = new_infected  # Store new infections

        return {
            "time_series": {
                "S": S_ts,
                "I": I_ts,
                "R": R_ts,
                "new_infections": new_infections_ts,  # Include new infections
                "days": np.arange(days)
            },
            "summary": {
                "peak_infected": np.max(I_ts),
                "peak_day": np.argmax(I_ts),
                "total_infected": R_ts[-1] - state["R0"],
                "attack_rate": (R_ts[-1] - state["R0"]) / N
            }
        }

    # === OUTPUT EXTRACTORS via decorators ===

    @model_output("prevalence")
    def extract_prevalence(self, raw: Any, seed: int) -> pl.DataFrame:
        """Extract daily prevalence time series.

        Note: Framework adds SEED_COL automatically.
        DO NOT add seed column here!
        """
        ts = raw["time_series"]
        return pl.DataFrame({
            "day": ts["days"],
            "prevalence": ts["I"] / np.sum([ts["S"], ts["I"], ts["R"]], axis=0)
        })

    @model_output("incidence")
    def extract_incidence(self, raw: Any, seed: int) -> pl.DataFrame:
        """Extract weekly incidence."""
        ts = raw["time_series"]

        # Use the tracked new infections directly
        daily_new = ts["new_infections"]

        # Aggregate to weekly
        weeks = len(daily_new) // 7
        weekly_incidence = []
        for w in range(weeks):
            week_sum = np.sum(daily_new[w*7:(w+1)*7])
            weekly_incidence.append({
                "week": w + 1,
                "incidence": week_sum
            })

        return pl.DataFrame(weekly_incidence)

    @model_output("summary")
    def extract_summary(self, raw: Any, seed: int) -> pl.DataFrame:
        """Extract simulation summary statistics."""
        summary = raw["summary"]
        return pl.DataFrame([{
            "peak_infected": summary["peak_infected"],
            "peak_day": summary["peak_day"],
            "total_infected": summary["total_infected"],
            "attack_rate": summary["attack_rate"]
        }])

    @model_output("compartments")
    def extract_compartments(self, raw: Any, seed: int) -> pl.DataFrame:
        """Extract full compartment time series."""
        ts = raw["time_series"]
        return pl.DataFrame({
            "day": ts["days"],
            "S": ts["S"],
            "I": ts["I"],
            "R": ts["R"]
        })


# === USAGE EXAMPLES ===

def example_basic_usage():
    """Basic model usage with complete parameter sets."""
    model = SIRModel()

    # Must provide ALL parameters (complete M-space)
    params = ParameterSet({
        "beta": 0.3,
        "gamma": 0.1,
        "population": 10000,
        "contact_rate": 4.0,
        "gamma": 0.1,
        "initial_infected": 10
    })

    # Run baseline scenario
    results = model.simulate(params, seed=42)
    print(f"Outputs: {results.keys()}")  # prevalence, incidence, summary, compartments

    # Each output has SEED_COL added automatically
    assert SEED_COL in results["prevalence"].columns
    assert results["prevalence"][SEED_COL][0] == 42


def example_scenario_usage():
    """Using scenarios to apply patches."""
    model = SIRModel()

    params = ParameterSet({
        "beta": 0.3,
        "gamma": 0.1,
        "population": 10000,
        "contact_rate": 8.0,  # Will be overridden by lockdown
        "gamma": 0.1,
        "initial_infected": 10
    })

    # Run lockdown scenario (patches contact_rate to 2.0)
    results = model.simulate_scenario("lockdown", params, seed=42)

    # The scenario's patch overrides the contact_rate
    # Internally: contact_rate becomes 2.0, not 8.0


def example_derived_model():
    """Using ModelVariant for ergonomic P-space interface."""
    from calabaria.core import ParameterView, ModelVariant

    model = SIRModel()

    # Fix experimental constants, leave beta/gamma free
    view = ParameterView.from_fixed(
        model.space,
        population=10000,
        contact_rate=4.0,
        gamma=14,
        initial_infected=10
    )

    # Create derived model for research
    derived = ModelVariant(model, view)

    # Now we can simulate with just free parameters!
    results = derived.simulate(seed=42, beta=0.3, gamma=0.1)

    # Apply scenario (reconciliation prevents double-specification)
    derived_lockdown = derived.scenario("lockdown")
    # contact_rate is now fixed by scenario, removed from free params

    # Only need to specify truly free parameters
    results = derived_lockdown.simulate(seed=42, beta=0.3, gamma=0.1)


def example_calibration_workflow():
    """Complete calibration workflow with transforms."""
    from calabaria.core import AffineSqueezedLogit

    model = SIRModel()

    # Set up calibration space
    view = ParameterView.from_fixed(
        model.space,
        population=10000,
        contact_rate=4.0,
        gamma=14,
        initial_infected=10
    )

    # Add transforms for bounded optimization
    derived_lockdown = derived_lockdown.with_transforms(
        beta=AffineSqueezedLogit(),  # [0,1] → ℝ
        gamma=AffineSqueezedLogit()
    )

    # Create derived model with transforms
    derived = ModelVariant(model, view, tview)

    # Compile for infrastructure
    sim_fn = derived.compile_local()

    # Optimizer works in transformed space
    from scipy.optimize import differential_evolution

    def objective(z_transformed):
        # Optimizer provides transformed coordinates
        free_z = {"beta": z_transformed[0], "gamma": z_transformed[1]}

        # Convert back to natural space
        free_p = derived.from_transformed(free_z)

        # Run simulation
        outputs = wire_fn(free_p, seed=42)

        # ... compute loss from outputs ...
        return loss

    # Get bounds in transformed space
    bounds_z = derived.bounds(transformed=True)
    bounds_list = [bounds_z["beta"], bounds_z["gamma"]]

    # Optimize in transformed space
    result = differential_evolution(objective, bounds_list)


def example_inheritance():
    """Extending models via inheritance preserves discoveries."""

    class ExtendedSIRModel(SIRModel):
        """Adds age structure to SIR model."""

        # Parent's outputs and scenarios are inherited!

        @model_output("age_distribution")
        def extract_age_dist(self, raw: Any, seed: int) -> pl.DataFrame:
            """Additional output for age structure."""
            # ... implementation ...
            return pl.DataFrame({"age_group": [1, 2, 3], "count": [100, 200, 300]})

        @model_scenario("schools_closed")
        def schools_closed_scenario(self) -> ScenarioSpec:
            """Additional scenario for school closures."""
            return ScenarioSpec(
                name="schools_closed",
                doc="Schools closed, reduced child contacts",
                param_patches={"contact_rate": 1.5}
            )

    extended = ExtendedSIRModel()

    # Has all parent outputs plus new ones
    params = ParameterSet({...})
    results = extended.simulate(params, seed=42)
    assert "prevalence" in results  # From parent
    assert "age_distribution" in results  # New in child

    # Has all parent scenarios plus new ones
    results = extended.simulate_scenario("lockdown", params, seed=42)  # Parent
    results = extended.simulate_scenario("schools_closed", params, seed=42)  # Child
```

## Part VI: Cloud Execution & Export

### Cloud Integration Types

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import hashlib
import inspect
import json
import yaml

@dataclass(frozen=True)
class WireResponse:
    """Response from cloud wire execution."""
    outputs: Dict[str, bytes]  # IPC-serialized DataFrames
    provenance: Dict[str, Any]  # sim_root, task_id, model_hash, variant_key

@dataclass(frozen=True)
class EntryRecord:
    """Registry entry for a model class with single canonical wire."""
    id: str                    # e.g., "examples.sir.SIR@a1b2c3d4"
    model_hash: str            # Full hash of code + registry
    wire: Callable             # Single RUN wire (M-space only, accepts stacks)
    scenarios: List[str]       # Sorted scenario names
    outputs: List[str]         # Sorted output names
    space: ParameterSpace      # For validation
    alias: Optional[str] = None # Display name if decorated

# Global registry for cloud lookup
ENTRY_REGISTRY: Dict[str, EntryRecord] = {}

@dataclass(frozen=True)
class ReparamSpec:
    """Reparameterization specification."""
    forward: Dict[str, Callable]  # P(reparam) → observation functions
    inverse: Dict[str, Callable]  # P(reparam) → natural P
    free: Tuple[str, ...]         # New free params in P(reparam)
    name: str = "reparam"

    def apply_inverse(self, reparam_vals: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Convert reparameterized values to natural P-space."""
        natural = {}
        for param_name, inverse_fn in self.inverse.items():
            natural[param_name] = inverse_fn(reparam_vals)
        return natural
```

### Registry Architecture Deep Dive

#### ENTRY_REGISTRY: The Global Model Catalog

The `ENTRY_REGISTRY` is a module-level dictionary serving as the global catalog of all compiled models available for cloud execution:

```python
# Global registry - lives at module level
ENTRY_REGISTRY: Dict[str, EntryRecord] = {}

# Purpose:
# 1. Cloud runner looks up models by entry_id
# 2. Models self-register when compile_entrypoint() is called
# 3. Provides introspection for UIs/dashboards
# 4. Single source of truth for available models
```

**How Registration Works:**

```python
# Models register themselves when compile_entrypoint() is called
class SIRModel(BaseModel):
    # ... model implementation ...
    pass

# Registration happens explicitly
record = SIRModel.compile_entrypoint(alias="SIR Epidemic Model")

# Now ENTRY_REGISTRY contains:
# {
#   "examples.sir.SIRModel@a1b2c3d4": EntryRecord(...),
#   "examples.seir.SEIRModel@5f6g7h8i": EntryRecord(...),
# }

# The registry persists for the Python session
# Cloud runners can look up any registered model
```

#### EntryRecord: Complete Model Interface

`EntryRecord` provides the complete interface for a model in the cloud environment:

```python
# Example of a fully populated EntryRecord
example_record = EntryRecord(
    id="examples.sir.SIRModel@a1b2c3d4",
    model_hash="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6",
    wire=<function run_wire at 0x...>,  # The execution callable
    scenarios=["baseline", "lockdown", "vaccination", "combined"],
    outputs=["prevalence", "incidence", "summary", "compartments"],
    space=ParameterSpace(specs=[
        ParameterSpec("beta", 0.0, 1.0),
        ParameterSpec("gamma", 0.0, 1.0),
        ParameterSpec("population", 100, 1000000, kind="int"),
        # ... more parameters
    ]),
    alias="SIR Epidemic Model"
)
```

**The Wire Function Closure:**

The `wire` field contains the actual execution function, which is a closure capturing the model class:

```python
def run_wire(*,
            params_M: Dict[str, Scalar],
            seed: int,
            scenario_stack: Union[List[str], Tuple[str, ...]] = ("baseline",),
            outputs: Optional[List[str]] = None) -> WireResponse:
    """
    This function is stored in EntryRecord.wire
    It's a closure that captures the model class from compile_entrypoint
    """
    # Create model instance (cls captured from outer scope)
    model = cls()

    # Apply scenario stack with LWW semantics
    stack = tuple(scenario_stack)
    pset = ParameterSet(params_M)
    config = dict(model.base_config)

    for scenario_name in stack:
        spec = model._scenarios[scenario_name]
        pset, config = spec.apply(pset, config, model.space)

    # Run simulation
    state = model.build_sim(pset, config)
    raw = model.run_sim(state, seed)
    outs = model.extract_outputs(raw, seed)

    # Filter outputs if requested
    if outputs:
        outs = {k: v for k, v in outs.items() if k in outputs}

    # Return WireResponse with outputs and provenance
    return WireResponse(
        outputs=serialize_ipc(outs),
        provenance={
            "entry_id": entry_id,
            "model_hash": model_hash,
            "scenario_stack": list(stack),
            "variant_key": variant_key(cls.__name__, stack, {}),
            # ... more provenance
        }
    )
```

**How Cloud Runner Uses EntryRecord:**

```python
def run_yaml_job(yaml_content: str) -> List[WireResponse]:
    config = yaml.safe_load(yaml_content)

    # Look up the model by entry_id
    entry_id = config["entry_id"]  # e.g., "examples.sir.SIRModel@a1b2c3d4"

    if entry_id not in ENTRY_REGISTRY:
        available = sorted(ENTRY_REGISTRY.keys())
        raise ValueError(
            f"Model {entry_id} not registered. "
            f"Available models: {available}"
        )

    entry = ENTRY_REGISTRY[entry_id]

    # Validate scenarios against what model provides
    for branch in config["scenario_branches"]:
        scenarios = [branch] if isinstance(branch, str) else branch
        for scenario in scenarios:
            if scenario not in entry.scenarios:
                raise ValueError(
                    f"Unknown scenario '{scenario}'. "
                    f"Available: {entry.scenarios}"
                )

    # Validate outputs if specified
    if "outputs" in config:
        unknown = set(config["outputs"]) - set(entry.outputs)
        if unknown:
            raise ValueError(
                f"Unknown outputs: {unknown}. "
                f"Available: {entry.outputs}"
            )

    # Execute using the wire function
    results = []
    for stack in branches:
        for run in config["runs"]:
            response = entry.wire(  # Call the registered wire
                params_M=run["params_M"],
                seed=run["seed"],
                scenario_stack=stack,
                outputs=config.get("outputs")
            )
            results.append(response)

    return results
```

#### ReparamSpec: Working in Alternative Parameter Spaces

`ReparamSpec` enables working in more natural parameter spaces:

```python
# Classic R0 reparameterization for epidemiology
r0_reparam = ReparamSpec(
    forward={
        "R0": lambda M: M["beta"] / M["gamma"],  # Basic reproduction number
        "infectious_period": lambda M: 1.0 / M["gamma"]
    },
    inverse={
        "beta": lambda P: P["R0"] * P["gamma"],  # Reconstruct beta from R0
        # gamma stays as-is in the reparam space
    },
    free=("R0", "gamma"),  # New free parameters
    name="R0_reparam"
)

# How it works in ModelVariant
dm = (SIRModel()
      .variant()
      .fix(population=10000)
      .reparam(
          forward={"R0": lambda M: M["beta"] / M["gamma"]},
          inverse={"beta": lambda P: P["R0"] * P["gamma"]},
          free=("R0", "gamma", "contact_rate")
      ))

# User works in R0 space
results = dm.simulate(seed=42, R0=2.5, gamma=0.1, contact_rate=4.0)

# Behind the scenes in paramset_for():
def paramset_for(self, **free_vals: Scalar) -> ParameterSet:
    # If reparameterized, convert back to natural space
    if self.reparam_spec:
        # free_vals = {"R0": 2.5, "gamma": 0.1, "contact_rate": 4.0}
        natural = self.reparam_spec.apply_inverse(free_vals)
        # natural = {"beta": 0.25, "gamma": 0.1, "contact_rate": 4.0}
        free_vals = natural

    # Combine with fixed to create complete M-space
    params_M = {**self.fixed, **free_vals}
    return ParameterSet(params_M)
```

**Advanced Reparameterization Examples:**

```python
# Coupled parameters
coupling_reparam = ReparamSpec(
    forward={
        "coupling_strength": lambda M: M["param1"] * M["param2"],
        "coupling_ratio": lambda M: M["param1"] / M["param2"]
    },
    inverse={
        "param1": lambda P: np.sqrt(P["coupling_strength"] * P["coupling_ratio"]),
        "param2": lambda P: np.sqrt(P["coupling_strength"] / P["coupling_ratio"])
    },
    free=("coupling_strength", "coupling_ratio", "param3"),
    name="coupled_params"
)

# Dimensionality reduction
reduced_reparam = ReparamSpec(
    forward={
        "R0": lambda M: M["beta"] / M["gamma"],
        # Reducing from (beta, gamma) to just R0
    },
    inverse={
        "beta": lambda P: P["R0"] * 0.1,  # Fix gamma=0.1
        "gamma": lambda P: 0.1             # Fixed in reparam
    },
    free=("R0",),  # Only R0 is free
    name="R0_only"
)

# Parameter grouping for hierarchical models
hierarchical_reparam = ReparamSpec(
    forward={
        "group_mean": lambda M: (M["param_A"] + M["param_B"]) / 2,
        "group_diff": lambda M: M["param_A"] - M["param_B"]
    },
    inverse={
        "param_A": lambda P: P["group_mean"] + P["group_diff"] / 2,
        "param_B": lambda P: P["group_mean"] - P["group_diff"] / 2
    },
    free=("group_mean", "group_diff"),
    name="hierarchical"
)
```

### Model Hashing and Entry IDs

```python
def hash_model_code_and_registry(model_cls: Type[BaseModel]) -> str:
    """Compute stable hash of model class including discovered items.

    Deterministic hash includes:
    - Calabaria version salt
    - Parameter space specification
    - Class source code
    - Discovered outputs (sorted)
    - Discovered scenarios (sorted)
    """
    hasher = hashlib.sha256()

    # Version salt for framework changes
    hasher.update(b"calabaria-v1.0.0")

    # Parameter space (if available)
    if hasattr(model_cls, 'PARAM_SPACE'):
        space_repr = repr(sorted(
            (s.name, s.lower, s.upper, s.kind)
            for s in model_cls.PARAM_SPACE.specs
        ))
        hasher.update(space_repr.encode('utf-8'))

    # Class source code
    source = inspect.getsource(model_cls)
    hasher.update(source.encode('utf-8'))

    # Discovered outputs (sorted for determinism)
    outputs = getattr(model_cls, "_discovered_outputs", {})
    for name in sorted(outputs.keys()):
        hasher.update(f"output:{name}".encode('utf-8'))
        method_name = outputs[name]
        method = getattr(model_cls, method_name)
        method_source = inspect.getsource(method)
        hasher.update(method_source.encode('utf-8'))

    # Discovered scenarios (sorted for determinism)
    scenarios = getattr(model_cls, "_discovered_scenarios", {})
    for name in sorted(scenarios.keys()):
        hasher.update(f"scenario:{name}".encode('utf-8'))
        method_name = scenarios[name]
        method = getattr(model_cls, method_name)
        method_source = inspect.getsource(method)
        hasher.update(method_source.encode('utf-8'))

    return hasher.hexdigest()

def compute_entry_id(model_cls: Type[BaseModel]) -> str:
    """Compute deterministic entry ID for model class."""
    model_hash = hash_model_code_and_registry(model_cls)
    module = model_cls.__module__
    name = model_cls.__name__
    return f"{module}.{name}@{model_hash[:8]}"

def scenario_spec_hash(spec: ScenarioSpec) -> str:
    """Hash a scenario spec for provenance."""
    data = {
        "name": spec.name,
        "param_patches": dict(spec.param_patches),
        "config_patches": dict(spec.config_patches)
    }
    return hashlib.sha256(
        json.dumps(data, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]
```

### Variant Key Generation

```python
def variant_key(*, class_name: str, stack: Tuple[str, ...], fixed: Dict) -> str:
    """Generate human-readable variant identifier.

    Format: ClassName:scenario1+scenario2|param1=val1|param2=val2@hash8

    Examples:
    - "SIR:baseline@a1b2c3d4"
    - "SIR:lockdown|pop=10k@5f6g7h8i"
    - "SIR:lockdown+vaccination|pop=10k|beta=0.3@9j0k1l2m"
    """
    def _short_num(v):
        """Shorten numbers for readability."""
        if isinstance(v, int) and v >= 1000:
            return f"{v//1000}k"
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v)

    # Scenario part
    scenario_part = '+'.join(stack) if stack else 'baseline'
    prefix = f"{class_name}:{scenario_part}"

    # Fixed params part (cap at 4 most important)
    if fixed:
        # Prioritize small params that vary often
        sorted_params = sorted(fixed.items(), key=lambda x: (len(str(x[1])), x[0]))
        fixed_bits = [f"{k}={_short_num(v)}" for k, v in sorted_params[:4]]
        base = "|".join(fixed_bits)
    else:
        base = ""

    # Hash for uniqueness (includes all fixed params for collision resistance)
    stack_hash = hashlib.sha256(
        json.dumps({"stack": stack, "fixed": fixed}, sort_keys=True).encode()
    ).hexdigest()[:8]

    if base:
        return f"{prefix}|{base}@{stack_hash}"
    else:
        return f"{prefix}@{stack_hash}"
```

### Provenance Utilities

```python
def canonical_json(obj: Any) -> str:
    """Serialize to canonical JSON for hashing."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))

def provenance_ids(*,
                   model_hash: str,
                   scenario_stack: List[str],
                   params_M: Dict[str, Scalar],
                   seed: int,
                   outputs: Tuple[str, ...]) -> Tuple[str, str]:
    """Compute provenance IDs for simulation.

    Returns:
        sim_root: Identity of simulation (params + seed + scenarios)
        task_id: Identity including outputs
    """
    # Scenario chain hash
    scen_chain = hashlib.sha256(
        canonical_json(scenario_stack).encode()
    ).hexdigest()

    # Simulation root (independent of outputs)
    sim_data = {
        "model": model_hash,
        "scenario_chain": scen_chain,
        "params": params_M,
        "seed": seed
    }
    sim_root = hashlib.sha256(
        canonical_json(sim_data).encode()
    ).hexdigest()

    # Task ID (includes outputs)
    task_data = {
        "sim_root": sim_root,
        "outputs": sorted(outputs)
    }
    task_id = hashlib.sha256(
        canonical_json(task_data).encode()
    ).hexdigest()[:16]

    return sim_root, task_id

def serialize_ipc(dataframes: Dict[str, pl.DataFrame]) -> Dict[str, bytes]:
    """Serialize DataFrames to Arrow IPC format."""
    import io
    serialized = {}
    for name, df in dataframes.items():
        bio = io.BytesIO()
        df.write_ipc(bio)
        serialized[name] = bio.getvalue()
    return serialized
```

### Complete Cloud Workflow

This section demonstrates the end-to-end workflow from model registration to cloud execution:

#### Step 1: Model Registration

Models self-register when `compile_entrypoint()` is called:

```python
# === Registration at Import Time ===

# In examples/sir.py
from calabaria.core import BaseModel

class SIRModel(BaseModel):
    # ... model implementation ...
    pass

# Register when module is imported
# This populates ENTRY_REGISTRY
_sir_entry = SIRModel.compile_entrypoint(alias="SIR Model")

# In examples/seir.py
class SEIRModel(BaseModel):
    # ... model implementation ...
    pass

_seir_entry = SEIRModel.compile_entrypoint(alias="SEIR Model")

# Now when someone imports the models:
import examples.sir
import examples.seir

# ENTRY_REGISTRY automatically contains both:
print(list(ENTRY_REGISTRY.keys()))
# Output: [
#   "examples.sir.SIRModel@a1b2c3d4",
#   "examples.seir.SEIRModel@5f6g7h8i"
# ]
```

#### Step 2: Local Development with Reparameterization

Researchers work locally in natural parameter spaces, using a fluent interface:

```python
from examples.sir import SIRModel

# Create reparameterized model for R0 space
dm = (SIRModel()
      .variant("lockdown")  # Start with scenario
      .fix(population=10000, gamma=14, initial_infected=10)
      .reparam(
          forward={"R0": lambda M: M["beta"] / M["gamma"]},
          inverse={"beta": lambda P: P["R0"] * P["gamma"]},
          free=("R0", "gamma", "contact_rate")
      ))

# Test locally in R0 space
results = dm.simulate(seed=42, R0=2.5, gamma=0.1, contact_rate=4.0)
print(f"Peak infections: {results['summary']['peak_infected'][0]}")

# The reparam automatically converts:
# User provides: R0=2.5, gamma=0.1
# apply_inverse produces: beta=0.25, gamma=0.1
# Model runs with natural parameters
```

#### Step 3: Export to Cloud

Export sweeps to YAML with automatic P→M conversion:

```python
# Define sweep in reparameterized space
yaml_text = dm.to_yaml_sweep(
    sweep={
        "R0": [1.5, 2.0, 2.5, 3.0],      # Basic reproduction number
        "gamma": [0.07, 0.1, 0.14],       # Recovery rate
        "contact_rate": [2.0, 4.0]        # Will be overridden by lockdown
    },
    seeds=[1, 2, 3],
    scenario_branches=[
        "baseline",                        # Single scenario
        "lockdown",                        # Overrides contact_rate
        ["lockdown", "vaccination"],      # Composed stack
    ]
)

# The YAML contains M-space parameters:
# - R0 values are converted to beta via inverse mapping
# - All parameter combinations are explicit
# - Total: 4×3×2 params × 3 seeds × 3 branches = 216 simulations

print(yaml_text)
# Output:
# schema: calabaria.job.v1
# entry_id: examples.sir.SIRModel@a1b2c3d4
# scenario_branches:
#   - baseline
#   - lockdown
#   - [lockdown, vaccination]
# runs:
#   - params_M:
#       beta: 0.105    # From R0=1.5, gamma=0.07
#       gamma: 0.07
#       population: 10000
#       contact_rate: 2.0
#       gamma: 14
#       initial_infected: 10
#     seed: 1
#   # ... 215 more runs ...
```

#### Step 4: Cloud Execution

Cloud runner uses the registry to execute YAML:

```python
from calabaria.cloud import run_yaml_job

# Cloud execution (happens on cluster)
responses = run_yaml_job(yaml_text)

# Process returns:
# 1. Look up "examples.sir.SIRModel@a1b2c3d4" in ENTRY_REGISTRY
# 2. Validate all scenarios exist in model
# 3. Execute each run × branch combination
# 4. Return WireResponse objects with provenance

for resp in responses[:3]:
    prov = resp.provenance
    print(f"Variant: {prov['variant_key']}")
    print(f"Stack: {prov['scenario_stack']}")
    print(f"Sim root: {prov['sim_root'][:8]}...")

# Output:
# Variant: SIRModel:baseline|pop=10k@3f8a9c21
# Stack: ['baseline']
# Sim root: a7b8c9d0...
#
# Variant: SIRModel:lockdown|pop=10k@5f6g7h8i
# Stack: ['lockdown']
# Sim root: e1f2g3h4...
#
# Variant: SIRModel:lockdown+vaccination|pop=10k@9j0k1l2m
# Stack: ['lockdown', 'vaccination']
# Sim root: i5j6k7l8...
```

### Registry Management Patterns

#### Discovery and Introspection

```python
def list_available_models() -> List[str]:
    """List all registered models with their capabilities."""
    output = []
    for entry_id, record in ENTRY_REGISTRY.items():
        output.append(
            f"{record.alias or entry_id}:\n"
            f"  ID: {entry_id}\n"
            f"  Scenarios: {', '.join(record.scenarios)}\n"
            f"  Outputs: {', '.join(record.outputs)}\n"
            f"  Parameters: {len(record.space.specs)} total"
        )
    return output

def get_model_info(entry_id: str) -> dict:
    """Get detailed information about a model."""
    if entry_id not in ENTRY_REGISTRY:
        # Try partial match
        matches = [k for k in ENTRY_REGISTRY if entry_id in k]
        if len(matches) == 1:
            entry_id = matches[0]
        else:
            raise ValueError(f"Model {entry_id} not found")

    record = ENTRY_REGISTRY[entry_id]
    return {
        "id": record.id,
        "hash": record.model_hash,
        "alias": record.alias,
        "scenarios": record.scenarios,
        "outputs": record.outputs,
        "parameters": [
            {
                "name": spec.name,
                "bounds": [spec.lower, spec.upper],
                "type": spec.kind,
                "doc": spec.doc
            }
            for spec in record.space.specs
        ]
    }
```

#### YAML Validation

```python
def validate_yaml_compatibility(yaml_config: dict) -> Tuple[bool, List[str]]:
    """Check if YAML can be executed, return errors if not."""
    errors = []

    # Check entry_id
    entry_id = yaml_config.get("entry_id")
    if not entry_id:
        errors.append("Missing entry_id")
        return False, errors

    if entry_id not in ENTRY_REGISTRY:
        errors.append(f"Unknown model: {entry_id}")
        available = sorted(ENTRY_REGISTRY.keys())[:3]
        errors.append(f"Available models include: {available}")
        return False, errors

    entry = ENTRY_REGISTRY[entry_id]

    # Check scenarios
    branches = yaml_config.get("scenario_branches", ["baseline"])
    for branch in branches:
        scenarios = [branch] if isinstance(branch, str) else branch
        for scenario in scenarios:
            if scenario not in entry.scenarios:
                errors.append(
                    f"Unknown scenario '{scenario}'. "
                    f"Available: {entry.scenarios}"
                )

    # Check outputs
    requested_outputs = yaml_config.get("outputs", [])
    unknown_outputs = set(requested_outputs) - set(entry.outputs)
    if unknown_outputs:
        errors.append(
            f"Unknown outputs: {unknown_outputs}. "
            f"Available: {entry.outputs}"
        )

    # Check parameters
    required_params = set(entry.space.names())
    for i, run in enumerate(yaml_config.get("runs", [])):
        provided = set(run.get("params_M", {}).keys())
        missing = required_params - provided
        extra = provided - required_params

        if missing:
            errors.append(f"Run {i}: missing parameters {missing}")
        if extra:
            errors.append(f"Run {i}: unknown parameters {extra}")

    return len(errors) == 0, errors
```

#### Lazy Registration Pattern

```python
def ensure_models_registered(*modules):
    """Ensure specified model modules are registered."""
    for module_name in modules:
        # Check if already registered
        if any(module_name in entry_id for entry_id in ENTRY_REGISTRY):
            continue

        # Import to trigger registration
        try:
            module = __import__(module_name, fromlist=[''])
            print(f"Registered models from {module_name}")
        except ImportError as e:
            print(f"Failed to import {module_name}: {e}")

# Usage
ensure_models_registered(
    "examples.sir",
    "examples.seir",
    "examples.sirs"
)

def auto_discover_models(package_name: str):
    """Auto-discover and register all models in a package."""
    import pkgutil
    import importlib

    package = importlib.import_module(package_name)

    for importer, modname, ispkg in pkgutil.walk_packages(
        path=package.__path__,
        prefix=package.__name__ + ".",
        onerror=lambda x: None
    ):
        if not ispkg:
            # Import module to trigger registration
            try:
                importlib.import_module(modname)
            except Exception as e:
                print(f"Failed to import {modname}: {e}")

    return sorted(ENTRY_REGISTRY.keys())

# Auto-discover all models
discovered = auto_discover_models("examples")
print(f"Discovered {len(discovered)} models")
```

#### Registry Serialization for Distributed Systems

```python
def export_registry_metadata() -> dict:
    """Export registry metadata for sharing (without wire functions)."""
    return {
        entry_id: {
            "model_hash": record.model_hash,
            "scenarios": record.scenarios,
            "outputs": record.outputs,
            "parameters": [
                {
                    "name": spec.name,
                    "lower": spec.lower,
                    "upper": spec.upper,
                    "kind": spec.kind,
                    "doc": spec.doc
                }
                for spec in record.space.specs
            ],
            "alias": record.alias
        }
        for entry_id, record in ENTRY_REGISTRY.items()
    }

def verify_registry_compatibility(remote_metadata: dict) -> dict:
    """Verify local registry matches remote metadata."""
    compatibility = {}

    for entry_id, remote_meta in remote_metadata.items():
        if entry_id not in ENTRY_REGISTRY:
            compatibility[entry_id] = "missing_local"
            continue

        local_record = ENTRY_REGISTRY[entry_id]
        if local_record.model_hash != remote_meta["model_hash"]:
            compatibility[entry_id] = "hash_mismatch"
        else:
            compatibility[entry_id] = "compatible"

    # Check for local models not in remote
    for entry_id in ENTRY_REGISTRY:
        if entry_id not in remote_metadata:
            compatibility[entry_id] = "missing_remote"

    return compatibility

# Usage for distributed setup
metadata = export_registry_metadata()
# Send metadata to remote workers...

# On remote worker:
compatibility = verify_registry_compatibility(metadata)
incompatible = [k for k, v in compatibility.items() if v != "compatible"]
if incompatible:
    raise ValueError(f"Registry mismatch for models: {incompatible}")
```

### Cloud Runner Implementation

```python
def _expand_branches(entry: EntryRecord, config: dict) -> List[Tuple[str, ...]]:
    """Expand branch specification to list of scenario stacks.

    Handles three forms:
    1. scenario_branches: List of strings or lists (explicit stacks)
    2. scenario_filter: Wildcard patterns with include/exclude
    3. Default: ["baseline"]
    """
    import fnmatch

    if "scenario_branches" in config:
        branches = config["scenario_branches"]
        out = []
        for b in branches:
            # Handle both "scenario" and ["scenario1", "scenario2"]
            stack = (b,) if isinstance(b, str) else tuple(b)

            # Validate all scenarios exist
            for nm in stack:
                if nm not in entry.scenarios:
                    raise ValueError(
                        f"Unknown scenario '{nm}' in stack {stack}. "
                        f"Available: {entry.scenarios}"
                    )
            out.append(stack)
        return out

    # Handle wildcard patterns
    sel = config.get("scenario_filter", {"include": ["baseline"]})
    include = sel.get("include", ["baseline"])
    exclude = sel.get("exclude", [])

    chosen = set()
    for pat in include:
        if pat == "*":
            chosen.update(entry.scenarios)
        else:
            chosen.update(n for n in entry.scenarios
                         if fnmatch.fnmatchcase(n, pat))

    for pat in exclude:
        if pat == "*":
            chosen.clear()
        else:
            chosen = {n for n in chosen
                     if not fnmatch.fnmatchcase(n, pat)}

    # Return as single-element stacks
    return [(n,) for n in sorted(chosen)]

def run_yaml_job(yaml_content: str) -> List[WireResponse]:
    """Execute YAML job specification with scenario stacks.

    Simple deterministic execution:
    - Cartesian product: scenario_branches × runs
    - All in M-space (P-space already expanded)
    - Returns list of WireResponse objects
    """
    import yaml

    config = yaml.safe_load(yaml_content)

    # Validate schema
    if config.get("schema") != "calabaria.job.v1":
        raise ValueError(
            f"Unsupported schema: {config.get('schema')}. "
            f"Expected: calabaria.job.v1"
        )

    # Look up model in registry
    entry_id = config["entry_id"]
    if entry_id not in ENTRY_REGISTRY:
        raise ValueError(
            f"Unknown entry_id: {entry_id}. "
            f"Available: {sorted(ENTRY_REGISTRY.keys())}"
        )

    entry = ENTRY_REGISTRY[entry_id]

    # Expand branches (handles stacks and wildcards)
    branches = _expand_branches(entry, config)

    # Optional output filtering
    outputs_filter = config.get("outputs")

    # Execute Cartesian product
    results = []
    for stack in branches:
        for run in config["runs"]:
            resp = entry.wire(
                params_M=run["params_M"],
                seed=run["seed"],
                scenario_stack=stack,
                outputs=outputs_filter
            )
            results.append(resp)

    return results
```

### YAML Schema Examples

#### Single Scenario Branches
```yaml
schema: calabaria.job.v1
bundle: examples.sir
entry_id: examples.sir.SIR@a1b2c3d4
variant_key: "SIR:baseline|pop=10k@3f8a9c21"

scenario_branches:
  - baseline              # Single scenario
  - lockdown             # Single scenario

runs:
  - params_M:
      beta: 0.3
      gamma: 0.1
      population: 10000
      contact_rate: 4.0
      gamma: 14
      initial_infected: 10
    seed: 42

outputs: [prevalence, incidence]
```

#### Composed Scenario Stacks
```yaml
schema: calabaria.job.v1
bundle: examples.sir
entry_id: examples.sir.SIR@a1b2c3d4
variant_key: "SIR:lockdown+vaccination|pop=10k|beta=0.3@5f6g7h8i"

scenario_branches:
  - baseline                    # Single scenario
  - [lockdown, vaccination]     # Composed stack (LWW order)
  - [vaccination, lockdown]     # Different order = different result

runs:
  - params_M: {beta: 0.3, gamma: 0.1, population: 10000, ...}
    seed: 1
  - params_M: {beta: 0.4, gamma: 0.12, population: 10000, ...}
    seed: 2

outputs: [prevalence]
```

#### Wildcard Patterns
```yaml
schema: calabaria.job.v1
bundle: examples.sir
entry_id: examples.sir.SIR@a1b2c3d4
variant_key: "SIR:all-scenarios|pop=10k@7j8k9m0n"

scenario_filter:
  include: ["*"]              # All scenarios
  exclude: ["test_*", "*_debug"]  # Exclude test/debug scenarios

runs:
  - params_M: {...}
    seed: 42
```

### Complete Usage Examples with Scenario Stacks and Registry

```python
# === Example 1: Model Registration and Discovery ===

from calabaria.core import SIRModel, ENTRY_REGISTRY

# Register model (typically done at module import)
entry = SIRModel.compile_entrypoint(alias="SIR Epidemic Model")
print(f"Registered: {entry.id}")
# Output: "Registered: examples.sir.SIRModel@a1b2c3d4"

# Discover available models
for entry_id, record in ENTRY_REGISTRY.items():
    print(f"{record.alias}: {len(record.scenarios)} scenarios")
# Output:
# SIR Epidemic Model: 4 scenarios
# SEIR Model: 3 scenarios

# Get model capabilities
sir_entry = ENTRY_REGISTRY["examples.sir.SIRModel@a1b2c3d4"]
print(f"Outputs: {sir_entry.outputs}")
print(f"Scenarios: {sir_entry.scenarios}")


# === Example 2: Basic fluent workflow with stacks ===

# Start with composed scenario stack
dm = (SIRModel()
      .variant("lockdown", "vaccination")  # Stack from the start
      .fix(population=10000, contact_rate=4.0,
           gamma=14, initial_infected=10))

# The stack applies in order:
# 1. lockdown patches contact_rate to 2.0
# 2. vaccination adds config patches
# contact_rate is now fixed at 2.0 (removed from free)

# Simulate with remaining free parameters
results = dm.simulate(seed=42, beta=0.3, gamma=0.1)

# Generate variant key for tracking
key = variant_key(
    class_name="SIRModel",
    stack=("lockdown", "vaccination"),
    fixed=dict(dm.fixed)
)
print(f"Running variant: {key}")
# Output: "SIRModel:lockdown+vaccination|pop=10k|contact=2@a1b2c3d4"


# === Example 2: Export sweep with multiple branches ===

dm = (SIRModel()
      .variant()  # Start with baseline
      .fix(population=10000, gamma=14, initial_infected=10))

# Export sweep with multiple scenario branches
yaml_text = dm.to_yaml_sweep(
    sweep={
        "beta": [0.2, 0.3, 0.4],
        "gamma": [0.08, 0.1, 0.12],
        "contact_rate": [2.0, 4.0, 8.0]
    },
    seeds=[1, 2, 3],
    scenario_branches=[
        "baseline",                      # Single scenario
        "lockdown",                      # Single scenario
        ["lockdown", "vaccination"],    # Composed stack
        ["vaccination", "lockdown"]     # Different order
    ]
)

# This generates:
# - 3×3×3 = 27 parameter combinations
# - × 3 seeds = 81 runs
# - × 4 branches = 324 total simulations


# === Example 3: Reparameterization with stacks ===

dm = (SIRModel()
      .variant()
      .fix(population=10000, gamma=14, initial_infected=10)
      .reparam(
          forward={"R0": lambda M: M["beta"] / M["gamma"]},
          inverse={
              "beta": lambda P: P["R0"] * P["gamma"],
              # gamma stays as-is
          },
          free=("R0", "gamma", "contact_rate")
      )
      .scenarios("lockdown", "vaccination"))  # Add stack after reparam

# Now working in (R0, gamma) space with scenario stack
results = dm.simulate(seed=42, R0=2.5, gamma=0.1)
# Note: contact_rate was free, but lockdown fixes it to 2.0

# Export reparam sweep
yaml_text = dm.to_yaml_reparam(
    reparam_sweep={
        "R0": [1.5, 2.0, 2.5, 3.0],
        "gamma": [0.07, 0.1, 0.14]
    },
    seeds=list(range(10)),
    scenario_branches=[
        ["lockdown", "vaccination"],
        ["vaccination", "lockdown", "boost"]  # 3-element stack
    ]
)


# === Example 4: Reconciliation prevents double-specification ===

dm = (SIRModel()
      .variant()
      .fix(population=10000, gamma=14))

# Lockdown fixes contact_rate
dm_lockdown = dm.scenario("lockdown")
print(f"Free params: {dm_lockdown.free}")
# Output: ('beta', 'gamma', 'initial_infected')
# Note: contact_rate removed from free!

# This would fail - contact_rate already fixed by scenario
try:
    dm_lockdown.simulate(
        seed=42,
        beta=0.3,
        gamma=0.1,
        initial_infected=10,
        contact_rate=8.0  # ERROR: not free anymore
    )
except ValueError as e:
    print(f"Error: {e}")
    # "paramset_for() error: unexpected: ['contact_rate']"


# === Example 5: Programmatic branch exploration ===

def explore_scenario_combinations(model, max_stack_size=3):
    """Generate all possible scenario stacks up to size N."""
    import itertools

    scenarios = sorted(model._scenarios.keys())
    branches = []

    for size in range(1, max_stack_size + 1):
        for combo in itertools.combinations(scenarios, size):
            # Try both orders for 2-element stacks
            if size == 2:
                branches.append(list(combo))
                branches.append(list(reversed(combo)))
            else:
                branches.append(list(combo))

    return branches

model = SIRModel()
all_branches = explore_scenario_combinations(model, max_stack_size=2)

dm = model.variant().fix(population=10000)
yaml_text = dm.to_yaml_sweep(
    sweep={"beta": [0.3], "gamma": [0.1]},  # Simple test
    seeds=[42],
    scenario_branches=all_branches
)


# === Example 6: Cloud execution roundtrip ===

# Local: Create and test variant
dm = (SIRModel()
      .variant("lockdown", "vaccination")
      .fix(population=10000, gamma=14, initial_infected=10))

# Local test
local_results = dm.simulate(seed=42, beta=0.3, gamma=0.1, contact_rate=4.0)

# Export to YAML
yaml_text = dm.to_yaml_runs(
    runs=[({"beta": 0.3, "gamma": 0.1, "contact_rate": 4.0}, 42)],
    outputs=["prevalence", "incidence"]
)

# Cloud: Execute YAML
from calabaria.cloud import run_yaml_job
responses = run_yaml_job(yaml_text)

# Responses include variant keys for tracking
for resp in responses:
    print(f"Variant: {resp.provenance['variant_key']}")
    print(f"Sim root: {resp.provenance['sim_root']}")
    print(f"Outputs: {resp.provenance['outputs']}")


# === Example 7: Complete Workflow with Registry and Reparameterization ===

# Step 1: Ensure models are registered
from calabaria.core import auto_discover_models
discovered = auto_discover_models("examples")
print(f"Available models: {discovered}")

# Step 2: Create reparameterized variant
from examples.sir import SIRModel

# Work in epidemiologically meaningful space
dm = (SIRModel()
      .variant()
      .fix(population=50000, gamma=10, initial_infected=5)
      .reparam(
          # Define R0 and infectious period instead of beta/gamma
          forward={
              "R0": lambda M: M["beta"] / M["gamma"],
              "infectious_period": lambda M: 1.0 / M["gamma"]
          },
          inverse={
              "beta": lambda P: P["R0"] / P["infectious_period"],
              "gamma": lambda P: 1.0 / P["infectious_period"]
          },
          free=("R0", "infectious_period", "contact_rate")
      ))

# Step 3: Test locally with meaningful parameters
results = dm.simulate(
    seed=42,
    R0=2.5,  # Basic reproduction number
    infectious_period=10.0,  # Days infectious
    contact_rate=4.0  # Contacts per day
)

# Step 4: Export sweep for cloud
yaml_text = dm.to_yaml_sweep(
    sweep={
        "R0": [1.5, 2.0, 2.5, 3.0, 3.5],
        "infectious_period": [7, 10, 14],
        "contact_rate": [2, 4, 8]
    },
    seeds=list(range(100)),  # 100 replicates
    scenario_branches=[
        "baseline",
        "lockdown",
        ["lockdown", "vaccination"],
        ["lockdown", "vaccination", "boost"]  # 3-level stack
    ]
)

# Step 5: Validate before submission
from calabaria.cloud import validate_yaml_compatibility
is_valid, errors = validate_yaml_compatibility(yaml.safe_load(yaml_text))
if not is_valid:
    print(f"YAML validation failed: {errors}")
else:
    print("YAML validated successfully")

# Step 6: Execute on cloud
responses = run_yaml_job(yaml_text)
print(f"Executed {len(responses)} simulations")

# Step 7: Track using variant keys
variant_counts = {}
for resp in responses:
    key = resp.provenance['variant_key']
    variant_counts[key] = variant_counts.get(key, 0) + 1

for variant, count in sorted(variant_counts.items()):
    print(f"{variant}: {count} runs")

# Output shows human-readable tracking:
# SIRModel:baseline|pop=50k|R0=1.5@a1b2c3d4: 100 runs
# SIRModel:baseline|pop=50k|R0=2.0@b2c3d4e5: 100 runs
# SIRModel:lockdown|pop=50k|R0=1.5@c3d4e5f6: 100 runs
# SIRModel:lockdown+vaccination|pop=50k|R0=1.5@d4e5f6g7: 100 runs
# ... etc


# === Example 8: Registry Synchronization for Distributed Teams ===

# Export registry for sharing with team
metadata = export_registry_metadata()
with open("model_registry.json", "w") as f:
    json.dump(metadata, f, indent=2)

# On another machine/cluster:
with open("model_registry.json", "r") as f:
    remote_metadata = json.load(f)

# Verify compatibility
compat = verify_registry_compatibility(remote_metadata)
if any(v != "compatible" for v in compat.values()):
    print("Registry mismatch detected:")
    for model, status in compat.items():
        if status != "compatible":
            print(f"  {model}: {status}")
else:
    print("Registry synchronized successfully")
```

## Part VII: Implementation Timeline

### Week 1-2: Core Parameter System
- Implement frozen dataclasses for all types
- ParameterSet, ParameterSpace, ParameterSpec
- ParameterView with bind() for P→M
- Transform system for optimization
- Comprehensive unit tests

### Week 3-4: Model Interface
- BaseModel with simulate() enforcement
- Scenario system as patches
- ModelVariant facade with reconciliation
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
- **Ergonomic research API** via ModelVariant
- **Reproducible science** through pure functions
- **Scalable execution** via clean contracts

The immutable parameter system with ParameterView/ModelVariant provides researchers
with intuitive workflows while maintaining architectural purity and preventing
entire classes of bugs.
