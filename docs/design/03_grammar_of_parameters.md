# Grammar of Parameters

## Intent

Like the Grammar of Graphics, this "Grammar of Parameters"
provides a small set of primitives, operators, and laws that
compose into all common parameter workflows—exploration,
scenarios, calibration—while staying pure, immutable, and
predictable.

The design patterns directly prevent state bugs through
immutability. Every primitive and operator enforces
immutability, ensuring parameters can never be accidentally
mutated, leaked between experiments, or silently corrupted.

## Mathematical Notation

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

## 1. Primitives (Objects)

### ParameterSpace (M)

- The model's complete set of named numeric parameters with
  domains
- **Intuition**: "The full list of knobs the model
  understands"
- **Names**: `Names(M)` - finite set of parameter names
- **Domain per name**: `Domain(name)` - interval or integer
  range (no categorical)
- **Cartesian space**: M is the product of all numeric domains
- **Note**: Categorical choices belong in configuration, not
  parameters

### ParameterSet (m ∈ M)

- A complete, immutable assignment for every name in M
- **Intuition**: "One concrete knob setting—what the model
  actually runs with"
- **Enforced at boundary**: BaseModel.simulate() only accepts
  complete ParameterSet
- **Invariant**: All parameters must have values (no missing,
  no defaults)

### ParameterView — Partial Valuation Inducing Calibration Subspace

- A partial valuation: mapping from subset of names(M) to
  values
- **Essential data**: `fixed`: {name → value} mapping
- **Derived data**: `free` = names(M) \ domain(fixed) (the
  complement)
- **Induced subspace**: Pᵥ = ∏_{n ∈ free} Domain(n)
- **Provides two canonical operations**:
  - **Projection π**: M → Pᵥ - extracts free coordinates
  - **Embedding ι (bind)**: Pᵥ → M - combines free values with
    fixed mapping
- **Formally**: view = fixed_mapping, which implicitly defines
  free by complement
- **Intuition**: "Pin some parameters, let others vary for
  calibration"

### Transform Set (T)

- Per-free-parameter bijections for optimization coordinates
- **Intuition**: "Change units for the optimizer without
  changing the model"
- **For each free name**: T[name]: Domain(name) ↔ Coord(name)
- **MVP constraint**: Independent transforms only, no coupling

### ScenarioPatch

- Two partial updates applied at execution:
  - `param_patches`: {name → value} - fix/override numeric
    parameters
  - `config_patches`: {key → value} - modify
    categorical/structural configuration
- **Intuition**: "Preset numeric knobs and categorical choices
  to embody a scenario"
- **Note**: Config patches handle categorical values that
  can't be calibration targets

## 2. Operators (Morphisms)

### Bind (Embedding ι: P → M)

```
bind(view, free_values) → ParameterSet
```

- The embedding operation ιᵥ: Pᵥ → M
- Takes values in calibration space P, merges with partial valuation, returns
  point in M
- **Definition**: bind(view, p) = ParameterSet(view.fixed ∪ {n → p[n] for n in
  free(view)})
- **Domain**: Defined iff keys(p) == free(view)
- **Purpose**: THE bridge from calibration subspace back to full M-space
- **Law**: π ∘ ι = id_P (project after bind recovers free values)
- **Intuition**: "Complete the partial valuation with calibration parameters"

### Project (Projection π: M → P)

```
project(view, params) → free_values
```

- The projection operation πᵥ: M → Pᵥ
- Extracts only the free coordinates from a complete ParameterSet
- **Returns**: Ordered dict matching view.free ordering
- **Law**: ι ∘ π = clamp_fixed (bind after project clamps to fixed)
- **Intuition**: "Extract the free dimensions, forgetting fixed values"

### ApplyPatch

```
apply_patch(patch, m, cfg) → (m', cfg')
```

- Applies ScenarioSpec patches to parameters and config
- **Parameter patches**: Direct value overrides in M-space (no transforms/reparams)
- **Config patches**: Key-value updates to configuration dictionary
- **Semantics**: Last-write-wins (LWW) for overlapping keys in stacked scenarios
- **Immutability**: Returns updated (m', cfg') without modifying originals
- **Intuition**: "Bake scenario overrides into params/config"

### Transform/Inverse

```
to_transformed(T, natural_P) → transformed_Z
from_transformed(T, transformed_Z) → natural_P
```

- Applied only to free parameters
- No dimensionality change
- **Invariant**: `from_transformed ∘ to_transformed = id` (up to floating point)
- **Intuition**: "Switch to optimizer space and back"
- **Caution**: Bounds handling needs care (e.g., log space boundaries)

### ComposePatches (Monoid)

```
compose(p₂, p₁) = "apply p₁ then p₂"
```

- **Identity**: The empty patch
- **Associative**: compose(p₃, compose(p₂, p₁)) = compose(compose(p₃, p₂), p₁)
- **Last-write-wins**: Later patches override earlier ones

## 3. Laws and Invariants

### Fundamental Laws

1. **Embedding-Projection Identity**:
   ```
   π ∘ ι = id_Pᵥ
   ```
   Projecting after embedding recovers the free values exactly.

2. **Embedding-Projection Clamping**:
   ```
   ι ∘ π = clamp_to_fixed
   ```
   Embedding after projecting clamps to the fixed values.

3. **Transform Invertibility**:
   ```
   from_transformed ∘ to_transformed = id
   to_transformed ∘ from_transformed = id
   ```
   Transforms are bijections (modulo floating point).

4. **Patch Monoid Laws**:
   ```
   compose(empty, p) = p                    # Left identity
   compose(p, empty) = p                    # Right identity
   compose(p₃, compose(p₂, p₁)) =          # Associativity
     compose(compose(p₃, p₂), p₁)
   ```

### Validation Invariants

1. **ParameterSet Completeness**: Every parameter in M must have a value
2. **ParameterView Consistency**: Fixed parameters must exist in M
3. **Bind Exactness**: Free values must match view.free exactly
4. **Transform Domain**: Transforms apply only to free parameters

## 4. Design Rationale: Why ModelVariant Uses kwargs

### The Design Choice
ModelVariant takes free parameters as kwargs (`**free`), not a ParameterSet.

```python
# ModelVariant uses:
variant.simulate(seed=42, beta=0.3, gamma=0.1)

# NOT:
variant.simulate(ParameterSet(...), seed=42)
```

### Core Invariant We Preserve
- `ParameterSet` means "complete M-space point" - ONLY
- BaseModel.simulate accepts ONLY complete ParameterSets
- This invariant prevents parameter state bugs

### What Goes Wrong with P-space ParameterSets

If we allowed partial ParameterSets:

1. **Conceptual leakage**: People try to reuse objects between P and M calls
2. **Runtime bugs**: A "P-ParameterSet" can't be validated without the view
3. **Transform confusion**: Are transforms applied? To which params?
4. **Cache muddle**: M-space has deterministic identity; P-space is view-dependent

### Why kwargs Are Right

1. **Ergonomics**: `variant.simulate(seed=1, beta=0.3, gamma=0.1)` reads clearly
2. **Clear semantics**: "I'm giving only the free parameters"
3. **Preserves barrier**: Internally binds via ι to produce ParameterSet
4. **Type safety**: Each API matches its abstraction level

### The Internal Bind

```python
def simulate(self, seed: int, **free_values):
    params_M = self.view.bind(**free_values)  # P → M via embedding
    # ... passes complete ParameterSet to BaseModel
```

This gives researchers ergonomic P-space operations while maintaining the architectural constraint that ParameterSet = complete M-space point only.

## 5. Composition Patterns

### Scenario Stacking

```python
# Scenarios compose with LWW semantics
base_scenario = ScenarioSpec("base", param_patches={"beta": 0.3})
lockdown = ScenarioSpec("lockdown", param_patches={"beta": 0.1, "contact": 2})

# Composition: base then lockdown
combined = compose(lockdown, base_scenario)
# Result: {"beta": 0.1, "contact": 2}  # lockdown's beta wins
```

### View Fixing

```python
# Views compose by accumulating fixed parameters
view1 = ParameterView.from_fixed(space, {"alpha": 0.5})
view2 = view1.fix(beta=0.3)
view3 = view2.fix(gamma=0.1)
# view3.fixed = {"alpha": 0.5, "beta": 0.3, "gamma": 0.1}
# view3.free = remaining parameters
```

### Transform Chaining

```python
# Transforms compose pointwise (per parameter)
natural_p = {"beta": 0.3, "gamma": 0.1}
z = to_transformed(T, natural_p)      # Natural → Optimizer space
p_back = from_transformed(T, z)       # Optimizer → Natural space
assert p_back ≈ natural_p             # Round-trip property
```

## 6. Reconciliation

When scenarios patch parameters that are free in a view, reconciliation moves them from free to fixed:

```python
view = ParameterView.from_fixed(space, {"alpha": 0.5})
# free = ["beta", "gamma", "contact_rate"]

scenario = ScenarioSpec("lockdown", param_patches={"contact_rate": 2.0})

# After reconciliation:
# fixed = {"alpha": 0.5, "contact_rate": 2.0}
# free = ["beta", "gamma"]
```

This prevents double-specification errors and ensures consistency.

## 7. Examples

### Calibration Workflow

```python
# 1. Define calibration subspace
view = ParameterView.from_fixed(space, {
    "population": 10000,
    "vaccination_rate": 0.7
})
# free = ["beta", "gamma", "contact_rate"]

# 2. Optimizer works in P-space
def objective(free_params):
    m = view.bind(**free_params)  # P → M
    results = model.simulate(m, seed=42)
    return compute_loss(results, targets)

# 3. Optimize over free parameters only
optimal_p = optimizer.minimize(objective, bounds=get_bounds(view.free))

# 4. Final model uses complete parameters
final_m = view.bind(**optimal_p)
```

### Scenario Application

```python
# Base parameters
base = ParameterSet(space, all_values)

# Apply scenario
lockdown = ScenarioSpec("lockdown",
    param_patches={"contact_rate": 2.0},
    config_patches={"schools": "closed"}
)

# Get modified parameters and config
params_new, config_new = lockdown.apply(base, base_config)
# params_new has contact_rate=2.0
# config_new has schools="closed"
```

## Summary

The Grammar of Parameters provides:

1. **Immutable primitives** that prevent state bugs
2. **Composable operators** with clear mathematical semantics
3. **Validated transformations** between spaces
4. **Clean separation** between M-space and P-space
5. **Ergonomic APIs** that maintain invariants

Every design choice reinforces the core principle: parameters are immutable
data that flow through pure functions, never mutated in place.
