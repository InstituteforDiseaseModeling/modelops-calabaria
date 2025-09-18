# Core Principles: One Model, Infinite Variants, One Wire Function

## Critical Caveat: Code Identity Matters

**⚠️ This entire architecture assumes the model code is immutable.** If the simulation engine changes (bug fixes, algorithm updates, refactoring), you have a DIFFERENT model that produces DIFFERENT results. This is why:

1. **Model identity = code hash**: Every model is identified by its content hash (e.g., `SIRModel@a1b2c3d4`)
2. **Bundle system required**: The `modelops-bundle` project captures and versions the exact code
3. **Cache invalidation**: When code changes, the hash changes, invalidating all cached results
4. **Reproducibility**: Results are only reproducible with the EXACT code version

Without proper code versioning via bundles, you could have:
- **Silent bugs**: Updated code returning cached results from old code
- **Irreproducibility**: Can't reproduce old results after code changes
- **Cache pollution**: Mixed results from different code versions

**The bundle system (separate project) solves this by:**
- Creating content-addressed bundles of model code
- Computing deterministic digests
- Shipping exact code from laptop to cloud
- Ensuring cache keys include code hash

## The Fundamental Insight

**Every possible variant of a model - regardless of fixed parameters, scenarios, or transforms - compiles to calls to the SAME wire function (for the SAME code version).**

This is the architectural keystone of Calabaria. It enables:
- **Deploy once, explore infinitely**: Ship model code once, explore any parameter subspace
- **Complete decoupling**: Research configuration is separate from model logic
- **Perfect caching**: Same inputs always produce same outputs (for same code)
- **Infinite scalability**: One wire function handles all variant combinations

## The Three Unities

### 1. One Model
A single `BaseModel` subclass contains all the computational logic:
- Parameter definitions (M-space)
- Simulation dynamics (`build_sim`, `run_sim`)
- Scenario definitions
- Output extractors

### 2. Infinite Variants
Unlimited research configurations through `VariantSpec`:
- Different fixed/free parameter splits
- Various scenario combinations
- Alternative parameter transforms
- Distinct output selections

### 3. One Wire Function
A single cloud entry point that accepts M-space parameters:
```python
def wire(params_M: Dict, seed: int, scenario_stack: Tuple, outputs: List) -> WireResponse
```

## The Core Principle in Action

### Local Research Environment
```python
# Researcher creates a variant for their specific analysis
spec = VariantSpec(
    view=ParameterView.from_fixed(model.space, {
        "population": 10000,
        "vaccination_rate": 0.7
    }),
    scenarios=("lockdown", "masks"),
    transforms={"beta": LogTransform()}  # For optimization
)

# They work in P-space (free parameters only)
runner = ModelRunner(model, spec)
result = runner.simulate({"beta": 0.3, "gamma": 0.1}, seed=42)
```

### Cloud Execution Environment
```python
# The SAME model wire function handles ALL variants
wire = REGISTRY.get_wire("examples.sir.SIRModel@a1b2c3d4")

# Variant 1: Different fixed parameters
response1 = wire(
    params_M={"population": 10000, "vaccination_rate": 0.7, "beta": 0.3, "gamma": 0.1},
    seed=42,
    scenario_stack=("lockdown", "masks"),
    outputs=None
)

# Variant 2: Different scenarios
response2 = wire(
    params_M={"population": 5000, "vaccination_rate": 0.0, "beta": 0.4, "gamma": 0.15},
    seed=42,
    scenario_stack=("baseline",),
    outputs=None
)

# Variant 3: Different everything - SAME WIRE FUNCTION
response3 = wire(
    params_M={"population": 100000, "vaccination_rate": 0.9, "beta": 0.2, "gamma": 0.2},
    seed=123,
    scenario_stack=("aggressive_intervention", "school_closure", "travel_ban"),
    outputs=["incidence", "hospitalizations"]
)
```

## The Translation Layer

The magic happens in the P→M translation:

```
┌─────────────────────────────────────────────────────────────┐
│                    LOCAL (Research)                         │
├─────────────────────────────────────────────────────────────┤
│  VariantSpec                                               │
│  ├── Fixed: {population: 10k}                              │
│  ├── Free: {beta, gamma}                                   │
│  └── Scenarios: ["lockdown"]                               │
│                                                             │
│  researcher.simulate({"beta": 0.3, "gamma": 0.1})          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    YAML/JSON Export
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLOUD (Execution)                        │
├─────────────────────────────────────────────────────────────┤
│  # Reconstruct M-space parameters                          │
│  params_M = {                                               │
│      **fixed_from_variant,  # {population: 10k}            │
│      **free_from_researcher  # {beta: 0.3, gamma: 0.1}     │
│  }                                                          │
│                                                             │
│  wire(params_M, seed, scenario_stack, outputs)             │
└─────────────────────────────────────────────────────────────┘
```

## Why This Matters

### For Researchers
- **Work in natural space**: Only specify parameters they're varying
- **Infinite flexibility**: Any combination of fixed/free/scenarios works
- **Zero deployment**: Never touch cloud infrastructure
- **Perfect reproducibility**: Same spec = same results

### For Infrastructure
- **Single deployment**: One model registration serves all variants
- **Perfect caching**: `hash(model + params + scenarios) = result`
- **No variant code**: Variants are just configuration, not code
- **Trivial scaling**: All variants go through same wire function

### For Science
- **Clean separation**: Model logic vs research configuration
- **Composability**: Scenarios compose, transforms compose, variants compose
- **Provenance**: Every result traces back to exact configuration
- **Collaboration**: Share variants as YAML/JSON, not code

## Common Misconceptions

### ❌ "Each variant needs its own wire function"
**Reality**: All variants use the same wire. The wire accepts complete M-space parameters; variants just specify how to construct those parameters.

### ❌ "Scenarios are different entry points"
**Reality**: Scenarios are data (patches) passed to the single wire function via the `scenario_stack` parameter.

### ❌ "Transforms happen in the cloud"
**Reality**: Transforms happen locally for optimization. The cloud always receives natural M-space parameters.

### ❌ "Fixed parameters are baked into deployed code"
**Reality**: Fixed parameters are configuration. The wire function receives ALL parameters every time.

## The Deployment Story

```python
# Step 1: Model developer creates model (ONCE)
class SIRModel(BaseModel):
    def __init__(self):
        space = ParameterSpace([
            ParameterSpec("beta", 0, 1),
            ParameterSpec("gamma", 0, 1),
            ParameterSpec("population", 100, 1000000, "int"),
            ParameterSpec("vaccination_rate", 0, 1),
        ])
        super().__init__(space)

    # ... implement dynamics ...

# Step 2: Create bundle and register model (ONCE per code version)
# Bundle captures exact code and dependencies, creates digest
bundle = Bundle.from_model(SIRModel)  # Creates digest: "a1b2c3d4..."
bundle.push()  # Uploads to cloud

# Step 3: Register model with its bundle digest
SIRModel.compile_entrypoint(
    alias="SIR with Vaccination",
    bundle_digest=bundle.digest  # Links model to exact code version
)

# Step 3: Infinite research variations (NO DEPLOYMENT)
variant_A = VariantSpec(...)  # Studying urban dynamics
variant_B = VariantSpec(...)  # Studying rural dynamics
variant_C = VariantSpec(...)  # Studying vaccination strategies
variant_D = VariantSpec(...)  # Studying seasonal effects
# ... unlimited variants, same wire function ...
```

## Implementation Hierarchy

```
┌──────────────────────────────────────┐
│          BaseModel (Logic)           │  ← Deployed once
├──────────────────────────────────────┤
│      Wire Function (Interface)       │  ← Single entry point
├──────────────────────────────────────┤
│       VariantSpec (Configuration)    │  ← Infinite variations
├──────────────────────────────────────┤
│      ModelRunner (Local Execution)   │  ← Research convenience
└──────────────────────────────────────┘
```

## Cache Efficiency

Because all variants go through the same wire function:

```python
# These two research approaches produce IDENTICAL cache keys:

# Approach 1: Fixed population in variant
variant1 = VariantSpec(
    view=ParameterView.from_fixed(space, {"population": 10000}),
    scenarios=("lockdown",)
)
runner1.simulate({"beta": 0.3, "gamma": 0.1}, seed=42)

# Approach 2: All parameters free
variant2 = VariantSpec(
    view=ParameterView.all_free(space),
    scenarios=("lockdown",)
)
runner2.simulate({"population": 10000, "beta": 0.3, "gamma": 0.1}, seed=42)

# Both compile to the SAME wire call:
wire(
    params_M={"population": 10000, "beta": 0.3, "gamma": 0.1, "vaccination_rate": 0},
    seed=42,
    scenario_stack=("lockdown",),
    outputs=None
)

# Therefore, the cache key includes:
# - Model hash (e.g., "a1b2c3d4" from bundle digest)
# - Parameter values (complete M-space)
# - Scenario stack
# - Seed
# cache_key = hash(model_hash + params_M + scenario_stack + seed)
# Result: SAME cache key, SAME cached result (for same code version)
```

## The Purity Principle

The wire function is **pure**:
- **Deterministic**: Same inputs → same outputs
- **Stateless**: No hidden state between calls
- **Side-effect free**: Only computation, no I/O
- **Complete**: All inputs explicit in signature

This purity is what enables:
- Perfect caching
- Trivial parallelization
- Time-travel debugging
- Reproducible science

## Practical Benefits

### Development Workflow
1. **Model developer** writes and tests `BaseModel` locally
2. **Model developer** deploys model once with `compile_entrypoint()`
3. **Researchers** create unlimited variants without deployment
4. **Researchers** share variants as configuration files
5. **Cloud** executes any variant through the single wire

### Optimization Workflow
1. **Researcher** defines P-space with `VariantSpec`
2. **Optimizer** works in transformed P-space locally
3. **Each iteration** translates P→M for simulation
4. **Cloud** receives only M-space parameters
5. **Results** flow back through same path

### Collaboration Workflow
```yaml
# Researcher A shares their variant as YAML
variant:
  model_id: "examples.sir.SIRModel@a1b2c3d4"
  fixed_params:
    population: 10000
    vaccination_rate: 0.7
  scenarios:
    - lockdown
    - masks
  transforms:
    beta: log
    gamma: logit

# Researcher B can reproduce EXACTLY without code changes
```

## Design Invariants

These invariants MUST be maintained:

1. **Wire functions accept ONLY M-space parameters** (complete)
2. **Variants are pure configuration** (no code)
3. **Scenarios are data** (patches, not logic)
4. **Transforms are local** (optimization, not execution)
5. **One model class → One wire function** (no proliferation)

## Performance Implications

### What's Fast
- **Cache hits**: Identical (model, params, scenarios) → instant
- **Parallel variants**: All independent, perfect scaling
- **Scenario composition**: Just dictionary merging
- **Parameter validation**: One-time at wire entry

### What's Slower (But Worth It)
- **P→M translation**: Small overhead for huge flexibility
- **Scenario stack application**: Linear in stack depth
- **Output filtering**: Happens after simulation

## Security Benefits

- **No code injection**: Variants can't introduce code
- **Parameter validation**: Wire function validates everything
- **Scenario validation**: Only registered scenarios allowed
- **Output filtering**: Only registered outputs accessible

## Future Extensions That Preserve The Principle

### ✅ Compatible Extensions
- **New scenarios**: Just add to model, same wire
- **New outputs**: Just add extractors, same wire
- **New parameters**: Extend M-space, same wire pattern
- **Reparameterizations**: Still compile to M-space calls

### ⚠️ Incompatible Changes (Don't Do These)
- **Variant-specific wires**: Breaks the principle
- **Dynamic scenario code**: Scenarios must be data
- **Parameter preprocessing in cloud**: Keep cloud pure
- **Stateful variants**: All state must be explicit

## Summary

The "One Model, Infinite Variants, One Wire Function" principle is not just an implementation detail - it's the foundational architecture that enables:

1. **Separation of concerns** between model logic and research configuration
2. **Infinite flexibility** without deployment complexity
3. **Perfect reproducibility** through pure functions
4. **Efficient caching** through canonical representations
5. **Seamless scaling** through stateless execution

Every design decision should be evaluated against this principle: **Does it maintain the unity of the wire function while enabling variant flexibility?**

If yes, proceed. If no, reconsider.

## The Slogan

> **"Write Once, Explore Everywhere, Cache Everything"**

- Write the model once
- Explore any parameter subspace
- Cache every unique computation

This is the promise of Calabaria's architecture.