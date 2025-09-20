# Core Principles: One BaseModel, Infinite Variants (and One Wire Function)

## The Fundamental Insight

**Most common "variants" of a model a user will want to create --regardless of
fixed parameters, scenarios, or transforms-- compile to calls to the SAME wire
function (for the same code version).**

This is the architectural keystone of Calabaria. It enables:

- **Deploy once, explore infinitely**: Ship model code once, explore any
  parameter subspace
- **Complete decoupling**: Research configuration is separate from model logic
- **Perfect caching**: Same inputs always produce same outputs (for same code)
- **Infinite scalability**: One wire function handles all variant combinations

## The Software Design of BaseModel

Calabaria's `BaseModel` class function is the most important in Calabaria, and
it's worth explaining the underlying idea and what it brings. In software
engineering terms, `BaseModel` is a *Facade-style Service Provider Interface
(SPI)*. It's worth unpacking this jargon and explaining the two facets of
`BaseModel`

 1. **Research-user-facing facade**: a *facade* class serves as a simple,
    front-facing interface to mask complex operations in highly usable layer.
    The `BaseModel` class masks engine specifics behind a thin shell, leaving
    only *ergonomic* modeling-specific semantics exposed, such as parameter
    space adjustments, scenarios, transforms, and reparameterizations. Part of
    `BaseModel`'s design is as a *template method* host: high-level methods
    like `simulate()` are *fixed*, as they are constructed from subclass's
    `build_sim()` and `run_sim()` methods automatically.

 2. **Service Provider Interface**: behind the `BaseModel` facade is a Service
    Provider Interface. Specifically, we can think of this as a *port* to a
    service (simulation of a model, through the `simulate()` method) that is
    equipped with all the extras outside frameworks need to interact with the
    underlying model (such as the parameters, their bounds, etc). **Any outside
    framework that needs to interact with a simulation model can do so by
    *calling into* the model via its `BaseModel` class.** This is classic
    *inversion of control* (or the "Hollywood Principle: don’t call us, we’ll
    call you"). This allows an outside framework (e.g. for calibration) to
    orchestrate calls to the simulation model externally -- exactly what they
    need to do!


## The Three Unities

### 1. One Model A single `BaseModel` subclass contains all the computational
logic:
- Parameter definitions (the *model parameter space*, or M-space)
- Simulation dynamics (`build_sim`, `run_sim`)
- Scenario definitions
- Output extractors

### 2. Infinite Variants

Unlimited research configurations through `VariantSpec`:
- Different fixed/free parameter splits (which are a subspace of M-space, known
  as the P-space)
- Various scenario combinations
- Alternative parameter transforms
- Distinct output selections

### 3. One Wire Function

A single cloud entry point that only accepts M-space parameters:

```python
def wire(params_M: Dict, seed: int, scenario_stack: Tuple, outputs: List) -> WireResponse
```

There are many benefits of keeping all simulations in the *full* model parameter
space (M-space):

- **Fixing a parameter does not invalidate the cache**: Whether beta is fixed
  or free, the wire call with beta=0.3 produces the same result and hits the
  same cache entry
- **Research flexibility without redeployment**: Any researcher can fix any
  parameter combination without changing the deployed code
- **Consistent provenance**: The same (model_hash, params_M, scenarios, seed)
  tuple always produces the same result, regardless of how you arrived at those
  parameters
- **Simplified infrastructure**: Cloud systems only need to understand M-space,
  not the complexities of different P-space configurations
- **Perfect result sharing**: Two researchers using different fixed/free splits
  but the same final parameters get identical results from cache
- **Deterministic identity**: Cache keys depend only on actual computation
  inputs, not on research methodology

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
variant = ModelVariant(model, spec)
result = variant.simulate(seed=42, beta=0.3, gamma=0.1)
```

### Cloud Execution Environment

```python
# The SAME model wire function handles ALL variants
# Created on-demand from manifest data
wire = make_wire_from_manifest("examples.sir.SIRModel@a1b2c3d4", manifest)

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
│  VariantSpec                                                │
│  ├── Fixed: {population: 10k}                               │
│  ├── Free: {beta, gamma}                                    │
│  └── Scenarios: ["lockdown"]                                │
│                                                             │
│  researcher.simulate({"beta": 0.3, "gamma": 0.1})           │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    YAML/JSON Export
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    CLOUD (Execution)                        │
├─────────────────────────────────────────────────────────────┤
│  # Reconstruct M-space parameters                           │
│  params_M = {                                               │
│      **fixed_from_variant,  # {population: 10k}             │
│      **free_from_researcher  # {beta: 0.3, gamma: 0.1}      │
│  }                                                          │
│                                                             │
│  wire(params_M, seed, scenario_stack, outputs)              │
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

**Reality**: All variants use the same wire. The wire accepts complete M-space
parameters; variants just specify how to construct those parameters. Only when
the underlying model code changes or alters its M-space model parameters do we
need a new wire function. These changes are not *model variants* but *different
models*.

### ❌ "Scenarios are different entry points"

**Reality**: Scenarios are data (config or parameter patches) passed to the
single wire function via the `scenario_stack` parameter.

### ❌ "Transforms happen in the cloud"

**Reality**: Transforms happen locally (when it is more natural for a user to
work in a transformed parameter space) or for optimization (when it is more
natural to calibrate in a transformed parameter space). The cloud simulation
service always receives natural M-space parameters. 

### ❌ "Fixed parameters are baked into deployed code"

**Reality**: Fixed parameters are configuration. The wire function receives
*all* parameters every time (in M-space).

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

# Step 2: Create bundle and manifest (ONCE per code version)
# Bundle captures exact code and dependencies, creates digest
bundle = Bundle.from_model(SIRModel)  # Creates digest: "a1b2c3d4..."
bundle.push()  # Uploads to cloud

# Step 3: Create manifest entry with bundle digest
# Manifest is the single source of truth for model metadata
manifest = {
    "models": {
        "SIRModel@a1b2c3d4": {
            "class": "examples.sir:SIRModel",
            "model_digest": bundle.digest,
            "param_specs": [...],  # Serialized parameter specifications
            "scenarios": [...],    # Available scenarios
            "outputs": [...]       # Available outputs
        }
    }
}

# Step 4: Infinite research variations (NO DEPLOYMENT)
variant_A = VariantSpec(...)  # Studying urban dynamics
variant_B = VariantSpec(...)  # Studying rural dynamics
variant_C = VariantSpec(...)  # Studying vaccination strategies
variant_D = VariantSpec(...)  # Studying seasonal effects
# ... unlimited variants, same wire function ...
```

## Implementation Hierarchy

```
┌──────────────────────────────────────┐
│         Manifest (Discovery)         │  ← Single source of truth
├──────────────────────────────────────┤
│          BaseModel (Logic)           │  ← Deployed once
├──────────────────────────────────────┤
│      Wire Function (Interface)       │  ← Single entry point (stateless)
├──────────────────────────────────────┤
│       VariantSpec (Configuration)    │  ← Infinite variations
├──────────────────────────────────────┤
│      ModelVariant (Local Execution)  │  ← Research convenience
└──────────────────────────────────────┘
```

## Stateless Wire Loader: Enabling Discovery and Deployment

The stateless wire loader enables "One Model, Infinite Variants" without global state:

### Manifest-Driven Discovery

```python
# Manifest contains all model metadata
manifest = {
    "models": {
        "examples.sir.SIRModel@a1b2c3d4": {
            "class": "examples.sir:SIRModel",
            "model_digest": "sha256:a1b2c3d4...",
            "param_specs": [
                {"name": "beta", "min": 0.0, "max": 1.0, "kind": "float"},
                {"name": "gamma", "min": 0.0, "max": 1.0, "kind": "float"}
            ],
            "scenarios": ["baseline", "lockdown", "vaccination"],
            "outputs": ["incidence", "prevalence", "summary"]
        }
    }
}

# Create wire function on-demand
wire = make_wire_from_manifest("examples.sir.SIRModel@a1b2c3d4", manifest)
```

### EntryRecord: Pure Value Object
Each model has an immutable `EntryRecord` containing:
- **Identity**: Class path and model digest (e.g., `examples.sir:SIRModel`)
- **ABI Version**: Wire protocol version for compatibility (e.g., `calabria.wire.v1`)
- **Import Path**: Module and class names (no heavy closures)
- **Discovery Info**: Available scenarios, outputs, parameter specs
- **Stateless**: No global state or registry

This design ensures:
- **No global state**: Wire functions created from manifest data
- **Full serialization**: Everything can be JSON-encoded for persistence
- **Version safety**: ABI versioning allows protocol evolution
- **Lazy loading**: Models imported only when needed
- **Stateless execution**: Fresh model instances for each call

### Wire Protocol Hardening

The production wire protocol addresses critical issues:

1. **Protocol Versioning via WireABI**
   ```python
   class WireABI(str, Enum):
       V1 = "calabaria.wire.v1"  # (params_M, seed, scenario_stack, outputs)
       V2 = "calabaria.wire.v2"  # Future: adds config overrides
   ```
   Different versions can coexist, enabling protocol evolution without breaking deployments.

2. **Serializable Parameter Specifications**
   ```python
   @dataclass(frozen=True)
   class SerializedParameterSpec:
       """Guaranteed JSON-serializable specification."""
       name: str
       min: float
       max: float
       kind: str  # "float" or "int"
   ```
   Unlike runtime ParameterSpec, these are designed for storage and transmission.

3. **Import Path Pattern (No Heavy Closures)**
   ```python
   def make_wire(entry: EntryRecord) -> Callable:
       """Creates wire function from entry record."""
       # Parse class path
       module_name, class_name = entry.class_path.split(":")

       def wire_v1(params_M, seed, scenario_stack=("baseline",), outputs=None):
           # Import model fresh for each call (stateless)
           module = importlib.import_module(module_name)
           model_class = getattr(module, class_name)
           # ... execute simulation ...
       return wire_v1
   ```
   Models are imported fresh for each call, ensuring stateless execution.

4. **Stateless Execution**
   ```python
   # No global registry - everything from manifest
   entry = entry_from_manifest(model_id, manifest)
   wire = make_wire(entry)
   # Each call creates fresh model instance
   ```

### Manifest Persistence and Federation

```python
# Manifest is already JSON-serializable
manifest = {
    "models": {
        "examples.sir.SIRModel@a1b2c3d4": {
            "class": "examples.sir:SIRModel",
            "model_digest": "sha256:...",
            # ... metadata ...
        }
    }
}

with open("model_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)

# Load in cloud environment - no global state needed
with open("model_manifest.json") as f:
    manifest = json.load(f)

# Execute any model - created on-demand
wire = make_wire_from_manifest("examples.sir.SIRModel@a1b2c3d4", manifest)
response = wire(
    params_M={...},  # Always complete M-space
    seed=42,
    scenario_stack=("baseline", "lockdown"),
    outputs=["incidence"]
)
```

### Wire Response Format

The wire function returns a standardized `WireResponse`:

```python
@dataclass(frozen=True)
class WireResponse:
    """Standardized response from any wire function."""
    outputs: Dict[str, bytes]        # Output name → Arrow IPC bytes
    provenance: Dict[str, Any]       # Execution metadata

    def get_dataframe(self, name: str) -> pl.DataFrame:
        """Deserialize specific output to DataFrame."""
        return pl.read_ipc(io.BytesIO(self.outputs[name]))
```

Provenance includes:
- Model identity and hash
- Complete parameters used
- Scenario stack applied
- Seed and timestamp
- ABI version for reproducibility

This standardized format ensures:
- **Efficient serialization**: Arrow IPC for DataFrames
- **Complete provenance**: Full audit trail
- **Language agnostic**: Can be consumed by any system
- **Cache-friendly**: Deterministic serialization

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
2. **Model developer** creates manifest entry with model metadata
3. **Researchers** create unlimited variants without deployment
4. **Researchers** share variants as configuration files
5. **Cloud** creates wire functions on-demand and executes variants

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

## Production Deployment Flow

The complete deployment story leveraging all components:

```python
# 1. Developer writes model ONCE
class EpidemicModel(BaseModel):
    def __init__(self):
        # Define complete parameter space
        super().__init__(ParameterSpace([...]))

    def build_sim(self, params, config):
        # Model dynamics
        pass

# 2. Bundle and create manifest (ONCE per code version)
bundle = Bundle.from_model(EpidemicModel)
bundle.push()  # Upload to cloud storage

# Create manifest entry
manifest = {
    "models": {
        "EpidemicModel@abc123": {
            "class": "epidemic.models:EpidemicModel",
            "model_digest": bundle.digest,
            "param_specs": [...],  # Model parameter specifications
            "scenarios": [...],    # Available scenarios
            "outputs": [...]       # Available outputs
        }
    }
}

# 3. Manifest persists to cloud
with open("manifest.json", "w") as f:
    json.dump(manifest, f)
# Upload to cloud storage/database

# 4. Cloud execution creates wire functions on-demand
with open("manifest.json") as f:
    manifest = json.load(f)

# 5. Any researcher can now use ANY variant
wire = make_wire_from_manifest("EpidemicModel@abc123", manifest)

# Variant A: Urban dynamics
response_a = wire(
    params_M={...},  # Complete parameters for urban scenario
    seed=42,
    scenario_stack=("urban", "high_density"),
    outputs=["hospitalizations"]
)

# Variant B: Rural dynamics (SAME WIRE)
response_b = wire(
    params_M={...},  # Complete parameters for rural scenario
    seed=43,
    scenario_stack=("rural", "low_density"),
    outputs=["cases", "deaths"]
)

# Both hit cache if parameters match previous runs
```

## Summary

The "One Model, Infinite Variants, One Wire Function" principle is not just an implementation detail - it's the foundational architecture that enables:

1. **Separation of concerns** between model logic and research configuration
2. **Infinite flexibility** without deployment complexity
3. **Perfect reproducibility** through pure functions and versioned bundles
4. **Efficient caching** through canonical M-space representations
5. **Seamless scaling** through stateless, serializable execution
6. **Production safety** through stateless execution and ABI versioning

Every design decision should be evaluated against this principle: **Does it maintain the unity of the wire function while enabling variant flexibility?**

The stateless wire protocol ensures this principle scales to production:
- **No heavy closures** → Import paths and on-demand creation
- **Protocol evolution** → ABI versioning
- **No global state** → Manifest-driven, stateless execution
- **Full persistence** → JSON serialization

## The Architecture Promise

> **"Write Once, Explore Everywhere, Cache Everything"**

This is achieved through:
- **Write once**: Single BaseModel implementation
- **Deploy once**: One wire function per code version
- **Configure infinitely**: Unlimited variants via VariantSpec
- **Cache perfectly**: Deterministic M-space identity
- **Scale effortlessly**: Stateless, pure execution

The combination of:
- Immutable parameter system
- Pure wire functions
- Content-addressed bundles
- Stateless execution
- Versioned protocols

Creates a system where **research flexibility meets production reliability**.

This is the promise of Calabaria's architecture.
