# ModelOps-Calabaria

[![Tests](https://github.com/institutefordiseasemodeling/modelops-calabaria/actions/workflows/tests.yml/badge.svg)](https://github.com/institutefordiseasemodeling/modelops-calabaria/actions/workflows/tests.yml)

A science framework for distributed epidemic modeling and calibration on ModelOps infrastructure.

## What is Calabaria?

Calabaria provides the modeling framework layer for ModelOps, enabling:
- **Parameter space exploration** with Sobol and grid sampling
- **Scenario-based modeling** with parameter overrides
- **Model outputs** extraction and aggregation
- **Calibration targets** for model fitting
- **Wire protocol** for distributed execution
- **Automatic imports** - no PYTHONPATH configuration needed!

It implements the contracts defined in `modelops-contracts` to bridge epidemic models with the ModelOps infrastructure.

## Installation

```bash
pip install git+https://github.com/institutefordiseasemodeling/modelops-calabaria.git
```

Or for development:
```bash
git clone https://github.com/institutefordiseasemodeling/modelops-calabaria.git
cd modelops-calabaria
pip install -e .
```

**Requirements**: Python 3.12+

## Quick Start

### Starsim-SIR CLI flow (no prompts, no flags)

The Starsim SIR example in `modelops/examples/starsim-sir` now works end-to-end
with four commands—no `PYTHONPATH` tricks, no `--outputs`, and no confirmation
prompts.

1. **Register the model.** We auto-discover the outputs directly from the
   decorators, so nothing extra is required.

```console
$ mops bundle register-model models/sir.py
+ sir_starsimsir       entry=models.sir:StarsimSIR
✓ Models updated: +1 ~0 -0
```

2. **See what’s in the bundle.** The table makes it obvious that we now have a
   model but no targets yet.

```console
$ mops bundle list
                                      Registered Models (1)
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Model          ┃ Entrypoint            ┃ Outputs                           ┃ Labels ┃ Aliases ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ sir_starsimsir │ models.sir:StarsimSIR │ incidence, prevalence, cumulative │ -      │ -       │
└────────────────┴───────────────────────┴───────────────────────────────────┴────────┴─────────┘

  (no targets)
```

3. **Register targets (with regeneration).** Again, pure autodetection.

```console
$ mops bundle register-target --regen-all targets/incidence.py
+ incidence_per_replicate_target entry=targets.incidence:incidence_per_replicate_target
+ incidence_replicate_mean_target entry=targets.incidence:incidence_replicate_mean_target
✓ Targets updated: +2 ~0 -0
```

4. **Confirm everything is wired up.**

```console
$ mops bundle list
                                      Registered Models (1)
┏━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┓
┃ Model          ┃ Entrypoint            ┃ Outputs                           ┃ Labels ┃ Aliases ┃
┡━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━┩
│ sir_starsimsir │ models.sir:StarsimSIR │ incidence, prevalence, cumulative │ -      │ -       │
└────────────────┴───────────────────────┴───────────────────────────────────┴────────┴─────────┘

                                                 Registered Targets (2)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┓
┃ Target                          ┃ Entrypoint                                        ┃ Model Output ┃ Labels ┃ Weight ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━┩
│ incidence_per_replicate_target  │ targets.incidence:incidence_per_replicate_target  │ incidence    │ -      │ -      │
│ incidence_replicate_mean_target │ targets.incidence:incidence_replicate_mean_target │ incidence    │ -      │ -      │
└─────────────────────────────────┴───────────────────────────────────────────────────┴──────────────┴────────┴────────┘
```

5. **Generate samples directly from the CLI.** The `cb` command handles the
   import pathing, uses the bundle registry to resolve the model, and emits a
   friendly summary.

```console
$ cb sampling sobol sir_starsimsir --n-samples 1024 --name sobol --n-replicates 100
Resolved model id 'sir_starsimsir' → models.sir:StarsimSIR
Generated 1024 Sobol samples for 2 parameters
Using default output path sobol.json (set --output to override)
✓ Generated SimulationStudy with 1024 parameter sets

Study Summary
  Name       : sobol
  Model      : sir_starsimsir → models.sir:StarsimSIR (scenario=baseline)
  Sampling   : Sobol (scramble=on, seed=42)
  Parameters : 1024 sets × 100 replicates = 102,400 simulations
  Tags       : -
  Output     : sobol.json
  Parameter Space:
    • beta ∈ [0.01, 0.2]
    • dur_inf ∈ [3.0, 10.0]
```

6. **Submit the study** (handled by the `modelops` CLI, but shown here so you
   can see the exact output users will experience).

```console
$ mops jobs submit sobol.json

Loading job specification
  Type: SimulationStudy
  Model: models.sir/baseline
  Sampling: sobol
  Parameters: 1024 unique sets
  Replicates: 100 per parameter set
  Total simulations: 102400

Auto-pushing bundle
  Building and pushing bundle from current directory...
Successfully pushed modelopsdevacrvsb.azurecr.io/starsim-sir:latest
  ✓ Pushed bundle: starsim-sir@sha256:b94198d364c820303701615e702066f...

Submitting simulation job

✓ Job submitted successfully!
  Job ID: job-47179d43
  Environment: dev
  Status: Running

 To monitor job execution:
  # Port-forward to access Dask dashboard (run in separate terminals or use &)
  kubectl port-forward -n modelops-dask-dev svc/dask-scheduler 8787:8787 &
  kubectl port-forward -n modelops-dask-dev svc/dask-scheduler 8786:8786 &
  # Then open http://localhost:8787 in your browser

 To check job status:
  kubectl -n modelops-dask-dev get job job-47179d43

 To see logs:
  kubectl -n modelops-dask-dev logs job/job-47179d43
  kubectl -n modelops-dask-dev logs deployment/dask-workers
```

That's the entire happy path; the rest of this README covers the Python API if
you prefer to integrate directly.

## Fluent Builder API (Grammar of Parameters)

Calabaria implements the **Grammar of Model Parameters**, a formal framework for working with parameter spaces, coordinate transformations, and model simulators. The fluent builder API provides an expressive way to create calibration-ready simulators:

```python
import numpy as np
from modelops_calabaria import StochasticSEIR

# Create model
model = StochasticSEIR()

# Build simulator with fluent API
sim = (model
       .builder("baseline")                              # Select scenario
       .fix(population=100000, initial_infected=10)     # Fix some parameters
       .with_transforms(beta="log", gamma="log")        # Transform others
       .build())                                         # Create ModelSimulator

# Now sim is a callable: z × seed → outputs
# z is in transformed space (log-space for beta, gamma)
z = np.array([np.log(0.5), np.log(0.2)])  # log(beta), log(gamma)
outputs = sim(z, seed=42)

print(f"Dimension: {sim.dim}")                # 2 (only beta, gamma free)
print(f"Free parameters: {sim.free_param_names}")  # ('beta', 'gamma')
print(f"Bounds (transformed): {sim.bounds()}")     # Bounds in log-space
```

### Why Transforms?

Transforms map between natural parameter space and inference space:

- **`"log"`**: For positive parameters (rates, counts) - maps (0, ∞) → (-∞, ∞)
- **`"logit"`**: For probabilities [0,1] - maps (0, 1) → (-∞, ∞)
- **`"identity"`**: No transformation (default)

Benefits for optimization/calibration:
- Unbounded inference space (easier for optimizers)
- Normalized scales (better gradient behavior)
- Automatic constraint satisfaction (rates stay positive, probabilities in [0,1])
- Uniform sampling in inference space → good coverage in natural space

### Complete Examples

See comprehensive examples in [`examples/`](examples/):

- **[`fluent_api_complete.py`](examples/fluent_api_complete.py)** - Complete workflow with scenarios, transforms, and reusable builders
- **[`coordinate_system_demo.py`](examples/coordinate_system_demo.py)** - Deep dive into coordinate transforms and their effects
- **[`epi_models/src/models/seir.py`](examples/epi_models/src/models/seir.py)** - Production SEIR model using the fluent API

### 1. Import the Framework

```python
import modelops_calabaria as cb
import polars as pl

# Everything you need is available through 'cb'
# cb.BaseModel, cb.ParameterSpec, cb.ParameterSpace, etc.
```

### 2. Define Your Model

```python
class StochasticSEIR(cb.BaseModel):
    """Example SEIR epidemic model."""

    @classmethod
    def parameter_space(cls):
        """Define parameter ranges for exploration."""
        return cb.ParameterSpace([
            cb.ParameterSpec("beta", 0.1, 2.0, "float", doc="Transmission rate"),
            cb.ParameterSpec("sigma", 0.05, 0.5, "float", doc="Incubation rate"),
            cb.ParameterSpec("gamma", 0.05, 0.5, "float", doc="Recovery rate"),
            cb.ParameterSpec("population", 10000, 100000, "int"),
            cb.ParameterSpec("initial_infected", 1, 10, "int"),
        ])

    def build_sim(self, params: cb.ParameterSet, config):
        """Prepare simulation state from parameters."""
        N = int(params["population"])
        I0 = int(params["initial_infected"])

        return {
            "initial_state": {"S": N - I0, "E": 0, "I": I0, "R": 0},
            "params": dict(params.values),
            "config": dict(config)
        }

    def run_sim(self, state, seed: int):
        """Run the simulation with given state and seed."""
        # Your simulation logic here
        import numpy as np
        rng = np.random.RandomState(seed)

        # ... simulation code ...

        return {
            "times": list(range(100)),
            "infected": [10, 15, 22, 35, 50, ...],  # Your results
            "recovered": [0, 0, 1, 3, 5, ...]
        }

    @cb.model_output("prevalence")
    def extract_prevalence(self, raw, seed):
        """Extract infection prevalence time series."""
        return pl.DataFrame({
            "day": raw["times"],
            "infected": raw["infected"]
        })

    @cb.model_output("summary")
    def extract_summary(self, raw, seed):
        """Extract summary statistics."""
        return pl.DataFrame({
            "metric": ["peak_infections", "final_size"],
            "value": [max(raw["infected"]), raw["recovered"][-1]]
        })
```

### 3. Generate Parameter Samples

**No PYTHONPATH needed!** Calabaria automatically handles imports:

```bash
# Sobol sampling - works from your project directory
cb sampling sobol "models.seir:StochasticSEIR" \
  --n-samples 256 \
  --n-replicates 10 \
  --scenario baseline \
  --output study.json

# Grid sampling
cb sampling grid "models.seir:StochasticSEIR" \
  --grid-points 5 \
  --output grid-study.json

# File path syntax also works
cb sampling sobol "./models/seir.py:StochasticSEIR" \
  --n-samples 100 \
  --output study.json
```

### 4. Submit to ModelOps

```bash
# Register your model with the bundle system
modelops-bundle register-model models/seir.py

# Submit study for distributed execution
mops jobs submit study.json --auto
```

## Key Features

### Clean Python API

```python
import modelops_calabaria as cb

# Everything is available through the 'cb' namespace
model = MyModel(cb.BaseModel)
space = cb.ParameterSpace([...])
params = cb.ParameterSet(space, {...})

# Sampling
sampler = cb.SobolSampler(space)
samples = sampler.sample(n=100)

```

### Sampling Strategies

```python
# Programmatic sampling
sampler = cb.SobolSampler(space, scramble=True, seed=42)
samples = sampler.sample(n=256)

# Grid sampling
grid = cb.GridSampler(space, n_points_per_param=5)
grid_samples = grid.sample()
```

## CLI Reference

### Sampling

```bash
# Sobol sampling with options
cb sampling sobol "models.seir:MyModel" \
  --n-samples 512 \
  --n-replicates 10 \
  --output study.json

# Grid sampling
cb sampling grid "models.seir:MyModel" \
  --grid-points 10 \
  --output grid.json
```

Tip: inside a ModelOps bundle you can also reference models by their registry IDs
(e.g., `cb sampling sobol sir_starsimsir`), and Calabaria will look them up in
`.modelops-bundle/registry.yaml` automatically.

### Calibration

```bash
# Build an Optuna calibration spec
cb calibration optuna "models.seir:MyModel" \
  data/observed_incidence.parquet \
  beta:0.2:1.0,gamma:0.05:0.3,sigma:0.05:0.4 \
  --target-set incidence \
  --max-trials 500 \
  --n-replicates 16 \
  --name seir-calibration \
  --output studies/seir-calibration.json
```

Target metadata comes from `.modelops-bundle/registry.yaml`. Use `--target-set
<name>` to reference a named group created via `mops-bundle target-set set`, or
repeat `--target <id>` to select specific target IDs. If you omit both flags,
all registered targets are included.

### Diagnostics

```bash
# Generate a diagnostics PDF from a ModelOps results parquet
cb diagnostics report results/optuna_results.parquet --output reports/study.pdf
```

## Integration with ModelOps

Calabaria works seamlessly with the ModelOps ecosystem:

- **[modelops-contracts](https://github.com/institutefordiseasemodeling/modelops-contracts)**: Protocol definitions
- **[modelops-bundle](https://github.com/institutefordiseasemodeling/modelops-bundle)**: Model packaging
- **[modelops](https://github.com/institutefordiseasemodeling/modelops)**: Infrastructure orchestration

## Examples

See the [ModelOps examples](https://github.com/institutefordiseasemodeling/modelops/tree/main/examples) for complete working models.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type checking
mypy src/modelops_calabaria
```

## License

MIT



## Documentation

Design docs live under [`docs/`](docs/). See [`docs/index.md`](docs/index.md) for the curated list of active specs plus the archived planning notes.
