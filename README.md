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

### 🎯 Automatic Import Handling

No more `PYTHONPATH` configuration! Calabaria automatically finds your models:

```bash
# Module path - automatically adds cwd to Python path
cb sampling sobol "models.seir:MyModel"

# File path - loads directly without sys.path changes
cb sampling sobol "./models/seir.py:MyModel"

# Control import behavior if needed
cb sampling sobol "models.seir:MyModel" --no-cwd-import  # Requires PYTHONPATH
cb sampling sobol "models.seir:MyModel" --project-root /path/to/project
```

### 📊 Clean Python API

```python
import modelops_calabaria as cb

# Everything is available through the 'cb' namespace
model = MyModel(cb.BaseModel)
space = cb.ParameterSpace([...])
params = cb.ParameterSet(space, {...})

# Sampling
sampler = cb.SobolSampler(space)
samples = sampler.sample(n=100)

# Utilities
ModelClass = cb.load_symbol("models.seir:StochasticSEIR")
```

### 🔄 Sampling Strategies

```python
# Programmatic sampling
sampler = cb.SobolSampler(space, scramble=True, seed=42)
samples = sampler.sample(n=256)

# Grid sampling
grid = cb.GridSampler(space, n_points_per_param=5)
grid_samples = grid.sample()
```

## CLI Reference

### Model Commands

```bash
# Discover models in your project
cb models discover

# Export model configuration
cb models export "models.seir:StochasticSEIR" --files "models/*.py"

# Verify import boundaries
cb models verify
```

### Sampling Commands

```bash
# Sobol sampling with options
cb sampling sobol "models.seir:MyModel" \
  --n-samples 512 \
  --n-replicates 10 \
  --scenario high_transmission \
  --output study.json

# Grid sampling
cb sampling grid "models.seir:MyModel" \
  --grid-points 10 \
  --output grid.json
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


