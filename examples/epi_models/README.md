# Example Epidemiological Models

This directory contains example epidemiological models for testing and demonstrating the ModelOps-Calabria framework. The models implement canonical stochastic SEIR (Susceptible-Exposed-Infected-Recovered) compartmental models commonly used in epidemiology.

## Models

### 1. StochasticSEIR (`src/models/seir.py`)

A basic stochastic SEIR compartmental model with:

**Parameters:**
- `beta`: Transmission rate (0.1 - 2.0)
- `sigma`: Incubation rate (0.05 - 0.5)
- `gamma`: Recovery rate (0.05 - 0.5)
- `population`: Total population (1,000 - 1,000,000)
- `initial_infected`: Initial infections (1 - 100)
- `initial_exposed`: Initial exposed (0 - 100)
- `simulation_days`: Simulation duration (100 - 365 days)

**Outputs:**
- `prevalence`: Daily infected count
- `incidence`: Daily new infections
- `compartments`: Full SEIR time series
- `summary`: Epidemic summary statistics

**Scenarios:**
- `baseline`: Default parameters
- `lockdown`: 50% reduction in transmission
- `high_transmission`: 50% increase in transmission

### 2. AgeStratifiedSEIR (`src/models/seir_age.py`)

An age-structured SEIR model with three age groups (children 0-17, adults 18-64, elderly 65+):

**Additional Parameters:**
- `contact_matrix_scale`: Scale factor for age-specific contact patterns
- `child_susceptibility`: Relative susceptibility of children
- `elderly_susceptibility`: Relative susceptibility of elderly
- `initial_infected_*`: Initial infections by age group

**Additional Outputs:**
- `prevalence_by_age`: Infections by age group
- `total_infections`: Aggregate infections
- `incidence_by_age`: New infections by age group
- `summary_by_age`: Summary statistics by age group

**Additional Scenarios:**
- `school_closure`: 80% reduction in child-child contacts
- `protect_elderly`: 70% reduction in contacts with elderly

## Usage

### 1. Model Discovery

Discover available models in the project:

```bash
cd examples/epi_models
cb models discover
```

This will find and display information about both models including their parameters, outputs, and scenarios.

### 2. Export Model Configuration

Add models to the project configuration:

```bash
# Export the basic SEIR model
cb models export models.seir:StochasticSEIR --files src/models/seir.py

# Export the age-structured model
cb models export models.seir_age:AgeStratifiedSEIR --files src/models/seir_age.py
```

This updates `pyproject.toml` with model definitions.

### 3. Verify Import Boundaries

Verify that models only import from their declared dependencies:

```bash
cb models verify
```

### 4. Build Manifest

Generate the manifest file containing model metadata:

```bash
cb manifest build
```

This creates `manifest.json` with complete model specifications.

### 5. Test Wire Protocol

Test model execution via the wire protocol:

```python
from modelops_calabaria.wire_loader import make_wire_from_manifest
import json

# Load manifest
with open('manifest.json') as f:
    manifest = json.load(f)

# Create wire function for StochasticSEIR
wire = make_wire_from_manifest('models.seir:StochasticSEIR', manifest)

# Run simulation
result = wire(
    params_M={
        'beta': 0.5,
        'sigma': 0.2,
        'gamma': 0.1,
        'population': 100000,
        'initial_infected': 10,
        'initial_exposed': 5,
        'simulation_days': 200
    },
    seed=42,
    scenario_stack=['lockdown'],  # Apply lockdown scenario
    outputs=['prevalence', 'summary']  # Only return these outputs
)

print("Outputs:", list(result.outputs.keys()))
print("Provenance:", result.provenance)
```

## Model Details

### Stochastic Elements

Both models use stochastic transitions between compartments based on Poisson processes:
- Infections follow `Poisson(beta * S * I / N * dt)`
- Progressions follow `Poisson(sigma * E * dt)`
- Recoveries follow `Poisson(gamma * I * dt)`

This introduces realistic variability while maintaining reproducibility via seeds.

### Age Structure (AgeStratifiedSEIR)

The age-structured model uses a contact matrix to model differential mixing between age groups:

```
Contact Matrix (contacts per day):
           Children  Adults  Elderly
Children      8.0     2.0     0.5
Adults        2.0     6.0     1.0
Elderly       0.5     1.0     3.0
```

Age-specific susceptibilities allow modeling different infection risks by age.

### Scenarios

Scenarios modify model behavior without changing core parameters:
- **Configuration updates**: Modify transmission rates, contact patterns
- **Parameter modifications**: Can override parameter values
- **Composable**: Multiple scenarios can be applied in sequence

## Integration with ModelOps

These models are designed to work with the broader ModelOps ecosystem:

### With ModelOps-Bundle

```bash
# Bundle the project for distribution
mb bundle create epi-models-v1.0 .

# Push to registry
mb bundle push epi-models-v1.0
```

### With ModelOps Service

```python
from modelops.services import SimulationService
from modelops_contracts import ParameterSet

# Connect to distributed simulation service
sim_service = SimulationService(endpoint="http://localhost:8000")

# Submit simulation job
job = sim_service.submit_simulation(
    bundle_id="epi-models-v1.0",
    model_id="models.seir:StochasticSEIR",
    parameters=[
        ParameterSet(values={"beta": 0.3, "gamma": 0.1, ...}),
        ParameterSet(values={"beta": 0.5, "gamma": 0.1, ...}),
        # ... parameter sweep
    ],
    scenario="lockdown",
    outputs=["prevalence", "summary"]
)

# Retrieve results
results = sim_service.get_results(job.id)
```

## Development

### Adding New Models

1. Create new model file in `src/models/`
2. Implement `BaseModel` with required methods
3. Add `@model_output` and `@model_scenario` decorators
4. Export with `cb models export`
5. Test with `cb models verify`

### Testing Changes

Run the complete workflow to test changes:

```bash
make discover  # If you create a Makefile
make export
make verify
make manifest
make test
```

This example project demonstrates the complete ModelOps-Calabria workflow from model development to distributed execution.