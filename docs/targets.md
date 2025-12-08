# Calibration Targets

Calabaria provides two separate tracks for calibration:

1. **Loss Track**: For ad-hoc loss functions (MSE, MAE, etc.) using `LossTarget` and `LossTargetSet`
2. **Likelihood Track**: For proper probabilistic calibration using `LikelihoodTarget` and `LikelihoodTargetSet`

## Quick Start: Loss-Based Calibration

Use the Loss track for simple distance metrics like MSE. Loss targets use linear combination to aggregate.

```python
# targets/incidence.py
import polars as pl
import modelops_calabaria as cb
from modelops_calabaria.core.target import LossTarget
from modelops_calabaria.core.alignment import JoinAlignment
from modelops_calabaria.core.evaluation import mean_signal_mse, annealed_mse


@cb.calibration_target(
    model_output="incidence",
    data={'observed': "data/observed_incidence.csv"}
)
def incidence_mean_signal_target(data_paths):
    """
    Mean-signal MSE target for daily incidence.

    Uses mean_signal_mse which computes: (y - E_ε[sim])²
    This targets the observation-level mean WITHOUT penalizing simulator variance.
    """
    observed_data = pl.read_csv(data_paths['observed'])

    return LossTarget(
        name="incidence_mean_signal",
        model_output="incidence",
        data=observed_data,
        alignment=JoinAlignment(on_cols="day", mode="exact"),
        evaluator=mean_signal_mse(col="infected", weight=1.0),
    )


@cb.calibration_target(
    model_output="incidence",
    data={'observed': "data/observed_incidence.csv"}
)
def incidence_annealed_target(data_paths):
    """
    Annealed MSE target for daily incidence.

    Uses annealed_mse which computes: E_ε[(y - sim)²]
    This INCLUDES a variance penalty and penalizes simulator stochasticity.
    """
    observed_data = pl.read_csv(data_paths['observed'])

    return LossTarget(
        name="incidence_annealed",
        model_output="incidence",
        data=observed_data,
        alignment=JoinAlignment(on_cols="day", mode="exact"),
        evaluator=annealed_mse(col="infected", weight=1.0),
    )
```

Register targets:
```bash
modelops-bundle register-target targets/incidence.py --regen-all
```

Create a target set:
```bash
modelops-bundle target-set set incidence \
    --target incidence_mean_signal_target \
    --target incidence_annealed_target
```

## Quick Start: Likelihood-Based Calibration

Use the Likelihood track for proper probabilistic calibration. Likelihood targets use log-mean-exp for ε-marginalization.

```python
# targets/prevalence.py
import polars as pl
import modelops_calabaria as cb
from modelops_calabaria.core.target import LikelihoodTarget
from modelops_calabaria.core.alignment import JoinAlignment
from modelops_calabaria.core.evaluation import beta_binomial_loglik_per_rep


@cb.calibration_target(
    model_output="prevalence",
    data={'observed': "data/observed_prevalence.csv"}
)
def prevalence_likelihood_target(data_paths):
    """
    Beta-binomial likelihood for prevalence data.

    Uses proper ε-marginalization at the TargetSet level to avoid
    Jensen's inequality issues.
    """
    observed_data = pl.read_csv(data_paths['observed'])

    return LikelihoodTarget(
        name="prevalence_bb",
        model_output="prevalence",
        data=observed_data,
        alignment=JoinAlignment(on_cols="day", mode="exact"),
        evaluator=beta_binomial_loglik_per_rep(
            x_col="cases",
            n_col="total",
            alpha=1.0,
            beta=1.0
        ),
    )
```

## Loss Track: Available Evaluators

### Mean-Signal MSE (Recommended for most use cases)
Averages replicates first, then computes MSE. No variance penalty.

```python
from modelops_calabaria.core.evaluation import mean_signal_mse

evaluator = mean_signal_mse(col="infected", weight=1.0)
# Computes: (y - E_ε[sim])²
```

### Annealed MSE
Computes MSE per replicate, then averages. Includes variance penalty.

```python
from modelops_calabaria.core.evaluation import annealed_mse

evaluator = annealed_mse(col="infected", weight=1.0)
# Computes: E_ε[(y - sim)²]
```

### Custom Loss Evaluator
Build your own by composing strategies:

```python
from modelops_calabaria.core.evaluation import (
    LossEvaluator,
    IdentityAggregator,
    SquaredErrorLoss,
    MeanGroupedByReplicate,
)

my_evaluator = LossEvaluator(
    aggregator=IdentityAggregator(),      # Don't aggregate replicates
    loss_fn=SquaredErrorLoss(col="value"),  # Squared error loss
    reducer=MeanGroupedByReplicate(),     # Mean per replicate
    weight=1.0,
    name="custom_mse",
)
```

## Likelihood Track: Available Evaluators

### Beta-Binomial Log-Likelihood
For count data (e.g., number of cases out of total population).

```python
from modelops_calabaria.core.evaluation import beta_binomial_loglik_per_rep

evaluator = beta_binomial_loglik_per_rep(
    x_col="cases",      # Number of successes (observed)
    n_col="total",      # Number of trials (observed)
    alpha=1.0,          # Prior alpha (default: 1.0)
    beta=1.0,           # Prior beta (default: 1.0)
)
```

### Binomial Log-Likelihood
When you have probability estimates from simulation.

```python
from modelops_calabaria.core.evaluation import binomial_loglik_per_rep

evaluator = binomial_loglik_per_rep(
    p_col="prevalence",  # Probability from simulation
    x_col="cases",       # Number of successes (observed)
    n_col="total",       # Number of trials (observed)
)
```

## Alignment Strategies

Align simulation outputs with observed data:

```python
from modelops_calabaria.core.alignment import JoinAlignment

# Exact match on time column
JoinAlignment(on_cols="day", mode="exact")

# Nearest time match (finds closest simulation timestep)
JoinAlignment(on_cols="time", mode="nearest")

# Multiple join columns
JoinAlignment(on_cols=["day", "location"], mode="exact")
```

## Loss vs Likelihood: When to Use What

### Use Loss Track When:
- You want simple distance metrics (MSE, MAE)
- You're doing parameter sweeps or sensitivity analysis
- You don't need proper probabilistic calibration
- You want to combine targets with different units (weighted sum)

### Use Likelihood Track When:
- You have count data (cases, deaths, etc.)
- You want proper Bayesian calibration
- You need to marginalize over simulator stochasticity (ε)
- You want to avoid Jensen's inequality issues

**Important**: Don't mix Loss and Likelihood tracks in the same calibration. They use different aggregation methods:
- Loss: Linear combination (weighted sum)
- Likelihood: Log-sum-exp for ε-marginalization

## Target Sets

Group related targets for joint calibration:

```python
# Loss target set (linear combination)
loss_set = LossTargetSet(
    name="all_outcomes",
    targets=[incidence_target, prevalence_target],
)

# Likelihood target set (log-sum-exp marginalization)
likelihood_set = LikelihoodTargetSet(
    name="all_observations",
    targets=[prevalence_likelihood, deaths_likelihood],
)
```

Via CLI:
```bash
# Create target set
modelops-bundle target-set set all_outcomes \
    --target incidence_mean_signal_target \
    --target prevalence_target

# Use in calibration
mops jobs submit calibration_spec.json --target-set all_outcomes
```

## Complete Example

See `modelops/examples/starsim-sir/` for a complete working example with:
- Loss-based targets (mean-signal MSE and annealed MSE)
- Model registration
- Target registration
- Job submission workflow
- Makefile automation

Key files:
- `models/sir.py` - Starsim SIR model with outputs
- `targets/incidence.py` - Loss targets for incidence calibration
- `Makefile` - Complete workflow automation
