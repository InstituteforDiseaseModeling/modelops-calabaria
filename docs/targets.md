# Targets Wire Protocol

## Quick Start

Define targets in a Python file with an entrypoint function:

```python
# targets/prevalence.py
from modelops_calabaria.core.target import Target, Targets
from modelops_calabaria.core.alignment import JoinAlignment
import polars as pl

def get_targets():
    observed = pl.read_csv("data/observed_prevalence.csv")

    target = Target(
        model_output="prevalence",  # Which model output to use
        data=observed,
        alignment=JoinAlignment(on_cols="day", mode="exact"),
        evaluation=BetaBinomialNLL(x_col="cases", n_col="total"),
        weight=1.0
    )
    return Targets(targets=[target])
```

Register it:
```bash
modelops-bundle register-target targets/prevalence.py:get_targets
```

## Wire Protocol

The target wire function evaluates targets against simulation outputs:

**Input:**
- `entrypoint`: String like "targets.prevalence:get_targets"
- `sim_outputs`: Dict of output name → Arrow IPC bytes

**Output:**
- `total_loss`: Float
- `target_losses`: Dict of target name → loss

**Flow:**
1. Load target function via entrypoint
2. Convert sim outputs from bytes to DataFrames
3. Evaluate targets against sim data
4. Return losses

## API

```python
from modelops_calabaria.targets import wire_target_function

result = wire_target_function(
    entrypoint="targets.prevalence:get_targets",
    sim_outputs={
        "prevalence": b"<arrow_ipc_bytes>",
        "incidence": b"<arrow_ipc_bytes>"
    }
)
# result = {"total_loss": 123.4, "target_losses": {"prevalence_target": 123.4}}
```

## Alignment Strategies

```python
from modelops_calabaria.core.alignment import JoinAlignment, ExactJoin, AsofJoin

# Exact match on time column
JoinAlignment(on_cols="timestep", mode="exact")

# Nearest time match
JoinAlignment(on_cols="time", mode="nearest")

# Or use specific strategies directly
ExactJoin(on_cols=["time", "location"])
AsofJoin(on_column="time", strategy="backward")
```

## Built-in Evaluation Strategies

```python
from modelops_calabaria.core.evaluation import (
    beta_binomial_nll,
    mean_of_per_replicate_mse,
    replicate_mean_mse,
)

# Beta-binomial for count data
target = Target(
    model_output="prevalence",
    data=observed,
    alignment=JoinAlignment(on_cols="day"),
    evaluation=beta_binomial_nll(x_col="cases", n_col="total"),
)

# MSE computed per replicate then averaged
target = Target(
    model_output="incidence",
    data=observed,
    alignment=JoinAlignment(on_cols="week"),
    evaluation=mean_of_per_replicate_mse(col="new_cases"),
)

# Average replicates first, then compute MSE
target = Target(
    model_output="deaths",
    data=observed,
    alignment=JoinAlignment(on_cols="month"),
    evaluation=replicate_mean_mse(col="deaths"),
)
```

## Custom Evaluation

Build your own evaluator by composing strategies:

```python
from modelops_calabaria.core.evaluation import (
    Evaluator,
    IdentityAggregator,
    SquaredErrorLoss,
    MeanReducer,
)

# Compose aggregation, loss, and reduction
my_evaluator = Evaluator(
    aggregator=IdentityAggregator(),  # Don't aggregate replicates
    loss_fn=SquaredErrorLoss(col="value"),  # Squared error loss
    reducer=MeanReducer(),  # Simple mean of losses
)
```

That's it. Targets work just like models - defined in code, loaded via entrypoints, no serialization.