#!/usr/bin/env python
"""Create test data for diagnostics command."""

import polars as pl
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Create synthetic optimization results
n_samples = 500

# Generate parameter values
beta = np.random.uniform(0.1, 0.5, n_samples)
gamma = np.random.uniform(0.05, 0.2, n_samples)
initial_infected = np.random.uniform(1, 50, n_samples)

# Generate synthetic loss values (with some structure)
# Loss should be lower when beta ~0.3, gamma ~0.1, initial_infected ~10
loss = (
    1000 * ((beta - 0.3)**2 + (gamma - 0.1)**2 + ((initial_infected - 10)/40)**2)
    + np.random.normal(0, 50, n_samples)
)

# Create DataFrame
df = pl.DataFrame({
    'param_id': range(n_samples),
    'param_beta': beta,
    'param_gamma': gamma,
    'param_initial_infected': initial_infected,
    'loss': loss,
    'status': ['COMPLETED'] * n_samples
})

# Save to parquet
output_file = 'test_optimization_results.parquet'
df.write_parquet(output_file)
print(f"Created {output_file} with {n_samples} samples")
print(f"Columns: {df.columns}")
print(f"Min loss: {df['loss'].min():.2f}")
print(f"Max loss: {df['loss'].max():.2f}")