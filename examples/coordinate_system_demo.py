"""Demonstration of CoordinateSystem flexibility.

This example showcases how the CoordinateSystem abstraction enables
flexible parameter space transformations for different use cases:

1. Identity transforms (natural parameter space)
2. Log transforms (for positive rate parameters)
3. Logit transforms (for probability parameters)
4. Mixed transforms (different transforms per parameter)
5. Bounds calculations in transformed spaces

The CoordinateSystem provides bidirectional mappings:
- to_M: Z_V → P_V → M (inference space to model space)
- from_M: M → P_V → Z_V (model space to inference space)
"""

import numpy as np
import matplotlib.pyplot as plt

from modelops_calabaria import (
    BaseModel,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ParameterView,
    CoordinateSystem,
    ConfigSpec,
    ConfigurationSpace,
    ConfigurationSet,
    Identity,
    LogTransform,
    AffineSqueezedLogit,
    model_output,
)
import polars as pl


# ==============================================================================
# Example Model: Simple Growth Model
# ==============================================================================

class GrowthModel(BaseModel):
    """Simple exponential growth model for demonstration.

    Parameters:
    - rate: growth rate (positive)
    - initial: initial value (positive)
    - probability: success probability (0-1)
    - scale: scaling factor (positive)
    """

    def __init__(self):
        space = ParameterSpace([
            ParameterSpec("rate", 0.01, 5.0, "float", doc="Growth rate"),
            ParameterSpec("initial", 1.0, 100.0, "float", doc="Initial value"),
            ParameterSpec("probability", 0.01, 0.99, "float", doc="Success probability"),
            ParameterSpec("scale", 0.1, 10.0, "float", doc="Scaling factor"),
        ])
        config_space = ConfigurationSpace([
            ConfigSpec("steps", default=100, doc="Number of steps"),
        ])
        base_config = ConfigurationSet(config_space, {"steps": 100})
        super().__init__(space, config_space, base_config)

    def build_sim(self, params: ParameterSet, config: ConfigurationSet) -> dict:
        return dict(params.to_dict(), **config.to_dict())

    def run_sim(self, state: dict, seed: int) -> dict:
        rng = np.random.RandomState(seed)
        values = [state["initial"]]
        for _ in range(state["steps"]):
            # Stochastic growth with probability
            if rng.random() < state["probability"]:
                values.append(values[-1] * (1 + state["rate"]) * state["scale"])
            else:
                values.append(values[-1])
        return {"values": np.array(values)}

    @model_output("trajectory")
    def extract_trajectory(self, raw: dict, seed: int) -> pl.DataFrame:
        return pl.DataFrame({
            "step": np.arange(len(raw["values"])),
            "value": raw["values"],
        })


# ==============================================================================
# Example 1: Identity Transforms (Natural Space)
# ==============================================================================

def example_1_identity_transforms():
    """Demonstrate identity transforms - work directly in natural parameter space."""
    print("="*70)
    print("Example 1: Identity Transforms (Natural Space)")
    print("="*70)

    model = GrowthModel()

    # All parameters free, no transforms
    view = ParameterView.all_free(model.space)
    coords = CoordinateSystem(view, {})  # Empty transform dict = all Identity

    print(f"Free parameters: {view.free}")
    print(f"Dimension: {coords.dim}")
    print(f"\nBounds (natural space):")
    bounds = coords.bounds_transformed()
    for i, name in enumerate(coords.param_names):
        print(f"  {name}: [{bounds[i, 0]:.3f}, {bounds[i, 1]:.3f}]")

    # Test round-trip
    z = np.array([0.5, 10.0, 0.7, 2.0])
    params = coords.to_M(z)
    z_back = coords.from_M(params)

    print(f"\nRound-trip test:")
    print(f"  Input z:  {z}")
    print(f"  Output z: {z_back}")
    print(f"  Error: {np.linalg.norm(z - z_back):.2e}")
    print()


# ==============================================================================
# Example 2: Log Transforms for Positive Parameters
# ==============================================================================

def example_2_log_transforms():
    """Demonstrate log transforms for positive rate parameters."""
    print("="*70)
    print("Example 2: Log Transforms for Positive Parameters")
    print("="*70)

    model = GrowthModel()

    # Fix probability, transform rate and scale with log
    view = ParameterView.from_fixed(model.space, probability=0.8, initial=10.0)
    coords = CoordinateSystem(view, {
        "rate": LogTransform(),
        "scale": LogTransform(),
    })

    print(f"Free parameters: {coords.param_names}")
    print(f"Dimension: {coords.dim}")
    print(f"\nBounds (log space):")
    bounds = coords.bounds_transformed()
    for i, name in enumerate(coords.param_names):
        print(f"  log({name}): [{bounds[i, 0]:.3f}, {bounds[i, 1]:.3f}]")

    # Compare with natural bounds
    print(f"\nOriginal bounds (natural space):")
    for name in coords.param_names:
        spec = model.space.get_spec(name)
        print(f"  {name}: [{spec.lower}, {spec.upper}]")
        print(f"    → log: [{np.log(spec.lower):.3f}, {np.log(spec.upper):.3f}]")

    # Test mapping
    z_log = np.array([np.log(0.5), np.log(2.0)])  # log(rate), log(scale)
    params = coords.to_M(z_log)
    print(f"\nMapping test:")
    print(f"  z (log space): {z_log}")
    print(f"  rate (natural): {params['rate']:.3f} (expected: 0.500)")
    print(f"  scale (natural): {params['scale']:.3f} (expected: 2.000)")
    print()


# ==============================================================================
# Example 3: Logit Transforms for Probabilities
# ==============================================================================

def example_3_logit_transforms():
    """Demonstrate logit transforms for probability parameters."""
    print("="*70)
    print("Example 3: Logit Transforms for Probabilities")
    print("="*70)

    model = GrowthModel()

    # Only probability free, use logit transform
    view = ParameterView.from_fixed(
        model.space,
        rate=0.5,
        initial=10.0,
        scale=1.0
    )
    coords = CoordinateSystem(view, {
        "probability": AffineSqueezedLogit(),
    })

    print(f"Free parameter: {coords.param_names}")
    print(f"Dimension: {coords.dim}")

    # Test transformation across range
    print(f"\nProbability transformations:")
    test_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    for p in test_probs:
        # Create params with this probability
        params = ParameterSet(model.space, {
            "rate": 0.5,
            "initial": 10.0,
            "probability": p,
            "scale": 1.0,
        })
        z = coords.from_M(params)
        params_back = coords.to_M(z)
        p_back = params_back["probability"]

        print(f"  p={p:.1f} → z={z[0]:7.3f} → p={p_back:.6f}")

    print()


# ==============================================================================
# Example 4: Mixed Transforms
# ==============================================================================

def example_4_mixed_transforms():
    """Demonstrate mixed transforms - different types per parameter."""
    print("="*70)
    print("Example 4: Mixed Transforms")
    print("="*70)

    model = GrowthModel()

    # Mixed: log for rate/scale, logit for probability, identity for initial
    view = ParameterView.all_free(model.space)
    coords = CoordinateSystem(view, {
        "rate": LogTransform(),
        "initial": Identity(),  # Explicit identity
        "probability": AffineSqueezedLogit(),
        "scale": LogTransform(),
    })

    print(f"Free parameters: {coords.param_names}")
    print(f"Transforms:")
    for name in coords.param_names:
        transform = coords.transforms.get(name, Identity())
        print(f"  {name}: {transform.__class__.__name__}")

    print(f"\nBounds in transformed space:")
    bounds = coords.bounds_transformed()
    for i, name in enumerate(coords.param_names):
        print(f"  {name}: [{bounds[i, 0]:8.3f}, {bounds[i, 1]:8.3f}]")

    # Test full round-trip
    z = np.array([
        np.log(0.5),      # log(rate)
        50.0,             # initial (identity)
        0.0,              # logit(probability) ≈ 0.5
        np.log(2.0),      # log(scale)
    ])
    params = coords.to_M(z)
    z_back = coords.from_M(params)

    print(f"\nRound-trip with mixed transforms:")
    print(f"  Input z:       {z}")
    print(f"  Params (natural):")
    for name in coords.param_names:
        print(f"    {name}: {params[name]:.6f}")
    print(f"  Output z:      {z_back}")
    print(f"  Round-trip error: {np.linalg.norm(z - z_back):.2e}")
    print()


# ==============================================================================
# Example 5: Visualizing Transform Effects
# ==============================================================================

def example_5_visualize_transforms():
    """Visualize how transforms affect parameter space."""
    print("="*70)
    print("Example 5: Visualizing Transform Effects")
    print("="*70)

    model = GrowthModel()

    # Create coordinate systems with different transforms
    view = ParameterView.from_fixed(
        model.space,
        initial=10.0,
        scale=1.0
    )

    coords_identity = CoordinateSystem(view, {
        "rate": Identity(),
        "probability": Identity(),
    })

    coords_transformed = CoordinateSystem(view, {
        "rate": LogTransform(),
        "probability": AffineSqueezedLogit(),
    })

    # Sample uniform grid in transformed space (avoid exact endpoints)
    n_samples = 20
    bounds = coords_transformed.bounds_transformed()
    # Use 95% of the range to avoid numerical issues at boundaries
    z_rate = np.linspace(bounds[0, 0] * 0.95, bounds[0, 1] * 0.95, n_samples)
    z_prob = np.linspace(bounds[1, 0] * 0.95, bounds[1, 1] * 0.95, n_samples)

    # Map to natural space
    natural_rates = []
    natural_probs = []
    for zr in z_rate:
        for zp in z_prob:
            z = np.array([zr, zp])
            params = coords_transformed.to_M(z)
            natural_rates.append(params["rate"])
            natural_probs.append(params["probability"])

    natural_rates = np.array(natural_rates)
    natural_probs = np.array(natural_probs)

    print(f"Sampled {n_samples}x{n_samples} grid in transformed space")
    print(f"Rate range (natural): [{natural_rates.min():.3f}, {natural_rates.max():.3f}]")
    print(f"Prob range (natural): [{natural_probs.min():.3f}, {natural_probs.max():.3f}]")
    print(f"\nNote: Uniform grid in transformed space → non-uniform in natural space")
    print(f"This is the key benefit: uniform sampling in unbounded space!")
    print()


# ==============================================================================
# Example 6: Practical Use Case - Optimization
# ==============================================================================

def example_6_optimization_use_case():
    """Show why transforms are useful for optimization."""
    print("="*70)
    print("Example 6: Practical Use Case - Optimization")
    print("="*70)

    model = GrowthModel()

    # Scenario: We want to optimize over rate and probability
    # Rate has large range [0.01, 5.0] - use log
    # Probability is bounded [0.01, 0.99] - use logit

    view = ParameterView.from_fixed(model.space, initial=10.0, scale=1.0)
    coords = CoordinateSystem(view, {
        "rate": LogTransform(),
        "probability": AffineSqueezedLogit(),
    })

    print("Setup for optimization:")
    print(f"  Optimization dimension: {coords.dim}")
    print(f"  Free parameters: {coords.param_names}")
    print(f"\nBounds for optimizer (transformed space):")
    bounds = coords.bounds_transformed()
    for i, name in enumerate(coords.param_names):
        print(f"  {name}: [{bounds[i, 0]:.3f}, {bounds[i, 1]:.3f}]")

    print(f"\nBenefits:")
    print(f"  1. Unbounded space → easier for optimizers")
    print(f"  2. Rate scale normalized (log) → better gradient behavior")
    print(f"  3. Probability stays in (0,1) automatically")
    print(f"  4. Uniform sampling in z → good coverage in natural space")

    # Simulate optimization iteration
    print(f"\nSimulated optimization iteration:")
    z_current = np.array([0.0, 0.0])  # Start at center
    params = coords.to_M(z_current)
    print(f"  z = {z_current}")
    print(f"  → rate = {params['rate']:.4f}")
    print(f"  → probability = {params['probability']:.4f}")

    # Take a step in z space
    z_next = z_current + np.array([0.5, -0.3])
    params_next = coords.to_M(z_next)
    print(f"\n  z = {z_next}")
    print(f"  → rate = {params_next['rate']:.4f}")
    print(f"  → probability = {params_next['probability']:.4f}")
    print()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    example_1_identity_transforms()
    example_2_log_transforms()
    example_3_logit_transforms()
    example_4_mixed_transforms()
    example_5_visualize_transforms()
    example_6_optimization_use_case()

    print("="*70)
    print("All coordinate system examples completed!")
    print("="*70)
