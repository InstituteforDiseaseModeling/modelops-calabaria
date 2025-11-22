#!/usr/bin/env python3
"""Example usage of the modelops-calabaria programmatic API."""

from __future__ import annotations

from pathlib import Path

from modelops_calabaria import (
    GridSampler,
    ParameterSpace,
    ParameterSpec,
    SobolSampler,
    load_symbol,
)


def describe_samples(samples, title: str, limit: int = 3) -> None:
    """Pretty-print a few sampled parameter sets."""
    print(f"\n{title} (showing {min(limit, len(samples))} of {len(samples)})")
    for idx, params in enumerate(samples[:limit], start=1):
        values = ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
        print(f"  {idx:>2}: {values}")


def main() -> None:
    print("🔬 Calabaria API demo")

    # 1) Build a parameter space directly
    space = ParameterSpace([
        ParameterSpec("beta", 0.2, 1.6, "float", doc="Transmission rate"),
        ParameterSpec("gamma", 0.05, 0.5, "float", doc="Recovery rate"),
        ParameterSpec("sigma", 0.05, 0.4, "float", doc="Incubation rate"),
    ])

    # 2) Generate Sobol samples
    sobol = SobolSampler(space, scramble=True, seed=123)
    sobol_samples = sobol.sample(16)
    describe_samples(sobol_samples, "Sobol samples")

    # 3) Generate a coarse grid for the same space
    grid = GridSampler(space, n_points_per_param=3)
    grid_samples = grid.sample(None)
    describe_samples(grid_samples, "Grid samples")

    # 4) Load a model dynamically (if it exists) and introspect its parameter space
    demo_model_path = Path("examples/epi_models/src/models/seir.py")
    if demo_model_path.exists():
        try:
            Model = load_symbol("examples/epi_models/src/models/seir.py:StochasticSEIR")
            model_space = Model.parameter_space()
            print(f"\nLoaded {Model.__name__} with {len(model_space.specs)} parameters:")
            for spec in model_space.specs[:5]:
                rng = f"[{spec.min}, {spec.max}]" if spec.min != spec.max else f"= {spec.min}"
                print(f"  • {spec.name} {rng}")
        except Exception as exc:
            print(f"\nCould not load demo model: {exc}")
    else:
        print("\nDemo model not found (examples/epi_models not present)")

    print("\n✅ API demo complete")


if __name__ == "__main__":
    main()
