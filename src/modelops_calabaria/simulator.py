"""ModelSimulator: Packaged simulator interface for calibration and optimization.

This module implements ModelSimulator, which packages together:
- BaseModel (with configuration c ∈ C baked in)
- Scenario specification
- CoordinateSystem (for Z_V ↔ M mapping)

The ModelSimulator provides a clean callable interface: Z_V × S → Y
This is the primary interface for calibration and optimization workflows.
"""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import polars as pl

from .base_model import BaseModel
from .parameters import CoordinateSystem


@dataclass(frozen=True)
class ModelSimulator:
    """Packaged simulator interface: Z_V × S → Y

    Wraps together:
    - BaseModel (holds configuration c ∈ C internally via base_config)
    - scenario (ScenarioSpec name to apply)
    - coords (CoordinateSystem for Z_V ↔ M mapping)

    The configuration lives in BaseModel.base_config (as per whitepaper: Sim(m, c₀, s)).
    This interface only manipulates parameters via CoordinateSystem.

    Example:
        >>> # Create model with configuration
        >>> model = StochasticSEIR(space, config_space, base_config)
        >>>
        >>> # Create coordinate system (free params + transforms)
        >>> view = ParameterView.from_fixed(space, gamma=0.1)
        >>> coords = CoordinateSystem(view, {"beta": LogTransform()})
        >>>
        >>> # Create simulator
        >>> sim = ModelSimulator(model, "baseline", coords)
        >>>
        >>> # Execute: z vector → simulation outputs
        >>> z = np.array([0.0])  # log(beta) = 0 → beta = 1.0
        >>> outputs = sim(z, seed=42)
    """
    model: BaseModel
    scenario: str
    coords: CoordinateSystem

    def __post_init__(self):
        """Validate that scenario exists in model."""
        if self.scenario not in self.model.list_scenarios():
            available = self.model.list_scenarios()
            raise ValueError(
                f"Unknown scenario '{self.scenario}'. "
                f"Available scenarios: {available}"
            )

    def __call__(self, z: np.ndarray, seed: int) -> Dict[str, pl.DataFrame]:
        """Execute simulation: Z_V × S → Y

        Pipeline:
        1. coords.to_M(z) → params (ParameterSet in M-space)
        2. model.simulate_scenario(scenario, params, seed)
           - Model applies scenario patches internally
           - Uses model.base_config (c₀) from construction

        Args:
            z: Vector in inference space (length = coords.dim)
            seed: Random seed for reproducibility

        Returns:
            Dict of output DataFrames with SEED_COL added

        Example:
            >>> sim = ModelSimulator(model, "baseline", coords)
            >>> z = np.array([0.5, 0.3])
            >>> outputs = sim(z, seed=42)
            >>> outputs["prevalence"]  # DataFrame with results
        """
        # Step 1: Map z → M (inference space → model parameter space)
        params = self.coords.to_M(z)

        # Step 2: Run model with scenario
        # Model internally:
        # - Applies scenario patches to params and config
        # - Uses base_config as c₀
        # - Executes build_sim, run_sim, extract_outputs
        return self.model.simulate_scenario(self.scenario, params, seed)

    def bounds(self) -> np.ndarray:
        """Return bounds in transformed (Z_V) space.

        Returns:
            (n_free, 2) array of [lower, upper] bounds in Z-space

        Example:
            >>> sim.bounds()
            array([[-5.0, 2.3],   # log(beta) bounds
                   [ 0.0, 1.0]])   # gamma bounds (untransformed)
        """
        return self.coords.bounds_transformed()

    @property
    def dim(self) -> int:
        """Dimensionality of parameter space (number of free parameters).

        Returns:
            Number of dimensions in z vector

        Example:
            >>> sim.dim
            2  # Two free parameters
        """
        return self.coords.dim

    @property
    def free_param_names(self) -> tuple:
        """Ordered list of free parameter names.

        The order matches the z vector dimensions.

        Returns:
            Tuple of parameter names

        Example:
            >>> sim.free_param_names
            ('beta', 'gamma')
            >>> # z[0] corresponds to beta, z[1] to gamma
        """
        return self.coords.param_names

    def __repr__(self) -> str:
        """Compact representation for debugging."""
        return (
            f"ModelSimulator("
            f"model={self.model.__class__.__name__}, "
            f"scenario='{self.scenario}', "
            f"dim={self.dim})"
        )
