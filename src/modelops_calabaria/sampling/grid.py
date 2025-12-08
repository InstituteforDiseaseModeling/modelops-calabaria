"""Grid search sampling strategy."""

from typing import List, Dict, Any, Optional
import itertools
import numpy as np

from .base import SamplingStrategy


class GridSampler(SamplingStrategy):
    """Grid search parameter sampling.

    Generates a regular grid of parameter combinations, exhaustively
    covering the parameter space with evenly spaced points.
    """

    def __init__(self, parameter_space, n_points_per_param: Optional[int] = None):
        """Initialize grid sampler.

        Args:
            parameter_space: The parameter space to sample from
            n_points_per_param: Number of grid points per parameter.
                               If None, uses 3 for continuous, all values for discrete.
        """
        super().__init__(parameter_space)
        self.n_points_per_param = n_points_per_param

    def _get_param_values(self, spec, n_points: Optional[int]) -> List[Any]:
        """Get grid values for a single parameter.

        Args:
            spec: Parameter specification
            n_points: Number of points to generate

        Returns:
            List of parameter values
        """
        if spec.kind == "choice":
            # For discrete parameters, use all values
            return list(spec.lower)  # min holds the choices for choice params
        elif spec.kind in ["int", "integer"]:
            if n_points is None:
                # Default to 3 points for integers
                n_points = min(3, int(spec.upper - spec.lower + 1))
            # Generate evenly spaced integers
            values = np.linspace(spec.lower, spec.upper, n_points)
            return [int(round(v)) for v in values]
        elif spec.kind == "float":
            if n_points is None:
                n_points = 3  # Default to 3 points for floats
            # Generate evenly spaced floats
            return list(np.linspace(spec.lower, spec.upper, n_points))
        elif spec.kind == "bool":
            return [False, True]
        else:
            raise ValueError(f"Unsupported parameter kind: {spec.kind}")

    def sample(self, n_samples: int = None) -> List[Dict[str, Any]]:
        """Generate grid of parameter combinations.

        Args:
            n_samples: Ignored for grid search (uses all combinations)

        Returns:
            List of parameter dictionaries covering the grid
        """
        # Build list of values for each parameter
        param_grids = {}
        for spec in self.parameter_space.specs:
            param_grids[spec.name] = self._get_param_values(
                spec, self.n_points_per_param
            )

        # Generate all combinations
        param_names = list(param_grids.keys())
        param_values = list(param_grids.values())

        # Use itertools.product to get all combinations
        samples = []
        for combination in itertools.product(*param_values):
            sample = dict(zip(param_names, combination))
            samples.append(sample)

        # If n_samples specified and less than full grid, subsample
        if n_samples is not None and n_samples < len(samples):
            # Take evenly spaced samples from the grid
            indices = np.linspace(0, len(samples) - 1, n_samples, dtype=int)
            samples = [samples[i] for i in indices]

        return samples

    def method_name(self) -> str:
        """Return the name of this sampling method."""
        return "grid"

    def grid_size(self) -> int:
        """Calculate total number of points in the full grid.

        Returns:
            Total number of grid points
        """
        size = 1
        for spec in self.parameter_space.specs:
            values = self._get_param_values(spec, self.n_points_per_param)
            size *= len(values)
        return size