"""Sobol sequence sampling strategy."""

from typing import List, Dict, Any, Optional
import numpy as np
try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from .base import SamplingStrategy


class SobolSampler(SamplingStrategy):
    """Sobol sequence quasi-random sampling.

    Generates low-discrepancy Sobol sequences that provide better
    space-filling properties than random sampling, especially useful
    for sensitivity analysis and efficient parameter space exploration.
    """

    def __init__(self, parameter_space, scramble: bool = True, seed: Optional[int] = None):
        """Initialize Sobol sampler.

        Args:
            parameter_space: The parameter space to sample from
            scramble: Whether to scramble the sequence (adds randomization)
            seed: Random seed for scrambling

        Raises:
            ImportError: If scipy is not available
        """
        if not HAS_SCIPY:
            raise ImportError(
                "Sobol sampling requires scipy. Install with: pip install scipy>=1.7.0"
            )
        super().__init__(parameter_space)
        self.scramble = scramble
        self.seed = seed

    def _normalize_to_param(self, value: float, spec) -> Any:
        """Convert [0,1] value to parameter range.

        Args:
            value: Value in [0,1] from Sobol sequence
            spec: Parameter specification

        Returns:
            Value scaled to parameter range
        """
        if spec.kind == "choice":
            # Map to discrete choice
            choices = spec.min  # min holds the choices for choice params
            index = min(int(value * len(choices)), len(choices) - 1)
            return choices[index]
        elif spec.kind in ["int", "integer"]:
            # Map to integer range
            scaled = spec.min + value * (spec.max - spec.min)
            return int(round(scaled))
        elif spec.kind == "float":
            # Map to float range
            return spec.min + value * (spec.max - spec.min)
        elif spec.kind == "bool":
            # Map to boolean
            return value >= 0.5
        else:
            raise ValueError(f"Unsupported parameter kind: {spec.kind}")

    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate Sobol sequence samples.

        Args:
            n_samples: Number of parameter sets to generate

        Returns:
            List of parameter dictionaries from Sobol sequence
        """
        # Get continuous parameters (Sobol works in continuous space)
        continuous_params = []
        discrete_params = []

        for spec in self.parameter_space.specs:
            if spec.kind in ["float", "int", "integer"]:
                continuous_params.append(spec)
            else:
                discrete_params.append(spec)

        # Need at least one continuous parameter for Sobol
        n_dims = len(continuous_params) + len(discrete_params)
        if n_dims == 0:
            raise ValueError("Sobol sampling requires at least one parameter")

        # Generate Sobol sequence in [0,1]^d
        sampler = qmc.Sobol(d=n_dims, scramble=self.scramble, seed=self.seed)
        sobol_samples = sampler.random(n_samples)

        # Convert to parameter values
        samples = []
        for sobol_point in sobol_samples:
            sample = {}

            # Process all parameters in order
            idx = 0
            for spec in self.parameter_space.specs:
                sample[spec.name] = self._normalize_to_param(sobol_point[idx], spec)
                idx += 1

            samples.append(sample)

        return samples

    def method_name(self) -> str:
        """Return the name of this sampling method."""
        return "sobol"

    def estimate_convergence(self, n_samples: int) -> float:
        """Estimate convergence rate for Sobol sequence.

        Sobol sequences have convergence rate O((log N)^d / N) for
        d-dimensional problems, better than O(1/sqrt(N)) for random.

        Args:
            n_samples: Number of samples

        Returns:
            Estimated error bound
        """
        d = len(self.parameter_space.specs)
        if n_samples <= 1:
            return 1.0
        # Approximate error bound for Sobol
        return (np.log(n_samples) ** d) / n_samples