"""
Likelihood-based calibration components.

This module provides proper statistical calibration via log-likelihoods:
- LikelihoodEvaluator: Returns per-replicate log-likelihoods (no marginalization)
- LogLikelihoodFn: Protocol for log-likelihood functions

Key principle: Marginalization over simulator replicates (ε) occurs only at
the LikelihoodTargetSet level to avoid Jensen's inequality issues.
"""

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from ..alignment import AlignedData


class LogLikelihoodFn(Protocol):
    """
    Protocol for log-likelihood functions.

    Returns per-replicate log-likelihoods: np.ndarray of shape (R,) where R is
    the number of replicates. Each element is the sum of log-likelihoods for
    that replicate across all observations.

    The function should:
    1. Validate that aligned.replicate_col is not None
    2. Compute log-likelihood for each observation
    3. Group by replicate and sum
    4. Return numpy array of per-replicate sums
    """

    def __call__(self, aligned: AlignedData) -> np.ndarray:
        """
        Compute per-replicate log-likelihoods.

        Parameters:
            aligned: Aligned data with replicate column

        Returns:
            np.ndarray of shape (R,) with per-replicate log-likelihood sums

        Raises:
            ValueError: If replicate_col is None
        """
        ...


@dataclass(frozen=True, slots=True)
class TargetLikelihoodResult:
    """
    Result of evaluating a single likelihood target.

    This stores per-replicate log-likelihoods WITHOUT marginalization.
    Marginalization over simulator noise (ε) is performed at the
    LikelihoodTargetSet level.

    Attributes:
        name: Target identifier
        loglik_per_rep: Per-replicate log-likelihoods, shape (R,)
        aligned_data: Optional aligned data used for evaluation (for diagnostics)
    """

    name: str
    loglik_per_rep: np.ndarray  # shape (R,)
    aligned_data: Optional[AlignedData] = None


@dataclass
class LikelihoodEvaluator:
    """
    Per-target likelihood evaluator.

    Returns per-replicate log-likelihoods WITHOUT marginalization over replicates.
    This is critical: marginalization over simulator noise (ε) must happen at the
    LikelihoodTargetSet level to correctly handle Jensen's inequality.

    The evaluator simply calls the log-likelihood function and returns the
    per-replicate results. No aggregation or marginalization occurs here.

    Attributes:
        loglik_fn: Log-likelihood function that returns per-replicate values
        name: Name for this evaluator (for result identification)
    """

    loglik_fn: LogLikelihoodFn
    name: str

    def evaluate(self, aligned: AlignedData) -> TargetLikelihoodResult:
        """
        Evaluate log-likelihood on aligned data.

        Returns per-replicate log-likelihoods WITHOUT marginalization.

        Parameters:
            aligned: Aligned data with replicate column

        Returns:
            TargetLikelihoodResult with per-replicate log-likelihoods

        Raises:
            ValueError: If replicate_col is None (passed through from loglik_fn)
        """
        loglik_per_rep = self.loglik_fn(aligned)

        return TargetLikelihoodResult(
            name=self.name,
            loglik_per_rep=loglik_per_rep,
            aligned_data=aligned,
        )
