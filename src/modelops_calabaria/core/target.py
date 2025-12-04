"""Target definitions for calibration: Loss and Likelihood tracks."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import polars as pl
from scipy.special import logsumexp

from .alignment import AlignmentStrategy
from .evaluation.composable import LossEvaluator, TargetLossResult
from .types import SimOutput


# -------------------------------------------------------------------
# Loss Track (pedagogical/ad-hoc losses with flexible replicate handling)
# -------------------------------------------------------------------


@dataclass
class LossTarget:
    """
    A single calibration target scored by an arbitrary loss (MSE, MAE, etc.).

    This is the "ad hoc / pedagogical" track. Each target is internally
    responsible for how it treats replicates (annealed vs mean-signal).

    Attributes:
        name: Identifier for this target
        model_output: Name of the model output to extract from simulations
        data: Observed data for comparison
        alignment: Strategy for joining observed and simulated data
        evaluator: Loss evaluator that computes scalar loss
    """

    name: str
    model_output: str
    data: pl.DataFrame
    alignment: AlignmentStrategy
    evaluator: LossEvaluator

    def evaluate(self, sim_outputs: Sequence[SimOutput]) -> TargetLossResult:
        """
        Evaluate the target using the provided replicates of simulated data.

        Parameters:
            sim_outputs: Sequence of simulation outputs (one per replicate)

        Returns:
            TargetLossResult with scalar loss and weighted loss
        """
        # Extract the specific model output for this target
        replicates_df = [out[self.model_output] for out in sim_outputs]

        # Align observed and simulated data
        aligned = self.alignment.align(observed=self.data, simulated=replicates_df)

        # Evaluate loss using the target's evaluator
        return self.evaluator.evaluate(aligned)

    def __repr__(self) -> str:
        data_shape = self.data.shape
        return (
            f"LossTarget(name={self.name!r}, "
            f"model_output={self.model_output!r}, "
            f"data=shape={data_shape}, "
            f"alignment={self.alignment.__class__.__name__}, "
            f"evaluator={self.evaluator.__class__.__name__})"
        )


@dataclass
class LossEvaluationResult:
    """
    Result of evaluating all loss targets for a parameter set θ.

    Attributes:
        total_loss: Sum of weighted losses across all targets
        target_results: Individual results for each target
        seeds: Seeds used for simulation replicates
        metadata: Additional metadata (e.g., study_id, iteration)
    """

    total_loss: float
    target_results: List[TargetLossResult]
    seeds: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def per_target_loss(self) -> Dict[str, float]:
        """Return unweighted loss for each target."""
        return {r.name: r.loss for r in self.target_results}

    def per_target_weighted_loss(self) -> Dict[str, float]:
        """Return weighted loss for each target."""
        return {r.name: r.weighted_loss for r in self.target_results}

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "loss": self.total_loss,
            "per_target": self.per_target_loss(),
            "per_target_weighted": self.per_target_weighted_loss(),
            "seeds": self.seeds,
            **self.metadata,
        }

    def to_dataframe(self) -> pl.DataFrame:
        """Convert target results to a Polars DataFrame."""
        return pl.concat(
            [r.to_dataframe() for r in self.target_results], how="vertical"
        )


@dataclass
class LossTargetSet:
    """
    Container for multiple loss targets.

    Combines targets via linear combination: J_total = Σ_j w_j * J_j

    This ensures no new Jensen inequality effects beyond each target's own
    design. Any variance penalties or Jensen effects are encapsulated within
    each LossEvaluator's aggregation strategy (e.g., annealed vs mean-signal).

    Use this for pedagogical or ad-hoc calibrations with arbitrary loss functions.
    For proper likelihood-based calibration, use LikelihoodTargetSet.

    Attributes:
        targets: List of loss targets to evaluate
    """

    targets: List[LossTarget]

    def evaluate(
        self,
        sim_outputs: Sequence[SimOutput],
        seeds: List[int],
        metadata: Dict[str, Any] | None = None,
    ) -> LossEvaluationResult:
        """
        Evaluate all loss targets.

        Parameters:
            sim_outputs: Sequence of simulation outputs (one per replicate)
            seeds: Seeds used for the simulation replicates
            metadata: Optional metadata to attach to result

        Returns:
            LossEvaluationResult with total loss and per-target results
        """
        results: List[TargetLossResult] = []

        for target in self.targets:
            result = target.evaluate(sim_outputs)
            results.append(result)

        # Linear combination: sum of weighted losses
        total_loss = sum(r.weighted_loss for r in results)

        return LossEvaluationResult(
            total_loss=total_loss,
            target_results=results,
            seeds=seeds,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return f"LossTargetSet(n_targets={len(self.targets)})"


# -------------------------------------------------------------------
# Likelihood Track (proper scoring with ε-marginalization at set level)
# -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TargetLikelihoodResult:
    """
    Result of evaluating a single likelihood target.

    Attributes:
        name: Target identifier
        loglik_per_rep: Per-replicate log-likelihoods, shape (R,)
        aligned_data: Optional aligned data used for evaluation (for diagnostics)
    """

    name: str
    loglik_per_rep: np.ndarray  # shape (R,)
    aligned_data: Any = None  # Optional[AlignedData], but avoid circular import


@dataclass
class LikelihoodTarget:
    """
    A single calibration target scored by an observation likelihood.

    This is the "proper scoring" track. Returns per-replicate log-likelihoods.
    Epsilon (simulator noise) is marginalized only in LikelihoodTargetSet.

    Attributes:
        name: Identifier for this target
        model_output: Name of the model output to extract from simulations
        data: Observed data for comparison
        alignment: Strategy for joining observed and simulated data
        evaluator: Likelihood evaluator that returns per-replicate log-likelihoods
    """

    name: str
    model_output: str
    data: pl.DataFrame
    alignment: AlignmentStrategy
    evaluator: Any  # LikelihoodEvaluator (avoid circular import)

    def evaluate(self, sim_outputs: Sequence[SimOutput]) -> TargetLikelihoodResult:
        """
        Evaluate the target using the provided replicates of simulated data.

        Parameters:
            sim_outputs: Sequence of simulation outputs (one per replicate)

        Returns:
            TargetLikelihoodResult with per-replicate log-likelihoods (NO marginalization)
        """
        # Extract the specific model output for this target
        replicates_df = [out[self.model_output] for out in sim_outputs]

        # Align observed and simulated data
        aligned = self.alignment.align(observed=self.data, simulated=replicates_df)

        # Evaluate likelihood using the target's evaluator
        return self.evaluator.evaluate(aligned)

    def __repr__(self) -> str:
        data_shape = self.data.shape
        return (
            f"LikelihoodTarget(name={self.name!r}, "
            f"model_output={self.model_output!r}, "
            f"data=shape={data_shape}, "
            f"alignment={self.alignment.__class__.__name__}, "
            f"evaluator={self.evaluator.__class__.__name__})"
        )


@dataclass
class LikelihoodEvaluationResult:
    """
    Result of evaluating all likelihood targets for a parameter set θ.

    Attributes:
        log_marginal: Approximation to log p(y^{(1:J)} | θ) via Monte Carlo over replicates
        per_target: Per-target likelihood results with per-replicate log-likelihoods
        seeds: Seeds used for simulation replicates
        metadata: Additional metadata (e.g., study_id, iteration)
    """

    log_marginal: float
    per_target: Dict[str, TargetLikelihoodResult]
    seeds: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "log_marginal": self.log_marginal,
            "seeds": self.seeds,
            **self.metadata,
        }


@dataclass
class LikelihoodTargetSet:
    """
    Container for multiple likelihood targets with proper ε-marginalization.

    This implements proper likelihood-based calibration:
        p(y^{(1:J)} | θ) ≈ (1/R) Σ_r Π_j p_j(y^{(j)} | sim(θ, ε_r), φ_j)

    Key design: Marginalization over simulator noise (ε) is performed at THIS
    level via log-mean-exp, NOT within individual targets. This avoids Jensen's
    inequality issues that would arise from:
        E_ε[p(y|θ,ε)] vs p(y|θ,E_ε[sim(θ,ε)])

    Each LikelihoodTarget returns per-replicate log-likelihoods. This set:
    1. Sums per-target logliks for each replicate r: loglik_r = Σ_j loglik_j,r
    2. Marginalizes via log-mean-exp: log p(y|θ) = logsumexp(loglik_r) - log(R)

    Use this for statistically rigorous calibration. For ad-hoc losses, use LossTargetSet.

    Attributes:
        targets: List of likelihood targets to evaluate
    """

    targets: List[LikelihoodTarget]

    def evaluate(
        self,
        sim_outputs: Sequence[SimOutput],
        seeds: List[int],
        metadata: Dict[str, Any] | None = None,
    ) -> LikelihoodEvaluationResult:
        """
        Evaluate all likelihood targets and marginalize over simulator replicates.

        This implements proper likelihood-based calibration:
            p(y^{(1:J)} | θ) ≈ (1/R) Σ_r Π_j p_j(y^{(j)} | sim(θ, ε_r), φ_j)

        Marginalization over simulator noise (ε) is performed at this level via
        log-mean-exp (logsumexp - log(R)) to avoid Jensen's inequality issues
        that would arise from marginalizing within each target.

        Parameters:
            sim_outputs: Sequence of simulation outputs (one per replicate)
            seeds: Seeds used for the simulation replicates
            metadata: Optional metadata to attach to result

        Returns:
            LikelihoodEvaluationResult with log_marginal approximation

        Raises:
            ValueError: If no targets provided or no replicates available
        """
        if not self.targets:
            raise ValueError("No likelihood targets provided")

        per_target_results: Dict[str, TargetLikelihoodResult] = {}
        joint_loglik_per_rep: np.ndarray | None = None

        # Evaluate each target and accumulate joint log-likelihoods per replicate
        for target in self.targets:
            result = target.evaluate(sim_outputs)
            per_target_results[target.name] = result

            if joint_loglik_per_rep is None:
                joint_loglik_per_rep = result.loglik_per_rep.copy()
            else:
                # Sum log-likelihoods across targets for each replicate
                joint_loglik_per_rep += result.loglik_per_rep

        if joint_loglik_per_rep is None or len(joint_loglik_per_rep) == 0:
            raise ValueError("No replicates available for likelihood evaluation")

        R = len(joint_loglik_per_rep)
        if R <= 0:
            raise ValueError(f"Invalid replicate count: {R}")

        # Marginalize over replicates via log-mean-exp
        # log p(y|θ) ≈ log[(1/R) Σ_r exp(loglik_r)] = logsumexp(loglik_r) - log(R)
        log_marginal = float(logsumexp(joint_loglik_per_rep) - np.log(R))

        return LikelihoodEvaluationResult(
            log_marginal=log_marginal,
            per_target=per_target_results,
            seeds=seeds,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return f"LikelihoodTargetSet(n_targets={len(self.targets)})"
