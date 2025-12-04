"""Composable loss evaluator combining aggregation, loss, and reduction strategies."""

from dataclasses import dataclass
from typing import Any, Optional

import polars as pl

from ..alignment import AlignedData
from .aggregate import AggregateStrategy
from .loss.base import LossFunction
from .reduce import LossReducer


@dataclass(frozen=True, slots=True)
class TargetLossResult:
    """
    Result of evaluating a single loss target.

    Attributes:
        name: Name of the target
        loss: Unweighted loss value
        weight: Weight for this target
        weighted_loss: Loss multiplied by weight
        aligned_data: Optional aligned data used for evaluation (for diagnostics)
    """

    name: str
    loss: float
    weight: float
    weighted_loss: float
    aligned_data: Optional[AlignedData] = None

    def to_dataframe(self) -> pl.DataFrame:
        """Convert the result to a Polars DataFrame."""
        return pl.DataFrame(
            {
                "name": [self.name],
                "loss": [self.loss],
                "weight": [self.weight],
                "weighted_loss": [self.weighted_loss],
            }
        )


@dataclass
class LossEvaluator:
    """
    Composable loss evaluator that chains aggregation, loss computation, and reduction.

    This provides a flexible way to build loss evaluation strategies by combining:
    - Aggregator: How to handle multiple replicates before loss computation
      (e.g., identity for annealed MSE, mean across replicates for mean-signal MSE)
    - Loss function: How to compute pointwise losses (e.g., squared error, absolute error)
    - Reducer: How to reduce pointwise losses to a scalar
      (e.g., simple mean, mean grouped by replicate)

    The pipeline: aligned_data → aggregate → compute_loss → reduce → scalar

    Any variance/Jensen effects are encapsulated within this evaluator's design.
    """

    aggregator: AggregateStrategy
    loss_fn: LossFunction
    reducer: LossReducer
    weight: float = 1.0
    name: Optional[str] = None

    def evaluate(self, aligned: AlignedData) -> TargetLossResult:
        """
        Evaluate the loss on aligned data.

        Returns:
            TargetLossResult with scalar loss and weighted loss
        """
        aligned_agg = self.aggregator.aggregate(aligned)
        aligned_with_loss = self.loss_fn.compute(aligned_agg)
        loss = self.reducer.reduce(aligned_with_loss)

        evaluator_name = self.name or type(self.loss_fn).__name__

        return TargetLossResult(
            name=evaluator_name,
            loss=loss,
            weight=self.weight,
            weighted_loss=self.weight * loss,
            aligned_data=aligned_with_loss,
        )

    def parameters(self) -> Optional[Any]:
        """Return hyperparameters for this evaluator, if any."""
        return None


# Backward compatibility alias (deprecated)
Evaluator = LossEvaluator