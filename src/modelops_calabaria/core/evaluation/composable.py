"""Composable evaluator combining aggregation, loss, and reduction strategies."""

from dataclasses import dataclass
from typing import Any, Optional

from ..alignment import AlignedData
from .aggregate import AggregateStrategy
from .base import EvaluationStrategy, TargetEvaluation
from .loss.base import LossFunction
from .reduce import LossReducer


@dataclass
class Evaluator(EvaluationStrategy):
    """
    Composable evaluator that chains together aggregation, loss computation, and reduction.

    This provides a flexible way to build evaluation strategies by combining:
    - Aggregator: How to handle multiple replicates before loss computation
    - Loss function: How to compute pointwise losses
    - Reducer: How to reduce pointwise losses to a scalar
    """

    aggregator: AggregateStrategy
    loss_fn: LossFunction
    reducer: LossReducer
    weight: float = 1.0

    def evaluate(self, aligned: AlignedData) -> TargetEvaluation:
        aligned_agg = self.aggregator.aggregate(aligned)
        aligned_with_loss = self.loss_fn.compute(aligned_agg)
        loss = self.reducer.reduce(aligned_with_loss)

        return TargetEvaluation(
            name=type(self.loss_fn).__name__,
            loss=loss,
            weight=self.weight,
            weighted_loss=self.weight * loss if loss is not None else None,
            aligned_data=aligned_with_loss,
        )

    def parameters(self) -> Optional[Any]:
        return None