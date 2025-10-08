"""Base evaluation types and protocols."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import polars as pl

from ..alignment import AlignedData


@dataclass(slots=True, kw_only=True, frozen=True)
class TargetEvaluation:
    """
    A storage class for the result of evaluating a single target.

    Note that the `loss` and `weighted_loss` fields are optional, as users
    may want to use this class for diagnostics or purposes other than as part
    of the objective function in a calibration task.
    """

    name: str
    loss: Optional[float]
    weight: float
    weighted_loss: Optional[float]
    aligned_data: Optional[AlignedData] = None  # optional, useful for diagnostics

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert the TargetEvaluation to a Polars DataFrame.
        """
        data = {
            "name": self.name,
            "loss": self.loss,
            "weight": self.weight,
            "weighted_loss": self.weighted_loss,
        }
        return pl.DataFrame(data)


@dataclass
class EvaluationResult:
    """Result of evaluating all targets for a parameter set."""

    total_loss: Optional[float]
    target_results: List[TargetEvaluation]
    study_id: str
    seeds: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def per_target_loss(self) -> Dict[str, Optional[float]]:
        return {t.name: t.loss for t in self.target_results}

    def per_target_weighted_loss(self) -> Dict[str, Optional[float]]:
        return {t.name: t.weighted_loss for t in self.target_results}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "loss": self.total_loss,
            "per_target": self.per_target_loss(),
            "per_target_weighted": self.per_target_weighted_loss(),
            "study_id": self.study_id,
            "seeds": self.seeds,
            **self.metadata,
        }


class EvaluationStrategy(Protocol):
    """
    Interface for computing loss between simulation and observation data after
    alignment, for a single Target. The incoming data is a single AlignedData
    object, which had already (1) joined the observation and simulation data
    (possibly multiple replicates), and (2) concatenated the replicates into a
    single long table.
    """

    def evaluate(self, aligned: AlignedData) -> TargetEvaluation: ...

    def parameters(self) -> Optional[Any]:
        """
        Return a set of hyperparameters for this evaluation strategy,
        if any.
        """
        return None