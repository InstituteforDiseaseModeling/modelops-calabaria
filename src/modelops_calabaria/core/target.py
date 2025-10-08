"""Target definitions for calibration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import polars as pl

from .alignment import AlignmentStrategy
from .evaluation.base import EvaluationStrategy, TargetEvaluation
from .types import SimOutput


@dataclass
class Target:
    """
    Target represents a single target for calibration, encapsulating the
    observed data, an alignment strategy that defines how to join observed and
    simulated data, and an evaluation strategy that computes the loss between
    the aligned data.
    """

    model_output: str
    data: pl.DataFrame
    alignment: AlignmentStrategy
    evaluation: EvaluationStrategy
    source_path: Optional[Path] = None
    weight: float = 1.0

    def evaluate(self, replicated_sim_outputs: Sequence[SimOutput]) -> TargetEvaluation:
        """
        Evaluate the target using the provided replicates of simulated data,
        using the alignment and evaluation strategies.
        """

        # Pull the specific extractor output for the target we are evaluating
        replicates_df = [out[self.model_output] for out in replicated_sim_outputs]

        # (1) Align the observed and simulated data
        aligned = self.alignment.align(observed=self.data, simulated=replicates_df)

        # (2) Evaluate the loss using the target's evaluation strategy
        target_eval = self.evaluation.evaluate(aligned)

        return target_eval

    def __repr__(self) -> str:
        data_shape = self.data.shape
        return (
            f"Target(model_output={self.model_output!r}, "
            f"data=shape={data_shape}, "
            f"alignment={self.alignment.__class__.__name__}, "
            f"evaluation={self.evaluation.__class__.__name__}, "
            f"weight={self.weight})"
        )


class TargetEvaluations(list[TargetEvaluation]):
    """
    Thin wrapper around a list of TargetEvaluation objects, providing
    a method to convert the evaluations into a Polars DataFrame.
    """

    def to_dataframe(self) -> pl.DataFrame:
        """
        Convert the list of TargetEvaluation objects into a Polars DataFrame.
        """
        return pl.concat([ev.to_dataframe() for ev in self], how="vertical")


@dataclass
class Targets:
    """Container for multiple calibration targets."""

    targets: list[Target]

    def evaluate_all(self, replicates: Sequence[SimOutput]) -> TargetEvaluations:
        """
        Evaluate all targets against replicated simulation outputs.

        Parameters
        ----------
        replicates : Sequence[SimOutput]
            Simulation outputs for each replicate.

        Returns
        -------
        TargetEvaluations
        """
        results: list[TargetEvaluation] = []

        for target in self.targets:
            result = target.evaluate(replicates)
            results.append(result)

        return TargetEvaluations(results)