"""Aggregation strategies for evaluation."""

from typing import Protocol

import polars as pl

from ..alignment import AlignedData


class AggregateStrategy(Protocol):
    """Protocol for aggregation strategies."""

    def aggregate(self, aligned: AlignedData) -> AlignedData: ...


class IdentityAggregator(AggregateStrategy):
    """Pass-through aggregator that returns data unchanged."""

    def aggregate(self, aligned: AlignedData) -> AlignedData:
        return aligned


class MeanAcrossReplicates(AggregateStrategy):
    """
    Averages simulation columns across replicates, keeps one copy of observed values.
    """

    def __init__(self, cols: list[str]):
        self.cols = cols

    def aggregate(self, aligned: AlignedData) -> AlignedData:
        if aligned.replicate_col is None:
            raise ValueError(
                "AlignedData must include replicate_col for MeanAcrossReplicates."
            )

        sim_cols = [aligned.sim_colname(c) for c in self.cols]
        obs_cols = [aligned.obs_colname(c) for c in self.cols]
        group_cols = aligned.on_cols

        # Mean simulation across replicates
        sim_avg = aligned.data.group_by(group_cols).agg(
            [pl.col(c).mean().alias(c) for c in sim_cols]
        )

        # Extract one copy of the observed values
        obs = aligned.data.filter(pl.col(aligned.replicate_col) == 0).select(
            group_cols + obs_cols
        )

        merged = sim_avg.join(obs, on=group_cols, how="inner")

        return AlignedData(
            data=merged,
            on_cols=aligned.on_cols,
            replicate_col=None,  # No longer multiple replicates after this point
        )