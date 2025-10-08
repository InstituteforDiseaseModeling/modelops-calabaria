"""Reduction strategies for evaluation."""

from typing import Optional, Protocol, cast

import polars as pl

from ..alignment import AlignedData
from ..constants import LOSS_COL


class LossReducer(Protocol):
    """Protocol for loss reduction strategies."""

    def reduce(self, aligned: AlignedData) -> Optional[float]: ...


class MeanReducer(LossReducer):
    """Reduce losses by taking the mean."""

    def reduce(self, aligned: AlignedData) -> Optional[float]:
        return cast(Optional[float], aligned.get_loss_col().mean())


class MeanGroupedByReplicate(LossReducer):
    """Reduce losses by first averaging within replicates, then across replicates."""

    def reduce(self, aligned: AlignedData) -> Optional[float]:
        if aligned.replicate_col is None:
            return cast(Optional[float], aligned.get_loss_col().mean())

        grouped = aligned.data.group_by(aligned.replicate_col).agg(
            pl.col(LOSS_COL).mean().alias("rep_loss")
        )
        rep_loss = grouped["rep_loss"].mean()
        return cast(Optional[float], rep_loss)