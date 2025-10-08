"""Base loss functions."""

from typing import Protocol

from ...alignment import AlignedData
from ...constants import LOSS_COL


class LossFunction(Protocol):
    """Protocol for loss functions."""

    def compute(self, aligned: AlignedData) -> AlignedData: ...


class SquaredErrorLoss(LossFunction):
    """Squared error loss function."""

    def __init__(self, col: str):
        self.col = col

    def compute(self, aligned: AlignedData) -> AlignedData:
        residuals = aligned.get_obs_col(self.col) - aligned.get_sim_col(self.col)
        aligned_with_loss = aligned.data.with_columns((residuals**2).alias(LOSS_COL))
        return AlignedData(
            data=aligned_with_loss,
            on_cols=aligned.on_cols,
            replicate_col=aligned.replicate_col,
        )