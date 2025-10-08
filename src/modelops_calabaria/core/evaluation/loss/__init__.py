"""Loss functions for evaluation."""

from .base import LossFunction, SquaredErrorLoss
from .nll import BetaBinomialNLL, BinomialNLL

__all__ = [
    "LossFunction",
    "SquaredErrorLoss",
    "BetaBinomialNLL",
    "BinomialNLL",
]