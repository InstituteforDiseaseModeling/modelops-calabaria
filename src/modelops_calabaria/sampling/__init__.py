"""Parameter sampling strategies for simulation experiments.

This module provides various sampling methods for exploring parameter spaces,
generating simulation tasks for calibration, sensitivity analysis, and optimization.
"""

from .base import SamplingStrategy
from .grid import GridSampler
from .sobol import SobolSampler

__all__ = [
    "SamplingStrategy",
    "GridSampler",
    "SobolSampler",
]