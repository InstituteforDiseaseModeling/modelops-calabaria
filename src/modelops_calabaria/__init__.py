"""ModelOps-Calabaria: Science framework for distributed epidemic modeling.

This package provides the modeling framework layer for ModelOps, implementing
the contracts defined in modelops-contracts for distributed simulation and
calibration on Kubernetes infrastructure.
"""

# Export the public API
from .api import *  # noqa: F403, F401
from .api import __all__, __version__  # noqa: F401