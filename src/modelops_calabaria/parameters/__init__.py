"""Parameter system for Calabaria models.

This module provides the core parameter types and operations for the
Grammar of Parameters, ensuring immutability and type safety.
"""

from .types import (
    Scalar,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
)
from .view import ParameterView
from .transforms import (
    Transform,
    Identity,
    LogTransform,
    LogitTransform,
    AffineSqueezedLogit,
)

__all__ = [
    # Types
    "Scalar",
    "ParameterSpec",
    "ParameterSpace",
    "ParameterSet",
    # View
    "ParameterView",
    # Transforms
    "Transform",
    "Identity",
    "LogTransform",
    "LogitTransform",
    "AffineSqueezedLogit",
]