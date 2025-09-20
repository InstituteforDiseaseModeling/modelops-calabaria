"""ModelOps-Calabaria bridge package.

This package provides the integration layer between the Calabaria science
framework and the ModelOps infrastructure, implementing the protocols
defined in modelops-contracts.
"""

from .parameters import (
    Scalar,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ParameterView,
    Identity,
    LogTransform,
    LogitTransform,
    AffineSqueezedLogit,
)
from .scenarios import (
    ScenarioSpec,
    compose_scenarios,
    scenario_hash,
)
from .decorators import (
    model_output,
    model_scenario,
)
from .constants import SEED_COL
from .base_model import BaseModel
from .wire_protocol import (
    WireABI,
    SerializedParameterSpec,
    WireResponse,
)
from .api import (
    CalabariaCLI,
    quick_discover,
    quick_manifest,
    quick_verify,
    quick_bundle_id,
)

__version__ = "0.1.0"

__all__ = [
    # Parameters
    "Scalar",
    "ParameterSpec",
    "ParameterSpace",
    "ParameterSet",
    "ParameterView",
    # Transforms
    "Identity",
    "LogTransform",
    "LogitTransform",
    "AffineSqueezedLogit",
    # Scenarios
    "ScenarioSpec",
    "compose_scenarios",
    "scenario_hash",
    # Decorators
    "model_output",
    "model_scenario",
    "SEED_COL",
    # Base Model
    "BaseModel",
    # Wire Protocol
    "WireABI",
    "SerializedParameterSpec",
    "WireResponse",
    # API
    "CalabariaCLI",
    "quick_discover",
    "quick_manifest",
    "quick_verify",
    "quick_bundle_id",
]