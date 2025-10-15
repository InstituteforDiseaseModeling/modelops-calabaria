"""Public API for modelops-calabaria.

This module provides the complete public API for the Calabaria modeling framework,
including model components, parameter management, sampling, and utilities.
"""

# Core modeling
from .base_model import BaseModel

# Parameters
from .parameters import (
    Scalar,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ParameterView,
    # Transforms
    Identity,
    LogTransform,
    LogitTransform,
    AffineSqueezedLogit,
)

# Scenarios
from .scenarios import (
    ScenarioSpec,
    compose_scenarios,
    scenario_hash,
)

# Decorators
from .decorators import (
    model_output,
    model_scenario,
    calibration_target,
    get_registered_targets,
)

# Sampling
from .sampling import (
    SamplingStrategy,
    GridSampler,
    SobolSampler,
)

# Wire Protocol (advanced users)
from .wire_protocol import (
    WireABI,
    SerializedParameterSpec,
    WireResponse,
)

# Wire loading
from .wire_loader import (
    EntryRecord,
    entry_from_manifest,
    make_wire,
    make_wire_from_manifest,
)

# Constants
from .constants import SEED_COL

# Import utilities (new!)
from .utils.imports import load_symbol

# CLI utilities (for programmatic use)
from .cli.discover import discover_models, suggest_model_config
from .cli.config import read_pyproject, write_model_config, validate_config
from .cli.verify import verify_all_models, verify_model

# Version
try:
    from importlib.metadata import version
    __version__ = version("modelops-calabaria")
except Exception:
    __version__ = "0.1.0"

# Public API Export List
__all__ = [
    # Core
    "BaseModel",

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
    "calibration_target",
    "get_registered_targets",

    # Sampling
    "SamplingStrategy",
    "GridSampler",
    "SobolSampler",

    # Wire Protocol
    "WireABI",
    "SerializedParameterSpec",
    "WireResponse",

    # Wire Loading
    "EntryRecord",
    "entry_from_manifest",
    "make_wire",
    "make_wire_from_manifest",

    # Constants
    "SEED_COL",

    # Utilities
    "load_symbol",

    # CLI utilities
    "discover_models",
    "suggest_model_config",
    "read_pyproject",
    "write_model_config",
    "validate_config",
    "verify_all_models",
    "verify_model",

    # Version
    "__version__",
]