"""Factory for creating calibration algorithm adapters.

This module provides a unified interface for creating different types
of calibration algorithm adapters based on configuration.
"""

import logging
from typing import Any, Dict

from modelops_calabaria.calibration.base import AlgorithmAdapter, ParameterSpec
from modelops_calabaria.calibration.optuna_adapter import OptunaAdapter

logger = logging.getLogger(__name__)


def create_algorithm_adapter(
    algorithm_type: str,
    parameter_specs: Dict[str, ParameterSpec] = None,
    config: Dict[str, Any] = None,
) -> AlgorithmAdapter:
    """Create an algorithm adapter based on type.

    Args:
        algorithm_type: Type of algorithm ("optuna", "abc", "mcmc", etc.)
        parameter_specs: Parameter specifications for the optimization
        config: Algorithm-specific configuration

    Returns:
        Algorithm adapter instance

    Raises:
        ValueError: If algorithm_type is not supported
    """
    if parameter_specs is None:
        parameter_specs = {}
    if config is None:
        config = {}

    logger.info(f"Creating adapter for algorithm: {algorithm_type}")

    # Normalize algorithm type
    algorithm_type = algorithm_type.lower()

    if algorithm_type == "optuna":
        # Create Optuna adapter
        max_trials = config.get("max_trials", 100)
        adapter = OptunaAdapter(
            parameter_specs=parameter_specs,
            max_trials=max_trials,
        )
        logger.info("Created OptunaAdapter")
        return adapter

    elif algorithm_type == "mcmc":
        # Placeholder for MCMC adapter
        raise NotImplementedError(
            "MCMC adapter not yet implemented. Use 'optuna' for now."
        )

    elif algorithm_type == "grid" or algorithm_type == "gridsearch":
        # Placeholder for GridSearch adapter
        raise NotImplementedError(
            "GridSearch adapter not yet implemented. Use 'optuna' for now."
        )

    else:
        raise ValueError(
            f"Unknown algorithm type: {algorithm_type}. "
            f"Supported types: optuna, abc, mcmc, grid"
        )


def parse_parameter_specs(specs_config: Dict[str, Any]) -> Dict[str, ParameterSpec]:
    """Parse parameter specifications from configuration.

    Args:
        specs_config: Dictionary of parameter configurations
            e.g., {"beta": {"lower": 0.1, "upper": 0.9, "transform": "logit"}}

    Returns:
        Dictionary of ParameterSpec objects
    """
    parameter_specs = {}

    for name, spec_dict in specs_config.items():
        spec = ParameterSpec(
            name=name,
            lower=spec_dict["lower"],
            upper=spec_dict["upper"],
            transform=spec_dict.get("transform"),
        )
        parameter_specs[name] = spec

    return parameter_specs