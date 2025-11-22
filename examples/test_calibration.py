#!/usr/bin/env python
"""Test calibration adapter functionality."""

import sys
import logging
from modelops_calabaria.calibration import create_algorithm_adapter
from modelops_calabaria.calibration.base import ParameterSpec
from modelops_contracts import TrialResult, TrialStatus

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Test basic calibration adapter functionality."""

    # Define parameter specifications
    param_specs = {
        "beta": ParameterSpec(name="beta", lower=0.1, upper=0.9),
        "gamma": ParameterSpec(name="gamma", lower=0.0, upper=1.0),
    }

    # Create Optuna adapter
    logger.info("Creating Optuna adapter...")
    adapter = create_algorithm_adapter(
        algorithm_type="optuna",
        parameter_specs=param_specs,
        config={"max_trials": 10},
    )

    # Initialize adapter
    logger.info("Initializing adapter...")
    adapter.initialize(
        job_id="test_job_001",
        config={
            "sampler": {"type": "tpe", "n_startup_trials": 5},
            "max_trials": 10,
        },
    )

    # Connect to infrastructure (in-memory for testing)
    logger.info("Connecting to infrastructure...")
    adapter.connect_infrastructure({})  # Empty = use in-memory storage

    # Run a few iterations
    for iteration in range(3):
        logger.info(f"\n--- Iteration {iteration + 1} ---")

        # Ask for parameters
        param_sets = adapter.ask(n=2)
        logger.info(f"Got {len(param_sets)} parameter sets:")
        for ps in param_sets:
            logger.info(f"  {ps.param_id[:8]}: {ps.params}")

        # Simulate evaluation (normally done by simulation service)
        results = []
        for ps in param_sets:
            # Fake loss calculation
            loss = ps.params["beta"] * 0.5 + ps.params["gamma"] * 0.3
            result = TrialResult(
                param_id=ps.param_id,
                loss=loss,
                status=TrialStatus.COMPLETED,
                diagnostics={"iteration": iteration + 1},
            )
            results.append(result)
            logger.info(f"  Evaluated {ps.param_id[:8]}: loss = {loss:.4f}")

        # Tell results
        adapter.tell(results)

        # Check if finished
        if adapter.finished():
            logger.info("Optimization finished!")
            break

    # Get best parameters
    if hasattr(adapter, "get_best_parameters"):
        best = adapter.get_best_parameters()
        if best:
            logger.info(f"\nBest parameters: {best}")

    # Get summary
    if hasattr(adapter, "get_study_summary"):
        summary = adapter.get_study_summary()
        logger.info(f"\nStudy summary:")
        logger.info(f"  Total trials: {summary.get('n_trials', 0)}")
        logger.info(f"  Completed: {summary.get('n_completed', 0)}")
        logger.info(f"  Best value: {summary.get('best_value', 'N/A')}")

    logger.info("\nTest completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())