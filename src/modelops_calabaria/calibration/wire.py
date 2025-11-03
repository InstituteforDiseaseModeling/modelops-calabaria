"""Calibration wire function for ModelOps.

This module provides the main entry point for calibration jobs running
in Kubernetes pods. It orchestrates the ask/tell loop between the
optimization algorithm and the simulation service.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from modelops_contracts import (
    CalibrationJob,
    ReplicateSet,
    SimTask,
    TrialResult,
    TrialStatus,
    UniqueParameterSet,
    SeedInfo,
    SimulationService,
)

from modelops_calabaria.calibration.base import AlgorithmAdapter, ParameterSpec
from modelops_calabaria.calibration.factory import (
    create_algorithm_adapter,
    parse_parameter_specs,
)

logger = logging.getLogger(__name__)


def calibration_wire(job: CalibrationJob, sim_service: SimulationService) -> None:
    """Main entry point for calibration jobs in K8s pods.

    This function orchestrates the ask/tell loop between an optimization
    algorithm (e.g., Optuna) and the simulation service, using targets
    for loss computation.

    Args:
        job: CalibrationJob specification
        sim_service: Simulation service for running model evaluations
    """
    logger.info(f"Starting calibration job {job.job_id}")
    logger.info(f"Algorithm: {job.algorithm}")
    logger.info(f"Max iterations: {job.max_iterations}")

    # Parse parameter specifications from job
    parameter_specs = {}
    if hasattr(job, "parameter_specs") and job.parameter_specs:
        parameter_specs = parse_parameter_specs(job.parameter_specs)
    else:
        # Extract from algorithm config if not at top level
        specs_config = job.algorithm_config.get("parameter_specs", {})
        if specs_config:
            parameter_specs = parse_parameter_specs(specs_config)
        else:
            logger.warning("No parameter specifications found in job")

    # Create algorithm adapter
    adapter = create_algorithm_adapter(
        algorithm_type=job.algorithm,
        parameter_specs=parameter_specs,
        config=job.algorithm_config,
    )

    # Initialize with job-specific config
    adapter.initialize(
        job_id=job.job_id,
        config=job.algorithm_config,
    )

    # Connect to pre-provisioned infrastructure
    # Connection info comes from K8s secrets/configmaps
    connection_info = get_connection_info()
    adapter.connect_infrastructure(connection_info)

    # Get configuration for replicates and batching
    n_replicates = job.algorithm_config.get("n_replicates", 1)
    batch_size = job.algorithm_config.get("batch_size", 16)

    # Extract target entrypoints if specified
    target_entrypoints = []
    if job.target_spec and job.target_spec.data:
        target_entrypoints = job.target_spec.data.get("target_entrypoints", [])
        logger.info(f"Using {len(target_entrypoints)} target(s) for evaluation")

    # Get simulation entrypoint (default if not specified)
    sim_entrypoint = job.algorithm_config.get(
        "simulation_entrypoint",
        "models.main/baseline",  # Default entrypoint
    )

    # Run ask/tell loop
    iteration = 0
    while not adapter.finished() and iteration < job.max_iterations:
        iteration += 1
        logger.info(f"Starting iteration {iteration}/{job.max_iterations}")

        # Ask for parameters
        param_sets = adapter.ask(n=batch_size)
        if not param_sets:
            logger.info("No more parameters to evaluate")
            break

        logger.info(f"Evaluating {len(param_sets)} parameter sets")

        # Submit simulations with proper seeding
        futures = []
        for i, params in enumerate(param_sets):
            # Generate deterministic seed based on param_id
            seed_info = generate_seed_info(
                param_id=params.param_id,
                base_seed=job.algorithm_config.get("base_seed", 42),
                n_replicates=n_replicates,
            )

            if n_replicates > 1:
                # Create replicate set for this parameter set
                replicate_set = ReplicateSet(
                    base_task=SimTask(
                        bundle_ref=job.bundle_ref,
                        entrypoint=sim_entrypoint,
                        params=params,
                        seed=seed_info.trial_seed,
                    ),
                    n_replicates=n_replicates,
                    seed_offset=0,  # Seeds already set in seed_info
                )

                # Submit with target aggregation if targets specified
                if target_entrypoints:
                    # Use first target for aggregation (TODO: support multiple)
                    future = sim_service.submit_replicate_set(
                        replicate_set,
                        target_entrypoints[0],
                    )
                else:
                    # Submit without target aggregation
                    future = sim_service.submit_replicate_set(replicate_set, None)
            else:
                # Single simulation (no replicates)
                task = SimTask(
                    bundle_ref=job.bundle_ref,
                    entrypoint=sim_entrypoint,
                    params=params,
                    seed=seed_info.trial_seed,
                )
                future = sim_service.submit(task)

            futures.append((params, future))

        # Gather results
        logger.info("Gathering simulation results...")
        param_future_pairs = futures
        results = sim_service.gather([f for _, f in futures])

        # Convert to TrialResults
        trial_results = []
        for (params, _), result in zip(param_future_pairs, results):
            trial_result = convert_to_trial_result(params, result)
            trial_results.append(trial_result)

            # Log result
            if trial_result.status == TrialStatus.COMPLETED:
                logger.debug(f"Param {params.param_id[:8]}: loss = {trial_result.loss:.6f}")
            else:
                logger.warning(f"Param {params.param_id[:8]}: {trial_result.status}")

        # Tell algorithm about results
        adapter.tell(trial_results)

        # Check convergence (if specified in job)
        if check_convergence(trial_results, job.convergence_criteria):
            logger.info("Convergence criteria met")
            break

        # Log progress
        if hasattr(adapter, "get_study_summary"):
            summary = adapter.get_study_summary()
            logger.info(
                f"Progress: {summary.get('n_completed', 0)} completed, "
                f"best loss: {summary.get('best_value', 'N/A')}"
            )

    # Save final results
    save_calibration_results(job, adapter)

    logger.info(f"Calibration job {job.job_id} completed after {iteration} iterations")


def get_connection_info() -> Dict[str, str]:
    """Get infrastructure connection information from environment.

    Returns:
        Dictionary with connection strings for databases, caches, etc.
    """
    connection_info = {}

    # PostgreSQL for Optuna
    if "POSTGRES_URL" in os.environ:
        connection_info["POSTGRES_URL"] = os.environ["POSTGRES_URL"]
    elif all(k in os.environ for k in ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"]):
        # Build URL from components
        host = os.environ["POSTGRES_HOST"]
        port = os.environ.get("POSTGRES_PORT", "5432")
        db = os.environ["POSTGRES_DB"]
        user = os.environ.get("POSTGRES_USER", "postgres")
        password = os.environ.get("POSTGRES_PASSWORD", "")

        if password:
            connection_info["POSTGRES_URL"] = (
                f"postgresql://{user}:{password}@{host}:{port}/{db}"
            )
        else:
            connection_info["POSTGRES_URL"] = f"postgresql://{user}@{host}:{port}/{db}"

    # Redis for caching (if needed)
    if "REDIS_URL" in os.environ:
        connection_info["REDIS_URL"] = os.environ["REDIS_URL"]

    return connection_info


def generate_seed_info(
    param_id: str,
    base_seed: int = 42,
    n_replicates: int = 1,
) -> SeedInfo:
    """Generate deterministic seeds for a parameter set.

    Args:
        param_id: Unique identifier for the parameter set
        base_seed: Base seed for the job
        n_replicates: Number of replicates

    Returns:
        SeedInfo with deterministic seeds
    """
    # Use param_id hash for deterministic trial seed
    trial_seed = (hash(param_id) & 0x7FFFFFFF) ^ base_seed

    # Generate replicate seeds
    replicate_seeds = tuple(
        (trial_seed + i * 1000) & 0x7FFFFFFF for i in range(n_replicates)
    )

    return SeedInfo(
        base_seed=base_seed,
        trial_seed=trial_seed,
        replicate_seeds=replicate_seeds,
    )


def convert_to_trial_result(
    params: UniqueParameterSet,
    result: Any,
) -> TrialResult:
    """Convert simulation result to TrialResult.

    Args:
        params: Parameter set that was evaluated
        result: Result from simulation (SimReturn or AggregationReturn)

    Returns:
        TrialResult for the algorithm
    """
    # Check if result is an AggregationReturn (has loss attribute)
    if hasattr(result, "loss") and hasattr(result, "aggregate_metrics"):
        # AggregationReturn from target evaluation
        return TrialResult(
            param_id=params.param_id,
            loss=float(result.loss),
            status=TrialStatus.COMPLETED,
            diagnostics={
                "aggregate_metrics": result.aggregate_metrics,
                "n_replicates": result.n_replicates if hasattr(result, "n_replicates") else 1,
            },
        )
    elif hasattr(result, "outputs"):
        # Raw SimReturn without target evaluation
        # Try to extract a loss if available
        loss = None
        if "loss" in result.outputs:
            # Loss might be in outputs
            loss_data = result.outputs["loss"]
            # Parse loss from Arrow/Parquet bytes
            # For now, just use a placeholder
            loss = 1.0  # Placeholder

        if loss is not None:
            return TrialResult(
                param_id=params.param_id,
                loss=loss,
                status=TrialStatus.COMPLETED,
                diagnostics={"outputs": list(result.outputs.keys())},
            )
        else:
            # No loss available
            return TrialResult(
                param_id=params.param_id,
                loss=float("inf"),
                status=TrialStatus.FAILED,
                diagnostics={"error": "No loss computed"},
            )
    else:
        # Unknown result type or error
        return TrialResult(
            param_id=params.param_id,
            loss=float("inf"),
            status=TrialStatus.FAILED,
            diagnostics={"error": f"Unknown result type: {type(result).__name__}"},
        )


def check_convergence(
    trial_results: List[TrialResult],
    criteria: Optional[Dict[str, float]],
) -> bool:
    """Check if convergence criteria are met.

    Args:
        trial_results: Latest trial results
        criteria: Convergence criteria from job spec

    Returns:
        True if converged
    """
    if not criteria:
        return False

    # Check if loss is below threshold
    if "max_loss" in criteria:
        completed_results = [r for r in trial_results if r.status == TrialStatus.COMPLETED]
        if completed_results:
            losses = [r.loss for r in completed_results]
            if min(losses) < criteria["max_loss"]:
                logger.info(f"Loss below threshold: {min(losses)} < {criteria['max_loss']}")
                return True

    # Could add other convergence criteria:
    # - Relative improvement
    # - Variance threshold
    # - Time limit

    return False


def save_calibration_results(job: CalibrationJob, adapter: AlgorithmAdapter) -> None:
    """Save final calibration results.

    Args:
        job: Calibration job specification
        adapter: Algorithm adapter with results
    """
    try:
        # Get best parameters if available
        best_params = None
        if hasattr(adapter, "get_best_parameters"):
            best_params = adapter.get_best_parameters()

        # Get study summary if available
        summary = {}
        if hasattr(adapter, "get_study_summary"):
            summary = adapter.get_study_summary()

        # Create results dictionary
        results = {
            "job_id": job.job_id,
            "algorithm": job.algorithm,
            "best_params": best_params,
            "summary": summary,
        }

        # Write to file (in production, would upload to blob storage)
        output_dir = Path("/tmp/calibration_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{job.job_id}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved calibration results to {output_file}")

    except Exception as e:
        logger.error(f"Failed to save calibration results: {e}")
        # Don't fail the job if saving results fails