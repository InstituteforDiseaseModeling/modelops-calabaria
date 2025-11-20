"""Calibration wire function for ModelOps.

This module provides the main entry point for calibration jobs running
in Kubernetes pods. It orchestrates the ask/tell loop between the
optimization algorithm and the simulation service.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from collections import defaultdict

from modelops_contracts import (
    CalibrationJob,
    ReplicateSet,
    SeedInfo,
    SimTask,
    SimulationService,
    TrialResult,
    TrialStatus,
    UniqueParameterSet,
)

from modelops_calabaria.calibration.base import AlgorithmAdapter, ParameterSpec
from modelops_calabaria.calibration.factory import (
    create_algorithm_adapter,
    parse_parameter_specs,
)

logger = logging.getLogger(__name__)


def calibration_wire(job: CalibrationJob, sim_service: SimulationService, prov_store=None) -> None:
    """Main entry point for calibration jobs in K8s pods.

    This function orchestrates the ask/tell loop between an optimization
    algorithm (e.g., Optuna) and the simulation service, using targets
    for loss computation.

    Args:
        job: CalibrationJob specification
        sim_service: Simulation service for running model evaluations
        prov_store: Optional ProvenanceStore for Azure uploads
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
        evaluation_plan: list[tuple[tuple[UniqueParameterSet, str | None], Any]] = []
        param_lookup: dict[str, UniqueParameterSet] = {}
        param_order: list[str] = []
        targets_to_run = target_entrypoints or [None]
        multi_target = len(target_entrypoints) > 1

        for params in param_sets:
            if params.param_id not in param_lookup:
                param_lookup[params.param_id] = params
                param_order.append(params.param_id)

            seed_info = generate_seed_info(
                param_id=params.param_id,
                base_seed=job.algorithm_config.get("base_seed", 42),
                n_replicates=n_replicates,
            )

            replicate_set = ReplicateSet(
                base_task=SimTask(
                    bundle_ref=job.bundle_ref,
                    entrypoint=sim_entrypoint,
                    params=params,
                    seed=seed_info.trial_seed,
                ),
                n_replicates=n_replicates,
                seed_offset=0,
            )

            for target in targets_to_run:
                future = sim_service.submit_replicate_set(replicate_set, target)
                evaluation_plan.append(((params, target), future))

        # Gather results
        logger.info("Gathering simulation results...")
        param_future_pairs = evaluation_plan

        # Handle potential failures in gather
        try:
            results = sim_service.gather([f for _, f in evaluation_plan])
            logger.info(f"Gathered {len(results)} results, types: {[type(r).__name__ for r in results]}")
        except Exception as e:
            logger.error(f"Failed to gather results: {e}")
            trial_results = [
                _failed_trial(
                    param_lookup[param_id],
                    f"Aggregation failed: {str(e)[:200]}",
                )
                for param_id in param_order
            ]
        else:
            param_results_map: dict[str, TrialResult] = {}
            param_failures: dict[str, TrialResult] = {}
            param_target_results: dict[str, list[tuple[str, TrialResult]]] = defaultdict(list)

            for ((params, target), _), result in zip(param_future_pairs, results):
                if params.param_id in param_failures:
                    continue

                if isinstance(result, Exception):
                    logger.error(
                        f"Param {params.param_id[:8]}: Got Exception: {type(result).__name__}: {str(result)[:200]}"
                    )
                    param_failures[params.param_id] = _failed_trial(
                        params,
                        str(result)[:200],
                    )
                    continue

                trial_result = convert_to_trial_result(params, result)
                if multi_target and target:
                    param_target_results[params.param_id].append((target, trial_result))
                else:
                    param_results_map[params.param_id] = trial_result

            for param_id, target_results in param_target_results.items():
                if param_id in param_failures:
                    continue
                combined = _combine_target_results(param_lookup[param_id], target_results)
                if combined.status == TrialStatus.FAILED:
                    param_failures[param_id] = combined
                else:
                    param_results_map[param_id] = combined

            trial_results = []
            for param_id in param_order:
                if param_id in param_failures:
                    trial_results.append(param_failures[param_id])
                elif param_id in param_results_map:
                    trial_results.append(param_results_map[param_id])

        # Log results
        for trial_result in trial_results:
            if trial_result.status == TrialStatus.COMPLETED:
                logger.debug(f"Param {trial_result.param_id[:8]}: loss = {trial_result.loss:.6f}")
            else:
                logger.warning(f"Param {trial_result.param_id[:8]}: {trial_result.status}")

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
    save_calibration_results(job, adapter, prov_store=prov_store)

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


def _stable_seed(namespace: str, base_seed: int) -> int:
    """Generate a stable uint31 seed from arbitrary identifiers."""
    h = hashlib.blake2b(digest_size=8)
    h.update(namespace.encode("utf-8"))
    h.update(str(base_seed).encode("utf-8"))
    return int.from_bytes(h.digest(), "big") & 0x7FFFFFFF


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
    trial_seed = _stable_seed(f"{param_id}:trial", base_seed)
    replicate_seeds = tuple(
        _stable_seed(f"{param_id}:replicate:{i}", base_seed) for i in range(n_replicates)
    )

    return SeedInfo(
        base_seed=base_seed,
        trial_seed=trial_seed,
        replicate_seeds=replicate_seeds,
    )


def _failed_trial(params: UniqueParameterSet, message: str, extra_diag: dict | None = None) -> TrialResult:
    diag = {"error": message}
    if extra_diag:
        diag.update(extra_diag)
    return TrialResult(
        param_id=params.param_id,
        loss=float("nan"),
        status=TrialStatus.FAILED,
        diagnostics=diag,
    )


def _combine_target_results(
    param: UniqueParameterSet,
    target_results: list[tuple[str, TrialResult]],
) -> TrialResult:
    failed = {
        target: result for target, result in target_results if result.status != TrialStatus.COMPLETED
    }
    if failed:
        diag = {
            "targets": {
                target: {
                    "status": result.status.value,
                    "diagnostics": result.diagnostics,
                }
                for target, result in target_results
            }
        }
        return _failed_trial(param, "One or more targets failed", diag)

    combined_loss = sum(result.loss for _, result in target_results) / len(target_results)
    diag_targets = {
        target: {
            "loss": result.loss,
            **result.diagnostics,
        }
        for target, result in target_results
    }
    return TrialResult(
        param_id=param.param_id,
        loss=combined_loss,
        status=TrialStatus.COMPLETED,
        diagnostics={"targets": diag_targets},
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
    # Log what we actually got for debugging
    logger.debug(f"Converting result of type {type(result).__name__}")

    # Handle dictionary results (common when deserialized from JSON)
    if isinstance(result, dict):
        if "loss" in result and "aggregation_id" in result:
            # AggregationReturn as dictionary from target evaluation
            return TrialResult(
                param_id=params.param_id,
                loss=float(result["loss"]),
                status=TrialStatus.COMPLETED,
                diagnostics={
                    "aggregation_id": result.get("aggregation_id"),
                    "n_replicates": result.get("n_replicates", 1),
                    "target_diagnostics": result.get("diagnostics", {}),
                },
            )
        elif "outputs" in result:
            outputs = result.get("outputs", {})
            if "loss" not in outputs:
                return _failed_trial(
                    params,
                    "No loss provided in outputs",
                    {"outputs": list(outputs.keys())},
                )

            return TrialResult(
                param_id=params.param_id,
                loss=float(outputs["loss"]),
                status=TrialStatus.COMPLETED,
                diagnostics={"outputs": list(outputs.keys())},
            )

    # Check if result is an AggregationReturn object (has loss and aggregation_id attributes)
    if hasattr(result, "loss") and hasattr(result, "aggregation_id"):
        # AggregationReturn from target evaluation
        return TrialResult(
            param_id=params.param_id,
            loss=float(result.loss),
            status=TrialStatus.COMPLETED,
            diagnostics={
                "aggregation_id": result.aggregation_id,
                "n_replicates": result.n_replicates if hasattr(result, "n_replicates") else 1,
                "target_diagnostics": result.diagnostics if hasattr(result, "diagnostics") else {},
            },
        )
    elif hasattr(result, "outputs"):
        # Raw SimReturn without target evaluation
        if "loss" not in result.outputs:
            return _failed_trial(
                params,
                "No loss computed",
                {"outputs": list(result.outputs.keys())},
            )

        return _failed_trial(
            params,
            "Loss extraction for raw SimReturn outputs is not supported; "
            "provide AggregationReturn with explicit loss.",
            {"outputs": list(result.outputs.keys())},
        )
    else:
        # Unknown result type or error
        return _failed_trial(
            params,
            f"Unknown result type: {type(result).__name__}",
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


def save_calibration_results(job: CalibrationJob, adapter: AlgorithmAdapter, prov_store=None) -> None:
    """Save final calibration results and upload to Azure.

    Args:
        job: Calibration job specification
        adapter: Algorithm adapter with results
        prov_store: Optional ProvenanceStore for Azure uploads
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

        # Write to local directory first (matches simulation job pattern)
        local_dir = Path("/tmp/modelops/provenance/token/v1/views/jobs") / job.job_id / "calibration"
        local_dir.mkdir(parents=True, exist_ok=True)

        output_file = local_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved calibration results to {output_file}")

        # Upload to Azure if ProvenanceStore is available (matches simulation job pattern)
        if prov_store and hasattr(prov_store, "supports_remote_uploads") and prov_store.supports_remote_uploads():
            try:
                logger.info("Uploading calibration results to Azure...")

                # Upload the calibration directory
                remote_prefix = f"views/jobs/{job.job_id}/calibration"
                prov_store.upload_directory(local_dir, remote_prefix)

                logger.info(f"Calibration results uploaded to Azure: {remote_prefix}")
            except Exception as e:
                logger.error(f"Failed to upload calibration results to Azure: {e}")
                # Continue without upload - results are still saved locally

    except Exception as e:
        logger.error(f"Failed to save calibration results: {e}")
        # Don't fail the job if saving results fails
