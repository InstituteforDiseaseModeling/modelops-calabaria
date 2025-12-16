"""Optuna adapter for ModelOps calibration.

This module implements the AlgorithmAdapter interface for Optuna,
enabling distributed hyperparameter optimization using PostgreSQL backend.
"""

import logging
from typing import Any, Dict, List, Optional

import optuna
from optuna.samplers import TPESampler, BaseSampler
from optuna.storages import BaseStorage

from modelops_contracts.types import (
    UniqueParameterSet,
    TrialResult,
    TrialStatus,
)
from modelops_calabaria.calibration.base import (
    AlgorithmAdapter,
    InfrastructureRequirements,
)
from modelops_calabaria.parameters import ParameterSpec

logger = logging.getLogger(__name__)


class OptunaAdapter(AlgorithmAdapter):
    """Adapter for Optuna optimization algorithm.

    This adapter connects Optuna to ModelOps infrastructure, enabling
    distributed optimization with PostgreSQL backend for persistence.
    """

    def __init__(
        self,
        parameter_specs: Dict[str, ParameterSpec],
        max_trials: int = 100,
    ):
        """Initialize Optuna adapter.

        Args:
            parameter_specs: Parameter specifications (name -> spec)
            max_trials: Maximum number of trials to run
        """
        self.parameter_specs = parameter_specs
        self.max_trials = max_trials
        self.study: Optional[optuna.Study] = None
        self.storage: Optional[BaseStorage] = None
        self.study_name: Optional[str] = None
        self.sampler: Optional[BaseSampler] = None
        # Track pending trials (asked but not yet told)
        self._pending_trials: Dict[str, optuna.Trial] = {}

    def initialize(self, job_id: str, config: Dict[str, Any]) -> None:
        """Initialize Optuna study with job configuration.

        Args:
            job_id: Unique identifier for this calibration job
            config: Optuna-specific configuration
        """
        self.study_name = f"study_{job_id}"

        # Configure sampler
        sampler_config = config.get("sampler", {})
        sampler_type = sampler_config.get("type", "tpe")

        if sampler_type == "tpe":
            self.sampler = TPESampler(
                n_startup_trials=sampler_config.get("n_startup_trials", 10),
                n_ei_candidates=sampler_config.get("n_ei_candidates", 24),
            )
        else:
            # Default to TPE for unknown samplers
            logger.warning(f"Unknown sampler type: {sampler_type}, using TPE")
            self.sampler = TPESampler()

        # Update max trials if specified
        if "max_trials" in config:
            self.max_trials = config["max_trials"]

        logger.info(
            f"Initialized Optuna adapter for job {job_id} with {sampler_type} sampler"
        )

    def connect_infrastructure(self, connection_info: Dict[str, str]) -> None:
        """Connect to PostgreSQL provisioned by adaptive infra.

        Args:
            connection_info: Connection details from K8s secrets
        """
        # Get database URL from connection info
        db_url = connection_info.get("POSTGRES_URL")
        if not db_url:
            # Fallback to in-memory storage for development
            logger.warning("No POSTGRES_URL provided, using in-memory storage")
            self.storage = optuna.storages.InMemoryStorage()
        else:
            # Connect to PostgreSQL
            self.storage = optuna.storages.RDBStorage(
                url=db_url,
                heartbeat_interval=60,  # Heartbeat for distributed optimization
                grace_period=120,  # Grace period before considering trial stale
            )
            logger.info(f"Connected to PostgreSQL for study {self.study_name}")

        # Create or load existing study
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage,
            sampler=self.sampler,
            direction="minimize",
            load_if_exists=True,  # Resume if exists (fault tolerance)
        )

        # Log study status
        n_trials = len(self.study.trials)
        if n_trials > 0:
            logger.info(f"Resuming study {self.study_name} with {n_trials} existing trials")
        else:
            logger.info(f"Starting new study {self.study_name}")

    def ask(self, n: int) -> List[UniqueParameterSet]:
        """Ask Optuna for n parameter sets to evaluate.

        Args:
            n: Number of parameter sets to generate

        Returns:
            List of parameter sets to evaluate
        """
        if not self.study:
            raise RuntimeError("Study not initialized. Call connect_infrastructure first.")

        param_sets = []
        for _ in range(n):
            # Ask Optuna for a new trial
            trial = self.study.ask()

            # Sample parameters according to specifications
            # NOTE: Transforms now handled at CoordinateSystem level (Phase 5)
            params = {}
            for name, spec in self.parameter_specs.items():
                # Sample in natural space
                value = trial.suggest_float(name, spec.lower, spec.upper)
                params[name] = value

            # Create parameter set with stable ID
            param_set = UniqueParameterSet.from_dict(params)

            # Track this trial for later tell()
            self._pending_trials[param_set.param_id] = trial

            param_sets.append(param_set)

        logger.info(f"Generated {len(param_sets)} parameter sets for evaluation")
        return param_sets

    def tell(self, results: List[TrialResult]) -> None:
        """Report evaluation results back to Optuna.

        Args:
            results: List of trial results from evaluation
        """
        if not self.study:
            raise RuntimeError("Study not initialized. Call connect_infrastructure first.")

        for result in results:
            # Find the corresponding trial
            trial = self._pending_trials.get(result.param_id)
            if not trial:
                logger.warning(f"No pending trial found for param_id {result.param_id}")
                continue

            # Report result to Optuna
            if result.status == TrialStatus.COMPLETED:
                # Store additional information as user attributes BEFORE telling
                if result.diagnostics:
                    for key, value in result.diagnostics.items():
                        # Optuna requires JSON-serializable values
                        if isinstance(value, (int, float, str, bool, list, dict)):
                            trial.set_user_attr(key, value)

                # Report successful evaluation
                self.study.tell(trial, result.loss)

                logger.debug(f"Reported loss {result.loss} for param_id {result.param_id}")
            else:
                # Report failed evaluation
                state = (
                    optuna.trial.TrialState.FAIL
                    if result.status == TrialStatus.FAILED
                    else optuna.trial.TrialState.PRUNED
                )
                self.study.tell(trial, state=state)
                logger.debug(f"Reported {result.status} for param_id {result.param_id}")

            # Remove from pending
            del self._pending_trials[result.param_id]

        logger.info(f"Reported {len(results)} results to Optuna")

    def finished(self) -> bool:
        """Check if optimization is complete.

        Returns:
            True if optimization should stop
        """
        if not self.study:
            return False

        # Check if we've reached max trials
        n_trials = len(self.study.trials)
        if n_trials >= self.max_trials:
            logger.info(f"Reached max trials ({self.max_trials})")
            return True

        # Could add other stopping criteria here:
        # - Convergence detection
        # - Time limit
        # - Best value threshold
        # - No improvement for N trials

        return False

    @classmethod
    def get_infrastructure_requirements(cls) -> InfrastructureRequirements:
        """Declare infrastructure requirements for Optuna.

        Returns:
            Infrastructure requirements (PostgreSQL database)
        """
        return InfrastructureRequirements(
            databases={
                "optuna": {
                    "type": "postgresql",
                    "size": "small",
                    "storage": "10Gi",
                }
            },
            secrets={"POSTGRES_URL"},
        )

    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found so far.

        Returns:
            Best parameters or None if no successful trials
        """
        if not self.study or len(self.study.trials) == 0:
            return None

        best_trial = self.study.best_trial
        return best_trial.params if best_trial else None

    def get_study_summary(self) -> Dict[str, Any]:
        """Get summary of the optimization study.

        Returns:
            Summary statistics about the study
        """
        if not self.study:
            return {}

        trials = self.study.trials
        completed_trials = [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE]

        return {
            "study_name": self.study_name,
            "n_trials": len(trials),
            "n_completed": len(completed_trials),
            "n_failed": len([t for t in trials if t.state == optuna.trial.TrialState.FAIL]),
            "n_pruned": len([t for t in trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_value": self.study.best_value if completed_trials else None,
            "best_params": self.study.best_params if completed_trials else None,
        }