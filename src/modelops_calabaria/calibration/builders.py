"""Builder classes for creating CalibrationSpec objects programmatically."""

from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json

from modelops_contracts import CalibrationSpec


class CalibrationSpecBuilder:
    """Base builder for creating CalibrationSpec objects.

    Provides a fluent interface for constructing calibration specifications
    that can be saved to JSON or used directly with ModelOps.
    """

    def __init__(self, model: str, scenario: str = "baseline", algorithm: str = "optuna"):
        """Initialize calibration builder.

        Args:
            model: Model class path (e.g., "models.sir:StarsimSIR")
            scenario: Scenario name (default: "baseline")
            algorithm: Algorithm type (default: "optuna")
        """
        # Convert model path format if needed
        model_path = model.split(":")[0]
        if model_path.endswith(".py"):
            model_path = model_path[:-3].replace("/", ".")

        self._model = model_path
        self._model_class = model  # Store original for metadata
        self._scenario = scenario
        self._algorithm = algorithm
        self._target_data = {}
        self._max_iterations = 100
        self._convergence_criteria = {}
        self._algorithm_config = {}
        self._outputs = None
        self._metadata = {
            "model_class": model,
            "created_by": f"{self.__class__.__name__}"
        }

    def set_targets(self, targets: Union[str, List[str]]) -> "CalibrationSpecBuilder":
        """Set target entrypoints for calibration.

        Args:
            targets: Single target or list of target entrypoints

        Returns:
            Self for chaining
        """
        if isinstance(targets, str):
            targets = [targets]
        self._target_data["target_entrypoints"] = targets
        return self

    def set_observed_data(self, path: Union[str, Path]) -> "CalibrationSpecBuilder":
        """Set path to observed data file.

        Args:
            path: Path to observed data (parquet/csv)

        Returns:
            Self for chaining
        """
        self._target_data["observed_file"] = str(path)
        return self

    def set_max_iterations(self, max_iterations: int) -> "CalibrationSpecBuilder":
        """Set maximum iterations for optimization.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            Self for chaining
        """
        self._max_iterations = max_iterations
        return self

    def set_outputs(self, outputs: List[str]) -> "CalibrationSpecBuilder":
        """Set specific model outputs to track.

        Args:
            outputs: List of output names

        Returns:
            Self for chaining
        """
        self._outputs = outputs
        return self

    def add_metadata(self, key: str, value: Any) -> "CalibrationSpecBuilder":
        """Add metadata to the specification.

        Args:
            key: Metadata key
            value: Metadata value

        Returns:
            Self for chaining
        """
        self._metadata[key] = value
        return self

    def build(self) -> CalibrationSpec:
        """Build the CalibrationSpec object.

        Returns:
            CalibrationSpec ready for use

        Raises:
            ValueError: If required fields are missing
        """
        # Validate required fields
        if not self._target_data.get("observed_file"):
            raise ValueError("Observed data file must be specified")
        if not self._target_data.get("target_entrypoints"):
            raise ValueError("At least one target entrypoint must be specified")

        return CalibrationSpec(
            model=self._model,
            scenario=self._scenario,
            algorithm=self._algorithm,
            target_data=self._target_data,
            max_iterations=self._max_iterations,
            convergence_criteria=self._convergence_criteria,
            algorithm_config=self._algorithm_config,
            outputs=self._outputs,
            metadata=self._metadata
        )

    def save(self, path: Union[str, Path]) -> None:
        """Save CalibrationSpec to JSON file.

        Args:
            path: Output file path
        """
        spec = self.build()
        output_path = Path(path)

        with open(output_path, "w") as f:
            spec_dict = {
                "model": spec.model,
                "scenario": spec.scenario,
                "algorithm": spec.algorithm,
                "target_data": spec.target_data,
                "max_iterations": spec.max_iterations,
                "convergence_criteria": spec.convergence_criteria,
                "algorithm_config": spec.algorithm_config,
                "outputs": spec.outputs,
                "metadata": spec.metadata
            }
            json.dump(spec_dict, f, indent=2)


class OptunaCalibrationBuilder(CalibrationSpecBuilder):
    """Builder specifically for Optuna calibration specifications.

    Provides Optuna-specific configuration methods while maintaining
    the general builder interface.
    """

    def __init__(self, model: str, scenario: str = "baseline"):
        """Initialize Optuna calibration builder.

        Args:
            model: Model class path
            scenario: Scenario name
        """
        super().__init__(model, scenario, algorithm="optuna")
        # Initialize with Optuna defaults
        self._algorithm_config = {
            "max_trials": 100,
            "batch_size": 4,
            "n_replicates": 3,
            "parameter_specs": {},
            "sampler": {"type": "tpe"}
        }

    def add_parameter(
        self,
        name: str,
        lower: float,
        upper: float,
        transform: Optional[str] = None
    ) -> "OptunaCalibrationBuilder":
        """Add a parameter to calibrate.

        Args:
            name: Parameter name
            lower: Lower bound
            upper: Upper bound
            transform: Optional transform ("log", "logit", None)

        Returns:
            Self for chaining
        """
        param_spec = {
            "lower": lower,
            "upper": upper
        }
        if transform:
            param_spec["transform"] = transform

        self._algorithm_config["parameter_specs"][name] = param_spec
        self._metadata["n_parameters"] = len(self._algorithm_config["parameter_specs"])
        return self

    def set_max_trials(self, max_trials: int) -> "OptunaCalibrationBuilder":
        """Set maximum number of Optuna trials.

        Args:
            max_trials: Maximum trials

        Returns:
            Self for chaining
        """
        self._algorithm_config["max_trials"] = max_trials
        return self

    def set_batch_size(self, batch_size: int) -> "OptunaCalibrationBuilder":
        """Set batch size for parallel evaluations.

        Args:
            batch_size: Number of parallel evaluations

        Returns:
            Self for chaining
        """
        self._algorithm_config["batch_size"] = batch_size
        return self

    def set_n_replicates(self, n_replicates: int) -> "OptunaCalibrationBuilder":
        """Set number of replicates per parameter set.

        Args:
            n_replicates: Number of replicates

        Returns:
            Self for chaining
        """
        self._algorithm_config["n_replicates"] = n_replicates
        return self

    def set_sampler(
        self,
        sampler_type: str,
        n_startup_trials: Optional[int] = None,
        **kwargs
    ) -> "OptunaCalibrationBuilder":
        """Configure Optuna sampler.

        Args:
            sampler_type: Sampler type ("tpe", "random", "grid")
            n_startup_trials: Number of random startup trials (TPE only)
            **kwargs: Additional sampler options

        Returns:
            Self for chaining
        """
        sampler_config = {"type": sampler_type}

        if sampler_type == "tpe" and n_startup_trials is not None:
            sampler_config["n_startup_trials"] = n_startup_trials

        sampler_config.update(kwargs)
        self._algorithm_config["sampler"] = sampler_config
        return self

    def set_pruner(self, pruner_type: str, **kwargs) -> "OptunaCalibrationBuilder":
        """Configure Optuna pruner for early stopping.

        Args:
            pruner_type: Pruner type ("median", "percentile", "hyperband")
            **kwargs: Pruner-specific options

        Returns:
            Self for chaining
        """
        pruner_config = {"type": pruner_type}
        pruner_config.update(kwargs)
        self._algorithm_config["pruner"] = pruner_config
        return self

    def set_convergence(self, **kwargs) -> "OptunaCalibrationBuilder":
        """Set Optuna-specific convergence criteria.

        For MVP, these go in algorithm_config to avoid confusion.

        Args:
            **kwargs: Convergence options (e.g., early_stopping_patience=20)

        Returns:
            Self for chaining
        """
        # Put in algorithm_config for clarity in MVP
        self._algorithm_config.update(kwargs)
        return self

    def build(self) -> CalibrationSpec:
        """Build the CalibrationSpec with Optuna configuration.

        Returns:
            CalibrationSpec ready for use

        Raises:
            ValueError: If required fields are missing
        """
        # Add simulation entrypoint if not set
        if "simulation_entrypoint" not in self._algorithm_config:
            self._algorithm_config["simulation_entrypoint"] = self._model_class

        # Validate parameters specified
        if not self._algorithm_config["parameter_specs"]:
            raise ValueError("At least one parameter must be specified for calibration")

        return super().build()


class ABCCalibrationBuilder(CalibrationSpecBuilder):
    """Builder for ABC-SMC calibration specifications."""

    def __init__(self, model: str, scenario: str = "baseline"):
        """Initialize ABC-SMC calibration builder.

        Args:
            model: Model class path
            scenario: Scenario name
        """
        super().__init__(model, scenario, algorithm="abc-smc")
        self._algorithm_config = {
            "n_particles": 1000,
            "epsilon_schedule": [10, 5, 2, 1],
            "parameter_specs": {},
            "distance_function": "euclidean",
            "kernel": "uniform"
        }

    def add_parameter(
        self,
        name: str,
        lower: float,
        upper: float,
        prior: str = "uniform"
    ) -> "ABCCalibrationBuilder":
        """Add a parameter with prior distribution.

        Args:
            name: Parameter name
            lower: Lower bound
            upper: Upper bound
            prior: Prior distribution type

        Returns:
            Self for chaining
        """
        self._algorithm_config["parameter_specs"][name] = {
            "lower": lower,
            "upper": upper,
            "prior": prior
        }
        self._metadata["n_parameters"] = len(self._algorithm_config["parameter_specs"])
        return self

    def set_particles(self, n_particles: int) -> "ABCCalibrationBuilder":
        """Set number of particles.

        Args:
            n_particles: Number of particles in population

        Returns:
            Self for chaining
        """
        self._algorithm_config["n_particles"] = n_particles
        return self

    def set_epsilon_schedule(self, epsilons: List[float]) -> "ABCCalibrationBuilder":
        """Set epsilon tolerance schedule.

        Args:
            epsilons: Decreasing sequence of epsilon values

        Returns:
            Self for chaining
        """
        self._algorithm_config["epsilon_schedule"] = epsilons
        return self

    def set_distance_function(self, distance: str) -> "ABCCalibrationBuilder":
        """Set distance function for ABC.

        Args:
            distance: Distance metric ("euclidean", "manhattan", "custom")

        Returns:
            Self for chaining
        """
        self._algorithm_config["distance_function"] = distance
        return self

    def set_kernel(self, kernel: str, **kwargs) -> "ABCCalibrationBuilder":
        """Set perturbation kernel.

        Args:
            kernel: Kernel type ("uniform", "gaussian", "multivariate_normal")
            **kwargs: Kernel-specific parameters

        Returns:
            Self for chaining
        """
        self._algorithm_config["kernel"] = kernel
        if kwargs:
            self._algorithm_config["kernel_params"] = kwargs
        return self


__all__ = [
    "CalibrationSpecBuilder",
    "OptunaCalibrationBuilder",
    "ABCCalibrationBuilder",
]