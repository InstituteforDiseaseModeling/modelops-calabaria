"""Base class for parameter sampling strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import uuid

from modelops_contracts import SimTask, format_entrypoint

from ..parameters import ParameterSpace


class SamplingStrategy(ABC):
    """Base class for parameter sampling strategies.

    All sampling strategies should inherit from this class and implement
    the sample method to generate parameter sets according to their
    specific algorithm.
    """

    def __init__(self, parameter_space: ParameterSpace):
        """Initialize sampling strategy.

        Args:
            parameter_space: The parameter space to sample from
        """
        self.parameter_space = parameter_space
        self._validate_space()

    def _validate_space(self) -> None:
        """Validate that the parameter space is suitable for sampling."""
        if not self.parameter_space.specs:
            raise ValueError("Parameter space must contain at least one parameter")

    @abstractmethod
    def sample(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate parameter samples.

        Args:
            n_samples: Number of parameter sets to generate

        Returns:
            List of parameter dictionaries
        """
        pass

    def generate_tasks(
        self,
        model_class: str,
        scenario: str,
        bundle_ref: str,
        n_samples: int,
        base_seed: int = 42,
        outputs: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[SimTask]:
        """Generate simulation tasks from parameter samples.

        Args:
            model_class: Python import path to model class
            scenario: Scenario name
            bundle_ref: Bundle reference for code version
            n_samples: Number of tasks to generate
            base_seed: Base random seed (each task gets base_seed + index)
            outputs: Optional list of outputs to extract
            config: Optional runtime configuration

        Returns:
            List of SimTask objects
        """
        # Generate parameter samples
        param_samples = self.sample(n_samples)

        # Create tasks
        tasks = []
        for i, params in enumerate(param_samples):
            task = SimTask.from_components(
                import_path=model_class,
                scenario=scenario,
                bundle_ref=bundle_ref,
                params=params,
                seed=base_seed + i,
                outputs=outputs,
                config=config,
            )
            tasks.append(task)

        return tasks

    @abstractmethod
    def method_name(self) -> str:
        """Return the name of this sampling method."""
        pass