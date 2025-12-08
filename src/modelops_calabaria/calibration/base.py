"""Base classes for calibration algorithm adapters.

This module defines the interface that all calibration algorithms must implement
to work with ModelOps infrastructure.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from modelops_contracts.adaptive import AdaptiveAlgorithm
from modelops_contracts.types import UniqueParameterSet, TrialResult


@dataclass(frozen=True)
class InfrastructureRequirements:
    """Declares infrastructure requirements for an algorithm.

    This allows ModelOps to provision the necessary infrastructure
    (databases, caches, etc.) before running the calibration job.

    Attributes:
        databases: Database requirements (name -> config)
        caches: Cache/Redis requirements (name -> config)
        volumes: Persistent volume requirements (name -> config)
        secrets: Names of secrets that will be provided
    """
    databases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    caches: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    volumes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    secrets: Set[str] = field(default_factory=set)


class AlgorithmAdapter(AdaptiveAlgorithm, ABC):
    """Base adapter for all calibration algorithms.

    This class bridges between the ModelOps infrastructure and specific
    optimization algorithms (Optuna, ABC-SMC, MCMC, etc.).

    The lifecycle is:
    1. Algorithm declares infrastructure requirements (class method)
    2. Infrastructure is provisioned by `mops adaptive up`
    3. Job starts, adapter is created
    4. initialize() is called with job configuration
    5. connect_infrastructure() is called with connection details
    6. ask/tell loop runs until finished()
    """

    @abstractmethod
    def initialize(self, job_id: str, config: Dict[str, Any]) -> None:
        """Initialize algorithm with job-specific configuration.

        Args:
            job_id: Unique identifier for this calibration job
            config: Algorithm-specific configuration from the job spec
        """
        pass

    @abstractmethod
    def connect_infrastructure(self, connection_info: Dict[str, str]) -> None:
        """Connect to pre-provisioned infrastructure.

        Args:
            connection_info: Connection details from K8s secrets/configmaps
                e.g., {"POSTGRES_URL": "postgresql://...", "REDIS_URL": "redis://..."}
        """
        pass

    @classmethod
    @abstractmethod
    def get_infrastructure_requirements(cls) -> InfrastructureRequirements:
        """Declare what infrastructure this algorithm needs.

        This is used by `mops adaptive up` to provision the necessary
        infrastructure before any jobs are run.

        Returns:
            Infrastructure requirements for this algorithm
        """
        pass

    # The ask/tell/finished methods are inherited from AdaptiveAlgorithm protocol
    # and must be implemented by concrete adapters
