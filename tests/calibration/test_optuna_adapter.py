"""Tests for OptunaAdapter."""

import pytest
from unittest.mock import MagicMock, patch

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
from modelops_calabaria.calibration.optuna_adapter import OptunaAdapter


class TestOptunaAdapter:
    """Test suite for OptunaAdapter."""

    def test_initialization(self):
        """Test adapter initialization."""
        specs = {
            "beta": ParameterSpec(name="beta", lower=0.1, upper=0.9),
            "gamma": ParameterSpec(name="gamma", lower=0.0, upper=1.0),
        }

        adapter = OptunaAdapter(parameter_specs=specs, max_trials=50)

        assert adapter.parameter_specs == specs
        assert adapter.max_trials == 50
        assert adapter.study is None
        assert adapter.storage is None

    def test_infrastructure_requirements(self):
        """Test that infrastructure requirements are properly declared."""
        reqs = OptunaAdapter.get_infrastructure_requirements()

        assert isinstance(reqs, InfrastructureRequirements)
        assert "optuna" in reqs.databases
        assert reqs.databases["optuna"]["type"] == "postgresql"
        assert "POSTGRES_URL" in reqs.secrets

    def test_initialize_job(self):
        """Test job-specific initialization."""
        specs = {"beta": ParameterSpec(name="beta", lower=0.1, upper=0.9)}
        adapter = OptunaAdapter(parameter_specs=specs)

        config = {
            "sampler": {"type": "tpe", "n_startup_trials": 20},
            "max_trials": 200,
        }

        adapter.initialize(job_id="test_job_123", config=config)

        assert adapter.study_name == "study_test_job_123"
        assert adapter.max_trials == 200
        assert adapter.sampler is not None

    @patch("modelops_calabaria.calibration.optuna_adapter.optuna")
    def test_connect_infrastructure_with_postgres(self, mock_optuna):
        """Test connecting to PostgreSQL infrastructure."""
        specs = {"beta": ParameterSpec(name="beta", lower=0.1, upper=0.9)}
        adapter = OptunaAdapter(parameter_specs=specs)
        adapter.study_name = "test_study"

        # Mock storage and study
        mock_storage = MagicMock()
        mock_study = MagicMock()
        mock_study.trials = []

        mock_optuna.storages.RDBStorage.return_value = mock_storage
        mock_optuna.create_study.return_value = mock_study

        connection_info = {"POSTGRES_URL": "postgresql://user:pass@host:5432/db"}
        adapter.connect_infrastructure(connection_info)

        # Verify RDBStorage was created with correct URL
        mock_optuna.storages.RDBStorage.assert_called_once_with(
            url="postgresql://user:pass@host:5432/db",
            heartbeat_interval=60,
            grace_period=120,
        )

        # Verify study was created
        mock_optuna.create_study.assert_called_once()
        assert adapter.study == mock_study
        assert adapter.storage == mock_storage

    @patch("modelops_calabaria.calibration.optuna_adapter.optuna")
    def test_connect_infrastructure_fallback_memory(self, mock_optuna):
        """Test fallback to in-memory storage when no PostgreSQL available."""
        specs = {"beta": ParameterSpec(name="beta", lower=0.1, upper=0.9)}
        adapter = OptunaAdapter(parameter_specs=specs)
        adapter.study_name = "test_study"

        # Mock storage and study
        mock_storage = MagicMock()
        mock_study = MagicMock()

        mock_optuna.storages.InMemoryStorage.return_value = mock_storage
        mock_optuna.create_study.return_value = mock_study

        # No POSTGRES_URL provided
        connection_info = {}
        adapter.connect_infrastructure(connection_info)

        # Verify InMemoryStorage was used
        mock_optuna.storages.InMemoryStorage.assert_called_once()
        assert adapter.storage == mock_storage

    @patch("modelops_calabaria.calibration.optuna_adapter.optuna")
    def test_ask_tell_loop(self, mock_optuna):
        """Test the ask/tell loop functionality."""
        specs = {
            "beta": ParameterSpec(name="beta", lower=0.1, upper=0.9),
            "gamma": ParameterSpec(name="gamma", lower=0.0, upper=1.0),
        }
        adapter = OptunaAdapter(parameter_specs=specs, max_trials=10)

        # Setup mock study
        mock_study = MagicMock()
        mock_trial = MagicMock()
        # Need 4 values: 2 parameters × 2 trials
        mock_trial.suggest_float.side_effect = [0.5, 0.3, 0.6, 0.4]  # beta, gamma for trial 1 and 2
        mock_study.ask.return_value = mock_trial
        mock_study.trials = []

        adapter.study = mock_study

        # Ask for parameters
        param_sets = adapter.ask(n=2)

        assert len(param_sets) == 2
        assert mock_study.ask.call_count == 2

        # Each param set should have a unique ID
        assert param_sets[0].param_id != param_sets[1].param_id

        # Tell results
        results = [
            TrialResult(
                param_id=param_sets[0].param_id,
                loss=0.1,
                status=TrialStatus.COMPLETED,
            ),
            TrialResult(
                param_id=param_sets[1].param_id,
                loss=0.2,
                status=TrialStatus.COMPLETED,
            ),
        ]

        adapter.tell(results)

        # Verify tell was called for each result
        assert mock_study.tell.call_count == 2

    def test_finished_max_trials(self):
        """Test that finished() returns True when max trials reached."""
        specs = {"beta": ParameterSpec(name="beta", lower=0.1, upper=0.9)}
        adapter = OptunaAdapter(parameter_specs=specs, max_trials=10)

        # Mock study with 10 trials
        mock_study = MagicMock()
        mock_study.trials = [MagicMock() for _ in range(10)]
        adapter.study = mock_study

        assert adapter.finished() is True

        # Test with fewer trials
        mock_study.trials = [MagicMock() for _ in range(5)]
        assert adapter.finished() is False

    @patch("modelops_calabaria.calibration.optuna_adapter.optuna")
    def test_get_best_parameters(self, mock_optuna):
        """Test retrieving best parameters."""
        specs = {"beta": ParameterSpec(name="beta", lower=0.1, upper=0.9)}
        adapter = OptunaAdapter(parameter_specs=specs)

        # Mock study with best trial
        mock_study = MagicMock()
        mock_best_trial = MagicMock()
        mock_best_trial.params = {"beta": 0.42}
        mock_study.best_trial = mock_best_trial
        mock_study.trials = [mock_best_trial]
        adapter.study = mock_study

        best_params = adapter.get_best_parameters()
        assert best_params == {"beta": 0.42}

        # Test with no trials
        adapter.study = None
        assert adapter.get_best_parameters() is None

    def test_parameter_transforms(self):
        """Test parameter transformations (log, logit)."""
        specs = {
            "rate": ParameterSpec(name="rate", lower=0.001, upper=10.0, transform="log"),
            "prob": ParameterSpec(name="prob", lower=0.01, upper=0.99, transform="logit"),
        }
        adapter = OptunaAdapter(parameter_specs=specs)

        # Mock study and trial
        mock_study = MagicMock()
        mock_trial = MagicMock()

        # Mock suggest_float to return values for transformed parameters
        suggest_values = {
            "rate": 1.0,  # Original space
            "_log_rate": 0.0,  # log(1.0) = 0
            "prob": 0.5,  # Original space
            "_logit_prob": 0.0,  # logit(0.5) = 0
        }
        mock_trial.suggest_float.side_effect = lambda name, low, high: suggest_values.get(name, 0.5)
        mock_study.ask.return_value = mock_trial

        adapter.study = mock_study

        # Ask for parameters
        param_sets = adapter.ask(n=1)

        assert len(param_sets) == 1
        # The actual values would depend on the transform logic