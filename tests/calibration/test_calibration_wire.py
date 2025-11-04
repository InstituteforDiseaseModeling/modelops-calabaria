"""Tests for calibration wire function."""

import pytest
from unittest.mock import MagicMock, patch, call
import os

from modelops_contracts import (
    CalibrationJob,
    TargetSpec,
    TrialResult,
    TrialStatus,
    UniqueParameterSet,
)

from modelops_calabaria.calibration.wire import (
    calibration_wire,
    get_connection_info,
    generate_seed_info,
    convert_to_trial_result,
    check_convergence,
)


class TestCalibrationWire:
    """Test suite for calibration wire functionality."""

    def test_get_connection_info_from_url(self):
        """Test getting connection info from POSTGRES_URL."""
        with patch.dict(os.environ, {"POSTGRES_URL": "postgresql://test:pass@localhost:5432/db"}):
            conn_info = get_connection_info()
            assert conn_info["POSTGRES_URL"] == "postgresql://test:pass@localhost:5432/db"

    def test_get_connection_info_from_components(self):
        """Test building connection URL from components."""
        with patch.dict(
            os.environ,
            {
                "POSTGRES_HOST": "pg-host",
                "POSTGRES_PORT": "5433",
                "POSTGRES_DB": "optuna_db",
                "POSTGRES_USER": "optuna_user",
                "POSTGRES_PASSWORD": "secret",
            },
            clear=True,
        ):
            conn_info = get_connection_info()
            assert conn_info["POSTGRES_URL"] == "postgresql://optuna_user:secret@pg-host:5433/optuna_db"

    def test_generate_seed_info(self):
        """Test seed generation from param_id."""
        seed_info = generate_seed_info(
            param_id="abc123",
            base_seed=42,
            n_replicates=3,
        )

        assert seed_info.base_seed == 42
        assert seed_info.trial_seed != 42  # Should be hashed
        assert len(seed_info.replicate_seeds) == 3
        assert all(isinstance(s, int) for s in seed_info.replicate_seeds)

        # Test determinism - same param_id should give same seeds
        seed_info2 = generate_seed_info("abc123", 42, 3)
        assert seed_info.trial_seed == seed_info2.trial_seed
        assert seed_info.replicate_seeds == seed_info2.replicate_seeds

    def test_convert_to_trial_result_with_aggregation(self):
        """Test converting AggregationReturn to TrialResult."""
        params = UniqueParameterSet.from_dict({"beta": 0.5})

        # Mock AggregationReturn
        agg_result = MagicMock()
        agg_result.loss = 0.123
        agg_result.aggregation_id = "agg-123"
        agg_result.n_replicates = 10
        agg_result.diagnostics = {"target_info": "test"}

        trial_result = convert_to_trial_result(params, agg_result)

        assert trial_result.param_id == params.param_id
        assert trial_result.loss == 0.123
        assert trial_result.status == TrialStatus.COMPLETED
        assert trial_result.diagnostics["aggregation_id"] == "agg-123"
        assert trial_result.diagnostics["n_replicates"] == 10
        assert trial_result.diagnostics["target_diagnostics"] == {"target_info": "test"}

    def test_convert_to_trial_result_without_aggregation(self):
        """Test converting raw SimReturn to TrialResult."""
        params = UniqueParameterSet.from_dict({"beta": 0.5})

        # Mock SimReturn
        sim_result = MagicMock()
        sim_result.outputs = {"trajectories": b"data", "metrics": b"data"}
        del sim_result.loss  # No loss attribute

        trial_result = convert_to_trial_result(params, sim_result)

        assert trial_result.param_id == params.param_id
        assert trial_result.status == TrialStatus.FAILED  # No loss available
        assert trial_result.diagnostics.get("outputs") == ["trajectories", "metrics"]

    def test_check_convergence_max_loss(self):
        """Test convergence based on max_loss threshold."""
        results = [
            TrialResult(param_id="1", loss=0.5, status=TrialStatus.COMPLETED),
            TrialResult(param_id="2", loss=0.05, status=TrialStatus.COMPLETED),
            TrialResult(param_id="3", loss=0.3, status=TrialStatus.COMPLETED),
        ]

        # Should converge when min loss < threshold
        criteria = {"max_loss": 0.1}
        assert check_convergence(results, criteria) is True

        # Should not converge when min loss >= threshold
        criteria = {"max_loss": 0.01}
        assert check_convergence(results, criteria) is False

        # No convergence without criteria
        assert check_convergence(results, None) is False

    @patch("modelops_calabaria.calibration.wire.create_algorithm_adapter")
    @patch("modelops_calabaria.calibration.wire.save_calibration_results")
    def test_calibration_wire_basic_flow(self, mock_save, mock_create_adapter):
        """Test basic calibration wire flow."""
        # Create mock job
        job = CalibrationJob(
            job_id="test_job",
            bundle_ref="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            algorithm="optuna",
            target_spec=TargetSpec(
                data={"target_entrypoints": ["targets/sir"]},
                loss_function="mse",
            ),
            max_iterations=2,
            convergence_criteria={},
            algorithm_config={
                "max_trials": 10,
                "batch_size": 2,
                "n_replicates": 3,
                "parameter_specs": {
                    "beta": {"lower": 0.1, "upper": 0.9},
                },
            },
        )

        # Create mock simulation service
        mock_sim_service = MagicMock()
        mock_agg_result = MagicMock()
        mock_agg_result.loss = 0.15
        mock_agg_result.aggregation_id = "agg-test-123"
        mock_agg_result.n_replicates = 3
        mock_agg_result.diagnostics = {"mean": 50}
        mock_sim_service.gather.return_value = [mock_agg_result, mock_agg_result]

        # Create mock adapter
        mock_adapter = MagicMock()
        mock_adapter.finished.side_effect = [False, True]  # Run 1 iteration then finish
        mock_param_set = UniqueParameterSet.from_dict({"beta": 0.5})
        mock_adapter.ask.return_value = [mock_param_set, mock_param_set]
        mock_create_adapter.return_value = mock_adapter

        # Run calibration wire
        with patch("modelops_calabaria.calibration.wire.get_connection_info") as mock_conn:
            mock_conn.return_value = {"POSTGRES_URL": "postgresql://test"}
            calibration_wire(job, mock_sim_service)

        # Verify adapter was initialized
        mock_adapter.initialize.assert_called_once_with(
            job_id="test_job",
            config=job.algorithm_config,
        )

        # Verify infrastructure was connected
        mock_adapter.connect_infrastructure.assert_called_once()

        # Verify ask was called
        mock_adapter.ask.assert_called_with(n=2)

        # Verify simulations were submitted
        assert mock_sim_service.submit_replicate_set.call_count == 2

        # Verify results were gathered
        mock_sim_service.gather.assert_called_once()

        # Verify tell was called with results
        mock_adapter.tell.assert_called_once()
        tell_args = mock_adapter.tell.call_args[0][0]
        assert len(tell_args) == 2  # 2 trial results
        assert all(isinstance(r, TrialResult) for r in tell_args)

        # Verify results were saved
        mock_save.assert_called_once_with(job, mock_adapter, prov_store=None)

    @patch("modelops_calabaria.calibration.wire.create_algorithm_adapter")
    def test_calibration_wire_convergence(self, mock_create_adapter):
        """Test calibration wire stops on convergence."""
        # Create mock job with convergence criteria
        job = CalibrationJob(
            job_id="test_job",
            bundle_ref="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            algorithm="optuna",
            target_spec=TargetSpec(
                data={},
                loss_function="mse",
            ),
            max_iterations=10,  # High limit
            convergence_criteria={"max_loss": 0.1},  # Will converge
            algorithm_config={
                "batch_size": 1,
                "n_replicates": 1,
            },
        )

        # Create mock simulation service that returns good result
        mock_sim_service = MagicMock()
        mock_result = MagicMock()
        mock_result.loss = 0.05  # Below convergence threshold
        mock_result.aggregation_id = "agg-convergence-test"
        mock_result.n_replicates = 1
        mock_result.diagnostics = {}
        mock_sim_service.gather.return_value = [mock_result]

        # Create mock adapter
        mock_adapter = MagicMock()
        mock_adapter.finished.return_value = False
        mock_param_set = UniqueParameterSet.from_dict({"beta": 0.5})
        mock_adapter.ask.return_value = [mock_param_set]
        mock_create_adapter.return_value = mock_adapter

        # Run calibration wire
        with patch("modelops_calabaria.calibration.wire.get_connection_info") as mock_conn:
            with patch("modelops_calabaria.calibration.wire.save_calibration_results"):
                mock_conn.return_value = {}
                calibration_wire(job, mock_sim_service)

        # Should have stopped after 1 iteration due to convergence
        mock_adapter.ask.assert_called_once()
        mock_adapter.tell.assert_called_once()