"""Tests for calibration wire function."""

import math
import os
from unittest.mock import MagicMock, patch, call

import pytest

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
        assert seed_info.trial_seed == 183415352
        assert len(seed_info.replicate_seeds) == 3
        assert all(isinstance(s, int) for s in seed_info.replicate_seeds)
        assert seed_info.replicate_seeds == (2068545870, 1805355463, 1722438088)

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
        assert math.isnan(trial_result.loss)
        assert trial_result.status == TrialStatus.FAILED  # No loss available
        assert trial_result.diagnostics.get("outputs") == ["trajectories", "metrics"]

    def test_convert_to_trial_result_dict_without_loss(self):
        params = UniqueParameterSet.from_dict({"gamma": 0.2})
        raw_dict = {"outputs": {"metrics": 5}}

        trial_result = convert_to_trial_result(params, raw_dict)

        assert trial_result.status == TrialStatus.FAILED
        assert math.isnan(trial_result.loss)
        assert trial_result.diagnostics["error"] == "No loss provided in outputs"

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

    @patch("modelops_calabaria.calibration.wire.create_algorithm_adapter")
    @patch("modelops_calabaria.calibration.wire.save_calibration_results")
    def test_calibration_wire_multi_target_combines_results(self, mock_save, mock_create_adapter):
        job = CalibrationJob(
            job_id="multi_target_job",
            bundle_ref="sha256:0000000000000000000000000000000000000000000000000000000000000000",
            algorithm="optuna",
            target_spec=TargetSpec(
                data={"target_entrypoints": ["targets/A", "targets/B"]},
                loss_function="mse",
            ),
            max_iterations=1,
            convergence_criteria={},
            algorithm_config={
                "max_trials": 1,
                "batch_size": 1,
                "n_replicates": 2,
                "parameter_specs": {"beta": {"lower": 0.1, "upper": 0.9}},
            },
        )

        mock_sim_service = MagicMock()
        future_a = MagicMock(name="future_a")
        future_b = MagicMock(name="future_b")
        mock_sim_service.submit_replicate_set.side_effect = [future_a, future_b]

        agg_a = MagicMock()
        agg_a.loss = 0.1
        agg_a.aggregation_id = "agg-a"
        agg_a.n_replicates = 2
        agg_a.diagnostics = {"target": "A"}

        agg_b = MagicMock()
        agg_b.loss = 0.3
        agg_b.aggregation_id = "agg-b"
        agg_b.n_replicates = 2
        agg_b.diagnostics = {"target": "B"}

        mock_sim_service.gather.return_value = [agg_a, agg_b]

        mock_adapter = MagicMock()
        mock_adapter.finished.side_effect = [False, True]
        mock_param_set = UniqueParameterSet.from_dict({"beta": 0.5})
        mock_adapter.ask.return_value = [mock_param_set]
        mock_create_adapter.return_value = mock_adapter

        calibration_wire(job, mock_sim_service, prov_store=None)

        # Ensure both targets submitted
        assert mock_sim_service.submit_replicate_set.call_count == 2
        mock_sim_service.gather.assert_called_once_with([future_a, future_b])

        # Verify combined TrialResult
        tell_args = mock_adapter.tell.call_args[0][0]
        assert len(tell_args) == 1
        combined_result = tell_args[0]
        assert combined_result.status == TrialStatus.COMPLETED
        assert combined_result.loss == pytest.approx((0.1 + 0.3) / 2)
        assert "targets" in combined_result.diagnostics
        assert set(combined_result.diagnostics["targets"].keys()) == {"targets/A", "targets/B"}

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
                data={"target_entrypoints": ["targets/sir"]},
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
        future = MagicMock(name="future")
        mock_sim_service.submit_replicate_set.return_value = future
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
