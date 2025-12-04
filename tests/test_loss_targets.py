"""Tests for loss-based calibration targets."""

import polars as pl
import pytest

from modelops_calabaria.core.alignment import ExactJoin
from modelops_calabaria.core.evaluation.aggregate import (
    IdentityAggregator,
    MeanAcrossReplicates,
)
from modelops_calabaria.core.evaluation.composable import LossEvaluator, TargetLossResult
from modelops_calabaria.core.evaluation.factories import annealed_mse, mean_signal_mse
from modelops_calabaria.core.evaluation.loss.base import SquaredErrorLoss
from modelops_calabaria.core.evaluation.reduce import MeanGroupedByReplicate, MeanReducer
from modelops_calabaria.core.target import (
    LossEvaluationResult,
    LossTarget,
    LossTargetSet,
)


class TestLossEvaluator:
    """Test the LossEvaluator class."""

    def test_basic_squared_error_evaluation(self, simple_aligned_data):
        """Test basic squared error evaluation without replicates."""
        evaluator = LossEvaluator(
            aggregator=IdentityAggregator(),
            loss_fn=SquaredErrorLoss(col="value"),
            reducer=MeanReducer(),
            weight=1.0,
            name="test_mse",
        )

        result = evaluator.evaluate(simple_aligned_data)

        assert isinstance(result, TargetLossResult)
        assert result.name == "test_mse"
        assert result.weight == 1.0
        assert result.loss == pytest.approx(result.weighted_loss)
        assert result.loss >= 0  # MSE is always non-negative

    def test_annealed_mse_with_replicates(self, multi_replicate_aligned_data):
        """Test annealed MSE (mean of per-replicate MSE)."""
        evaluator = LossEvaluator(
            aggregator=IdentityAggregator(),
            loss_fn=SquaredErrorLoss(col="value"),
            reducer=MeanGroupedByReplicate(),
            weight=1.0,
            name="annealed_mse",
        )

        result = evaluator.evaluate(multi_replicate_aligned_data)

        assert isinstance(result, TargetLossResult)
        assert result.loss >= 0
        # With replicates, this should include variance penalty

    def test_mean_signal_mse_with_replicates(self, multi_replicate_aligned_data):
        """Test mean-signal MSE (average sims first, then MSE)."""
        evaluator = LossEvaluator(
            aggregator=MeanAcrossReplicates(["value"]),
            loss_fn=SquaredErrorLoss(col="value"),
            reducer=MeanReducer(),
            weight=1.0,
            name="mean_signal_mse",
        )

        result = evaluator.evaluate(multi_replicate_aligned_data)

        assert isinstance(result, TargetLossResult)
        assert result.loss >= 0

    def test_custom_weight(self, simple_aligned_data):
        """Test that weights are applied correctly."""
        weight = 2.5
        evaluator = LossEvaluator(
            aggregator=IdentityAggregator(),
            loss_fn=SquaredErrorLoss(col="value"),
            reducer=MeanReducer(),
            weight=weight,
        )

        result = evaluator.evaluate(simple_aligned_data)

        assert result.weight == weight
        assert result.weighted_loss == pytest.approx(result.loss * weight)

    def test_evaluator_pipeline_composition(self, multi_replicate_aligned_data):
        """Test that the aggregate → loss → reduce pipeline works correctly."""
        # This should match mean_signal_mse behavior
        evaluator = LossEvaluator(
            aggregator=MeanAcrossReplicates(["value"]),
            loss_fn=SquaredErrorLoss(col="value"),
            reducer=MeanReducer(),
        )

        result = evaluator.evaluate(multi_replicate_aligned_data)

        # After aggregation, replicate_col should be None in aligned_data
        assert result.aligned_data.replicate_col is None


class TestLossTarget:
    """Test the LossTarget class."""

    def test_single_target_evaluation(self, mock_sim_outputs, observed_data):
        """Test evaluating a single loss target."""
        evaluator = annealed_mse("value", weight=1.0)
        alignment = ExactJoin(on_cols=["time"])

        target = LossTarget(
            name="test_target",
            model_output="output1",
            data=observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        result = target.evaluate(mock_sim_outputs)

        assert isinstance(result, TargetLossResult)
        assert result.name == "annealed_mse[value]"
        assert result.loss >= 0

    def test_alignment_integration(self, mock_sim_outputs, observed_data):
        """Test that alignment is correctly integrated."""
        evaluator = mean_signal_mse("value")
        alignment = ExactJoin(on_cols=["time"])

        target = LossTarget(
            name="alignment_test",
            model_output="output1",
            data=observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        result = target.evaluate(mock_sim_outputs)

        # Should have successfully aligned and evaluated
        assert result.aligned_data is not None
        assert result.aligned_data.on_cols == ["time"]

    def test_model_output_extraction(self, mock_sim_outputs, observed_data):
        """Test that the correct model output is extracted."""
        evaluator = annealed_mse("value")
        alignment = ExactJoin(on_cols=["time"])

        # Create target for "output1"
        target = LossTarget(
            name="output1_target",
            model_output="output1",
            data=observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        result = target.evaluate(mock_sim_outputs)
        assert isinstance(result, TargetLossResult)

    def test_weight_application(self, mock_sim_outputs, observed_data):
        """Test that target weight is passed through."""
        weight = 3.0
        evaluator = annealed_mse("value", weight=weight)
        alignment = ExactJoin(on_cols=["time"])

        target = LossTarget(
            name="weighted_target",
            model_output="output1",
            data=observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        result = target.evaluate(mock_sim_outputs)
        assert result.weight == weight
        assert result.weighted_loss == pytest.approx(result.loss * weight)


class TestLossTargetSet:
    """Test the LossTargetSet class."""

    def test_multiple_targets_linear_combination(
        self, mock_sim_outputs, observed_data, observed_data_alt
    ):
        """Test that multiple targets are combined linearly."""
        target1 = LossTarget(
            name="target1",
            model_output="output1",
            data=observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=annealed_mse("value", weight=1.0),
        )

        target2 = LossTarget(
            name="target2",
            model_output="output1",
            data=observed_data_alt,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=mean_signal_mse("value", weight=1.0),
        )

        target_set = LossTargetSet(targets=[target1, target2])
        result = target_set.evaluate(mock_sim_outputs, seeds=[1, 2, 3])

        assert isinstance(result, LossEvaluationResult)
        assert len(result.target_results) == 2
        # Total loss should be sum of weighted losses
        expected_total = sum(r.weighted_loss for r in result.target_results)
        assert result.total_loss == pytest.approx(expected_total)

    def test_weighted_sum_computation(self, mock_sim_outputs, observed_data):
        """Test that weights are correctly applied in the sum."""
        target1 = LossTarget(
            name="target1",
            model_output="output1",
            data=observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=annealed_mse("value", weight=2.0),
        )

        target2 = LossTarget(
            name="target2",
            model_output="output1",
            data=observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=mean_signal_mse("value", weight=0.5),
        )

        target_set = LossTargetSet(targets=[target1, target2])
        result = target_set.evaluate(mock_sim_outputs, seeds=[1, 2, 3])

        # Manually compute expected total
        expected_total = (
            result.target_results[0].weighted_loss
            + result.target_results[1].weighted_loss
        )
        assert result.total_loss == pytest.approx(expected_total)

    def test_per_target_results(self, mock_sim_outputs, observed_data):
        """Test per-target result retrieval."""
        target1 = LossTarget(
            name="target_A",
            model_output="output1",
            data=observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=annealed_mse("value", weight=1.0),
        )

        target2 = LossTarget(
            name="target_B",
            model_output="output1",
            data=observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=mean_signal_mse("value", weight=1.0),
        )

        target_set = LossTargetSet(targets=[target1, target2])
        result = target_set.evaluate(mock_sim_outputs, seeds=[1, 2, 3])

        per_target_loss = result.per_target_loss()
        assert "annealed_mse[value]" in per_target_loss
        assert "mean_signal_mse[value]" in per_target_loss

        per_target_weighted = result.per_target_weighted_loss()
        assert len(per_target_weighted) == 2

    def test_as_dict_serialization(self, mock_sim_outputs, observed_data):
        """Test conversion to dictionary for serialization."""
        target = LossTarget(
            name="test_target",
            model_output="output1",
            data=observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=annealed_mse("value", weight=1.0),
        )

        target_set = LossTargetSet(targets=[target])
        result = target_set.evaluate(
            mock_sim_outputs, seeds=[1, 2, 3], metadata={"iteration": 42}
        )

        result_dict = result.as_dict()

        assert "loss" in result_dict
        assert "per_target" in result_dict
        assert "per_target_weighted" in result_dict
        assert "seeds" in result_dict
        assert result_dict["seeds"] == [1, 2, 3]
        assert result_dict["iteration"] == 42

    def test_empty_target_set(self, mock_sim_outputs):
        """Test that empty target set evaluates to zero loss."""
        target_set = LossTargetSet(targets=[])
        result = target_set.evaluate(mock_sim_outputs, seeds=[1, 2, 3])

        assert result.total_loss == 0.0
        assert len(result.target_results) == 0


class TestLossFactories:
    """Test factory functions for common loss evaluators."""

    def test_annealed_mse_factory(self):
        """Test annealed_mse factory creates correct evaluator."""
        evaluator = annealed_mse("test_col", weight=2.0)

        assert isinstance(evaluator, LossEvaluator)
        assert isinstance(evaluator.aggregator, IdentityAggregator)
        assert isinstance(evaluator.loss_fn, SquaredErrorLoss)
        assert isinstance(evaluator.reducer, MeanGroupedByReplicate)
        assert evaluator.weight == 2.0
        assert evaluator.name == "annealed_mse[test_col]"

    def test_mean_signal_mse_factory(self):
        """Test mean_signal_mse factory creates correct evaluator."""
        evaluator = mean_signal_mse("test_col", weight=0.5)

        assert isinstance(evaluator, LossEvaluator)
        assert isinstance(evaluator.aggregator, MeanAcrossReplicates)
        assert isinstance(evaluator.loss_fn, SquaredErrorLoss)
        assert isinstance(evaluator.reducer, MeanReducer)
        assert evaluator.weight == 0.5
        assert evaluator.name == "mean_signal_mse[test_col]"

    def test_factory_returns_loss_evaluator(self):
        """Test that factories return LossEvaluator instances."""
        ev1 = annealed_mse("col")
        ev2 = mean_signal_mse("col")

        assert isinstance(ev1, LossEvaluator)
        assert isinstance(ev2, LossEvaluator)
