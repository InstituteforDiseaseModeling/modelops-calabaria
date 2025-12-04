"""Tests for likelihood-based calibration targets."""

import numpy as np
import polars as pl
import pytest
from scipy.special import logsumexp

from modelops_calabaria.core.alignment import ExactJoin
from modelops_calabaria.core.evaluation.likelihood import (
    LikelihoodEvaluator,
    TargetLikelihoodResult,
)
from modelops_calabaria.core.evaluation.loglik_fns import (
    beta_binomial_loglik_factory,
    beta_binomial_loglik_per_rep,
    binomial_loglik_per_rep,
)
from modelops_calabaria.core.target import (
    LikelihoodEvaluationResult,
    LikelihoodTarget,
    LikelihoodTargetSet,
)


class TestLogLikelihoodFunctions:
    """Test log-likelihood functions."""

    def test_beta_binomial_per_rep_basic(self, beta_binomial_aligned_data):
        """Test basic beta-binomial log-likelihood computation."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)

        result = loglik_fn(beta_binomial_aligned_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)  # 3 replicates
        # All values should be <= 0 (log-likelihood is non-positive)
        assert np.all(result <= 0)

    def test_beta_binomial_per_rep_multiple_replicates(
        self, beta_binomial_aligned_data
    ):
        """Test that per-replicate log-likelihoods are correctly summed."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=2.0, beta=2.0)

        result = loglik_fn(beta_binomial_aligned_data)

        # Should have one value per replicate
        assert len(result) == 3

        # Each should be the sum of log-likelihoods for that replicate
        # (we can't test exact values without knowing the data, but we can test properties)
        assert np.all(np.isfinite(result))

    def test_beta_binomial_per_rep_no_replicate_col_raises(self, simple_aligned_data):
        """Test that missing replicate column raises ValueError."""
        loglik_fn = beta_binomial_loglik_per_rep("value", "n", alpha=1.0, beta=1.0)

        with pytest.raises(ValueError, match="requires a replicate column"):
            loglik_fn(simple_aligned_data)

    def test_binomial_per_rep_basic(self, binomial_aligned_data):
        """Test basic binomial log-likelihood computation."""
        loglik_fn = binomial_loglik_per_rep("p", "x", "n")

        result = loglik_fn(binomial_aligned_data)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)  # 3 replicates
        assert np.all(result <= 0)

    def test_beta_binomial_factory_validates_prior(self):
        """Test that factory function validates prior parameters."""
        # Valid prior
        loglik_fn = beta_binomial_loglik_factory("x", "n", prior=(1.0, 1.0))
        assert callable(loglik_fn)

        # Invalid priors
        with pytest.raises(ValueError, match="Prior must be a tuple"):
            beta_binomial_loglik_factory("x", "n", prior=(1.0,))  # Too short

        with pytest.raises(ValueError, match="Prior must be a tuple"):
            beta_binomial_loglik_factory("x", "n", prior=(1.0, 0.0))  # Zero value

        with pytest.raises(ValueError, match="Prior must be a tuple"):
            beta_binomial_loglik_factory("x", "n", prior=(1.0, -1.0))  # Negative


class TestLikelihoodEvaluator:
    """Test the LikelihoodEvaluator class."""

    def test_evaluator_returns_per_rep_logliks(self, beta_binomial_aligned_data):
        """Test that evaluator returns per-replicate log-likelihoods."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test_beta_binomial")

        result = evaluator.evaluate(beta_binomial_aligned_data)

        assert isinstance(result, TargetLikelihoodResult)
        assert result.name == "test_beta_binomial"
        assert isinstance(result.loglik_per_rep, np.ndarray)
        assert result.loglik_per_rep.shape == (3,)

    def test_no_marginalization_at_evaluator_level(self, beta_binomial_aligned_data):
        """Test that evaluator does NOT marginalize over replicates."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")

        result = evaluator.evaluate(beta_binomial_aligned_data)

        # Should still have per-replicate values, not a single scalar
        assert len(result.loglik_per_rep) == 3
        assert result.loglik_per_rep.ndim == 1

    def test_replicate_validation(self, simple_aligned_data):
        """Test that missing replicate column is caught."""
        loglik_fn = beta_binomial_loglik_per_rep("value", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")

        with pytest.raises(ValueError, match="requires a replicate column"):
            evaluator.evaluate(simple_aligned_data)


class TestLikelihoodTarget:
    """Test the LikelihoodTarget class."""

    def test_single_likelihood_target_evaluation(
        self, mock_sim_outputs_likelihood, beta_binomial_observed_data
    ):
        """Test evaluating a single likelihood target."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="beta_binomial")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="test_likelihood_target",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        result = target.evaluate(mock_sim_outputs_likelihood)

        assert isinstance(result, TargetLikelihoodResult)
        assert result.name == "beta_binomial"
        assert isinstance(result.loglik_per_rep, np.ndarray)

    def test_alignment_integration(
        self, mock_sim_outputs_likelihood, beta_binomial_observed_data
    ):
        """Test that alignment is correctly integrated."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="alignment_test",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        result = target.evaluate(mock_sim_outputs_likelihood)

        # Should have successfully aligned and evaluated
        assert result.aligned_data is not None
        assert result.aligned_data.on_cols == ["time"]
        assert result.aligned_data.replicate_col is not None

    def test_returns_per_replicate_results(
        self, mock_sim_outputs_likelihood, beta_binomial_observed_data
    ):
        """Test that target returns per-replicate results without marginalization."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="test",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        result = target.evaluate(mock_sim_outputs_likelihood)

        # Should have per-replicate values, not marginalized
        assert result.loglik_per_rep.ndim == 1
        assert len(result.loglik_per_rep) == len(mock_sim_outputs_likelihood)


class TestLikelihoodTargetSet:
    """Test the LikelihoodTargetSet class."""

    def test_single_target_marginalization(
        self, mock_sim_outputs_likelihood, beta_binomial_observed_data
    ):
        """Test marginalization for a single target."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="single_target",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        target_set = LikelihoodTargetSet(targets=[target])
        result = target_set.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])

        assert isinstance(result, LikelihoodEvaluationResult)
        assert isinstance(result.log_marginal, float)
        assert np.isfinite(result.log_marginal)
        assert "single_target" in result.per_target

    def test_multiple_targets_joint_loglik(
        self,
        mock_sim_outputs_likelihood,
        beta_binomial_observed_data,
        beta_binomial_observed_data_alt,
    ):
        """Test that multiple targets are combined via joint log-likelihood."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)

        target1 = LikelihoodTarget(
            name="target1",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=LikelihoodEvaluator(loglik_fn=loglik_fn, name="eval1"),
        )

        target2 = LikelihoodTarget(
            name="target2",
            model_output="output1",
            data=beta_binomial_observed_data_alt,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=LikelihoodEvaluator(loglik_fn=loglik_fn, name="eval2"),
        )

        target_set = LikelihoodTargetSet(targets=[target1, target2])
        result = target_set.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])

        # Should have marginalized log-likelihood
        assert isinstance(result.log_marginal, float)
        assert len(result.per_target) == 2

        # log_marginal should be logsumexp of joint per-rep minus log(R)
        joint_loglik_per_rep = (
            result.per_target["target1"].loglik_per_rep
            + result.per_target["target2"].loglik_per_rep
        )
        R = len(joint_loglik_per_rep)
        expected_log_marginal = logsumexp(joint_loglik_per_rep) - np.log(R)
        assert result.log_marginal == pytest.approx(expected_log_marginal)

    def test_logsumexp_numerical_stability(
        self, mock_sim_outputs_likelihood, beta_binomial_observed_data
    ):
        """Test that logsumexp is numerically stable."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="stability_test",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        target_set = LikelihoodTargetSet(targets=[target])
        result = target_set.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])

        # Result should be finite (not inf or nan)
        assert np.isfinite(result.log_marginal)

    def test_empty_replicates_raises(
        self, mock_empty_sim_outputs, beta_binomial_observed_data
    ):
        """Test that empty replicates raises ValueError."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="test",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        target_set = LikelihoodTargetSet(targets=[target])

        # Empty sim_outputs raises ValueError (from alignment step)
        with pytest.raises(ValueError):
            target_set.evaluate(mock_empty_sim_outputs, seeds=[])

    def test_zero_targets_raises(self, mock_sim_outputs_likelihood):
        """Test that empty target list raises ValueError."""
        target_set = LikelihoodTargetSet(targets=[])

        with pytest.raises(ValueError, match="No likelihood targets provided"):
            target_set.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])

    def test_per_target_results_structure(
        self, mock_sim_outputs_likelihood, beta_binomial_observed_data
    ):
        """Test that per_target results have correct structure."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)

        target1 = LikelihoodTarget(
            name="target_A",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=LikelihoodEvaluator(loglik_fn=loglik_fn, name="eval_A"),
        )

        target2 = LikelihoodTarget(
            name="target_B",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=LikelihoodEvaluator(loglik_fn=loglik_fn, name="eval_B"),
        )

        target_set = LikelihoodTargetSet(targets=[target1, target2])
        result = target_set.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])

        assert "target_A" in result.per_target
        assert "target_B" in result.per_target
        assert isinstance(result.per_target["target_A"], TargetLikelihoodResult)
        assert isinstance(result.per_target["target_B"], TargetLikelihoodResult)

    def test_log_marginal_computation(
        self, mock_sim_outputs_likelihood, beta_binomial_observed_data
    ):
        """Test that log_marginal is computed correctly."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="test",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        target_set = LikelihoodTargetSet(targets=[target])
        result = target_set.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])

        # Manually compute expected log_marginal
        loglik_per_rep = result.per_target["test"].loglik_per_rep
        R = len(loglik_per_rep)
        expected = logsumexp(loglik_per_rep) - np.log(R)

        assert result.log_marginal == pytest.approx(expected)

    def test_joint_marginalization_vs_independent(
        self,
        mock_sim_outputs_likelihood,
        beta_binomial_observed_data,
        beta_binomial_observed_data_alt,
    ):
        """
        Verify that marginalizing jointly (correct) differs from
        marginalizing independently (incorrect Jensen's inequality).

        This is an integration test to verify the design principle.
        """
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)

        target1 = LikelihoodTarget(
            name="target1",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=LikelihoodEvaluator(loglik_fn=loglik_fn, name="eval1"),
        )

        target2 = LikelihoodTarget(
            name="target2",
            model_output="output1",
            data=beta_binomial_observed_data_alt,
            alignment=ExactJoin(on_cols=["time"]),
            evaluator=LikelihoodEvaluator(loglik_fn=loglik_fn, name="eval2"),
        )

        # Correct: joint marginalization
        target_set = LikelihoodTargetSet(targets=[target1, target2])
        joint_result = target_set.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])

        # Incorrect (for comparison): independent marginalization
        target_set1 = LikelihoodTargetSet(targets=[target1])
        target_set2 = LikelihoodTargetSet(targets=[target2])
        result1 = target_set1.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])
        result2 = target_set2.evaluate(mock_sim_outputs_likelihood, seeds=[1, 2, 3])
        independent_sum = result1.log_marginal + result2.log_marginal

        # These should differ (unless data happens to be independent)
        # We can't assert they're different in general, but we can verify
        # that the joint computation is doing the right thing
        loglik1 = joint_result.per_target["target1"].loglik_per_rep
        loglik2 = joint_result.per_target["target2"].loglik_per_rep
        joint_per_rep = loglik1 + loglik2
        R = len(joint_per_rep)
        expected_joint = logsumexp(joint_per_rep) - np.log(R)

        assert joint_result.log_marginal == pytest.approx(expected_joint)
        # The joint and independent may differ (due to Jensen's inequality)
        # but we just verify the joint computation is correct
