"""Tests for edge cases and validation in target evaluation."""

import numpy as np
import pytest

from modelops_calabaria.core.alignment import ExactJoin
from modelops_calabaria.core.evaluation.likelihood import LikelihoodEvaluator
from modelops_calabaria.core.evaluation.loglik_fns import beta_binomial_loglik_per_rep
from modelops_calabaria.core.target import LikelihoodTargetSet, LikelihoodTarget


class TestEdgeCases:
    """Test edge cases in target evaluation."""

    def test_single_replicate_likelihood(
        self, mock_single_replicate_sim_outputs, beta_binomial_observed_data
    ):
        """Test likelihood evaluation with only a single replicate."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="single_rep_target",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        target_set = LikelihoodTargetSet(targets=[target])
        result = target_set.evaluate(mock_single_replicate_sim_outputs, seeds=[1])

        # Should work with single replicate
        assert isinstance(result.log_marginal, float)
        assert np.isfinite(result.log_marginal)

        # With R=1, log_marginal should equal the single replicate's loglik
        expected = result.per_target["single_rep_target"].loglik_per_rep[0]
        assert result.log_marginal == pytest.approx(expected)

    def test_inf_handling(
        self, mock_sim_outputs_with_extreme_values, beta_binomial_observed_data
    ):
        """Test handling of -inf log-likelihood values."""
        loglik_fn = beta_binomial_loglik_per_rep("x", "n", alpha=1.0, beta=1.0)
        evaluator = LikelihoodEvaluator(loglik_fn=loglik_fn, name="test")
        alignment = ExactJoin(on_cols=["time"])

        target = LikelihoodTarget(
            name="inf_test",
            model_output="output1",
            data=beta_binomial_observed_data,
            alignment=alignment,
            evaluator=evaluator,
        )

        target_set = LikelihoodTargetSet(targets=[target])

        # This should not crash even if some log-likelihoods are -inf
        # (logsumexp handles this correctly)
        result = target_set.evaluate(mock_sim_outputs_with_extreme_values, seeds=[1, 2, 3])

        # Result might be -inf, but should not be nan
        assert not np.isnan(result.log_marginal)
