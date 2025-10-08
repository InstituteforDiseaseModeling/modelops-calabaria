"""Tests for evaluation strategies."""

import numpy as np
import polars as pl
import pytest

from modelops_calabaria.core.alignment import ExactJoin
from modelops_calabaria.core.evaluation import (
    Evaluator,
    IdentityAggregator,
    MeanAcrossReplicates,
    MeanReducer,
    MeanGroupedByReplicate,
    SquaredErrorLoss,
    beta_binomial_nll,
    mean_of_per_replicate_mse,
    replicate_mean_mse,
)
from modelops_calabaria.core.evaluation.loss.nll import BetaBinomialNLL, BinomialNLL


@pytest.fixture
def aligned_data():
    """Create aligned data for testing."""
    observed = pl.DataFrame(
        {
            "timestep": [0, 1, 2],
            "infected": [10, 15, 20],
            "popsize": [60, 62, 65],
        }
    )
    simulated = pl.DataFrame(
        {
            "timestep": [0, 1, 2],
            "infected": [12, 14, 19],
            "popsize": [71, 72, 74],
        }
    )
    simulated = simulated.with_columns(
        (simulated["infected"] / simulated["popsize"]).alias("p")
    )

    aligner = ExactJoin(on_cols=["timestep"])
    aligned = aligner.align(observed, [simulated])
    return aligned


@pytest.fixture
def multi_replicate_data():
    """Create aligned data with multiple replicates."""
    observed = pl.DataFrame(
        {
            "timestep": [0, 1, 2],
            "value": [10.0, 20.0, 30.0],
        }
    )

    # Two replicates with slightly different values
    sim1 = pl.DataFrame(
        {
            "timestep": [0, 1, 2],
            "value": [11.0, 19.0, 31.0],
        }
    )
    sim2 = pl.DataFrame(
        {
            "timestep": [0, 1, 2],
            "value": [9.0, 21.0, 29.0],
        }
    )

    aligner = ExactJoin(on_cols=["timestep"])
    aligned = aligner.align(observed, [sim1, sim2])
    return aligned


class TestLossFunctions:
    """Test individual loss functions."""

    def test_squared_error_loss(self, aligned_data):
        """Test squared error loss computation."""
        loss_fn = SquaredErrorLoss(col="infected")
        result = loss_fn.compute(aligned_data)

        assert result.has_loss()
        losses = result.get_loss_col().to_numpy()
        assert np.isfinite(losses).all()
        # Check actual squared error values
        expected = np.array([(10-12)**2, (15-14)**2, (20-19)**2])
        np.testing.assert_array_equal(losses, expected)

    def test_beta_binomial_nll(self, aligned_data):
        """Test beta-binomial NLL computation."""
        nll = BetaBinomialNLL(x_col="infected", n_col="popsize")
        result = nll.compute(aligned_data)

        assert result.has_loss()
        losses = result.get_loss_col().to_numpy()
        assert np.isfinite(losses).all()
        assert (losses >= 0).all()  # NLL should be non-negative

    def test_beta_binomial_with_prior(self, aligned_data):
        """Test beta-binomial with different prior."""
        nll = BetaBinomialNLL(x_col="infected", n_col="popsize", prior=(2, 2))
        result = nll.compute(aligned_data)

        losses = result.get_loss_col().to_numpy()
        assert np.isfinite(losses).all()

    def test_binomial_nll(self, aligned_data):
        """Test binomial NLL computation."""
        nll = BinomialNLL(p_col="p", x_col="infected", n_col="popsize")
        result = nll.compute(aligned_data)

        losses = result.get_loss_col().to_numpy()
        assert np.isfinite(losses).all()
        assert (losses >= 0).all()


class TestAggregators:
    """Test aggregation strategies."""

    def test_identity_aggregator(self, multi_replicate_data):
        """Test identity aggregator passes data through unchanged."""
        agg = IdentityAggregator()
        result = agg.aggregate(multi_replicate_data)

        assert result.data.shape == multi_replicate_data.data.shape
        assert result.replicate_col == multi_replicate_data.replicate_col

    def test_mean_across_replicates(self, multi_replicate_data):
        """Test averaging across replicates."""
        agg = MeanAcrossReplicates(cols=["value"])
        result = agg.aggregate(multi_replicate_data)

        # Should have one row per timestep (no replicates)
        assert result.replicate_col is None
        assert result.data.shape[0] == 3  # 3 timesteps

        # Check that values are averaged
        sim_col = result.sim_colname("value")
        sim_values = result.data[sim_col].to_numpy()
        # Average of [11, 9] = 10, [19, 21] = 20, [31, 29] = 30
        expected = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(sim_values, expected)


class TestReducers:
    """Test reduction strategies."""

    def test_mean_reducer(self, aligned_data):
        """Test mean reducer."""
        loss_fn = SquaredErrorLoss(col="infected")
        aligned_with_loss = loss_fn.compute(aligned_data)

        reducer = MeanReducer()
        result = reducer.reduce(aligned_with_loss)

        assert isinstance(result, float)
        assert np.isfinite(result)
        # Mean of [4, 1, 1] = 2.0
        assert result == pytest.approx(2.0)

    def test_mean_grouped_by_replicate(self, multi_replicate_data):
        """Test mean grouped by replicate."""
        loss_fn = SquaredErrorLoss(col="value")
        aligned_with_loss = loss_fn.compute(multi_replicate_data)

        reducer = MeanGroupedByReplicate()
        result = reducer.reduce(aligned_with_loss)

        assert isinstance(result, float)
        assert np.isfinite(result)


class TestComposableEvaluator:
    """Test the composable evaluator."""

    def test_evaluator_basic(self, aligned_data):
        """Test basic evaluator composition."""
        evaluator = Evaluator(
            aggregator=IdentityAggregator(),
            loss_fn=SquaredErrorLoss(col="infected"),
            reducer=MeanReducer(),
            weight=1.0
        )

        result = evaluator.evaluate(aligned_data)

        assert result.name == "SquaredErrorLoss"
        assert result.loss == pytest.approx(2.0)
        assert result.weight == 1.0
        assert result.weighted_loss == pytest.approx(2.0)

    def test_evaluator_with_weight(self, aligned_data):
        """Test evaluator with custom weight."""
        evaluator = Evaluator(
            aggregator=IdentityAggregator(),
            loss_fn=SquaredErrorLoss(col="infected"),
            reducer=MeanReducer(),
            weight=2.5
        )

        result = evaluator.evaluate(aligned_data)

        assert result.weight == 2.5
        assert result.weighted_loss == pytest.approx(2.0 * 2.5)


class TestFactoryFunctions:
    """Test factory functions for common evaluation strategies."""

    def test_mean_of_per_replicate_mse(self, multi_replicate_data):
        """Test MSE computed per replicate then averaged."""
        evaluator = mean_of_per_replicate_mse(col="value")
        result = evaluator.evaluate(multi_replicate_data)

        assert result.loss is not None
        assert np.isfinite(result.loss)
        # Each replicate has MSE, then we average them

    def test_replicate_mean_mse(self, multi_replicate_data):
        """Test averaging replicates first then MSE."""
        evaluator = replicate_mean_mse(col="value")
        result = evaluator.evaluate(multi_replicate_data)

        assert result.loss is not None
        assert np.isfinite(result.loss)
        # Should be 0 since average of [11,9]=10, [19,21]=20, [31,29]=30
        # which matches observed [10, 20, 30]
        assert result.loss == pytest.approx(0.0)

    def test_beta_binomial_nll_factory(self, aligned_data):
        """Test beta-binomial factory function."""
        evaluator = beta_binomial_nll(x_col="infected", n_col="popsize")
        result = evaluator.evaluate(aligned_data)

        assert result.loss is not None
        assert np.isfinite(result.loss)
        assert result.loss > 0  # NLL should be positive

    def test_beta_binomial_nll_with_prior(self, aligned_data):
        """Test beta-binomial factory with custom prior."""
        evaluator = beta_binomial_nll(
            x_col="infected",
            n_col="popsize",
            prior=(2, 2)
        )
        result = evaluator.evaluate(aligned_data)

        assert result.loss is not None
        assert np.isfinite(result.loss)