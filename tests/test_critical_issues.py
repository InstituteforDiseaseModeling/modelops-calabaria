"""Tests for critical issues identified in code review.

These tests are designed to catch specific edge cases and problems
that were identified in the code review. They should fail initially,
proving the issues exist, then pass after fixes are applied.
"""

import math
import pytest
import polars as pl

from modelops_calabaria.parameters import (
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    AffineSqueezedLogit,
)


class TestIntegerParameterValidation:
    """Test that integer parameters handle edge cases correctly."""

    def test_float_value_for_int_param_with_decimal(self):
        """Test that float with decimal for int param is rejected."""
        space = ParameterSpace([
            ParameterSpec("count", 1, 100, "int"),
        ])

        # This should fail - 3.9 is not integer-like
        with pytest.raises((TypeError, ValueError), match="must be int|integer"):
            ParameterSet(space, {"count": 3.9})

    def test_float_value_for_int_param_without_decimal(self):
        """Test that integer-like float for int param is accepted."""
        space = ParameterSpace([
            ParameterSpec("count", 1, 100, "int"),
        ])

        # This should work - 3.0 is integer-like
        pset = ParameterSet(space, {"count": 3.0})
        assert pset["count"] == 3
        assert isinstance(pset["count"], int)

    def test_boolean_for_int_param(self):
        """Test that boolean for int param is rejected."""
        space = ParameterSpace([
            ParameterSpec("count", 0, 100, "int"),
        ])

        # Booleans should be explicitly rejected
        with pytest.raises((TypeError, ValueError), match="requires int.*bool|bool"):
            ParameterSet(space, {"count": True})

        with pytest.raises((TypeError, ValueError), match="requires int.*bool|bool"):
            ParameterSet(space, {"count": False})

    def test_boolean_for_float_param(self):
        """Test that boolean for float param is rejected."""
        space = ParameterSpace([
            ParameterSpec("rate", 0.0, 1.0, "float"),
        ])

        # Booleans should be explicitly rejected for float too
        with pytest.raises((TypeError, ValueError), match="requires numeric.*bool|bool"):
            ParameterSet(space, {"rate": True})


class TestNonFiniteValues:
    """Test that NaN and Inf values are properly rejected."""

    def test_nan_rejected(self):
        """Test that NaN values are rejected."""
        space = ParameterSpace([
            ParameterSpec("value", 0.0, 1.0, "float"),
        ])

        with pytest.raises(ValueError, match="NaN|finite|invalid"):
            ParameterSet(space, {"value": float('nan')})

    def test_positive_inf_rejected(self):
        """Test that positive infinity is rejected."""
        space = ParameterSpace([
            ParameterSpec("value", 0.0, 1000.0, "float"),
        ])

        with pytest.raises(ValueError, match="Inf|finite|outside bounds"):
            ParameterSet(space, {"value": float('inf')})

    def test_negative_inf_rejected(self):
        """Test that negative infinity is rejected."""
        space = ParameterSpace([
            ParameterSpec("value", -1000.0, 1000.0, "float"),
        ])

        with pytest.raises(ValueError, match="Inf|finite|outside bounds"):
            ParameterSet(space, {"value": float('-inf')})


class TestParameterSetFactory:
    """Test ParameterSet factory method for convenience."""

    def test_factory_method_exists(self):
        """Test that ParameterSet.new() factory exists."""
        space = ParameterSpace([
            ParameterSpec("alpha", 0.0, 1.0, "float"),
            ParameterSpec("beta", 0.0, 1.0, "float"),
        ])

        # Should be able to use factory method with kwargs
        if hasattr(ParameterSet, 'new'):
            pset = ParameterSet.new(space, alpha=0.5, beta=0.3)
            assert pset["alpha"] == 0.5
            assert pset["beta"] == 0.3
        else:
            pytest.skip("ParameterSet.new() not implemented yet")


class TestAffineSqueezedLogitBoundaries:
    """Test AffineSqueezedLogit transform at boundaries."""

    def test_forward_at_zero(self):
        """Test forward transform at x=0."""
        transform = AffineSqueezedLogit(eps=1e-6)

        # Should not raise
        y = transform.forward(0.0)
        assert math.isfinite(y)

        # Should be large negative
        assert y < -10

    def test_forward_at_one(self):
        """Test forward transform at x=1."""
        transform = AffineSqueezedLogit(eps=1e-6)

        # Should not raise
        y = transform.forward(1.0)
        assert math.isfinite(y)

        # Should be large positive
        assert y > 10

    def test_round_trip_at_zero(self):
        """Test round-trip at x=0."""
        transform = AffineSqueezedLogit(eps=1e-6)

        x = 0.0
        y = transform.forward(x)
        x_recovered = transform.backward(y)

        # Should recover approximately (within epsilon tolerance)
        assert abs(x_recovered - x) < 1e-8

    def test_round_trip_at_one(self):
        """Test round-trip at x=1."""
        transform = AffineSqueezedLogit(eps=1e-6)

        x = 1.0
        y = transform.forward(x)
        x_recovered = transform.backward(y)

        # Should recover approximately (within epsilon tolerance)
        assert abs(x_recovered - x) < 1e-8

    def test_backward_extreme_negative(self):
        """Test backward transform with extreme negative value."""
        transform = AffineSqueezedLogit(eps=1e-6)

        # Very negative value
        x = transform.backward(-50)

        # Should be close to 0 but not negative
        assert 0 <= x <= 0.001
        assert math.isfinite(x)

    def test_backward_extreme_positive(self):
        """Test backward transform with extreme positive value."""
        transform = AffineSqueezedLogit(eps=1e-6)

        # Very positive value
        x = transform.backward(50)

        # Should be close to 1 but not above
        assert 0.999 <= x <= 1.0
        assert math.isfinite(x)


class TestSEEDCOLLocation:
    """Test that SEED_COL is properly centralized."""

    def test_seed_col_import_location(self):
        """Test that SEED_COL can be imported from constants."""
        try:
            from modelops_calabaria.constants import SEED_COL
            assert SEED_COL == "seed"
        except ImportError:
            # Currently it's in decorators
            from modelops_calabaria.decorators import SEED_COL
            assert SEED_COL == "seed"
            pytest.fail("SEED_COL should be in constants module, not decorators")