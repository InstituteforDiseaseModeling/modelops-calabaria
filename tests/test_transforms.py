"""Tests for parameter transforms.

Tests the transform system including:
- Forward/backward invertibility
- Domain validation
- Numerical stability at boundaries
- AffineSqueezedLogit epsilon handling
"""

import math
import pytest
from hypothesis import given, strategies as st, assume

from modelops_calabaria.parameters.transforms import (
    Identity,
    LogTransform,
    LogitTransform,
    AffineSqueezedLogit,
)


class TestIdentityTransform:
    """Tests for Identity transform."""

    def test_forward_backward_unchanged(self):
        """Test Identity transform leaves values unchanged."""
        transform = Identity()

        # Test various values
        for x in [-100, -1, 0, 0.5, 1, 100]:
            assert transform.forward(x) == x
            assert transform.backward(x) == x

    def test_bounds_unchanged(self):
        """Test Identity transform leaves bounds unchanged."""
        transform = Identity()
        bounds = (0.0, 10.0)

        assert transform.bounds(bounds, transformed=False) == bounds
        assert transform.bounds(bounds, transformed=True) == bounds


class TestLogTransform:
    """Tests for LogTransform."""

    def test_forward_valid_domain(self):
        """Test LogTransform forward on valid positive values."""
        transform = LogTransform()

        assert transform.forward(1.0) == 0.0
        assert transform.forward(math.e) == pytest.approx(1.0)
        assert transform.forward(0.1) == pytest.approx(math.log(0.1))

    def test_forward_invalid_domain_raises(self):
        """Test LogTransform forward raises on non-positive values."""
        transform = LogTransform()

        with pytest.raises(ValueError, match="requires x > 0"):
            transform.forward(0.0)

        with pytest.raises(ValueError, match="requires x > 0"):
            transform.forward(-1.0)

    def test_backward(self):
        """Test LogTransform backward (exp)."""
        transform = LogTransform()

        assert transform.backward(0.0) == 1.0
        assert transform.backward(1.0) == pytest.approx(math.e)
        assert transform.backward(-1.0) == pytest.approx(1/math.e)

    def test_invertibility(self):
        """Test LogTransform round-trip property."""
        transform = LogTransform()

        for x in [0.001, 0.1, 1.0, 10.0, 1000.0]:
            y = transform.forward(x)
            x_recovered = transform.backward(y)
            assert x_recovered == pytest.approx(x, rel=1e-10)

    def test_bounds_transform(self):
        """Test LogTransform bounds transformation."""
        transform = LogTransform()

        # Valid positive bounds
        bounds = (0.1, 10.0)
        natural = transform.bounds(bounds, transformed=False)
        assert natural == bounds

        transformed = transform.bounds(bounds, transformed=True)
        assert transformed[0] == pytest.approx(math.log(0.1))
        assert transformed[1] == pytest.approx(math.log(10.0))

        # Invalid bounds should raise
        with pytest.raises(ValueError, match="requires positive bounds"):
            transform.bounds((0.0, 1.0), transformed=True)


class TestLogitTransform:
    """Tests for LogitTransform."""

    def test_forward_valid_domain(self):
        """Test LogitTransform forward on valid [0,1] values."""
        transform = LogitTransform()

        assert transform.forward(0.5) == 0.0  # logit(0.5) = 0
        assert transform.forward(0.1) == pytest.approx(math.log(0.1/0.9))
        assert transform.forward(0.9) == pytest.approx(math.log(0.9/0.1))

    def test_forward_invalid_domain_raises(self):
        """Test LogitTransform forward raises outside (0,1)."""
        transform = LogitTransform()

        with pytest.raises(ValueError, match="requires 0 < x < 1"):
            transform.forward(0.0)

        with pytest.raises(ValueError, match="requires 0 < x < 1"):
            transform.forward(1.0)

        with pytest.raises(ValueError, match="requires 0 < x < 1"):
            transform.forward(1.5)

    def test_backward(self):
        """Test LogitTransform backward (sigmoid)."""
        transform = LogitTransform()

        assert transform.backward(0.0) == 0.5
        assert 0 < transform.backward(-10.0) < 0.01  # Very small
        assert 0.99 < transform.backward(10.0) < 1.0  # Very close to 1

    def test_invertibility(self):
        """Test LogitTransform round-trip property."""
        transform = LogitTransform()

        # Test interior points (avoiding exact 0 and 1)
        for x in [0.01, 0.1, 0.5, 0.9, 0.99]:
            y = transform.forward(x)
            x_recovered = transform.backward(y)
            assert x_recovered == pytest.approx(x, rel=1e-10)

    def test_bounds_transform(self):
        """Test LogitTransform bounds transformation."""
        transform = LogitTransform()

        bounds = (0.0, 1.0)
        natural = transform.bounds(bounds, transformed=False)
        assert natural == bounds

        # Transformed bounds are finite approximations of ±∞
        transformed = transform.bounds(bounds, transformed=True)
        assert transformed == (-10.0, 10.0)


class TestAffineSqueezedLogit:
    """Tests for AffineSqueezedLogit."""

    def test_init_validates_epsilon(self):
        """Test AffineSqueezedLogit validates epsilon parameter."""
        # Valid epsilon
        transform = AffineSqueezedLogit(eps=1e-6)
        assert transform.eps == 1e-6

        # Invalid epsilon
        with pytest.raises(ValueError, match="eps must be in"):
            AffineSqueezedLogit(eps=0.0)

        with pytest.raises(ValueError, match="eps must be in"):
            AffineSqueezedLogit(eps=0.6)

    def test_forward_valid_domain(self):
        """Test AffineSqueezedLogit forward on [0,1]."""
        transform = AffineSqueezedLogit(eps=1e-6)

        # Test boundary values (should work due to squeeze)
        y0 = transform.forward(0.0)
        y1 = transform.forward(1.0)
        assert math.isfinite(y0)
        assert math.isfinite(y1)

        # Test middle value
        assert abs(transform.forward(0.5)) < 1e-10  # Should be near 0

    def test_forward_invalid_domain_raises(self):
        """Test AffineSqueezedLogit forward raises outside [0,1]."""
        transform = AffineSqueezedLogit()

        with pytest.raises(ValueError, match="requires 0≤x≤1"):
            transform.forward(-0.1)

        with pytest.raises(ValueError, match="requires 0≤x≤1"):
            transform.forward(1.1)

    def test_backward(self):
        """Test AffineSqueezedLogit backward."""
        transform = AffineSqueezedLogit(eps=1e-6)

        # Middle value
        assert transform.backward(0.0) == pytest.approx(0.5, rel=1e-10)

        # Extreme values map back to near 0 and 1
        # Note: Due to numerical precision, may get tiny negative values
        assert transform.backward(-20) == pytest.approx(0.0, abs=1e-5)
        assert transform.backward(20) == pytest.approx(1.0, abs=1e-5)

    def test_invertibility_interior(self):
        """Test AffineSqueezedLogit round-trip for interior points."""
        transform = AffineSqueezedLogit(eps=1e-6)

        for x in [0.1, 0.3, 0.5, 0.7, 0.9]:
            y = transform.forward(x)
            x_recovered = transform.backward(y)
            assert x_recovered == pytest.approx(x, rel=1e-10)

    def test_invertibility_boundaries(self):
        """Test AffineSqueezedLogit round-trip at boundaries."""
        transform = AffineSqueezedLogit(eps=1e-6)

        # Test exact 0 and 1
        for x in [0.0, 1.0]:
            y = transform.forward(x)
            x_recovered = transform.backward(y)
            # Due to squeeze, won't recover exactly but should be very close
            assert x_recovered == pytest.approx(x, abs=1e-10)

    def test_different_epsilon_values(self):
        """Test AffineSqueezedLogit with different epsilon values."""
        for eps in [1e-8, 1e-6, 1e-4, 0.01]:
            transform = AffineSqueezedLogit(eps=eps)

            # Should handle boundaries
            y0 = transform.forward(0.0)
            y1 = transform.forward(1.0)
            assert math.isfinite(y0)
            assert math.isfinite(y1)

            # Round trip
            assert transform.backward(y0) == pytest.approx(0.0, abs=1e-10)
            assert transform.backward(y1) == pytest.approx(1.0, abs=1e-10)

    def test_bounds_transform(self):
        """Test AffineSqueezedLogit bounds transformation."""
        transform = AffineSqueezedLogit(eps=1e-6)

        bounds = (0.0, 1.0)
        natural = transform.bounds(bounds, transformed=False)
        assert natural == bounds

        # Transformed bounds
        transformed = transform.bounds(bounds, transformed=True)
        assert transformed[0] < transformed[1]
        assert math.isfinite(transformed[0])
        assert math.isfinite(transformed[1])

        # Bounds for subset of [0,1]
        bounds2 = (0.2, 0.8)
        transformed2 = transform.bounds(bounds2, transformed=True)
        assert transformed2[0] == transform.forward(0.2)
        assert transformed2[1] == transform.forward(0.8)


class TestPropertyTests:
    """Property-based tests for transforms."""

    @given(x=st.floats(min_value=0.001, max_value=1000))
    def test_log_transform_invertibility(self, x):
        """Property: LogTransform is invertible for positive values."""
        transform = LogTransform()
        y = transform.forward(x)
        x_recovered = transform.backward(y)
        assert x_recovered == pytest.approx(x, rel=1e-10)

    @given(x=st.floats(min_value=0.001, max_value=0.999))
    def test_logit_transform_invertibility(self, x):
        """Property: LogitTransform is invertible for (0,1) values."""
        transform = LogitTransform()
        y = transform.forward(x)
        x_recovered = transform.backward(y)
        assert x_recovered == pytest.approx(x, rel=1e-10)

    @given(
        x=st.floats(min_value=0.0, max_value=1.0),
        eps=st.floats(min_value=1e-8, max_value=0.1)
    )
    def test_affine_squeezed_logit_invertibility(self, x, eps):
        """Property: AffineSqueezedLogit is invertible for [0,1] values."""
        transform = AffineSqueezedLogit(eps=eps)
        y = transform.forward(x)
        x_recovered = transform.backward(y)
        # Slightly looser tolerance due to squeeze
        assert x_recovered == pytest.approx(x, abs=1e-8)

    @given(
        x=st.floats(min_value=0.0, max_value=1.0),
        eps1=st.floats(min_value=1e-8, max_value=0.01),
        eps2=st.floats(min_value=1e-8, max_value=0.01)
    )
    def test_affine_squeezed_logit_monotonicity(self, x, eps1, eps2):
        """Property: Smaller eps gives values closer to boundaries."""
        assume(eps1 < eps2)

        t1 = AffineSqueezedLogit(eps=eps1)
        t2 = AffineSqueezedLogit(eps=eps2)

        # For x=0, smaller eps should give more extreme negative value
        if x < 0.01:
            assert t1.forward(x) <= t2.forward(x)

        # For x=1, smaller eps should give more extreme positive value
        if x > 0.99:
            assert t1.forward(x) >= t2.forward(x)