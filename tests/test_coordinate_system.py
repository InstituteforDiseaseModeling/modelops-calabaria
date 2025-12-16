"""Tests for CoordinateSystem - the view + transforms packaging abstraction.

These tests verify that CoordinateSystem correctly implements:
- Bidirectional coordinate mappings (Z_V ↔ P_V ↔ M)
- Transform application per parameter
- Bounds transformation
- Vector ordering preservation (CRITICAL)
- Immutability guarantees
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st

from modelops_calabaria.parameters import (
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ParameterView,
    CoordinateSystem,
    Identity,
    LogTransform,
    LogitTransform,
    AffineSqueezedLogit,
)


@pytest.fixture
def space():
    """Create test parameter space."""
    return ParameterSpace([
        ParameterSpec("alpha", 0.0, 1.0, "float"),
        ParameterSpec("beta", 0.01, 10.0, "float"),
        ParameterSpec("gamma", 0.0, 1.0, "float"),
        ParameterSpec("delta", 0.1, 5.0, "float"),
    ])


@pytest.fixture
def view_all_free(space):
    """Create view with all parameters free."""
    return ParameterView.all_free(space)


@pytest.fixture
def view_some_fixed(space):
    """Create view with some parameters fixed."""
    return ParameterView.from_fixed(space, alpha=0.5, delta=1.0)


class TestCoordinateSystemCreation:
    """Tests for creating CoordinateSystem instances."""

    def test_create_with_no_transforms(self, view_all_free):
        """Test creating coordinate system with no transforms."""
        coords = CoordinateSystem(view_all_free, {})

        assert coords.view is view_all_free
        assert len(coords.transforms) == 0
        assert coords.dim == 4  # All params free

    def test_create_with_some_transforms(self, view_all_free):
        """Test creating coordinate system with some transforms."""
        transforms = {
            "beta": LogTransform(),
            "gamma": LogitTransform(),
        }
        coords = CoordinateSystem(view_all_free, transforms)

        assert len(coords.transforms) == 2
        assert "beta" in coords.transforms
        assert "gamma" in coords.transforms

    def test_create_with_partial_view_and_transforms(self, view_some_fixed):
        """Test creating coordinate system with fixed params and transforms."""
        # Only beta and gamma are free (alpha and delta are fixed)
        transforms = {
            "beta": LogTransform(),
        }
        coords = CoordinateSystem(view_some_fixed, transforms)

        assert coords.dim == 2  # beta, gamma free
        assert len(coords.transforms) == 1

    def test_transforms_for_non_free_params_raises(self, view_some_fixed):
        """Test that specifying transform for fixed param raises error."""
        # alpha is fixed, so transform shouldn't be allowed
        transforms = {
            "alpha": LogTransform(),
        }

        with pytest.raises(ValueError, match="non-free parameters"):
            CoordinateSystem(view_some_fixed, transforms)

    def test_invalid_transform_raises(self, view_all_free):
        """Test that invalid transform object raises error."""
        class BadTransform:
            pass  # Missing forward, backward, bounds

        transforms = {
            "beta": BadTransform(),
        }

        with pytest.raises(TypeError, match="Transform protocol"):
            CoordinateSystem(view_all_free, transforms)

    def test_transforms_are_immutable(self, view_all_free):
        """Test that transforms mapping is immutable."""
        transforms = {
            "beta": LogTransform(),
        }
        coords = CoordinateSystem(view_all_free, transforms)

        with pytest.raises(TypeError):
            coords.transforms["gamma"] = LogitTransform()


class TestCoordinateMappings:
    """Tests for to_M and from_M coordinate mappings."""

    def test_to_M_with_identity_transforms(self, space, view_all_free):
        """Test to_M with no transforms (identity)."""
        coords = CoordinateSystem(view_all_free, {})

        # z = [0.5, 1.0, 0.3, 2.0] for [alpha, beta, gamma, delta]
        z = np.array([0.5, 1.0, 0.3, 2.0])

        params = coords.to_M(z)

        assert params["alpha"] == 0.5
        assert params["beta"] == 1.0
        assert params["gamma"] == 0.3
        assert params["delta"] == 2.0

    def test_to_M_with_log_transform(self, space, view_all_free):
        """Test to_M with log transform."""
        transforms = {
            "beta": LogTransform(),
        }
        coords = CoordinateSystem(view_all_free, transforms)

        # z[1] = 0.0 (log space) → beta = exp(0.0) = 1.0
        z = np.array([0.5, 0.0, 0.3, 2.0])

        params = coords.to_M(z)

        assert params["alpha"] == 0.5
        assert params["beta"] == pytest.approx(1.0)
        assert params["gamma"] == 0.3
        assert params["delta"] == 2.0

    def test_to_M_with_logit_transform(self, space, view_all_free):
        """Test to_M with logit transform."""
        transforms = {
            "gamma": AffineSqueezedLogit(),  # For [0,1] parameters
        }
        coords = CoordinateSystem(view_all_free, transforms)

        # z[2] = 0.0 (logit space) → gamma = 0.5
        z = np.array([0.5, 1.0, 0.0, 2.0])

        params = coords.to_M(z)

        assert params["alpha"] == 0.5
        assert params["beta"] == 1.0
        assert params["gamma"] == pytest.approx(0.5)
        assert params["delta"] == 2.0

    def test_to_M_with_partial_view(self, space, view_some_fixed):
        """Test to_M with partial view (some params fixed)."""
        coords = CoordinateSystem(view_some_fixed, {})

        # Only beta and gamma are free
        z = np.array([2.0, 0.7])  # [beta, gamma]

        params = coords.to_M(z)

        # Fixed values preserved
        assert params["alpha"] == 0.5
        assert params["delta"] == 1.0
        # Free values from z
        assert params["beta"] == 2.0
        assert params["gamma"] == 0.7

    def test_to_M_wrong_dimensionality_raises(self, view_all_free):
        """Test that wrong z dimensionality raises error."""
        coords = CoordinateSystem(view_all_free, {})

        z = np.array([0.5, 1.0])  # Too short (need 4)

        with pytest.raises(ValueError, match="Expected z vector of length 4"):
            coords.to_M(z)

    def test_from_M_with_identity_transforms(self, space, view_all_free):
        """Test from_M with no transforms."""
        coords = CoordinateSystem(view_all_free, {})

        params = ParameterSet(space, {
            "alpha": 0.5,
            "beta": 1.0,
            "gamma": 0.3,
            "delta": 2.0,
        })

        z = coords.from_M(params)

        assert z.shape == (4,)
        assert z[0] == pytest.approx(0.5)
        assert z[1] == pytest.approx(1.0)
        assert z[2] == pytest.approx(0.3)
        assert z[3] == pytest.approx(2.0)

    def test_from_M_with_log_transform(self, space, view_all_free):
        """Test from_M with log transform."""
        transforms = {
            "beta": LogTransform(),
        }
        coords = CoordinateSystem(view_all_free, transforms)

        params = ParameterSet(space, {
            "alpha": 0.5,
            "beta": 1.0,  # log(1.0) = 0.0
            "gamma": 0.3,
            "delta": 2.0,
        })

        z = coords.from_M(params)

        assert z[0] == pytest.approx(0.5)
        assert z[1] == pytest.approx(0.0)  # log(1.0)
        assert z[2] == pytest.approx(0.3)
        assert z[3] == pytest.approx(2.0)

    def test_from_M_with_partial_view(self, space, view_some_fixed):
        """Test from_M with partial view."""
        coords = CoordinateSystem(view_some_fixed, {})

        params = ParameterSet(space, {
            "alpha": 0.5,
            "beta": 2.0,
            "gamma": 0.7,
            "delta": 1.0,
        })

        z = coords.from_M(params)

        # Only free params (beta, gamma) in z
        assert z.shape == (2,)
        assert z[0] == pytest.approx(2.0)  # beta
        assert z[1] == pytest.approx(0.7)  # gamma


class TestRoundTripInvertibility:
    """Tests for round-trip invertibility: from_M(to_M(z)) ≈ z"""

    def test_round_trip_identity_transforms(self, view_all_free):
        """Test round-trip with identity transforms."""
        coords = CoordinateSystem(view_all_free, {})

        z_original = np.array([0.5, 2.0, 0.7, 1.5])

        params = coords.to_M(z_original)
        z_recovered = coords.from_M(params)

        np.testing.assert_allclose(z_recovered, z_original, rtol=1e-10)

    def test_round_trip_log_transform(self, view_all_free):
        """Test round-trip with log transform."""
        transforms = {
            "beta": LogTransform(),
        }
        coords = CoordinateSystem(view_all_free, transforms)

        z_original = np.array([0.5, 0.0, 0.7, 1.5])  # beta in log space

        params = coords.to_M(z_original)
        z_recovered = coords.from_M(params)

        np.testing.assert_allclose(z_recovered, z_original, rtol=1e-10)

    def test_round_trip_logit_transform(self, view_all_free):
        """Test round-trip with logit transform."""
        transforms = {
            "gamma": AffineSqueezedLogit(),
        }
        coords = CoordinateSystem(view_all_free, transforms)

        z_original = np.array([0.5, 2.0, 0.0, 1.5])  # gamma in logit space

        params = coords.to_M(z_original)
        z_recovered = coords.from_M(params)

        np.testing.assert_allclose(z_recovered, z_original, rtol=1e-10)

    def test_round_trip_multiple_transforms(self, view_all_free):
        """Test round-trip with multiple transforms."""
        transforms = {
            "beta": LogTransform(),
            "gamma": AffineSqueezedLogit(),
        }
        coords = CoordinateSystem(view_all_free, transforms)

        z_original = np.array([0.5, 0.5, -1.0, 1.5])

        params = coords.to_M(z_original)
        z_recovered = coords.from_M(params)

        np.testing.assert_allclose(z_recovered, z_original, rtol=1e-10)

    def test_round_trip_with_partial_view(self, view_some_fixed):
        """Test round-trip with partial view."""
        transforms = {
            "beta": LogTransform(),
        }
        coords = CoordinateSystem(view_some_fixed, transforms)

        z_original = np.array([0.5, 0.3])  # [beta in log, gamma]

        params = coords.to_M(z_original)
        z_recovered = coords.from_M(params)

        np.testing.assert_allclose(z_recovered, z_original, rtol=1e-10)


class TestVectorOrdering:
    """Tests for vector ordering preservation - CRITICAL!"""

    def test_param_names_match_view_free_order(self, space):
        """Test that param_names matches view.free ordering."""
        view = ParameterView.from_fixed(space, alpha=0.5, delta=1.0)
        coords = CoordinateSystem(view, {})

        # view.free should be ("beta", "gamma") in space order
        assert view.free == ("beta", "gamma")
        assert coords.param_names == ("beta", "gamma")
        assert coords.param_names == view.free

    def test_z_vector_dimension_matches_free_count(self, view_some_fixed):
        """Test that z vector dimension matches number of free params."""
        coords = CoordinateSystem(view_some_fixed, {})

        assert coords.dim == len(view_some_fixed.free)
        assert coords.dim == 2  # beta, gamma

    def test_z_ordering_matches_free_ordering(self, space):
        """Test that z[i] corresponds to free[i]."""
        view = ParameterView.from_fixed(space, alpha=0.5, delta=1.0)
        coords = CoordinateSystem(view, {})

        # free = ("beta", "gamma")
        z = np.array([3.0, 0.8])  # [beta=3.0, gamma=0.8]

        params = coords.to_M(z)

        assert params["beta"] == 3.0
        assert params["gamma"] == 0.8

    def test_from_M_preserves_ordering(self, space):
        """Test that from_M returns z in correct order."""
        view = ParameterView.from_fixed(space, alpha=0.5, delta=1.0)
        coords = CoordinateSystem(view, {})

        params = ParameterSet(space, {
            "alpha": 0.5,
            "beta": 3.0,
            "gamma": 0.8,
            "delta": 1.0,
        })

        z = coords.from_M(params)

        # Should be [beta, gamma] order
        assert z[0] == pytest.approx(3.0)
        assert z[1] == pytest.approx(0.8)


class TestBoundsTransformation:
    """Tests for bounds_transformed method."""

    def test_bounds_identity_transforms(self, space, view_all_free):
        """Test bounds with identity transforms."""
        coords = CoordinateSystem(view_all_free, {})

        bounds = coords.bounds_transformed()

        assert bounds.shape == (4, 2)
        # alpha: [0.0, 1.0]
        assert bounds[0, 0] == pytest.approx(0.0)
        assert bounds[0, 1] == pytest.approx(1.0)
        # beta: [0.01, 10.0]
        assert bounds[1, 0] == pytest.approx(0.01)
        assert bounds[1, 1] == pytest.approx(10.0)

    def test_bounds_log_transform(self, space, view_all_free):
        """Test bounds with log transform."""
        transforms = {
            "beta": LogTransform(),
        }
        coords = CoordinateSystem(view_all_free, transforms)

        bounds = coords.bounds_transformed()

        # beta bounds in log space: [log(0.01), log(10.0)]
        import math
        assert bounds[1, 0] == pytest.approx(math.log(0.01))
        assert bounds[1, 1] == pytest.approx(math.log(10.0))

    def test_bounds_partial_view(self, space, view_some_fixed):
        """Test bounds with partial view."""
        coords = CoordinateSystem(view_some_fixed, {})

        bounds = coords.bounds_transformed()

        # Only beta and gamma (free params)
        assert bounds.shape == (2, 2)
        # beta: [0.01, 10.0]
        assert bounds[0, 0] == pytest.approx(0.01)
        assert bounds[0, 1] == pytest.approx(10.0)
        # gamma: [0.0, 1.0]
        assert bounds[1, 0] == pytest.approx(0.0)
        assert bounds[1, 1] == pytest.approx(1.0)


class TestProperties:
    """Tests for properties and helper methods."""

    def test_dim_property(self, view_all_free, view_some_fixed):
        """Test dim property."""
        coords_all = CoordinateSystem(view_all_free, {})
        assert coords_all.dim == 4

        coords_some = CoordinateSystem(view_some_fixed, {})
        assert coords_some.dim == 2

    def test_param_names_property(self, space):
        """Test param_names property."""
        view = ParameterView.from_fixed(space, alpha=0.5)
        coords = CoordinateSystem(view, {})

        assert coords.param_names == ("beta", "gamma", "delta")
        assert coords.param_names == view.free

    def test_repr(self, view_some_fixed):
        """Test __repr__ method."""
        transforms = {
            "beta": LogTransform(),
        }
        coords = CoordinateSystem(view_some_fixed, transforms)

        repr_str = repr(coords)
        assert "CoordinateSystem" in repr_str
        assert "dim=2" in repr_str
        assert "transforms=1" in repr_str
