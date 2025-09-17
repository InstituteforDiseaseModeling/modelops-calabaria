"""Tests for core parameter types.

Tests the fundamental parameter system including:
- ParameterSpec validation
- ParameterSpace immutability and completeness
- ParameterSet validation and freezing
- Error handling for typos and missing parameters
"""

import pytest
from hypothesis import given, strategies as st, assume
from typing import Dict

from modelops_calabaria.parameters import (
    Scalar,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
)


class TestParameterSpec:
    """Tests for ParameterSpec."""

    def test_create_float_spec(self):
        """Test creating a float parameter specification."""
        spec = ParameterSpec("beta", 0.0, 1.0, "float", "Transmission rate")
        assert spec.name == "beta"
        assert spec.min == 0.0
        assert spec.max == 1.0
        assert spec.kind == "float"
        assert spec.doc == "Transmission rate"

    def test_create_int_spec(self):
        """Test creating an integer parameter specification."""
        spec = ParameterSpec("population", 100, 10000, "int", "Population size")
        assert spec.name == "population"
        assert spec.min == 100
        assert spec.max == 10000
        assert spec.kind == "int"

    def test_invalid_kind_raises(self):
        """Test that invalid parameter kind raises error."""
        with pytest.raises(ValueError, match="Parameter kind must be"):
            ParameterSpec("param", 0, 1, "string")

    def test_min_greater_than_max_raises(self):
        """Test that min > max raises error."""
        with pytest.raises(ValueError, match="min .* > max"):
            ParameterSpec("param", 10.0, 5.0)

    def test_int_spec_requires_int_bounds(self):
        """Test that integer specs require integer bounds."""
        with pytest.raises(ValueError, match="must have integer bounds"):
            ParameterSpec("count", 0.0, 10.0, "int")

    def test_validate_value_in_bounds(self):
        """Test value validation for in-bounds values."""
        spec = ParameterSpec("beta", 0.0, 1.0, "float")
        spec.validate_value(0.5)  # Should not raise
        spec.validate_value(0.0)  # Min boundary
        spec.validate_value(1.0)  # Max boundary

    def test_validate_value_out_of_bounds(self):
        """Test value validation for out-of-bounds values."""
        spec = ParameterSpec("beta", 0.0, 1.0, "float")
        with pytest.raises(ValueError, match="outside bounds"):
            spec.validate_value(1.5)
        with pytest.raises(ValueError, match="outside bounds"):
            spec.validate_value(-0.1)

    def test_validate_value_wrong_type(self):
        """Test value validation for wrong type."""
        spec = ParameterSpec("population", 0, 100, "int")
        with pytest.raises(TypeError, match="requires int"):
            spec.validate_value(50.5)

    def test_spec_is_immutable(self):
        """Test that ParameterSpec is immutable."""
        spec = ParameterSpec("beta", 0.0, 1.0)
        with pytest.raises(AttributeError):
            spec.name = "gamma"
        with pytest.raises(AttributeError):
            spec.min = -1.0


class TestParameterSpace:
    """Tests for ParameterSpace."""

    def test_create_space(self):
        """Test creating a parameter space."""
        specs = [
            ParameterSpec("beta", 0.0, 1.0, "float"),
            ParameterSpec("gamma", 0.0, 1.0, "float"),
            ParameterSpec("population", 100, 10000, "int"),
        ]
        space = ParameterSpace(specs)
        assert len(space) == 3
        assert space.names() == ["beta", "gamma", "population"]

    def test_duplicate_names_raise(self):
        """Test that duplicate parameter names raise error."""
        specs = [
            ParameterSpec("beta", 0.0, 1.0),
            ParameterSpec("beta", 0.0, 2.0),  # Duplicate name
        ]
        with pytest.raises(ValueError, match="Duplicate parameter names.*beta"):
            ParameterSpace(specs)

    def test_get_spec(self):
        """Test retrieving parameter specification."""
        specs = [
            ParameterSpec("beta", 0.0, 1.0),
            ParameterSpec("gamma", 0.0, 1.0),
        ]
        space = ParameterSpace(specs)

        beta_spec = space.get_spec("beta")
        assert beta_spec.name == "beta"
        assert beta_spec.min == 0.0
        assert beta_spec.max == 1.0

    def test_get_spec_unknown_raises(self):
        """Test that getting unknown parameter raises error."""
        space = ParameterSpace([ParameterSpec("beta", 0.0, 1.0)])
        with pytest.raises(KeyError, match="Unknown parameter: gamma"):
            space.get_spec("gamma")

    def test_contains(self):
        """Test parameter membership check."""
        space = ParameterSpace([
            ParameterSpec("beta", 0.0, 1.0),
            ParameterSpec("gamma", 0.0, 1.0),
        ])
        assert "beta" in space
        assert "gamma" in space
        assert "alpha" not in space

    def test_space_is_immutable(self):
        """Test that ParameterSpace is immutable."""
        space = ParameterSpace([ParameterSpec("beta", 0.0, 1.0)])

        # Can't modify specs
        with pytest.raises(AttributeError):
            space.specs.append(ParameterSpec("gamma", 0.0, 1.0))

        # Can't reassign specs
        with pytest.raises(AttributeError):
            space.specs = []


class TestParameterSet:
    """Tests for ParameterSet."""

    @pytest.fixture
    def space(self):
        """Create a test parameter space."""
        return ParameterSpace([
            ParameterSpec("beta", 0.0, 1.0, "float"),
            ParameterSpec("gamma", 0.0, 1.0, "float"),
            ParameterSpec("population", 100, 10000, "int"),
        ])

    def test_create_valid_set(self, space):
        """Test creating a valid parameter set."""
        values = {"beta": 0.3, "gamma": 0.1, "population": 1000}
        pset = ParameterSet(space, values)

        assert pset["beta"] == 0.3
        assert pset["gamma"] == 0.1
        assert pset["population"] == 1000

    def test_missing_parameter_raises(self, space):
        """Test that missing parameters fail loudly."""
        values = {"beta": 0.3, "gamma": 0.1}  # Missing population
        with pytest.raises(ValueError, match="Missing required parameters.*population"):
            ParameterSet(space, values)

    def test_typo_in_parameter_name_raises(self, space):
        """Test that typos in parameter names fail loudly."""
        values = {
            "betta": 0.3,  # Typo! Should be "beta"
            "gamma": 0.1,
            "population": 1000
        }
        with pytest.raises(ValueError, match="Unknown parameters.*betta"):
            ParameterSet(space, values)

    def test_extra_parameter_raises(self, space):
        """Test that extra parameters raise error."""
        values = {
            "beta": 0.3,
            "gamma": 0.1,
            "population": 1000,
            "alpha": 0.5  # Not in space
        }
        with pytest.raises(ValueError, match="Unknown parameters.*alpha"):
            ParameterSet(space, values)

    def test_out_of_bounds_value_raises(self, space):
        """Test that out-of-bounds values raise error."""
        values = {"beta": 1.5, "gamma": 0.1, "population": 1000}  # beta > 1.0
        with pytest.raises(ValueError, match="Validation failed for beta"):
            ParameterSet(space, values)

    def test_wrong_type_value_raises(self, space):
        """Test that wrong type values raise error."""
        values = {"beta": 0.3, "gamma": 0.1, "population": 1000.5}  # float for int param
        with pytest.raises(ValueError, match="Validation failed for population"):
            ParameterSet(space, values)

    def test_values_are_immutable(self, space):
        """Test that ParameterSet values cannot be modified."""
        values = {"beta": 0.3, "gamma": 0.1, "population": 1000}
        pset = ParameterSet(space, values)

        # Can't modify values dict
        with pytest.raises(TypeError):
            pset.values["beta"] = 0.5

        # Can't reassign values
        with pytest.raises(AttributeError):
            pset.values = {"beta": 0.5}

    def test_with_updates_creates_new(self, space):
        """Test that with_updates creates a new ParameterSet."""
        pset1 = ParameterSet(space, {"beta": 0.3, "gamma": 0.1, "population": 1000})
        pset2 = pset1.with_updates(beta=0.5)

        # Original unchanged
        assert pset1["beta"] == 0.3
        # New has update
        assert pset2["beta"] == 0.5
        # Others unchanged
        assert pset2["gamma"] == 0.1
        assert pset2["population"] == 1000
        # Different objects
        assert pset1 is not pset2

    def test_with_updates_validates(self, space):
        """Test that with_updates validates new values."""
        pset = ParameterSet(space, {"beta": 0.3, "gamma": 0.1, "population": 1000})

        # Out of bounds update should fail
        with pytest.raises(ValueError, match="Validation failed for beta"):
            pset.with_updates(beta=1.5)

        # Unknown parameter should fail
        with pytest.raises(ValueError, match="Unknown parameters.*alpha"):
            pset.with_updates(alpha=0.5)

    def test_to_dict(self, space):
        """Test exporting to dict."""
        values = {"beta": 0.3, "gamma": 0.1, "population": 1000}
        pset = ParameterSet(space, values)
        exported = pset.to_dict()

        assert exported == values
        # Should be a regular dict, not MappingProxyType
        assert type(exported) is dict
        # Modifying export doesn't affect original
        exported["beta"] = 0.5
        assert pset["beta"] == 0.3

    def test_getitem_unknown_raises(self, space):
        """Test that accessing unknown parameter raises error."""
        pset = ParameterSet(space, {"beta": 0.3, "gamma": 0.1, "population": 1000})
        with pytest.raises(KeyError, match="Parameter alpha not in set"):
            pset["alpha"]


class TestPropertyTests:
    """Property-based tests using hypothesis."""

    @given(
        min_val=st.floats(min_value=-1000, max_value=1000),
        max_val=st.floats(min_value=-1000, max_value=1000),
    )
    def test_spec_bounds_ordering(self, min_val, max_val):
        """Property: ParameterSpec enforces min <= max."""
        assume(min_val <= max_val)
        spec = ParameterSpec("param", min_val, max_val, "float")
        assert spec.min <= spec.max

    @given(params=st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.floats(min_value=-999, max_value=999, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10
    ))
    def test_parameter_set_immutability(self, params):
        """Property: ParameterSet values cannot be modified after creation."""
        # Create space from params (with slightly wider bounds to ensure values fit)
        specs = [ParameterSpec(name, -1000, 1000, "float") for name in params]
        if not specs:  # Skip empty
            return

        space = ParameterSpace(specs)
        pset = ParameterSet(space, params)

        # Original values preserved
        original = dict(pset.values)

        # Attempt mutation (should fail or have no effect)
        with pytest.raises(TypeError):
            pset.values["new_key"] = 1.0

        # Values unchanged
        assert dict(pset.values) == original

    @given(
        base=st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False),
            min_size=1
        ),
        updates=st.dictionaries(
            st.text(min_size=1, max_size=10),
            st.floats(min_value=-1000, max_value=1000, allow_nan=False),
            min_size=0
        )
    )
    def test_with_updates_creates_new(self, base, updates):
        """Property: with_updates() creates new object, doesn't mutate original."""
        # Create space that accepts all values generated
        all_names = set(base.keys()) | set(updates.keys())
        specs = [ParameterSpec(name, -1001, 1001, "float") for name in all_names]
        space = ParameterSpace(specs)

        # Ensure base has all required params
        full_base = {spec.name: base.get(spec.name, 0.0) for spec in specs}
        pset1 = ParameterSet(space, full_base)

        # Apply valid updates (only those in base)
        valid_updates = {k: v for k, v in updates.items() if k in full_base}
        if not valid_updates:
            valid_updates = {list(full_base.keys())[0]: 1.0}

        pset2 = pset1.with_updates(**valid_updates)

        # Original unchanged
        assert dict(pset1.values) == full_base
        # New object created
        assert pset1 is not pset2
        # Updates applied to new only
        for k, v in valid_updates.items():
            assert pset2[k] == v


def test_no_silent_defaults():
    """Test that missing parameters fail loudly, no silent defaults."""
    space = ParameterSpace([
        ParameterSpec("beta", 0.0, 1.0, "float"),
        ParameterSpec("gamma", 0.0, 1.0, "float"),
    ])

    # Missing parameter fails
    with pytest.raises(ValueError, match="Missing required parameter"):
        ParameterSet(space, {"gamma": 0.1})  # no beta

    # Typo fails
    with pytest.raises(ValueError, match="Unknown parameter"):
        ParameterSet(space, {"betta": 0.3, "gamma": 0.1})  # typo in beta