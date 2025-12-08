"""Tests for ParameterView and P-space operations.

Tests the ParameterView implementation including:
- Fixed/free partitioning
- bind() embedding operation (P → M)
- project() projection operation (M → P)
- Immutability guarantees
- Mathematical properties (projection/embedding laws)
"""

import pytest
from hypothesis import given, strategies as st

from modelops_calabaria.parameters import (
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ParameterView,
)


class TestParameterView:
    """Tests for ParameterView."""

    @pytest.fixture
    def space(self):
        """Create a test parameter space."""
        return ParameterSpace([
            ParameterSpec("beta", 0.0, 1.0, "float"),
            ParameterSpec("gamma", 0.0, 1.0, "float"),
            ParameterSpec("population", 100, 10000, "int"),
            ParameterSpec("contact_rate", 0.0, 10.0, "float"),
        ])

    def test_all_free(self, space):
        """Test creating view with all parameters free."""
        view = ParameterView.all_free(space)

        assert len(view.fixed) == 0
        assert len(view.free) == 4
        assert set(view.free) == {"beta", "gamma", "population", "contact_rate"}

    def test_from_fixed(self, space):
        """Test creating view with specified fixed parameters."""
        view = ParameterView.from_fixed(space, population=1000, beta=0.3)

        assert len(view.fixed) == 2
        assert view.fixed["population"] == 1000
        assert view.fixed["beta"] == 0.3
        assert len(view.free) == 2
        assert set(view.free) == {"gamma", "contact_rate"}

    def test_fix_method(self, space):
        """Test fixing additional parameters returns new view."""
        view1 = ParameterView.all_free(space)
        view2 = view1.fix(population=1000)
        view3 = view2.fix(beta=0.3)

        # Original unchanged
        assert len(view1.fixed) == 0
        assert len(view1.free) == 4

        # Each fix creates new view
        assert view1 is not view2
        assert view2 is not view3

        # Progressive fixing
        assert len(view2.fixed) == 1
        assert len(view2.free) == 3
        assert len(view3.fixed) == 2
        assert len(view3.free) == 2

    def test_unknown_fixed_parameter_raises(self, space):
        """Test that fixing unknown parameter raises error."""
        with pytest.raises(ValueError, match="Unknown fixed parameters.*alpha"):
            ParameterView.from_fixed(space, alpha=0.5)

    def test_non_numeric_fixed_value_raises(self, space):
        """Test that non-numeric fixed value raises error."""
        with pytest.raises(TypeError, match="must be numeric"):
            ParameterView.from_fixed(space, beta="high")

    def test_fixed_is_immutable(self, space):
        """Test that fixed mapping cannot be modified."""
        view = ParameterView.from_fixed(space, beta=0.3)

        with pytest.raises(TypeError):
            view.fixed["gamma"] = 0.1

        with pytest.raises(AttributeError):
            view.fixed = {"beta": 0.5}

    def test_bind_complete_success(self, space):
        """Test successful bind with all free parameters."""
        view = ParameterView.from_fixed(space, population=1000)

        pset = view.bind(beta=0.3, gamma=0.1, contact_rate=4.0)

        assert isinstance(pset, ParameterSet)
        assert pset["population"] == 1000  # Fixed value
        assert pset["beta"] == 0.3  # Free value
        assert pset["gamma"] == 0.1  # Free value
        assert pset["contact_rate"] == 4.0  # Free value

    def test_bind_missing_parameter_raises(self, space):
        """Test bind with missing free parameter raises error."""
        view = ParameterView.from_fixed(space, population=1000)

        with pytest.raises(ValueError, match="bind\\(\\) error.*missing.*gamma"):
            view.bind(beta=0.3, contact_rate=4.0)  # Missing gamma

    def test_bind_extra_parameter_raises(self, space):
        """Test bind with extra parameter raises error."""
        view = ParameterView.from_fixed(space, population=1000, beta=0.3)

        with pytest.raises(ValueError, match="bind\\(\\) error.*unexpected.*beta"):
            view.bind(beta=0.5, gamma=0.1, contact_rate=4.0)  # Beta already fixed

    def test_bind_validates_bounds(self, space):
        """Test that bind validates parameter bounds."""
        view = ParameterView.from_fixed(space, population=1000)

        # Out of bounds value should fail
        with pytest.raises(ValueError, match="outside bounds"):
            view.bind(beta=1.5, gamma=0.1, contact_rate=4.0)

    def test_project_extracts_free(self, space):
        """Test project extracts only free parameters."""
        view = ParameterView.from_fixed(space, population=1000, beta=0.3)
        pset = ParameterSet(space, {
            "beta": 0.3,
            "gamma": 0.1,
            "population": 1000,
            "contact_rate": 4.0
        })

        free_vals = view.project(pset)

        assert free_vals == {"gamma": 0.1, "contact_rate": 4.0}
        assert "beta" not in free_vals  # Fixed, not projected
        assert "population" not in free_vals  # Fixed, not projected

    def test_project_wrong_space_raises(self, space):
        """Test project with wrong space raises error."""
        view = ParameterView.all_free(space)

        # Create ParameterSet with different space
        other_space = ParameterSpace([ParameterSpec("alpha", lower=0, upper=1)])
        other_pset = ParameterSet(other_space, {"alpha": 0.5})

        with pytest.raises(ValueError, match="different space"):
            view.project(other_pset)

    def test_bind_project_round_trip(self, space):
        """Test that bind ∘ project = identity on P-space."""
        view = ParameterView.from_fixed(space, population=1000)

        # Start with free values
        free_values = {"beta": 0.3, "gamma": 0.1, "contact_rate": 4.0}

        # bind: P → M
        pset = view.bind(**free_values)

        # project: M → P
        projected = view.project(pset)

        # Should recover original free values
        assert projected == free_values

    def test_reconciliation_law(self, space):
        """Test reconciliation: fixing a free param moves it from free to fixed."""
        view1 = ParameterView.all_free(space)
        assert "beta" in view1.free
        assert "beta" not in view1.fixed

        # Fix beta
        view2 = view1.fix(beta=0.3)
        assert "beta" not in view2.free
        assert "beta" in view2.fixed
        assert view2.fixed["beta"] == 0.3

        # Now binding with beta should fail (it's no longer free)
        with pytest.raises(ValueError, match="unexpected.*beta"):
            view2.bind(beta=0.5, gamma=0.1, population=1000, contact_rate=4.0)


class TestProjectionEmbeddingLaws:
    """Test mathematical properties of projection and embedding."""

    @pytest.fixture
    def setup(self):
        """Create space and view for testing."""
        space = ParameterSpace([
            ParameterSpec("a", 0, 10, "float"),
            ParameterSpec("b", 0, 10, "float"),
            ParameterSpec("c", 0, 10, "float"),
        ])
        view = ParameterView.from_fixed(space, c=5.0)
        return space, view

    def test_projection_embedding_identity(self, setup):
        """Test π ∘ ι = id_P (project after bind is identity on P)."""
        space, view = setup

        # Start in P-space
        p_values = {"a": 2.0, "b": 3.0}

        # ι: P → M (bind)
        m_point = view.bind(**p_values)

        # π: M → P (project)
        p_recovered = view.project(m_point)

        # Should be identity
        assert p_recovered == p_values

    def test_embedding_projection_clamps_fixed(self, setup):
        """Test ι ∘ π clamps to fixed values."""
        space, view = setup

        # Start with M-space point
        m_point1 = ParameterSet(space, {"a": 2.0, "b": 3.0, "c": 7.0})

        # π: M → P (project to get free)
        p_values = view.project(m_point1)

        # ι: P → M (bind back)
        m_point2 = view.bind(**p_values)

        # Free values preserved
        assert m_point2["a"] == m_point1["a"]
        assert m_point2["b"] == m_point1["b"]

        # Fixed value clamped to view's fixed value (not original)
        assert m_point2["c"] == 5.0  # View's fixed value
        assert m_point1["c"] == 7.0  # Original had different value


class TestPropertyTests:
    """Property-based tests for ParameterView."""

    @given(
        fixed_params=st.dictionaries(
            st.sampled_from(["a", "b", "c"]),
            st.floats(min_value=0, max_value=10),
            min_size=0,
            max_size=2
        )
    )
    def test_fixed_free_partition(self, fixed_params):
        """Property: fixed and free form a partition of parameter names."""
        space = ParameterSpace([
            ParameterSpec("a", lower=0, upper=10),
            ParameterSpec("b", lower=0, upper=10),
            ParameterSpec("c", lower=0, upper=10),
        ])

        view = ParameterView(space, fixed_params)

        # Disjoint
        assert set(view.fixed.keys()).isdisjoint(set(view.free))

        # Complete
        all_params = set(view.fixed.keys()) | set(view.free)
        assert all_params == set(space.names())

    @given(
        n_params=st.integers(min_value=1, max_value=5),
        n_fixed=st.integers(min_value=0, max_value=5)
    )
    def test_fix_increases_fixed_decreases_free(self, n_params, n_fixed):
        """Property: fixing parameters moves them from free to fixed."""
        # Create space with n_params
        specs = [ParameterSpec(f"p{i}", 0, 10) for i in range(n_params)]
        if not specs:
            return
        space = ParameterSpace(specs)

        # Start with all free
        view = ParameterView.all_free(space)
        initial_free = len(view.free)

        # Fix up to n_fixed parameters
        n_to_fix = min(n_fixed, len(specs))
        for i in range(n_to_fix):
            view = view.fix(**{f"p{i}": float(i)})

        # Check counts
        assert len(view.fixed) == n_to_fix
        assert len(view.free) == initial_free - n_to_fix