"""Tests for scenario system.

Tests ScenarioSpec, scenario composition, and conflict resolution.
"""

import pytest
from hypothesis import given, strategies as st

from modelops_calabaria.parameters import (
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ConfigSpec,
    ConfigurationSpace,
    ConfigurationSet,
)
from modelops_calabaria.scenarios import (
    ScenarioSpec,
    compose_scenarios,
    scenario_hash,
)


# Module-level fixtures shared across all test classes
@pytest.fixture
def space():
    """Create test parameter space."""
    return ParameterSpace([
        ParameterSpec("beta", 0.0, 1.0, "float"),
        ParameterSpec("gamma", 0.0, 1.0, "float"),
        ParameterSpec("contact_rate", 0.0, 10.0, "float"),
        ParameterSpec("population", 100, 10000, "int"),
    ])


@pytest.fixture
def base_params(space):
    """Create base parameter set."""
    return ParameterSet(space, {
        "beta": 0.3,
        "gamma": 0.1,
        "contact_rate": 4.0,
        "population": 1000,
    })


@pytest.fixture
def config_space():
    """Create test configuration space."""
    return ConfigurationSpace([
        ConfigSpec("mode", default="normal"),
        ConfigSpec("output", default="detailed"),
        ConfigSpec("alerts", default=False),
        ConfigSpec("feature_a", default=False),
        ConfigSpec("feature_b", default=False),
        ConfigSpec("level", default=0),
    ])


@pytest.fixture
def base_config(config_space):
    """Create base configuration set."""
    return ConfigurationSet(config_space, {
        "mode": "normal",
        "output": "detailed",
        "alerts": False,
        "feature_a": False,
        "feature_b": False,
        "level": 0,
    })


@pytest.fixture
def empty_config():
    """Create empty configuration set for tests that don't need config."""
    space = ConfigurationSpace([])
    return ConfigurationSet(space, {})


class TestScenarioSpec:
    """Tests for ScenarioSpec."""

    def test_create_empty_scenario(self):
        """Test creating scenario with no patches."""
        spec = ScenarioSpec(name="baseline", doc="Default scenario")
        assert spec.name == "baseline"
        assert spec.doc == "Default scenario"
        assert len(spec.param_patches) == 0
        assert len(spec.config_patches) == 0
        assert spec.conflict_policy == "lww"
        assert spec.allow_overlap is None

    def test_create_scenario_with_patches(self):
        """Test creating scenario with parameter and config patches."""
        spec = ScenarioSpec(
            name="lockdown",
            doc="Lockdown scenario",
            param_patches={"contact_rate": 2.0, "beta": 0.1},
            config_patches={"mobility": "low", "schools": "closed"},
        )
        assert len(spec.param_patches) == 2
        assert spec.param_patches["contact_rate"] == 2.0
        assert spec.param_patches["beta"] == 0.1
        assert len(spec.config_patches) == 2
        assert spec.config_patches["mobility"] == "low"

    def test_patches_are_immutable(self):
        """Test that patches cannot be modified after creation."""
        spec = ScenarioSpec(
            name="test",
            param_patches={"beta": 0.5},
            config_patches={"setting": "value"},
        )

        # Cannot modify param_patches
        with pytest.raises(TypeError):
            spec.param_patches["gamma"] = 0.1

        # Cannot modify config_patches
        with pytest.raises(TypeError):
            spec.config_patches["new_setting"] = "new_value"

        # Cannot reassign patches
        with pytest.raises(AttributeError):
            spec.param_patches = {}

    def test_apply_param_patches(self, space, base_params):
        """Test applying parameter patches."""
        spec = ScenarioSpec(
            name="high_transmission",
            param_patches={"beta": 0.8, "contact_rate": 8.0},
        )

        new_params, new_config = spec.apply(base_params, {})

        # Check patches applied
        assert new_params["beta"] == 0.8
        assert new_params["contact_rate"] == 8.0
        # Check unchanged params preserved
        assert new_params["gamma"] == 0.1
        assert new_params["population"] == 1000
        # Check original unchanged
        assert base_params["beta"] == 0.3

    def test_apply_config_patches(self, space, base_params, base_config):
        """Test applying configuration patches."""
        spec = ScenarioSpec(
            name="test",
            config_patches={"mode": "emergency", "alerts": True},
        )

        new_params, new_config = spec.apply(base_params, base_config)

        # Check config patches applied
        assert new_config["mode"] == "emergency"
        assert new_config["alerts"] is True
        # Check unchanged config preserved
        assert new_config["output"] == "detailed"
        # Check original unchanged (immutable)
        assert base_config["mode"] == "normal"

    def test_apply_unknown_parameter_raises(self, space, base_params):
        """Test that patching unknown parameter raises error."""
        spec = ScenarioSpec(
            name="bad",
            param_patches={"alpha": 0.5},  # Unknown parameter
        )

        with pytest.raises(ValueError, match="unknown parameter: alpha"):
            spec.apply(base_params, {})

    def test_apply_out_of_bounds_raises(self, space, base_params):
        """Test that out-of-bounds values raise error."""
        spec = ScenarioSpec(
            name="bad",
            param_patches={"beta": 1.5},  # > 1.0 upper bound
        )

        with pytest.raises(ValueError, match="invalid value for beta"):
            spec.apply(base_params, {})

    def test_apply_wrong_type_raises(self, space, base_params):
        """Test that wrong type values raise error."""
        spec = ScenarioSpec(
            name="bad",
            param_patches={"population": 1000.5},  # Float for int param
        )

        with pytest.raises(ValueError, match="invalid value for population"):
            spec.apply(base_params, {})

    def test_conflict_policy_options(self):
        """Test different conflict policy options."""
        # Last-write-wins (default)
        spec1 = ScenarioSpec(name="s1")
        assert spec1.conflict_policy == "lww"

        # Strict mode
        spec2 = ScenarioSpec(name="s2", conflict_policy="strict")
        assert spec2.conflict_policy == "strict"

    def test_allow_overlap_converted_to_tuple(self):
        """Test that allow_overlap is converted to immutable tuple."""
        spec = ScenarioSpec(
            name="test",
            allow_overlap=["beta", "gamma"],  # List input
        )
        assert isinstance(spec.allow_overlap, tuple)
        assert spec.allow_overlap == ("beta", "gamma")


class TestScenarioComposition:
    """Tests for composing multiple scenarios."""

    @pytest.fixture
    def space(self):
        """Create test parameter space."""
        return ParameterSpace([
            ParameterSpec("beta", 0.0, 1.0, "float"),
            ParameterSpec("gamma", 0.0, 1.0, "float"),
            ParameterSpec("contact_rate", 0.0, 10.0, "float"),
        ])

    @pytest.fixture
    def base_params(self, space):
        """Create base parameter set."""
        return ParameterSet(space, {
            "beta": 0.3,
            "gamma": 0.1,
            "contact_rate": 4.0,
        })

    def test_compose_single_scenario(self, base_params):
        """Test composing single scenario."""
        spec = ScenarioSpec(
            name="lockdown",
            param_patches={"contact_rate": 2.0},
        )

        new_params, new_config, sources = compose_scenarios(
            [spec], base_params, {}
        )

        assert new_params["contact_rate"] == 2.0
        assert sources["contact_rate"] == "lockdown"

    def test_compose_multiple_scenarios_lww(self, base_params):
        """Test composing multiple scenarios with last-write-wins."""
        spec1 = ScenarioSpec(
            name="mild",
            param_patches={"beta": 0.2, "contact_rate": 3.0},
        )
        spec2 = ScenarioSpec(
            name="severe",
            param_patches={"beta": 0.1, "gamma": 0.2},  # Overrides beta
        )

        new_params, new_config, sources = compose_scenarios(
            [spec1, spec2], base_params, {}
        )

        # Last write wins for beta
        assert new_params["beta"] == 0.1  # From spec2
        assert new_params["contact_rate"] == 3.0  # From spec1
        assert new_params["gamma"] == 0.2  # From spec2

        # Sources tracked
        assert sources["beta"] == "severe"
        assert sources["contact_rate"] == "mild"
        assert sources["gamma"] == "severe"

    def test_compose_order_matters(self, base_params, empty_config):
        """Test that scenario order matters for overlapping patches."""
        spec_a = ScenarioSpec(name="A", param_patches={"beta": 0.4})
        spec_b = ScenarioSpec(name="B", param_patches={"beta": 0.6})

        # A then B
        params_ab, _, _ = compose_scenarios([spec_a, spec_b], base_params, empty_config)
        assert params_ab["beta"] == 0.6  # B wins

        # B then A
        params_ba, _, _ = compose_scenarios([spec_b, spec_a], base_params, empty_config)
        assert params_ba["beta"] == 0.4  # A wins

        # Order matters!
        assert params_ab["beta"] != params_ba["beta"]

    def test_compose_strict_mode_raises(self, base_params, empty_config):
        """Test that strict mode raises on conflicts."""
        spec1 = ScenarioSpec(
            name="first",
            param_patches={"beta": 0.4},
        )
        spec2 = ScenarioSpec(
            name="second",
            param_patches={"beta": 0.6},
            conflict_policy="strict",  # Strict mode
        )

        with pytest.raises(ValueError, match="set by multiple scenarios.*first.*second"):
            compose_scenarios([spec1, spec2], base_params, empty_config)

    def test_compose_strict_with_allow_overlap(self, base_params, empty_config):
        """Test that allow_overlap permits specific conflicts in strict mode."""
        spec1 = ScenarioSpec(
            name="first",
            param_patches={"beta": 0.4, "gamma": 0.15},
        )
        spec2 = ScenarioSpec(
            name="second",
            param_patches={"beta": 0.6, "gamma": 0.2},
            conflict_policy="strict",
            allow_overlap=("beta",),  # Allow beta to overlap
        )

        # Should raise for gamma (not in allow_overlap)
        with pytest.raises(ValueError, match="gamma.*set by multiple"):
            compose_scenarios([spec1, spec2], base_params, empty_config)

        # If we allow both, should work
        spec3 = ScenarioSpec(
            name="third",
            param_patches={"beta": 0.6, "gamma": 0.2},
            conflict_policy="strict",
            allow_overlap=("beta", "gamma"),
        )
        params, _, _ = compose_scenarios([spec1, spec3], base_params, empty_config)
        assert params["beta"] == 0.6
        assert params["gamma"] == 0.2

    def test_compose_config_patches(self, base_params, base_config):
        """Test composing config patches across scenarios."""
        spec1 = ScenarioSpec(
            name="s1",
            config_patches={"feature_a": True, "level": 1},
        )
        spec2 = ScenarioSpec(
            name="s2",
            config_patches={"feature_b": True, "level": 2},  # Overrides level
        )

        _, new_config, _ = compose_scenarios(
            [spec1, spec2], base_params, base_config
        )

        assert new_config["mode"] == "normal"  # Base preserved
        assert new_config["feature_a"] is True  # From s1
        assert new_config["feature_b"] is True  # From s2
        assert new_config["level"] == 2  # s2 overrides s1


class TestScenarioHash:
    """Tests for scenario hashing."""

    def test_hash_deterministic(self):
        """Test that hash is deterministic for same content."""
        spec1 = ScenarioSpec(
            name="test",
            param_patches={"beta": 0.5, "gamma": 0.1},
            config_patches={"mode": "test"},
        )
        spec2 = ScenarioSpec(
            name="test",
            param_patches={"beta": 0.5, "gamma": 0.1},
            config_patches={"mode": "test"},
        )

        hash1 = scenario_hash(spec1)
        hash2 = scenario_hash(spec2)
        assert hash1 == hash2

    def test_hash_order_independent(self):
        """Test that hash is independent of patch order."""
        spec1 = ScenarioSpec(
            name="test",
            param_patches={"beta": 0.5, "gamma": 0.1},
        )
        # Different order in dict (though Python 3.7+ preserves order)
        spec2 = ScenarioSpec(
            name="test",
            param_patches={"gamma": 0.1, "beta": 0.5},
        )

        hash1 = scenario_hash(spec1)
        hash2 = scenario_hash(spec2)
        assert hash1 == hash2  # Order shouldn't matter due to sorting

    def test_hash_changes_with_content(self):
        """Test that hash changes when content changes."""
        spec1 = ScenarioSpec(name="test", param_patches={"beta": 0.5})
        spec2 = ScenarioSpec(name="test", param_patches={"beta": 0.6})
        spec3 = ScenarioSpec(name="other", param_patches={"beta": 0.5})

        hash1 = scenario_hash(spec1)
        hash2 = scenario_hash(spec2)
        hash3 = scenario_hash(spec3)

        assert hash1 != hash2  # Different value
        assert hash1 != hash3  # Different name
        assert hash2 != hash3  # Both different

    def test_hash_format(self):
        """Test that hash has expected format."""
        spec = ScenarioSpec(name="test")
        hash_val = scenario_hash(spec)

        # Should be 16 character hex string
        assert len(hash_val) == 16
        assert all(c in "0123456789abcdef" for c in hash_val)