"""Tests for configuration space types (C-space).

Tests ConfigSpec, ConfigurationSpace, and ConfigurationSet for completeness,
immutability, and validation.
"""

import pytest
from modelops_calabaria.parameters.config import (
    ConfigSpec,
    ConfigurationSpace,
    ConfigurationSet,
)


class TestConfigSpec:
    """Tests for ConfigSpec."""

    def test_config_spec_creation(self):
        """Test basic ConfigSpec creation."""
        spec = ConfigSpec("dt", default=0.1, doc="Time step")
        assert spec.name == "dt"
        assert spec.default == 0.1
        assert spec.doc == "Time step"

    def test_config_spec_minimal(self):
        """Test ConfigSpec with minimal fields."""
        spec = ConfigSpec("output_freq", default=1.0)
        assert spec.name == "output_freq"
        assert spec.default == 1.0
        assert spec.doc == ""

    def test_config_spec_various_types(self):
        """Test ConfigSpec with different default types."""
        float_spec = ConfigSpec("dt", default=0.1)
        int_spec = ConfigSpec("max_steps", default=100)
        str_spec = ConfigSpec("output_dir", default="./results")
        bool_spec = ConfigSpec("verbose", default=False)

        assert float_spec.default == 0.1
        assert int_spec.default == 100
        assert str_spec.default == "./results"
        assert bool_spec.default is False

    def test_config_spec_empty_name_fails(self):
        """Test that empty name raises error."""
        with pytest.raises(ValueError, match="name cannot be empty"):
            ConfigSpec("", default=0.1)

    def test_config_spec_immutable(self):
        """Test that ConfigSpec is immutable."""
        spec = ConfigSpec("dt", default=0.1)
        with pytest.raises(Exception):  # FrozenInstanceError
            spec.name = "new_name"


class TestConfigurationSpace:
    """Tests for ConfigurationSpace."""

    def test_configuration_space_creation(self):
        """Test basic ConfigurationSpace creation."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1, doc="Time step"),
            ConfigSpec("output_freq", default=1.0, doc="Output frequency"),
        ])
        assert len(space) == 2
        assert "dt" in space
        assert "output_freq" in space

    def test_configuration_space_empty(self):
        """Test empty ConfigurationSpace."""
        space = ConfigurationSpace([])
        assert len(space) == 0
        assert space.names() == []

    def test_configuration_space_names(self):
        """Test names() returns ordered list."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("output_freq", default=1.0),
            ConfigSpec("verbose", default=False),
        ])
        names = space.names()
        assert names == ["dt", "output_freq", "verbose"]
        assert isinstance(names, list)

    def test_configuration_space_get_spec(self):
        """Test get_spec retrieves specifications."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1, doc="Time step"),
        ])
        spec = space.get_spec("dt")
        assert spec.name == "dt"
        assert spec.default == 0.1
        assert spec.doc == "Time step"

    def test_configuration_space_get_spec_missing(self):
        """Test get_spec raises for unknown config."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
        ])
        with pytest.raises(KeyError, match="Unknown configuration: foo"):
            space.get_spec("foo")

    def test_configuration_space_contains(self):
        """Test __contains__ for checking membership."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("output_freq", default=1.0),
        ])
        assert "dt" in space
        assert "output_freq" in space
        assert "foo" not in space

    def test_configuration_space_duplicate_names_fails(self):
        """Test that duplicate names raise error."""
        with pytest.raises(ValueError, match="Duplicate configuration names"):
            ConfigurationSpace([
                ConfigSpec("dt", default=0.1),
                ConfigSpec("dt", default=0.2),
            ])

    def test_configuration_space_immutable(self):
        """Test that ConfigurationSpace is immutable."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
        ])

        # specs should be frozen to tuple
        assert isinstance(space.specs, tuple)

        # Can't modify the frozen instance
        with pytest.raises(Exception):
            space.specs = []

    def test_configuration_space_to_dict(self):
        """Test to_dict serialization."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1, doc="Time step"),
            ConfigSpec("output_freq", default=1.0),
        ])
        result = space.to_dict()

        assert "configurations" in result
        assert len(result["configurations"]) == 2

        config = result["configurations"][0]
        assert config["name"] == "dt"
        assert config["default"] == 0.1
        assert config["doc"] == "Time step"


class TestConfigurationSet:
    """Tests for ConfigurationSet."""

    def test_configuration_set_creation(self):
        """Test basic ConfigurationSet creation."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("output_freq", default=1.0),
        ])
        config_set = ConfigurationSet(space, {"dt": 0.05, "output_freq": 2.0})

        assert config_set["dt"] == 0.05
        assert config_set["output_freq"] == 2.0

    def test_configuration_set_getitem(self):
        """Test __getitem__ access."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
        ])
        config_set = ConfigurationSet(space, {"dt": 0.05})

        assert config_set["dt"] == 0.05

    def test_configuration_set_getitem_missing(self):
        """Test __getitem__ raises for missing config."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
        ])
        config_set = ConfigurationSet(space, {"dt": 0.05})

        with pytest.raises(KeyError, match="Configuration foo not in set"):
            _ = config_set["foo"]

    def test_configuration_set_completeness_required(self):
        """Test that all configurations must be specified."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("output_freq", default=1.0),
        ])

        # Missing output_freq
        with pytest.raises(ValueError, match="Missing required configurations"):
            ConfigurationSet(space, {"dt": 0.05})

    def test_configuration_set_unknown_configs_fail(self):
        """Test that unknown configurations are rejected."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
        ])

        with pytest.raises(ValueError, match="Unknown configurations"):
            ConfigurationSet(space, {"dt": 0.05, "foo": "bar"})

    def test_configuration_set_immutable(self):
        """Test that ConfigurationSet values are immutable."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
        ])
        config_set = ConfigurationSet(space, {"dt": 0.05})

        # values should be MappingProxyType (immutable)
        with pytest.raises(TypeError):
            config_set.values["dt"] = 0.2

    def test_configuration_set_with_updates(self):
        """Test with_updates creates new ConfigurationSet."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("output_freq", default=1.0),
        ])
        config_set1 = ConfigurationSet(space, {"dt": 0.05, "output_freq": 1.0})
        config_set2 = config_set1.with_updates(dt=0.2)

        # Original unchanged
        assert config_set1["dt"] == 0.05
        # New set has update
        assert config_set2["dt"] == 0.2
        # Other values preserved
        assert config_set2["output_freq"] == 1.0

    def test_configuration_set_with_updates_unknown_fails(self):
        """Test with_updates rejects unknown configurations."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
        ])
        config_set = ConfigurationSet(space, {"dt": 0.05})

        with pytest.raises(ValueError, match="Unknown configurations"):
            config_set.with_updates(foo="bar")

    def test_configuration_set_to_dict(self):
        """Test to_dict returns plain dict."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("output_freq", default=1.0),
        ])
        config_set = ConfigurationSet(space, {"dt": 0.05, "output_freq": 2.0})

        result = config_set.to_dict()
        assert isinstance(result, dict)
        assert result == {"dt": 0.05, "output_freq": 2.0}

    def test_configuration_set_repr(self):
        """Test __repr__ is informative."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("output_freq", default=1.0),
        ])
        config_set = ConfigurationSet(space, {"dt": 0.05, "output_freq": 2.0})

        repr_str = repr(config_set)
        assert "ConfigurationSet" in repr_str
        assert "dt" in repr_str or "output_freq" in repr_str

    def test_configuration_set_new_factory(self):
        """Test new() factory method."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("output_freq", default=1.0),
        ])
        config_set = ConfigurationSet.new(space, dt=0.05, output_freq=2.0)

        assert config_set["dt"] == 0.05
        assert config_set["output_freq"] == 2.0

    def test_configuration_set_various_types(self):
        """Test ConfigurationSet with various value types."""
        space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("max_steps", default=100),
            ConfigSpec("output_dir", default="./results"),
            ConfigSpec("verbose", default=False),
        ])
        config_set = ConfigurationSet(space, {
            "dt": 0.05,
            "max_steps": 200,
            "output_dir": "./output",
            "verbose": True,
        })

        assert config_set["dt"] == 0.05
        assert config_set["max_steps"] == 200
        assert config_set["output_dir"] == "./output"
        assert config_set["verbose"] is True
