"""Tests for model decorators.

Tests @model_output and @model_scenario decorators.
"""

import pytest
from typing import Any

from modelops_calabaria.decorators import (
    model_output,
    model_scenario,
    discover_decorated_methods,
    SEED_COL,
)
from modelops_calabaria.scenarios import ScenarioSpec


class TestModelOutputDecorator:
    """Tests for @model_output decorator."""

    def test_basic_decoration(self):
        """Test basic @model_output decoration."""
        class TestModel:
            @model_output("prevalence")
            def extract_prevalence(self, raw, seed):
                return f"prevalence_{seed}"

        # Check markers added
        method = TestModel.extract_prevalence
        assert hasattr(method, "_is_model_output")
        assert method._is_model_output is True
        assert method._output_name == "prevalence"
        assert method._output_metadata == {}

    def test_decoration_with_metadata(self):
        """Test @model_output with metadata."""
        class TestModel:
            @model_output("incidence", metadata={"units": "cases", "frequency": "weekly"})
            def extract_incidence(self, raw, seed):
                return f"incidence_{seed}"

        method = TestModel.extract_incidence
        assert method._output_metadata == {"units": "cases", "frequency": "weekly"}

    def test_static_method_raises(self):
        """Test that @model_output on static method raises error."""
        with pytest.raises(TypeError, match="must be an instance method"):
            class TestModel:
                @model_output("bad")
                @staticmethod
                def extract_bad(raw, seed):
                    return "bad"

    def test_class_method_raises(self):
        """Test that @model_output on class method raises error."""
        with pytest.raises(TypeError, match="must be an instance method"):
            class TestModel:
                @model_output("bad")
                @classmethod
                def extract_bad(cls, raw, seed):
                    return "bad"

    def test_method_still_callable(self):
        """Test that decorated method remains callable."""
        class TestModel:
            @model_output("test")
            def extract_test(self, raw, seed):
                return f"result_{raw}_{seed}"

        model = TestModel()
        result = model.extract_test("data", 42)
        assert result == "result_data_42"


class TestModelScenarioDecorator:
    """Tests for @model_scenario decorator."""

    def test_basic_decoration(self):
        """Test basic @model_scenario decoration."""
        class TestModel:
            @model_scenario("lockdown")
            def lockdown_scenario(self):
                return ScenarioSpec(
                    name="lockdown",
                    doc="Lockdown scenario",
                    param_patches={"contact_rate": 2.0}
                )

        # Check markers added
        method = TestModel.lockdown_scenario
        assert hasattr(method, "_is_model_scenario")
        assert method._is_model_scenario is True
        assert method._scenario_name == "lockdown"

    def test_validates_return_type(self):
        """Test that @model_scenario validates return type."""
        class TestModel:
            @model_scenario("bad")
            def bad_scenario(self):
                return {"not": "a scenario"}  # Wrong type

        model = TestModel()
        with pytest.raises(TypeError, match="must return ScenarioSpec"):
            model.bad_scenario()

    def test_validates_name_match(self):
        """Test that @model_scenario validates name matches."""
        class TestModel:
            @model_scenario("expected")
            def mismatched_scenario(self):
                return ScenarioSpec(
                    name="different",  # Doesn't match decorator
                    doc="Test"
                )

        model = TestModel()
        with pytest.raises(ValueError, match="different name"):
            model.mismatched_scenario()

    def test_static_method_raises(self):
        """Test that @model_scenario on static method raises error."""
        with pytest.raises(TypeError, match="must be an instance method"):
            class TestModel:
                @model_scenario("bad")
                @staticmethod
                def bad_scenario():
                    return ScenarioSpec(name="bad")

    def test_preserves_metadata(self):
        """Test that decorator preserves function metadata."""
        class TestModel:
            @model_scenario("test")
            def test_scenario(self):
                """This is a test scenario."""
                return ScenarioSpec(name="test")

        method = TestModel.test_scenario
        assert method.__name__ == "test_scenario"
        assert method.__doc__ == "This is a test scenario."

    def test_method_callable_and_validated(self):
        """Test that decorated method is callable and validates."""
        class TestModel:
            @model_scenario("valid")
            def valid_scenario(self):
                return ScenarioSpec(
                    name="valid",
                    param_patches={"beta": 0.5}
                )

        model = TestModel()
        result = model.valid_scenario()
        assert isinstance(result, ScenarioSpec)
        assert result.name == "valid"
        assert result.param_patches["beta"] == 0.5


class TestDiscoverDecoratedMethods:
    """Tests for discover_decorated_methods function."""

    def test_discover_outputs(self):
        """Test discovering @model_output methods."""
        class TestModel:
            @model_output("prevalence")
            def extract_prevalence(self, raw, seed):
                pass

            @model_output("incidence")
            def extract_incidence(self, raw, seed):
                pass

            def not_decorated(self):
                pass

        outputs, scenarios = discover_decorated_methods(TestModel)

        assert len(outputs) == 2
        assert outputs["prevalence"] == "extract_prevalence"
        assert outputs["incidence"] == "extract_incidence"
        assert len(scenarios) == 0

    def test_discover_scenarios(self):
        """Test discovering @model_scenario methods."""
        class TestModel:
            @model_scenario("baseline")
            def baseline_scenario(self):
                return ScenarioSpec(name="baseline")

            @model_scenario("lockdown")
            def lockdown_scenario(self):
                return ScenarioSpec(name="lockdown")

            def not_decorated(self):
                pass

        outputs, scenarios = discover_decorated_methods(TestModel)

        assert len(scenarios) == 2
        assert scenarios["baseline"] == "baseline_scenario"
        assert scenarios["lockdown"] == "lockdown_scenario"
        assert len(outputs) == 0

    def test_discover_mixed(self):
        """Test discovering both output and scenario methods."""
        class TestModel:
            @model_output("data")
            def extract_data(self, raw, seed):
                pass

            @model_scenario("test")
            def test_scenario(self):
                return ScenarioSpec(name="test")

        outputs, scenarios = discover_decorated_methods(TestModel)

        assert len(outputs) == 1
        assert outputs["data"] == "extract_data"
        assert len(scenarios) == 1
        assert scenarios["test"] == "test_scenario"

    def test_discover_inheritance(self):
        """Test discovering methods across inheritance."""
        class BaseModel:
            @model_output("base_output")
            def extract_base(self, raw, seed):
                pass

            @model_scenario("base_scenario")
            def base_scenario(self):
                return ScenarioSpec(name="base_scenario")

        class DerivedModel(BaseModel):
            @model_output("derived_output")
            def extract_derived(self, raw, seed):
                pass

        outputs, scenarios = discover_decorated_methods(DerivedModel)

        # Should find both base and derived methods
        assert len(outputs) == 2
        assert "base_output" in outputs
        assert "derived_output" in outputs
        assert len(scenarios) == 1
        assert "base_scenario" in scenarios

    def test_discover_ignores_private(self):
        """Test that discovery ignores private methods."""
        class TestModel:
            @model_output("public")
            def extract_public(self, raw, seed):
                pass

            # Even if we could decorate private (which we shouldn't)
            def _extract_private(self, raw, seed):
                pass

        outputs, scenarios = discover_decorated_methods(TestModel)

        assert len(outputs) == 1
        assert "public" in outputs
        # Private method not discovered even if it existed


class TestSeedColumn:
    """Test SEED_COL constant."""

    def test_seed_col_value(self):
        """Test that SEED_COL has expected value."""
        assert SEED_COL == "seed"
        # This constant is used to ensure extractors don't add seed column
        # since framework adds it automatically