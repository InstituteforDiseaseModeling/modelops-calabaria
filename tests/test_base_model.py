"""Tests for BaseModel abstract class.

Tests the core model interface including:
- Abstract method enforcement
- Decorator discovery and registration
- Seal-on-first-use behavior
- Parameter validation
- Output extraction with SEED_COL
"""

import pytest
from typing import Dict, Any, Mapping
import polars as pl

from modelops_calabaria import (
    BaseModel,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ConfigSpec,
    ConfigurationSpace,
    ConfigurationSet,
    ScenarioSpec,
    model_output,
    model_scenario,
    SEED_COL,
)


class TestAbstractEnforcement:
    """Test that BaseModel enforces abstract methods and class attributes."""

    def test_cannot_instantiate_without_implementing_abstract(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseModel()

    def test_must_set_params(self):
        """Test that PARAMS must be set as a class attribute."""
        with pytest.raises(TypeError, match="PARAMS must be set"):
            class ModelWithoutParams(BaseModel):
                def build_sim(self, params, config):
                    return {}

                def run_sim(self, state, seed):
                    return {}

    def test_must_implement_build_sim(self):
        """Test that build_sim must be implemented."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            class IncompleteModel(BaseModel):
                PARAMS = ParameterSpace((ParameterSpec("x", lower=0, upper=1),))

                def run_sim(self, state, seed):
                    return {}

            IncompleteModel()

    def test_must_implement_run_sim(self):
        """Test that run_sim must be implemented."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            class IncompleteModel(BaseModel):
                PARAMS = ParameterSpace((ParameterSpec("x", lower=0, upper=1),))

                def build_sim(self, params, config):
                    return {}

            IncompleteModel()


class SimpleTestModel(BaseModel):
    """Concrete test model for testing BaseModel functionality."""

    PARAMS = ParameterSpace((
        ParameterSpec("alpha", 0.0, 1.0, "float"),
        ParameterSpec("beta", 0.0, 1.0, "float"),
        ParameterSpec("steps", 10, 100, "int"),
    ))

    CONFIG = ConfigurationSpace((
        ConfigSpec("mode", default="test"),
    ))

    def __init__(self):
        super().__init__()

    def build_sim(self, params: ParameterSet, config: ConfigurationSet) -> Dict:
        """Build simulation state."""
        return {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "steps": params["steps"],
            "config": config.to_dict(),
        }

    def run_sim(self, state: Dict, seed: int) -> Dict:
        """Run simulation."""
        # Simple deterministic "simulation"
        result = state["alpha"] * 100 + state["beta"] * 10 + seed
        return {
            "result": result,
            "state": state,
            "seed": seed,
        }

    @model_output("result")
    def extract_result(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract result output."""
        return pl.DataFrame({
            "time": [0, 1, 2],
            "value": [raw["result"], raw["result"] * 2, raw["result"] * 3],
        })

    @model_output("summary")
    def extract_summary(self, raw: Dict, seed: int) -> pl.DataFrame:
        """Extract summary output."""
        return pl.DataFrame({
            "metric": ["mean", "max"],
            "value": [raw["result"], raw["result"] * 3],
        })

    @model_scenario("high_alpha")
    def high_alpha_scenario(self) -> ScenarioSpec:
        """Scenario with high alpha value."""
        return ScenarioSpec(
            name="high_alpha",
            doc="High alpha scenario",
            param_patches={"alpha": 0.9},
        )


class TestModelInitialization:
    """Test model initialization and setup."""

    def test_basic_initialization(self):
        """Test basic model initialization."""
        model = SimpleTestModel()

        assert model.space is not None
        assert len(model.space.names()) == 3
        assert model.base_config["mode"] == "test"

    def test_scenarios_discovered(self):
        """Test that decorated scenarios are discovered."""
        model = SimpleTestModel()

        scenarios = model.list_scenarios()
        assert "baseline" in scenarios  # Default
        assert "high_alpha" in scenarios  # From decorator

    def test_outputs_discovered(self):
        """Test that decorated outputs are discovered."""
        model = SimpleTestModel()

        outputs = model.list_outputs()
        assert "result" in outputs
        assert "summary" in outputs
        assert len(outputs) == 2

    def test_base_config_immutable(self):
        """Test that base_config is immutable."""
        model = SimpleTestModel()

        with pytest.raises(TypeError):
            model.base_config["new_key"] = "value"


class TestSealing:
    """Test seal-on-first-use behavior."""

    def test_seal_on_simulate(self):
        """Test that registries seal on first simulate."""
        model = SimpleTestModel()
        params = ParameterSet(model.space, {
            "alpha": 0.5,
            "beta": 0.3,
            "steps": 50,
        })

        # Not sealed yet
        assert not model._sealed

        # Run simulation
        model.simulate(params, seed=42)

        # Now sealed
        assert model._sealed

    def test_cannot_modify_after_seal(self):
        """Test that registries cannot be modified after sealing."""
        model = SimpleTestModel()
        model._seal()

        # Try to add scenario - should fail
        with pytest.raises(TypeError):
            model._scenarios["new"] = ScenarioSpec("new")

        # Try to add output - should fail
        with pytest.raises(TypeError):
            model._outputs["new"] = lambda: None

    def test_seal_idempotent(self):
        """Test that sealing multiple times is safe."""
        model = SimpleTestModel()

        model._seal()
        assert model._sealed

        model._seal()  # Should not raise
        assert model._sealed


class TestParameterValidation:
    """Test parameter validation in simulate()."""

    def test_accepts_dict_params(self):
        """Test that simulate accepts dicts and converts to ParameterSet."""
        model = SimpleTestModel()

        # Dict params should work - gets converted to ParameterSet internally
        outputs = model.simulate({"alpha": 0.5, "beta": 0.3, "steps": 50}, seed=42)
        assert isinstance(outputs, dict)
        assert "result" in outputs or len(outputs) > 0  # Has outputs

    def test_validates_parameter_space(self):
        """Test that ParameterSet must be for correct space."""
        model = SimpleTestModel()

        # Create ParameterSet for different space
        other_space = ParameterSpace([ParameterSpec("x", lower=0, upper=1)])
        other_params = ParameterSet(other_space, {"x": 0.5})

        with pytest.raises(ValueError, match="different parameter space"):
            model.simulate(other_params, seed=42)

    def test_validates_completeness(self):
        """Test that ParameterSet completeness is validated."""
        # This is actually validated in ParameterSet construction,
        # but let's verify the flow works
        model = SimpleTestModel()

        # Missing parameter will fail at ParameterSet construction
        with pytest.raises(ValueError, match="Missing required parameters"):
            ParameterSet(model.space, {"alpha": 0.5, "beta": 0.3})


class TestSimulation:
    """Test simulation execution."""

    def test_basic_simulation(self):
        """Test basic simulation execution."""
        model = SimpleTestModel()
        params = ParameterSet(model.space, {
            "alpha": 0.5,
            "beta": 0.3,
            "steps": 50,
        })

        results = model.simulate(params, seed=42)

        # Check outputs returned
        assert "result" in results
        assert "summary" in results

        # Check DataFrames
        assert isinstance(results["result"], pl.DataFrame)
        assert isinstance(results["summary"], pl.DataFrame)

        # Check SEED_COL added
        assert SEED_COL in results["result"].columns
        assert SEED_COL in results["summary"].columns

    def test_scenario_simulation(self):
        """Test simulation with specific scenario."""
        model = SimpleTestModel()
        base_params = ParameterSet(model.space, {
            "alpha": 0.1,  # Will be overridden by scenario
            "beta": 0.3,
            "steps": 50,
        })

        results = model.simulate_scenario("high_alpha", base_params, seed=42)

        # Verify scenario was applied (alpha should be 0.9)
        # Result = alpha * 100 + beta * 10 + seed
        # = 0.9 * 100 + 0.3 * 10 + 42 = 90 + 3 + 42 = 135
        expected = 135
        assert results["result"]["value"][0] == expected

    def test_unknown_scenario_raises(self):
        """Test that unknown scenario raises error."""
        model = SimpleTestModel()
        params = ParameterSet(model.space, {
            "alpha": 0.5,
            "beta": 0.3,
            "steps": 50,
        })

        with pytest.raises(ValueError, match="Unknown scenario: nonexistent"):
            model.simulate_scenario("nonexistent", params, seed=42)

    def test_seed_affects_results(self):
        """Test that different seeds produce different results."""
        model = SimpleTestModel()
        params = ParameterSet(model.space, {
            "alpha": 0.5,
            "beta": 0.3,
            "steps": 50,
        })

        results1 = model.simulate(params, seed=42)
        results2 = model.simulate(params, seed=123)

        # Different seeds should produce different results
        assert results1["result"]["value"][0] != results2["result"]["value"][0]


class TestOutputExtraction:
    """Test output extraction with SEED_COL."""

    def test_seed_col_added_automatically(self):
        """Test that SEED_COL is added automatically."""
        model = SimpleTestModel()
        params = ParameterSet(model.space, {
            "alpha": 0.5,
            "beta": 0.3,
            "steps": 50,
        })

        results = model.simulate(params, seed=42)

        # Check seed column exists and has correct value
        assert results["result"][SEED_COL][0] == 42
        assert results["summary"][SEED_COL][0] == 42

    def test_extractor_cannot_add_seed_col(self):
        """Test that extractors cannot add SEED_COL themselves."""
        class BadModel(BaseModel):
            PARAMS = ParameterSpace((ParameterSpec("x", lower=0, upper=1),))

            def __init__(self):
                super().__init__()

            def build_sim(self, params, config):
                return {}

            def run_sim(self, state, seed):
                return {}

            @model_output("bad")
            def extract_bad(self, raw, seed):
                # This is bad - extractor shouldn't add SEED_COL
                return pl.DataFrame({
                    "value": [1, 2, 3],
                    SEED_COL: [seed, seed, seed],  # NOT ALLOWED
                })

        model = BadModel()
        params = ParameterSet(model.space, {"x": 0.5})

        with pytest.raises(ValueError, match="must not add.*seed.*column"):
            model.simulate(params, seed=42)

    def test_extractor_must_return_dataframe(self):
        """Test that extractors must return DataFrame."""
        class BadModel(BaseModel):
            PARAMS = ParameterSpace((ParameterSpec("x", lower=0, upper=1),))

            def __init__(self):
                super().__init__()

            def build_sim(self, params, config):
                return {}

            def run_sim(self, state, seed):
                return {}

            @model_output("bad")
            def extract_bad(self, raw, seed):
                return {"not": "a dataframe"}  # Wrong type

        model = BadModel()
        params = ParameterSet(model.space, {"x": 0.5})

        with pytest.raises(TypeError, match="must return pl.DataFrame"):
            model.simulate(params, seed=42)


class TestInheritance:
    """Test decorator discovery across inheritance."""

    def test_inherits_parent_outputs(self):
        """Test that child classes inherit parent outputs."""
        class DerivedModel(SimpleTestModel):
            @model_output("derived")
            def extract_derived(self, raw, seed):
                return pl.DataFrame({"x": [1]})

        model = DerivedModel()
        outputs = model.list_outputs()

        # Should have both parent and child outputs
        assert "result" in outputs  # From parent
        assert "summary" in outputs  # From parent
        assert "derived" in outputs  # From child

    def test_inherits_parent_scenarios(self):
        """Test that child classes inherit parent scenarios."""
        class DerivedModel(SimpleTestModel):
            @model_scenario("derived_scenario")
            def derived_scenario(self):
                return ScenarioSpec(
                    name="derived_scenario",
                    param_patches={"beta": 0.8},
                )

        model = DerivedModel()
        scenarios = model.list_scenarios()

        # Should have both parent and child scenarios
        assert "baseline" in scenarios  # Default
        assert "high_alpha" in scenarios  # From parent
        assert "derived_scenario" in scenarios  # From child


class TestHelperMethods:
    """Test helper methods."""

    def test_list_scenarios(self):
        """Test listing scenarios."""
        model = SimpleTestModel()
        scenarios = model.list_scenarios()

        assert isinstance(scenarios, list)
        assert "baseline" in scenarios
        assert "high_alpha" in scenarios
        assert scenarios == sorted(scenarios)  # Should be sorted

    def test_list_outputs(self):
        """Test listing outputs."""
        model = SimpleTestModel()
        outputs = model.list_outputs()

        assert isinstance(outputs, list)
        assert "result" in outputs
        assert "summary" in outputs
        assert outputs == sorted(outputs)  # Should be sorted

    def test_get_scenario(self):
        """Test getting scenario by name."""
        model = SimpleTestModel()

        # Get existing scenario
        spec = model.get_scenario("high_alpha")
        assert isinstance(spec, ScenarioSpec)
        assert spec.name == "high_alpha"
        assert spec.param_patches["alpha"] == 0.9

        # Get unknown scenario
        with pytest.raises(KeyError, match="not found"):
            model.get_scenario("nonexistent")

