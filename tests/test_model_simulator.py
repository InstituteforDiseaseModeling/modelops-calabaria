"""Tests for ModelSimulator - the packaged simulator interface.

Tests verify that ModelSimulator correctly:
- Packages model + scenario + coords
- Executes simulation pipeline: z → params → outputs
- Validates scenario existence
- Provides correct bounds and dimensions
- Works with real models end-to-end
"""

import pytest
import numpy as np
import polars as pl

from modelops_calabaria import (
    BaseModel,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ConfigSpec,
    ConfigurationSpace,
    ConfigurationSet,
    ParameterView,
    CoordinateSystem,
    ModelSimulator,
    ScenarioSpec,
    model_output,
    model_scenario,
    LogTransform,
    Identity,
    SEED_COL,
)


class SimpleTestModel(BaseModel):
    """Simple test model for ModelSimulator tests."""

    PARAMS = ParameterSpace((
        ParameterSpec("alpha", 0.0, 1.0, "float"),
        ParameterSpec("beta", 0.01, 10.0, "float"),
        ParameterSpec("gamma", 0.0, 1.0, "float"),
    ))

    CONFIG = ConfigurationSpace((
        ConfigSpec("dt", default=0.1),
        ConfigSpec("steps", default=100),
    ))

    def __init__(self):
        super().__init__()

    def build_sim(self, params: ParameterSet, config: ConfigurationSet) -> dict:
        """Build simulation state."""
        return {
            "alpha": params["alpha"],
            "beta": params["beta"],
            "gamma": params["gamma"],
            "dt": config["dt"],
            "steps": config["steps"],
        }

    def run_sim(self, state: dict, seed: int) -> dict:
        """Run simulation."""
        # Simple deterministic result based on params and seed
        result = state["alpha"] * 100 + state["beta"] * 10 + state["gamma"] + seed
        return {
            "result": result,
            "state": state,
        }

    @model_output("result")
    def extract_result(self, raw: dict, seed: int) -> pl.DataFrame:
        """Extract result output."""
        return pl.DataFrame({
            "time": [0, 1, 2],
            "value": [raw["result"], raw["result"] * 2, raw["result"] * 3],
        })

    @model_scenario("high_beta")
    def high_beta_scenario(self) -> ScenarioSpec:
        """Scenario with high beta value."""
        return ScenarioSpec(
            name="high_beta",
            param_patches={"beta": 5.0},
        )


@pytest.fixture
def simple_model():
    """Create simple test model."""
    return SimpleTestModel()


@pytest.fixture
def coords_all_free(simple_model):
    """Create coordinate system with all params free."""
    view = ParameterView.all_free(simple_model.space)
    return CoordinateSystem(view, {})


@pytest.fixture
def coords_with_transforms(simple_model):
    """Create coordinate system with transforms."""
    view = ParameterView.from_fixed(simple_model.space, gamma=0.5)
    transforms = {
        "beta": LogTransform(),
    }
    return CoordinateSystem(view, transforms)


class TestModelSimulatorCreation:
    """Tests for creating ModelSimulator instances."""

    def test_create_simulator(self, simple_model, coords_all_free):
        """Test creating basic ModelSimulator."""
        sim = ModelSimulator(simple_model, "baseline", coords_all_free)

        assert sim.model is simple_model
        assert sim.scenario == "baseline"
        assert sim.coords is coords_all_free
        assert sim.dim == 3  # All params free

    def test_create_with_specific_scenario(self, simple_model, coords_all_free):
        """Test creating with specific scenario."""
        sim = ModelSimulator(simple_model, "high_beta", coords_all_free)

        assert sim.scenario == "high_beta"

    def test_unknown_scenario_raises(self, simple_model, coords_all_free):
        """Test that unknown scenario raises error."""
        with pytest.raises(ValueError, match="Unknown scenario 'nonexistent'"):
            ModelSimulator(simple_model, "nonexistent", coords_all_free)

    def test_simulator_is_immutable(self, simple_model, coords_all_free):
        """Test that ModelSimulator is immutable."""
        sim = ModelSimulator(simple_model, "baseline", coords_all_free)

        with pytest.raises(AttributeError):
            sim.scenario = "high_beta"


class TestSimulatorExecution:
    """Tests for simulator execution."""

    def test_call_with_identity_transforms(self, simple_model, coords_all_free):
        """Test __call__ with no transforms."""
        sim = ModelSimulator(simple_model, "baseline", coords_all_free)

        # z = [alpha=0.5, beta=2.0, gamma=0.3]
        z = np.array([0.5, 2.0, 0.3])
        outputs = sim(z, seed=42)

        # Check outputs structure
        assert isinstance(outputs, dict)
        assert "result" in outputs
        assert isinstance(outputs["result"], pl.DataFrame)

        # Check SEED_COL added
        assert SEED_COL in outputs["result"].columns
        assert outputs["result"][SEED_COL][0] == 42

        # Check result value
        # result = alpha*100 + beta*10 + gamma + seed
        # = 0.5*100 + 2.0*10 + 0.3 + 42 = 50 + 20 + 0.3 + 42 = 112.3
        expected = 112.3
        assert outputs["result"]["value"][0] == pytest.approx(expected)

    def test_call_with_log_transform(self, simple_model):
        """Test __call__ with log transform."""
        view = ParameterView.from_fixed(simple_model.space, alpha=0.5, gamma=0.3)
        coords = CoordinateSystem(view, {"beta": LogTransform()})
        sim = ModelSimulator(simple_model, "baseline", coords)

        # z = [log(beta)]
        # z[0] = 0.0 → beta = exp(0.0) = 1.0
        z = np.array([0.0])
        outputs = sim(z, seed=42)

        # result = 0.5*100 + 1.0*10 + 0.3 + 42 = 50 + 10 + 0.3 + 42 = 102.3
        expected = 102.3
        assert outputs["result"]["value"][0] == pytest.approx(expected)

    def test_call_applies_scenario(self, simple_model, coords_all_free):
        """Test that __call__ applies scenario patches."""
        sim = ModelSimulator(simple_model, "high_beta", coords_all_free)

        # z = [alpha=0.5, beta=2.0, gamma=0.3]
        # But high_beta scenario patches beta=5.0
        z = np.array([0.5, 2.0, 0.3])
        outputs = sim(z, seed=42)

        # result = 0.5*100 + 5.0*10 + 0.3 + 42 = 50 + 50 + 0.3 + 42 = 142.3
        expected = 142.3
        assert outputs["result"]["value"][0] == pytest.approx(expected)

    def test_different_seeds_produce_different_results(self, simple_model, coords_all_free):
        """Test that different seeds affect results."""
        sim = ModelSimulator(simple_model, "baseline", coords_all_free)

        z = np.array([0.5, 2.0, 0.3])
        outputs1 = sim(z, seed=42)
        outputs2 = sim(z, seed=123)

        # Results should differ by seed delta
        val1 = outputs1["result"]["value"][0]
        val2 = outputs2["result"]["value"][0]
        assert val1 != val2
        assert val2 - val1 == pytest.approx(123 - 42)


class TestBoundsAndDimensions:
    """Tests for bounds() and dimension properties."""

    def test_bounds_identity_transforms(self, simple_model, coords_all_free):
        """Test bounds with identity transforms."""
        sim = ModelSimulator(simple_model, "baseline", coords_all_free)

        bounds = sim.bounds()

        assert bounds.shape == (3, 2)
        # alpha: [0.0, 1.0]
        assert bounds[0, 0] == pytest.approx(0.0)
        assert bounds[0, 1] == pytest.approx(1.0)
        # beta: [0.01, 10.0]
        assert bounds[1, 0] == pytest.approx(0.01)
        assert bounds[1, 1] == pytest.approx(10.0)

    def test_bounds_with_log_transform(self, simple_model):
        """Test bounds with log transform."""
        view = ParameterView.from_fixed(simple_model.space, alpha=0.5, gamma=0.3)
        coords = CoordinateSystem(view, {"beta": LogTransform()})
        sim = ModelSimulator(simple_model, "baseline", coords)

        bounds = sim.bounds()

        assert bounds.shape == (1, 2)
        # beta in log space: [log(0.01), log(10.0)]
        import math
        assert bounds[0, 0] == pytest.approx(math.log(0.01))
        assert bounds[0, 1] == pytest.approx(math.log(10.0))

    def test_dim_property(self, simple_model, coords_all_free, coords_with_transforms):
        """Test dim property."""
        sim_all = ModelSimulator(simple_model, "baseline", coords_all_free)
        assert sim_all.dim == 3

        sim_partial = ModelSimulator(simple_model, "baseline", coords_with_transforms)
        assert sim_partial.dim == 2  # alpha and beta free

    def test_free_param_names(self, simple_model):
        """Test free_param_names property."""
        view = ParameterView.from_fixed(simple_model.space, gamma=0.5)
        coords = CoordinateSystem(view, {})
        sim = ModelSimulator(simple_model, "baseline", coords)

        assert sim.free_param_names == ("alpha", "beta")

    def test_param_names_match_z_dimensions(self, simple_model, coords_with_transforms):
        """Test that param_names match z vector dimensions."""
        sim = ModelSimulator(simple_model, "baseline", coords_with_transforms)

        # coords_with_transforms fixes gamma, so free = (alpha, beta)
        assert sim.free_param_names == ("alpha", "beta")
        assert sim.dim == 2
        assert len(sim.free_param_names) == sim.dim


class TestIntegrationWithRealModel:
    """Integration tests with real model."""

    def test_end_to_end_workflow(self, simple_model):
        """Test complete workflow from z to outputs."""
        # Setup: fix some params, transform others
        view = ParameterView.from_fixed(simple_model.space, gamma=0.2)
        coords = CoordinateSystem(view, {"beta": LogTransform()})
        sim = ModelSimulator(simple_model, "baseline", coords)

        # Execute multiple times
        z1 = np.array([0.3, 0.0])  # alpha=0.3, log(beta)=0 → beta=1.0
        z2 = np.array([0.7, 1.0])  # alpha=0.7, log(beta)=1 → beta=e

        outputs1 = sim(z1, seed=10)
        outputs2 = sim(z2, seed=20)

        # Both should have result outputs
        assert "result" in outputs1
        assert "result" in outputs2

        # Different inputs should produce different outputs
        val1 = outputs1["result"]["value"][0]
        val2 = outputs2["result"]["value"][0]
        assert val1 != val2

    def test_scenario_workflow(self, simple_model):
        """Test workflow with scenarios."""
        view = ParameterView.all_free(simple_model.space)
        coords = CoordinateSystem(view, {})

        # Create simulators for different scenarios
        sim_baseline = ModelSimulator(simple_model, "baseline", coords)
        sim_high_beta = ModelSimulator(simple_model, "high_beta", coords)

        z = np.array([0.5, 2.0, 0.3])

        outputs_baseline = sim_baseline(z, seed=42)
        outputs_high_beta = sim_high_beta(z, seed=42)

        # high_beta patches beta=5.0, so results should differ
        val_baseline = outputs_baseline["result"]["value"][0]
        val_high_beta = outputs_high_beta["result"]["value"][0]

        # Difference should be (5.0 - 2.0) * 10 = 30
        assert val_high_beta - val_baseline == pytest.approx(30.0)


class TestRepr:
    """Tests for __repr__ method."""

    def test_repr(self, simple_model, coords_all_free):
        """Test __repr__ output."""
        sim = ModelSimulator(simple_model, "baseline", coords_all_free)

        repr_str = repr(sim)
        assert "ModelSimulator" in repr_str
        assert "SimpleTestModel" in repr_str
        assert "baseline" in repr_str
        assert "dim=3" in repr_str
