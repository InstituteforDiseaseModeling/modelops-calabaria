"""Tests for SimulatorBuilder - the fluent API for creating ModelSimulator instances.

Tests verify that:
- Fluent API chaining works correctly
- Parameter fixing accumulates properly
- Transform resolution works (strings and instances)
- build() creates valid ModelSimulator
- Immutability is preserved
- Error handling is clear
"""

import pytest
import numpy as np

from modelops_calabaria import (
    BaseModel,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    ConfigSpec,
    ConfigurationSpace,
    ConfigurationSet,
    SimulatorBuilder,
    ModelSimulator,
    ScenarioSpec,
    model_output,
    model_scenario,
    LogTransform,
    AffineSqueezedLogit,
    Identity,
    SEED_COL,
)
import polars as pl


class BuilderTestModel(BaseModel):
    """Test model for builder tests."""

    PARAMS = ParameterSpace((
        ParameterSpec("alpha", 0.0, 1.0, "float"),
        ParameterSpec("beta", 0.01, 10.0, "float"),
        ParameterSpec("gamma", 0.0, 1.0, "float"),
        ParameterSpec("delta", 0.1, 5.0, "float"),
    ))

    CONFIG = ConfigurationSpace((
        ConfigSpec("dt", default=0.1),
    ))

    def __init__(self):
        super().__init__()

    def build_sim(self, params: ParameterSet, config: ConfigurationSet) -> dict:
        return {"params": params.to_dict(), "config": config.to_dict()}

    def run_sim(self, state: dict, seed: int) -> dict:
        return {"result": sum(state["params"].values()) + seed}

    @model_output("result")
    def extract_result(self, raw: dict, seed: int) -> pl.DataFrame:
        return pl.DataFrame({"value": [raw["result"]]})

    @model_scenario("special")
    def special_scenario(self) -> ScenarioSpec:
        return ScenarioSpec("special", param_patches={"alpha": 0.9})


@pytest.fixture
def model():
    """Create test model."""
    return BuilderTestModel()


class TestBuilderCreation:
    """Tests for creating SimulatorBuilder."""

    def test_create_from_model_builder(self, model):
        """Test creating builder via model.builder()."""
        builder = model.builder()

        assert isinstance(builder, SimulatorBuilder)
        assert builder._model is model
        assert builder._scenario == "baseline"
        assert len(builder._fixed) == 0
        assert len(builder._transforms) == 0

    def test_create_with_scenario(self, model):
        """Test creating builder with specific scenario."""
        builder = model.builder("special")

        assert builder._scenario == "special"

    def test_builder_repr(self, model):
        """Test builder __repr__."""
        builder = model.builder()
        repr_str = repr(builder)

        assert "SimulatorBuilder" in repr_str
        assert "BuilderTestModel" in repr_str
        assert "baseline" in repr_str


class TestFluentAPIChaining:
    """Tests for fluent API chaining."""

    def test_fix_single_parameter(self, model):
        """Test fixing a single parameter."""
        builder = model.builder().fix(alpha=0.5)

        assert builder._fixed["alpha"] == 0.5
        assert len(builder._fixed) == 1

    def test_fix_multiple_parameters_at_once(self, model):
        """Test fixing multiple parameters in one call."""
        builder = model.builder().fix(alpha=0.5, beta=2.0)

        assert builder._fixed["alpha"] == 0.5
        assert builder._fixed["beta"] == 2.0
        assert len(builder._fixed) == 2

    def test_fix_multiple_calls_accumulate(self, model):
        """Test that multiple fix() calls accumulate."""
        builder = (model
                   .builder()
                   .fix(alpha=0.5)
                   .fix(beta=2.0)
                   .fix(gamma=0.3))

        assert len(builder._fixed) == 3
        assert builder._fixed["alpha"] == 0.5
        assert builder._fixed["beta"] == 2.0
        assert builder._fixed["gamma"] == 0.3

    def test_fix_overwrites_previous(self, model):
        """Test that fixing same param twice uses latest value."""
        builder = (model
                   .builder()
                   .fix(alpha=0.3)
                   .fix(alpha=0.7))  # Overwrite

        assert builder._fixed["alpha"] == 0.7

    def test_with_transforms_string_names(self, model):
        """Test with_transforms using string names."""
        builder = (model
                   .builder()
                   .with_transforms(beta="log", gamma="logit"))

        assert len(builder._transforms) == 2
        assert isinstance(builder._transforms["beta"], LogTransform)
        assert isinstance(builder._transforms["gamma"], AffineSqueezedLogit)

    def test_with_transforms_instances(self, model):
        """Test with_transforms using Transform instances."""
        log_transform = LogTransform()
        logit_transform = AffineSqueezedLogit(eps=1e-5)

        builder = (model
                   .builder()
                   .with_transforms(
                       beta=log_transform,
                       gamma=logit_transform
                   ))

        assert builder._transforms["beta"] is log_transform
        assert builder._transforms["gamma"] is logit_transform

    def test_with_transforms_mixed(self, model):
        """Test with_transforms with mix of strings and instances."""
        custom_logit = AffineSqueezedLogit(eps=1e-4)

        builder = (model
                   .builder()
                   .with_transforms(
                       beta="log",
                       gamma=custom_logit
                   ))

        assert isinstance(builder._transforms["beta"], LogTransform)
        assert builder._transforms["gamma"] is custom_logit

    def test_full_fluent_chain(self, model):
        """Test full fluent API chain."""
        builder = (model
                   .builder("special")
                   .fix(delta=1.0)
                   .fix(gamma=0.5)
                   .with_transforms(beta="log")
                   .with_transforms(alpha="identity"))

        assert builder._scenario == "special"
        assert len(builder._fixed) == 2
        assert len(builder._transforms) == 2


class TestBuilderImmutability:
    """Tests for builder immutability."""

    def test_fix_returns_new_builder(self, model):
        """Test that fix() returns a new builder."""
        builder1 = model.builder()
        builder2 = builder1.fix(alpha=0.5)

        assert builder1 is not builder2
        assert len(builder1._fixed) == 0
        assert len(builder2._fixed) == 1

    def test_with_transforms_returns_new_builder(self, model):
        """Test that with_transforms() returns a new builder."""
        builder1 = model.builder()
        builder2 = builder1.with_transforms(beta="log")

        assert builder1 is not builder2
        assert len(builder1._transforms) == 0
        assert len(builder2._transforms) == 1

    def test_chaining_preserves_previous_builders(self, model):
        """Test that chaining preserves previous builder states."""
        builder1 = model.builder()
        builder2 = builder1.fix(alpha=0.5)
        builder3 = builder2.fix(beta=2.0)

        # Each builder has its own state
        assert len(builder1._fixed) == 0
        assert len(builder2._fixed) == 1
        assert len(builder3._fixed) == 2


class TestTransformResolution:
    """Tests for transform name resolution."""

    def test_resolve_log_string(self, model):
        """Test resolving 'log' string."""
        builder = model.builder().with_transforms(beta="log")
        assert isinstance(builder._transforms["beta"], LogTransform)

    def test_resolve_logit_string(self, model):
        """Test resolving 'logit' string."""
        builder = model.builder().with_transforms(gamma="logit")
        assert isinstance(builder._transforms["gamma"], AffineSqueezedLogit)

    def test_resolve_identity_string(self, model):
        """Test resolving 'identity' string."""
        builder = model.builder().with_transforms(alpha="identity")
        assert isinstance(builder._transforms["alpha"], Identity)

    def test_case_insensitive_resolution(self, model):
        """Test that transform names are case-insensitive."""
        builder = model.builder().with_transforms(
            alpha="LOG",
            beta="Logit",
            gamma="IDENTITY"
        )
        assert isinstance(builder._transforms["alpha"], LogTransform)
        assert isinstance(builder._transforms["beta"], AffineSqueezedLogit)
        assert isinstance(builder._transforms["gamma"], Identity)

    def test_unknown_transform_name_raises(self, model):
        """Test that unknown transform name raises clear error."""
        with pytest.raises(ValueError, match="Unknown transform 'invalid'"):
            model.builder().with_transforms(beta="invalid")


class TestBuild:
    """Tests for build() method."""

    def test_build_creates_simulator(self, model):
        """Test that build() creates a ModelSimulator."""
        sim = model.builder().build()

        assert isinstance(sim, ModelSimulator)
        assert sim.model is model
        assert sim.scenario == "baseline"

    def test_build_with_fixed_params(self, model):
        """Test build() with fixed parameters."""
        sim = (model
               .builder()
               .fix(alpha=0.5, delta=1.0)
               .build())

        assert sim.dim == 2  # Only beta and gamma free
        assert sim.free_param_names == ("beta", "gamma")

    def test_build_with_transforms(self, model):
        """Test build() with transforms."""
        sim = (model
               .builder()
               .fix(alpha=0.5, delta=1.0)
               .with_transforms(beta="log")
               .build())

        # Check that transform was applied
        assert sim.dim == 2
        # Bounds should be in log space for beta
        bounds = sim.bounds()
        import math
        assert bounds[0, 0] == pytest.approx(math.log(0.01))  # log(beta.lower)

    def test_build_with_scenario(self, model):
        """Test build() with scenario."""
        sim = model.builder("special").build()

        assert sim.scenario == "special"

    def test_build_and_execute(self, model):
        """Test building and executing simulator."""
        sim = (model
               .builder()
               .fix(gamma=0.3, delta=1.0)
               .build())

        # z = [alpha, beta]
        z = np.array([0.5, 2.0])
        outputs = sim(z, seed=42)

        assert "result" in outputs
        assert SEED_COL in outputs["result"].columns

    def test_build_transform_for_fixed_param_raises(self, model):
        """Test that specifying transform for fixed param raises error."""
        builder = (model
                   .builder()
                   .fix(beta=2.0)
                   .with_transforms(beta="log"))  # beta is fixed!

        with pytest.raises(ValueError, match="non-free parameters"):
            builder.build()


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    def test_complete_workflow(self, model):
        """Test complete workflow from model to execution."""
        # Build simulator
        sim = (model
               .builder("baseline")
               .fix(delta=1.0)
               .with_transforms(beta="log")
               .build())

        # Execute
        z = np.array([0.5, 0.0, 0.3])  # [alpha, log(beta), gamma]
        outputs = sim(z, seed=10)

        # Verify outputs
        assert "result" in outputs
        assert isinstance(outputs["result"], pl.DataFrame)
        assert len(outputs["result"]) == 1

    def test_multiple_scenarios(self, model):
        """Test creating simulators for different scenarios."""
        sim_baseline = model.builder("baseline").build()
        sim_special = model.builder("special").build()

        z = np.array([0.5, 2.0, 0.3, 1.0])

        outputs_baseline = sim_baseline(z, seed=42)
        outputs_special = sim_special(z, seed=42)

        # Special scenario patches alpha=0.9, so results should differ
        val_baseline = outputs_baseline["result"]["value"][0]
        val_special = outputs_special["result"]["value"][0]

        assert val_baseline != val_special

    def test_reusable_builder_pattern(self, model):
        """Test that builders can be reused and extended."""
        # Create base builder
        base = model.builder().fix(delta=1.0)

        # Create two different simulators from same base
        sim1 = base.fix(gamma=0.2).build()
        sim2 = base.fix(alpha=0.8).build()

        # sim1 fixes delta and gamma (alpha, beta free)
        assert sim1.dim == 2
        assert set(sim1.free_param_names) == {"alpha", "beta"}

        # sim2 fixes delta and alpha (beta, gamma free)
        assert sim2.dim == 2
        assert set(sim2.free_param_names) == {"beta", "gamma"}
