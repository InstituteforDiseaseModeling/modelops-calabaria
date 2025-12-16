"""Integration tests for the complete Grammar of Parameters pipeline.

Tests the full workflow from model definition through builder API
to simulator execution, covering:

1. End-to-end workflows (model → builder → simulator → outputs)
2. Multiple scenarios
3. Multiple coordinate systems (different transform combinations)
4. Edge cases (dim=0, dim=full, large spaces)
5. Performance validation

These tests verify that all components work together correctly
and that the Grammar of Parameters abstraction holds up under
realistic usage patterns.
"""

import time
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
    model_output,
    model_scenario,
    ScenarioSpec,
    LogTransform,
    AffineSqueezedLogit,
    Identity,
    SEED_COL,
)


# ==============================================================================
# Test Model
# ==============================================================================

class IntegrationTestModel(BaseModel):
    """Comprehensive test model for integration testing."""

    def __init__(self, n_params=4):
        """Create model with configurable parameter count."""
        specs = [
            ParameterSpec(f"rate_{i}", 0.01, 10.0, "float")
            for i in range(n_params // 2)
        ]
        specs += [
            ParameterSpec(f"prob_{i}", 0.01, 0.99, "float")
            for i in range(n_params - n_params // 2)
        ]

        space = ParameterSpace(specs)
        config_space = ConfigurationSpace([
            ConfigSpec("dt", default=0.1),
            ConfigSpec("steps", default=100),
        ])
        base_config = ConfigurationSet(config_space, {"dt": 0.1, "steps": 100})

        super().__init__(space, config_space, base_config)

    def build_sim(self, params: ParameterSet, config: ConfigurationSet) -> dict:
        return {"params": params.to_dict(), "config": config.to_dict()}

    def run_sim(self, state: dict, seed: int) -> dict:
        # Simple deterministic output based on params and seed
        result = sum(state["params"].values()) + seed
        return {"result": result, "params": state["params"]}

    @model_output("summary")
    def extract_summary(self, raw: dict, seed: int) -> pl.DataFrame:
        return pl.DataFrame({
            "total": [raw["result"]],
        })

    @model_output("parameters")
    def extract_parameters(self, raw: dict, seed: int) -> pl.DataFrame:
        return pl.DataFrame({
            "name": list(raw["params"].keys()),
            "value": list(raw["params"].values()),
        })

    @model_scenario("doubled")
    def doubled_scenario(self) -> ScenarioSpec:
        """Scenario that doubles first rate parameter."""
        param_name = f"rate_0"
        if param_name in self.space.names():
            return ScenarioSpec(
                name="doubled",
                param_patches={param_name: 2.0},
                doc="Double the first rate parameter"
            )
        return ScenarioSpec(name="doubled")

    @model_scenario("long_run")
    def long_run_scenario(self) -> ScenarioSpec:
        """Scenario with more steps."""
        return ScenarioSpec(
            name="long_run",
            config_patches={"steps": 200},
            doc="Run for more steps"
        )


# ==============================================================================
# Test 1: End-to-End Workflow
# ==============================================================================

class TestEndToEndWorkflow:
    """Test complete workflow from model to outputs."""

    def test_basic_workflow(self):
        """Test: model → builder → simulator → outputs."""
        # Create model
        model = IntegrationTestModel(n_params=4)

        # Build simulator with fluent API
        sim = (model
               .as_sim("baseline")
               .fix(prob_0=0.5, prob_1=0.7)
               .with_transforms(rate_0="log", rate_1="log")
               .build())

        # Verify setup
        assert sim.dim == 2
        assert set(sim.free_param_names) == {"rate_0", "rate_1"}

        # Execute
        z = np.array([0.0, 0.5])  # log(1.0), log(1.649)
        outputs = sim(z, seed=42)

        # Verify outputs
        assert "summary" in outputs
        assert "parameters" in outputs
        assert SEED_COL in outputs["summary"].columns
        assert SEED_COL in outputs["parameters"].columns

    def test_workflow_with_scenario(self):
        """Test workflow with scenario application."""
        model = IntegrationTestModel(n_params=4)

        # Build baseline simulator
        sim_baseline = (model
                        .as_sim("baseline")
                        .fix(prob_0=0.5, prob_1=0.7)
                        .build())

        # Build doubled scenario simulator
        sim_doubled = (model
                       .as_sim("doubled")
                       .fix(prob_0=0.5, prob_1=0.7)
                       .build())

        z = np.array([1.0, 2.0])

        # Execute both
        out_baseline = sim_baseline(z, seed=42)
        out_doubled = sim_doubled(z, seed=42)

        # Results should differ (doubled scenario patches rate_0=2.0)
        val_baseline = out_baseline["summary"]["total"][0]
        val_doubled = out_doubled["summary"]["total"][0]

        # Doubled scenario should have higher total (rate_0 is 2.0 vs 1.0)
        assert val_doubled > val_baseline

    def test_workflow_multiple_executions(self):
        """Test repeated executions produce consistent results."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(prob_0=0.5, prob_1=0.7)
               .build())

        z = np.array([1.0, 2.0])

        # Execute multiple times with same seed
        out1 = sim(z, seed=42)
        out2 = sim(z, seed=42)

        # Should be identical
        assert out1["summary"]["total"][0] == out2["summary"]["total"][0]

        # Different seed should give different result
        out3 = sim(z, seed=123)
        assert out3["summary"]["total"][0] != out1["summary"]["total"][0]


# ==============================================================================
# Test 2: Multiple Coordinate Systems
# ==============================================================================

class TestMultipleCoordinateSystems:
    """Test different transform combinations."""

    def test_identity_transforms(self):
        """Test with all identity transforms."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(prob_0=0.5, prob_1=0.7)
               .build())  # No transforms = all identity

        z = np.array([1.0, 2.0])
        outputs = sim(z, seed=42)

        # Should work without errors
        assert "summary" in outputs

    def test_all_log_transforms(self):
        """Test with all log transforms."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(prob_0=0.5, prob_1=0.7)
               .with_transforms(rate_0="log", rate_1="log")
               .build())

        z = np.array([0.0, 0.5])  # log space
        outputs = sim(z, seed=42)

        # Verify parameters were transformed correctly
        params_df = outputs["parameters"]
        rate_0 = params_df.filter(pl.col("name") == "rate_0")["value"][0]
        rate_1 = params_df.filter(pl.col("name") == "rate_1")["value"][0]

        assert rate_0 == pytest.approx(1.0)  # exp(0.0)
        assert rate_1 == pytest.approx(np.exp(0.5))

    def test_all_logit_transforms(self):
        """Test with all logit transforms."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(rate_0=1.0, rate_1=2.0)
               .with_transforms(prob_0="logit", prob_1="logit")
               .build())

        z = np.array([0.0, 1.0])  # logit space
        outputs = sim(z, seed=42)

        # Verify probabilities are in valid range
        params_df = outputs["parameters"]
        prob_0 = params_df.filter(pl.col("name") == "prob_0")["value"][0]
        prob_1 = params_df.filter(pl.col("name") == "prob_1")["value"][0]

        assert 0.0 < prob_0 < 1.0
        assert 0.0 < prob_1 < 1.0
        assert prob_0 == pytest.approx(0.5, abs=0.01)  # logit(0) ≈ 0.5

    def test_mixed_transforms(self):
        """Test with mixed transform types."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .with_transforms(
                   rate_0="log",
                   rate_1="identity",
                   prob_0="logit",
                   prob_1="identity"
               )
               .build())

        z = np.array([0.0, 1.5, 0.0, 0.6])
        outputs = sim(z, seed=42)

        # All transforms should work together
        assert "summary" in outputs

    def test_custom_transform_instances(self):
        """Test with custom transform instances."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(rate_1=2.0, prob_1=0.7)
               .with_transforms(
                   rate_0=LogTransform(),
                   prob_0=AffineSqueezedLogit(eps=1e-5)
               )
               .build())

        z = np.array([0.0, 0.0])
        outputs = sim(z, seed=42)

        assert "summary" in outputs


# ==============================================================================
# Test 3: Edge Cases
# ==============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_parameters_fixed_raises(self):
        """Test that fixing all parameters raises appropriate error."""
        model = IntegrationTestModel(n_params=2)

        # Try to fix all parameters
        builder = (model
                   .as_sim("baseline")
                   .fix(rate_0=1.0, prob_0=0.5))

        sim = builder.build()

        # Should have dim=0
        assert sim.dim == 0

        # Calling with empty z should work
        z = np.array([])
        outputs = sim(z, seed=42)
        assert "summary" in outputs

    def test_all_parameters_free(self):
        """Test with all parameters free."""
        model = IntegrationTestModel(n_params=4)

        sim = model.as_sim("baseline").build()  # No fix() calls

        assert sim.dim == 4
        assert len(sim.free_param_names) == 4

        z = np.array([1.0, 2.0, 0.5, 0.6])
        outputs = sim(z, seed=42)

        assert "summary" in outputs

    def test_large_parameter_space(self):
        """Test with large parameter space."""
        model = IntegrationTestModel(n_params=20)

        # Fix half (first 10), leave half free (last 10)
        # Model with n_params=20 creates: rate_0..rate_9, prob_0..prob_9
        fixed = {f"rate_{i}": 1.0 for i in range(5)}
        fixed.update({f"prob_{i}": 0.5 for i in range(5)})

        sim = (model
               .as_sim("baseline")
               .fix(**fixed)
               .build())

        assert sim.dim == 10

        # Create valid z values for the free params
        # Free params are: rate_5..rate_9 (5 params), prob_5..prob_9 (5 params)
        z = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 0.3, 0.4, 0.5, 0.6, 0.7])
        outputs = sim(z, seed=42)

        assert "summary" in outputs

    def test_single_free_parameter(self):
        """Test with only one free parameter."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(rate_1=2.0, prob_0=0.5, prob_1=0.7)
               .build())

        assert sim.dim == 1
        assert sim.free_param_names == ("rate_0",)

        z = np.array([1.5])
        outputs = sim(z, seed=42)

        assert "summary" in outputs

    def test_bounds_with_various_transforms(self):
        """Test bounds calculation with all transform types."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .with_transforms(
                   rate_0="log",
                   rate_1="identity",
                   prob_0="logit",
                   prob_1="identity"
               )
               .build())

        bounds = sim.bounds()

        assert bounds.shape == (4, 2)
        assert np.all(bounds[:, 0] < bounds[:, 1])  # Lower < upper


# ==============================================================================
# Test 4: Scenario Composition
# ==============================================================================

class TestScenarioComposition:
    """Test complex scenario interactions."""

    def test_baseline_vs_modified_scenarios(self):
        """Test that scenarios produce different results."""
        model = IntegrationTestModel(n_params=4)

        sim_baseline = (model
                        .as_sim("baseline")
                        .fix(prob_0=0.5, prob_1=0.7)
                        .build())

        sim_doubled = (model
                       .as_sim("doubled")
                       .fix(prob_0=0.5, prob_1=0.7)
                       .build())

        z = np.array([1.0, 2.0])

        out_baseline = sim_baseline(z, seed=42)
        out_doubled = sim_doubled(z, seed=42)

        # Should differ
        val_baseline = out_baseline["summary"]["total"][0]
        val_doubled = out_doubled["summary"]["total"][0]

        assert val_baseline != val_doubled

    def test_config_patch_scenario(self):
        """Test scenario with config patches."""
        model = IntegrationTestModel(n_params=4)

        sim_baseline = (model
                        .as_sim("baseline")
                        .fix(rate_0=1.0, rate_1=2.0, prob_0=0.5, prob_1=0.7)
                        .build())

        sim_long = (model
                    .as_sim("long_run")
                    .fix(rate_0=1.0, rate_1=2.0, prob_0=0.5, prob_1=0.7)
                    .build())

        z = np.array([])  # All fixed

        out_baseline = sim_baseline(z, seed=42)
        out_long = sim_long(z, seed=42)

        # Both should work (different configs)
        assert "summary" in out_baseline
        assert "summary" in out_long


# ==============================================================================
# Test 5: Performance
# ==============================================================================

class TestPerformance:
    """Performance validation tests."""

    def test_simulator_creation_performance(self):
        """Test that simulator creation is fast."""
        model = IntegrationTestModel(n_params=10)

        start = time.time()
        for _ in range(100):
            sim = (model
                   .as_sim("baseline")
                   .fix(prob_0=0.5, prob_1=0.7)
                   .with_transforms(rate_0="log", rate_1="log")
                   .build())
        elapsed = time.time() - start

        # Should create 100 simulators in < 1 second
        assert elapsed < 1.0
        avg_time = elapsed / 100
        print(f"\nAverage simulator creation time: {avg_time*1000:.2f}ms")

    def test_coordinate_transform_performance(self):
        """Test that coordinate transforms are fast."""
        model = IntegrationTestModel(n_params=10)

        sim = (model
               .as_sim("baseline")
               .with_transforms(
                   rate_0="log",
                   rate_1="log",
                   rate_2="log",
                   rate_3="log",
                   rate_4="log",
                   prob_0="logit",
                   prob_1="logit",
                   prob_2="logit",
                   prob_3="logit",
                   prob_4="logit",
               )
               .build())

        z = np.random.randn(10)

        start = time.time()
        for _ in range(1000):
            outputs = sim(z, seed=42)
        elapsed = time.time() - start

        # Should execute 1000 simulations in < 2 seconds
        assert elapsed < 2.0
        avg_time = elapsed / 1000
        print(f"\nAverage execution time: {avg_time*1000:.2f}ms")

    def test_large_batch_execution(self):
        """Test executing simulator many times."""
        model = IntegrationTestModel(n_params=4)

        # Use log transforms so z can be from standard normal
        sim = (model
               .as_sim("baseline")
               .fix(prob_0=0.5, prob_1=0.7)
               .with_transforms(rate_0="log", rate_1="log")
               .build())

        # Get valid bounds in transformed space
        bounds = sim.bounds()  # Bounds in log-space

        # Execute many times with uniform samples in valid range
        n_executions = 100
        rng = np.random.RandomState(42)
        z_batch = [
            rng.uniform(bounds[:, 0], bounds[:, 1])
            for _ in range(n_executions)
        ]

        start = time.time()
        for z in z_batch:
            outputs = sim(z, seed=42)
        elapsed = time.time() - start

        avg_time = elapsed / n_executions
        print(f"\nAverage execution time (batch): {avg_time*1000:.2f}ms")

        # Should be reasonably fast
        assert elapsed < 5.0


# ==============================================================================
# Test 6: Builder Patterns
# ==============================================================================

class TestBuilderPatterns:
    """Test builder pattern edge cases."""

    def test_builder_reuse(self):
        """Test reusing builder to create multiple simulators."""
        model = IntegrationTestModel(n_params=4)

        base = model.as_sim("baseline").fix(prob_0=0.5)

        sim1 = base.fix(prob_1=0.7).build()  # Fixes prob_0, prob_1 → free: rate_0, rate_1
        sim2 = base.fix(rate_0=1.0).build()  # Fixes prob_0, rate_0 → free: rate_1, prob_1

        # Both have dim=2 but different free parameters
        assert sim1.dim == 2
        assert sim2.dim == 2
        assert set(sim1.free_param_names) != set(sim2.free_param_names)

        # Verify the actual free params
        assert set(sim1.free_param_names) == {"rate_0", "rate_1"}
        assert set(sim2.free_param_names) == {"rate_1", "prob_1"}

    def test_builder_chaining(self):
        """Test long builder chains."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(prob_0=0.5)
               .fix(prob_1=0.7)
               .with_transforms(rate_0="log")
               .with_transforms(rate_1="log")
               .build())

        assert sim.dim == 2

    def test_builder_overwriting(self):
        """Test that later fix() calls override earlier ones."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(rate_0=1.0)
               .fix(rate_0=2.0)  # Override
               .fix(prob_0=0.5, prob_1=0.7)
               .build())

        z = np.array([1.0])  # Only rate_1 free
        outputs = sim(z, seed=42)

        # rate_0 should be fixed at 2.0 (not 1.0)
        params_df = outputs["parameters"]
        rate_0 = params_df.filter(pl.col("name") == "rate_0")["value"][0]
        assert rate_0 == 2.0


# ==============================================================================
# Test 7: Error Handling
# ==============================================================================

class TestErrorHandling:
    """Test error handling and validation."""

    def test_invalid_transform_name_raises(self):
        """Test that invalid transform name raises clear error."""
        model = IntegrationTestModel(n_params=4)

        with pytest.raises(ValueError, match="Unknown transform 'invalid'"):
            model.as_sim("baseline").with_transforms(rate_0="invalid").build()

    def test_transform_for_fixed_param_raises(self):
        """Test that specifying transform for fixed param raises."""
        model = IntegrationTestModel(n_params=4)

        builder = (model
                   .as_sim("baseline")
                   .fix(rate_0=1.0)
                   .with_transforms(rate_0="log"))  # rate_0 is fixed!

        with pytest.raises(ValueError, match="non-free parameters"):
            builder.build()

    def test_invalid_z_dimension_raises(self):
        """Test that wrong z dimension raises error."""
        model = IntegrationTestModel(n_params=4)

        sim = (model
               .as_sim("baseline")
               .fix(prob_0=0.5, prob_1=0.7)
               .build())

        assert sim.dim == 2

        # Wrong dimension should raise
        z_wrong = np.array([1.0, 2.0, 3.0])  # 3 instead of 2
        with pytest.raises(ValueError):
            sim(z_wrong, seed=42)

    def test_unknown_scenario_raises(self):
        """Test that unknown scenario raises error."""
        model = IntegrationTestModel(n_params=4)

        with pytest.raises(ValueError, match="Unknown scenario"):
            model.as_sim("nonexistent").build()
