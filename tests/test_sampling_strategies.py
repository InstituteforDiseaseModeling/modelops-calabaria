"""Tests for parameter sampling strategies."""

import pytest
import numpy as np
from modelops_calabaria.parameters import ParameterSpace, ParameterSpec
from modelops_calabaria.sampling.grid import GridSampler
from modelops_calabaria.sampling.sobol import SobolSampler


class TestGridSampler:
    """Tests for Grid sampling strategy."""

    def test_grid_basic(self):
        """Test basic grid generation."""
        specs = [
            ParameterSpec(name="x", kind="float", min=0.0, max=1.0),
            ParameterSpec(name="y", kind="float", min=0.0, max=1.0),
        ]
        space = ParameterSpace(specs)
        sampler = GridSampler(space, n_points_per_param=3)

        samples = sampler.sample(None)

        # 3 points per dimension = 3^2 = 9 total
        assert len(samples) == 9

        # Check all samples have required parameters
        for sample in samples:
            assert "x" in sample
            assert "y" in sample
            assert 0.0 <= sample["x"] <= 1.0
            assert 0.0 <= sample["y"] <= 1.0

    def test_grid_integer_params(self):
        """Test grid with integer parameters."""
        specs = [
            ParameterSpec(name="n", kind="int", min=1, max=5),
            ParameterSpec(name="m", kind="int", min=10, max=20),
        ]
        space = ParameterSpace(specs)
        sampler = GridSampler(space, n_points_per_param=3)

        samples = sampler.sample(None)

        # Check all samples have integer values
        for sample in samples:
            assert isinstance(sample["n"], int)
            assert isinstance(sample["m"], int)
            assert 1 <= sample["n"] <= 5
            assert 10 <= sample["m"] <= 20

    def test_grid_mixed_types(self):
        """Test grid with mixed parameter types."""
        specs = [
            ParameterSpec(name="rate", kind="float", min=0.1, max=1.0),
            ParameterSpec(name="count", kind="int", min=1, max=10),
        ]
        space = ParameterSpace(specs)
        sampler = GridSampler(space, n_points_per_param=2)

        samples = sampler.sample(None)
        assert len(samples) == 4  # 2x2 grid

        for sample in samples:
            assert isinstance(sample["rate"], float)
            assert isinstance(sample["count"], int)

    def test_grid_method_name(self):
        """Test method name is correct."""
        specs = [ParameterSpec(name="x", kind="float", min=0.0, max=1.0)]
        space = ParameterSpace(specs)
        sampler = GridSampler(space)
        assert sampler.method_name() == "grid"


class TestSobolSampler:
    """Tests for Sobol sampling strategy."""

    def test_sobol_basic(self):
        """Test basic Sobol sequence generation."""
        specs = [
            ParameterSpec(name="x", kind="float", min=0.0, max=1.0),
            ParameterSpec(name="y", kind="float", min=0.0, max=1.0),
        ]
        space = ParameterSpace(specs)
        sampler = SobolSampler(space, scramble=False, seed=42)

        samples = sampler.sample(16)  # Power of 2

        assert len(samples) == 16

        # Check all samples have required parameters
        for sample in samples:
            assert "x" in sample
            assert "y" in sample
            assert 0.0 <= sample["x"] <= 1.0
            assert 0.0 <= sample["y"] <= 1.0

    def test_sobol_deterministic(self):
        """Test Sobol sequence is deterministic with same seed."""
        specs = [
            ParameterSpec(name="x", kind="float", min=0.0, max=1.0),
            ParameterSpec(name="y", kind="float", min=0.0, max=1.0),
        ]
        space = ParameterSpace(specs)

        sampler1 = SobolSampler(space, scramble=True, seed=42)
        samples1 = sampler1.sample(8)

        sampler2 = SobolSampler(space, scramble=True, seed=42)
        samples2 = sampler2.sample(8)

        # Should produce identical results
        for s1, s2 in zip(samples1, samples2):
            assert s1["x"] == s2["x"]
            assert s1["y"] == s2["y"]

    def test_sobol_scrambling_different(self):
        """Test scrambled Sobol sequences are different with different seeds."""
        specs = [
            ParameterSpec(name="x", kind="float", min=0.0, max=1.0),
        ]
        space = ParameterSpace(specs)

        sampler1 = SobolSampler(space, scramble=True, seed=42)
        samples1 = sampler1.sample(8)

        sampler2 = SobolSampler(space, scramble=True, seed=43)
        samples2 = sampler2.sample(8)

        # Should produce different results (with high probability)
        different = False
        for s1, s2 in zip(samples1, samples2):
            if s1["x"] != s2["x"]:
                different = True
                break
        assert different

    def test_sobol_integer_params(self):
        """Test Sobol with integer parameters."""
        specs = [
            ParameterSpec(name="n", kind="int", min=1, max=100),
            ParameterSpec(name="m", kind="int", min=1000, max=2000),
        ]
        space = ParameterSpace(specs)
        sampler = SobolSampler(space, scramble=False, seed=42)

        samples = sampler.sample(16)

        for sample in samples:
            assert isinstance(sample["n"], int)
            assert isinstance(sample["m"], int)
            assert 1 <= sample["n"] <= 100
            assert 1000 <= sample["m"] <= 2000

    def test_sobol_method_name(self):
        """Test method name is correct."""
        specs = [ParameterSpec(name="x", kind="float", min=0.0, max=1.0)]
        space = ParameterSpace(specs)
        sampler = SobolSampler(space)
        assert sampler.method_name() == "sobol"

    def test_sobol_requires_scipy(self):
        """Test that Sobol requires scipy."""
        # This test will only work if scipy is installed
        # We're testing that the import works, not that it fails
        specs = [ParameterSpec(name="x", kind="float", min=0.0, max=1.0)]
        space = ParameterSpace(specs)

        # Should not raise ImportError since we have scipy installed
        sampler = SobolSampler(space)
        samples = sampler.sample(4)
        assert len(samples) == 4

    def test_sobol_convergence_estimate(self):
        """Test convergence estimation."""
        specs = [
            ParameterSpec(name="x", kind="float", min=0.0, max=1.0),
            ParameterSpec(name="y", kind="float", min=0.0, max=1.0),
        ]
        space = ParameterSpace(specs)
        sampler = SobolSampler(space)

        # Convergence should improve with more samples
        conv_10 = sampler.estimate_convergence(10)
        conv_100 = sampler.estimate_convergence(100)
        conv_1000 = sampler.estimate_convergence(1000)

        assert conv_10 > conv_100 > conv_1000
        assert conv_10 <= 1.0  # Should be bounded


class TestSamplingIntegration:
    """Integration tests for sampling strategies."""

    def test_numpy_type_conversion(self):
        """Test that numpy types are properly handled."""
        specs = [
            ParameterSpec(name="x", kind="float", min=0.0, max=1.0),
            ParameterSpec(name="n", kind="int", min=1, max=10),
        ]
        space = ParameterSpace(specs)
        sampler = SobolSampler(space)

        samples = sampler.sample(4)

        # The sampler might produce numpy types internally
        # but they should be convertible to Python types
        for sample in samples:
            # These should not raise errors
            float(sample["x"])
            int(sample["n"])

            # Should be JSON serializable
            import json
            # Convert numpy types if present
            clean_sample = {}
            for k, v in sample.items():
                if isinstance(v, (np.integer, np.floating)):
                    clean_sample[k] = v.item()
                else:
                    clean_sample[k] = v
            json.dumps(clean_sample)  # Should not raise

    def test_generate_tasks(self):
        """Test SimBatch generation from sampler."""
        specs = [
            ParameterSpec(name="beta", kind="float", min=0.1, max=2.0),
            ParameterSpec(name="gamma", kind="float", min=0.05, max=0.5),
        ]
        space = ParameterSpace(specs)
        sampler = GridSampler(space, n_points_per_param=2)

        tasks = sampler.generate_tasks(
            model_class="models.test.Model",
            scenario="baseline",
            bundle_ref="sha256:" + "a" * 64,
            n_samples=4,  # Will be ignored for grid, uses grid size
            base_seed=42
        )

        assert len(tasks) == 4  # 2x2 grid

        # Check tasks are properly formed
        for i, task in enumerate(tasks):
            assert task.entrypoint == "models.test.Model/baseline"
            assert task.bundle_ref == "sha256:" + "a" * 64
            assert task.seed == 42 + i
            assert "beta" in task.params.params
            assert "gamma" in task.params.params