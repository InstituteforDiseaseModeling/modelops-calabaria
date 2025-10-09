"""Tests for CLI sampling commands."""

import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from modelops_calabaria.cli.sampling import sobol_command, grid_command
from modelops_calabaria.parameters import ParameterSpace, ParameterSpec


class MockModel:
    """Mock model for testing."""

    @classmethod
    def parameter_space(cls):
        return ParameterSpace(specs=[
            ParameterSpec(name="alpha", min=0.1, max=1.0, kind="float", doc="Alpha parameter"),
            ParameterSpec(name="beta", min=0.5, max=2.0, kind="float", doc="Beta parameter"),
            ParameterSpec(name="count", min=10, max=100, kind="int", doc="Count parameter")
        ])


class TestCLISampling:
    """Tests for sampling CLI commands."""

    @patch("importlib.import_module")
    def test_sobol_command_basic(self, mock_import):
        """Test basic Sobol study generation."""
        # Mock the model import
        mock_module = MagicMock()
        mock_module.TestModel = MockModel
        mock_import.return_value = mock_module

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_study.json"

            # Run the command
            sobol_command(
                model_class="models.test:TestModel",
                scenario="baseline",
                n_samples=8,
                output=str(output_path),
                seed=42,
                scramble=False,
                targets=None,
                n_replicates=1
            )

            # Check output was generated
            assert output_path.exists()

            with open(output_path) as f:
                study_data = json.load(f)

            assert study_data["model"] == "models.test"
            assert study_data["scenario"] == "baseline"
            assert study_data["sampling_method"] == "sobol"
            assert len(study_data["parameter_sets"]) == 8
            assert study_data["metadata"]["model_class"] == "models.test:TestModel"

            # Check first parameter set structure
            param_set = study_data["parameter_sets"][0]
            assert "params" in param_set
            assert "alpha" in param_set["params"]
            assert "beta" in param_set["params"]
            assert "count" in param_set["params"]

    @patch("importlib.import_module")
    def test_grid_command_basic(self, mock_import):
        """Test basic Grid study generation."""
        # Mock the model import
        mock_module = MagicMock()
        mock_module.TestModel = MockModel
        mock_import.return_value = mock_module

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "grid_study.json"

            # Run the command
            grid_command(
                model_class="models.test:TestModel",
                scenario="variant",
                grid_points=2,
                output=str(output_path),
                seed=100,
                targets=None,
                n_replicates=1
            )

            # Check output
            assert output_path.exists()

            with open(output_path) as f:
                study_data = json.load(f)

            assert study_data["model"] == "models.test"
            assert study_data["scenario"] == "variant"
            assert study_data["sampling_method"] == "grid"
            # 2 points per param, 3 params = 2^3 = 8 parameter sets
            assert len(study_data["parameter_sets"]) == 8

    @patch("importlib.import_module")
    def test_invalid_model_error(self, mock_import):
        """Test error when model cannot be imported."""
        mock_import.side_effect = ImportError("Module not found")

        from typer import Exit
        with pytest.raises(Exit) as exc_info:
            sobol_command(
                model_class="models.nonexistent:Model",
                scenario="baseline",
                n_samples=8,
                output="study.json",
                seed=42,
                scramble=True,
                targets=None,
                n_replicates=1
            )

        assert exc_info.value.exit_code == 1

    @patch("importlib.import_module")
    def test_model_without_parameter_space(self, mock_import):
        """Test error when model doesn't have parameter_space method."""
        # Mock a model without parameter_space
        class BadModel:
            pass

        mock_module = MagicMock()
        mock_module.BadModel = BadModel
        mock_import.return_value = mock_module

        from typer import Exit
        with pytest.raises(Exit) as exc_info:
            sobol_command(
                model_class="models.test:BadModel",
                scenario="baseline",
                n_samples=8,
                output="study.json",
                seed=42,
                scramble=True,
                targets=None,
                n_replicates=1
            )

        assert exc_info.value.exit_code == 1

    @patch("importlib.import_module")
    def test_sobol_with_scrambling(self, mock_import):
        """Test Sobol with scrambling enabled."""
        mock_module = MagicMock()
        mock_module.TestModel = MockModel
        mock_import.return_value = mock_module

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scrambled_study.json"

            sobol_command(
                model_class="models.test:TestModel",
                scenario="baseline",
                n_samples=16,
                output=str(output_path),
                seed=123,
                scramble=True,
                targets=None,
                n_replicates=1
            )

            assert output_path.exists()

            with open(output_path) as f:
                study_data = json.load(f)

            assert study_data["metadata"]["scramble"] == True
            assert len(study_data["parameter_sets"]) == 16

    @patch("importlib.import_module")
    def test_grid_with_different_points(self, mock_import):
        """Test Grid sampling with different grid points."""
        mock_module = MagicMock()
        mock_module.TestModel = MockModel
        mock_import.return_value = mock_module

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "grid_3_study.json"

            grid_command(
                model_class="models.test:TestModel",
                scenario="baseline",
                grid_points=3,
                output=str(output_path),
                seed=200,
                targets=None,
                n_replicates=1
            )

            assert output_path.exists()

            with open(output_path) as f:
                study_data = json.load(f)

            # 3 points per param, 3 params = 3^3 = 27 parameter sets
            assert len(study_data["parameter_sets"]) == 27