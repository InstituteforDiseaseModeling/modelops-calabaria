"""Tests for CLI sampling commands."""

import json
import tempfile
import textwrap
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
            ParameterSpec(name="alpha", lower=0.1, upper=1.0, kind="float", doc="Alpha parameter"),
            ParameterSpec(name="beta", lower=0.5, upper=2.0, kind="float", doc="Beta parameter"),
            ParameterSpec(name="count", lower=10, upper=100, kind="int", doc="Count parameter")
        ])


class TestCLISampling:
    """Tests for sampling CLI commands."""

    @patch("modelops_calabaria.cli.sampling.load_symbol")
    def test_sobol_command_basic(self, mock_load_symbol):
        """Test basic Sobol study generation."""
        # Mock the model import
        mock_load_symbol.return_value = MockModel

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
            assert study_data["metadata"]["name"] == "test_study"
            assert study_data["metadata"].get("tags", {}) == {}

            # Check first parameter set structure
            param_set = study_data["parameter_sets"][0]
            assert "params" in param_set
            assert "alpha" in param_set["params"]
            assert "beta" in param_set["params"]
            assert "count" in param_set["params"]

    @patch("modelops_calabaria.cli.sampling.load_symbol")
    def test_grid_command_basic(self, mock_load_symbol):
        """Test basic Grid study generation."""
        # Mock the model import
        mock_load_symbol.return_value = MockModel

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "grid_study.json"

            # Run the command
            grid_command(
                model_class="models.test:TestModel",
                scenario="variant",
                grid_points=2,
                output=str(output_path),
                seed=100,
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
            assert study_data["metadata"]["name"] == "grid_study"
            assert study_data["metadata"].get("tags", {}) == {}

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
                n_replicates=1
            )

        assert exc_info.value.exit_code == 1

    @patch("modelops_calabaria.cli.sampling.load_symbol")
    def test_sobol_with_scrambling(self, mock_load_symbol):
        """Test Sobol with scrambling enabled."""
        mock_load_symbol.return_value = MockModel

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "scrambled_study.json"

            sobol_command(
                model_class="models.test:TestModel",
                scenario="baseline",
                n_samples=16,
                output=str(output_path),
                seed=123,
                scramble=True,
                n_replicates=1
            )

            assert output_path.exists()

            with open(output_path) as f:
                study_data = json.load(f)

            assert study_data["metadata"]["scramble"] == True
            assert len(study_data["parameter_sets"]) == 16

    @patch("modelops_calabaria.cli.sampling.load_symbol")
    def test_grid_with_different_points(self, mock_load_symbol):
        """Test Grid sampling with different grid points."""
        mock_load_symbol.return_value = MockModel

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "grid_3_study.json"

            grid_command(
                model_class="models.test:TestModel",
                scenario="baseline",
                grid_points=3,
                output=str(output_path),
                seed=200,
                n_replicates=1
            )

            assert output_path.exists()

            with open(output_path) as f:
                study_data = json.load(f)

            # 3 points per param, 3 params = 3^3 = 27 parameter sets
            assert len(study_data["parameter_sets"]) == 27

    @patch("modelops_calabaria.cli.sampling.load_symbol")
    def test_sobol_with_name_and_tags(self, mock_load_symbol, capsys):
        mock_load_symbol.return_value = MockModel

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_study.json"

            sobol_command(
                model_class="models.test:TestModel",
                scenario="baseline",
                n_samples=4,
                output=str(output_path),
                seed=10,
                scramble=True,
                n_replicates=2,
                name="baseline_sobol",
                tags=["phase=mlp", "priority=high"],
            )

            data = json.loads(output_path.read_text())
            assert data["metadata"]["name"] == "baseline_sobol"
            assert data["metadata"]["tags"] == {"phase": "mlp", "priority": "high"}

            captured = capsys.readouterr().out
            assert "Study Summary" in captured
            assert "phase=mlp" in captured
            assert "baseline_sobol" in captured

    @patch("modelops_calabaria.cli.sampling.load_symbol")
    def test_sobol_accepts_bundle_model_id(self, mock_load_symbol, capsys):
        """Model IDs from .modelops-bundle/registry.yaml should resolve automatically."""
        mock_load_symbol.return_value = MockModel

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_dir = root / ".modelops-bundle"
            registry_dir.mkdir()
            registry_file = registry_dir / "registry.yaml"
            registry_file.write_text(textwrap.dedent(
                """
                version: '1.0'
                models:
                  sir_starsimsir:
                    entrypoint: models.sir:StarsimSIR
                    path: models/sir.py
                    class_name: StarsimSIR
                """
            ).strip())

            output_path = root / "sobol.json"
            sobol_command(
                model_class="sir_starsimsir",
                n_samples=2,
                output=str(output_path),
                seed=1,
                scramble=False,
                n_replicates=1,
                project_root=str(root),
            )

            mock_load_symbol.assert_called_with(
                "models.sir:StarsimSIR",
                project_root=str(root),
                allow_cwd_import=True,
            )
            captured = capsys.readouterr().out
            assert "Resolved model id 'sir_starsimsir'" in captured
            data = json.loads(output_path.read_text())
            assert data["metadata"]["model_class"] == "models.sir:StarsimSIR"

    @patch("modelops_calabaria.cli.sampling.load_symbol")
    def test_bundle_model_id_not_found(self, mock_load_symbol):
        mock_load_symbol.return_value = MockModel

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            registry_dir = root / ".modelops-bundle"
            registry_dir.mkdir()
            (registry_dir / "registry.yaml").write_text(textwrap.dedent(
                """
                version: '1.0'
                models:
                  alt_model:
                    entrypoint: models.alt:AltModel
                    path: models/alt.py
                    class_name: AltModel
                """
            ).strip())

            from typer import Exit
            with pytest.raises(Exit):
                sobol_command(
                    model_class="sir_starsimsir",
                    n_samples=2,
                    output=str(root / "study.json"),
                    seed=1,
                    scramble=False,
                    n_replicates=1,
                    project_root=str(root),
                )
            assert not mock_load_symbol.called
