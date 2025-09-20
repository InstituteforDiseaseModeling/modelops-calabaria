"""Integration tests for the Calabaria CLI.

End-to-end testing of CLI commands and workflows,
including realistic project scenarios.
"""

import pytest
from pathlib import Path
import tempfile
import textwrap
import json
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner
from modelops_calabaria.cli.__main__ import app


class TestCLIIntegration:
    """Integration tests for CLI workflows."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    def test_models_discover_command(self):
        """Test 'cb models discover' command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a model file
            (tmpdir / "src").mkdir()
            (tmpdir / "src" / "model.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel

                class TestModel(BaseModel):
                    @model_output
                    def result(self, raw, seed):
                        return raw['data']

                    def build_sim(self, params, config):
                        pass

                    def run_sim(self, state, seed):
                        pass
            """))

            # Run discovery
            with patch('pathlib.Path.cwd', return_value=tmpdir):
                result = self.runner.invoke(app, ["models", "discover"])

            assert result.exit_code == 0
            assert "Found 1 models:" in result.stdout
            assert "model:TestModel" in result.stdout
            assert "cb models export" in result.stdout

    def test_models_discover_verbose(self):
        """Test verbose model discovery."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "src").mkdir()
            (tmpdir / "src" / "sir.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel

                class SIRModel(BaseModel):
                    @model_output
                    def infections(self, raw, seed):
                        return raw['I']

                    @model_scenario
                    def lockdown(self):
                        return ScenarioSpec("lockdown")

                    def build_sim(self, params, config):
                        pass

                    def run_sim(self, state, seed):
                        pass
            """))

            with patch('pathlib.Path.cwd', return_value=tmpdir):
                result = self.runner.invoke(app, ["models", "discover", "--verbose"])

            assert result.exit_code == 0
            assert "Found 1 models:" in result.stdout
            assert "File: src/sir.py" in result.stdout
            assert "Outputs: infections" in result.stdout
            assert "Scenarios: lockdown" in result.stdout

    def test_models_discover_no_models(self):
        """Test discovery when no models exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create file without BaseModel
            (tmpdir / "utils.py").write_text("def utility(): pass")

            with patch('pathlib.Path.cwd', return_value=tmpdir):
                result = self.runner.invoke(app, ["models", "discover"])

            assert result.exit_code == 0
            assert "No BaseModel subclasses found" in result.stdout

    def test_models_export_command(self):
        """Test 'cb models export' command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with patch('pathlib.Path.cwd', return_value=tmpdir):
                result = self.runner.invoke(app, [
                    "models", "export",
                    "models.sir:SIRModel",
                    "--files", "src/models/sir/**",
                    "--files", "src/common/*.py"
                ])

            assert result.exit_code == 0
            assert "Added model 'models.sir:SIRModel' to pyproject.toml" in result.stdout
            assert "cb models verify" in result.stdout
            assert "cb manifest build" in result.stdout

            # Check pyproject.toml was created
            pyproject = tmpdir / "pyproject.toml"
            assert pyproject.exists()

            content = pyproject.read_text()
            assert "[tool.calabaria]" in content
            assert 'class = "models.sir:SIRModel"' in content
            assert 'class = "models.sir:SIRModel"' in content

    def test_models_export_dry_run(self):
        """Test export with --dry-run flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with patch('pathlib.Path.cwd', return_value=tmpdir):
                result = self.runner.invoke(app, [
                    "models", "export",
                    "test:Model",
                    "--files", "src/test.py",
                    "--dry-run"
                ])

            assert result.exit_code == 0
            assert "Would add to pyproject.toml:" in result.stdout
            assert "[[tool.calabaria.model]]" in result.stdout

            # Should not create actual file
            pyproject = tmpdir / "pyproject.toml"
            assert not pyproject.exists()

    def test_models_export_invalid_class_path(self):
        """Test export with invalid class path format."""
        result = self.runner.invoke(app, [
            "models", "export",
            "InvalidFormat",  # Missing colon
            "--files", "src/test.py"
        ])

        assert result.exit_code == 1
        assert "must be in format 'module:Class'" in result.stderr

    def test_models_verify_command(self):
        """Test 'cb models verify' command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create pyproject.toml with model config
            (tmpdir / "pyproject.toml").write_text(textwrap.dedent("""
                [tool.calabaria]
                schema = 1
                abi = "model-entrypoint@1"

                [[tool.calabaria.model]]
                id = "test"
                class = "test.model:TestModel"
                files = ["src/model.py"]
            """))

            # Mock verification success
            mock_results = {
                "test": {
                    "ok": True,
                    "covered": ["src/model.py"],
                    "unexpected": [],
                    "unused": []
                }
            }

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.verify_all_models', return_value=mock_results):

                result = self.runner.invoke(app, ["models", "verify"])

            assert result.exit_code == 0
            assert "All models passed verification" in result.stdout

    def test_models_verify_with_failures(self):
        """Test verification with failing models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "pyproject.toml").write_text(textwrap.dedent("""
                [tool.calabaria]
                schema = 1
                abi = "model-entrypoint@1"

                [[tool.calabaria.model]]
                id = "bad"
                class = "test.bad:BadModel"
                files = ["src/model.py"]
            """))

            # Mock verification failure
            mock_results = {
                "bad": {
                    "ok": False,
                    "covered": ["src/model.py"],
                    "unexpected": ["src/utils.py", "src/extra.py"],
                    "unused": []
                }
            }

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.verify_all_models', return_value=mock_results), \
                 patch('modelops_calabaria.cli.verify.suggest_file_patterns', return_value=["src/**"]):

                result = self.runner.invoke(app, ["models", "verify"])

            assert result.exit_code == 1
            assert "Some models failed verification" in result.stderr
            assert "To fix bad, add these file patterns:" in result.stdout
            assert '"src/**"' in result.stdout

    def test_models_verify_no_config(self):
        """Test verification with no configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create empty pyproject.toml
            (tmpdir / "pyproject.toml").write_text("[project]\nname = 'test'")

            with patch('pathlib.Path.cwd', return_value=tmpdir):
                result = self.runner.invoke(app, ["models", "verify"])

            assert result.exit_code == 1
            assert "No [tool.calabaria] configuration found" in result.stderr

    def test_manifest_build_command(self):
        """Test 'cb manifest build' command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Mock manifest building
            mock_manifest = {
                "schema": 1,
                "bundle_id": "test123",
                "models": {"test": {"class": "test:Test"}}
            }

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.build_manifest', return_value=(mock_manifest, "test123")):

                result = self.runner.invoke(app, ["manifest", "build"])

            assert result.exit_code == 0
            assert "Generated manifest.json" in result.stdout
            assert "Bundle ID: test123" in result.stdout

    def test_manifest_build_verbose(self):
        """Test verbose manifest building."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            mock_manifest = {
                "schema": 1,
                "abi": "test@1",
                "bundle_id": "test123" + "0" * 57,
                "uv_lock_sha256": "locksha256" + "0" * 54,
                "models": {"test": {
                    "class": "test:Test",
                    "files": [{"path": "test.py", "sha256": "hash"}],
                    "scenarios": ["baseline"],
                    "outputs": ["result"],
                    "model_digest": "digest" + "0" * 58
                }}
            }

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.build_manifest', return_value=(mock_manifest, "test123")), \
                 patch('modelops_calabaria.cli.__main__.print_manifest_summary') as mock_print:

                result = self.runner.invoke(app, ["manifest", "build", "--verbose"])

            assert result.exit_code == 0
            mock_print.assert_called_once_with(mock_manifest)

    def test_manifest_check_command(self):
        """Test 'cb manifest check' command."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.check_manifest_drift', return_value=True):

                result = self.runner.invoke(app, ["manifest", "check"])

            assert result.exit_code == 0
            assert "manifest.json is up to date" in result.stdout

    def test_manifest_check_outdated(self):
        """Test manifest check when outdated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.check_manifest_drift', return_value=False):

                result = self.runner.invoke(app, ["manifest", "check"])

            assert result.exit_code == 1
            assert "manifest.json is out of date" in result.stderr
            assert "cb manifest build" in result.stderr

    def test_manifest_default_behavior(self):
        """Test manifest command default behavior (should build)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            mock_manifest = {
                "schema": 1,
                "abi": "test@1",
                "bundle_id": "test123",
                "uv_lock_sha256": "locksha256" + "0" * 54,
                "models": {}
            }

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.build_manifest', return_value=(mock_manifest, "test123")):

                result = self.runner.invoke(app, ["manifest"])

            assert result.exit_code == 0
            assert "Generated manifest.json" in result.stdout

    def test_manifest_check_flag(self):
        """Test manifest command with --check flag."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.check_manifest_drift', return_value=True):

                result = self.runner.invoke(app, ["manifest", "--check"])

            assert result.exit_code == 0
            assert "manifest.json is up to date" in result.stdout

    def test_version_command(self):
        """Test 'cb version' command."""
        with patch('modelops_calabaria.cli.__version__', "1.0.0"):
            result = self.runner.invoke(app, ["version"])

        assert result.exit_code == 0
        assert "Calabaria CLI version 1.0.0" in result.stdout

    def test_help_command(self):
        """Test help functionality."""
        result = self.runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Calabaria CLI for model export and bundle management" in result.stdout
        assert "models" in result.stdout
        assert "manifest" in result.stdout


class TestCLIWorkflows:
    """Test complete CLI workflows."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    def test_full_model_workflow(self):
        """Test complete workflow: discover -> export -> verify -> manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create project structure
            (tmpdir / "src" / "models").mkdir(parents=True)
            (tmpdir / "src" / "models" / "sir.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel

                class SIRModel(BaseModel):
                    @model_output
                    def infections(self, raw, seed):
                        return raw['I']

                    def build_sim(self, params, config):
                        pass

                    def run_sim(self, state, seed):
                        pass
            """))

            with patch('pathlib.Path.cwd', return_value=tmpdir):
                # 1. Discover models
                result = self.runner.invoke(app, ["models", "discover"])
                assert result.exit_code == 0
                assert "models.sir:SIRModel" in result.stdout

                # 2. Export model
                result = self.runner.invoke(app, [
                    "models", "export",
                    "src.models.sir:SIRModel",
                    "--files", "src/models/sir/**"
                ])
                assert result.exit_code == 0

                # 3. Verify model (mock successful verification)
                mock_results = {"src.models.sir:SIRModel": {"ok": True, "covered": ["src/models/sir.py"], "unexpected": []}}
                with patch('modelops_calabaria.cli.__main__.verify_all_models', return_value=mock_results):
                    result = self.runner.invoke(app, ["models", "verify"])
                    assert result.exit_code == 0
                    assert "All models passed verification" in result.stdout

                # 4. Build manifest (mock successful build)
                mock_manifest = {"schema": 1, "bundle_id": "abc123", "models": {"src.models.sir:SIRModel": {}}}
                with patch('modelops_calabaria.cli.__main__.build_manifest', return_value=(mock_manifest, "abc123")):
                    result = self.runner.invoke(app, ["manifest", "build"])
                    assert result.exit_code == 0
                    assert "Generated manifest.json" in result.stdout

    def test_error_handling_workflow(self):
        """Test workflow with various error conditions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with patch('pathlib.Path.cwd', return_value=tmpdir):
                # Try to verify without configuration
                result = self.runner.invoke(app, ["models", "verify"])
                assert result.exit_code == 1

                # Try to build manifest without configuration
                result = self.runner.invoke(app, ["manifest", "build"])
                assert result.exit_code == 1

                # Create invalid configuration
                (tmpdir / "pyproject.toml").write_text(textwrap.dedent("""
                    [tool.calabaria]
                    # Missing required fields
                    invalid = true
                """))

                # Should handle validation errors
                with patch('modelops_calabaria.cli.__main__.validate_config', return_value=["Missing schema"]):
                    result = self.runner.invoke(app, ["models", "verify"])
                    assert result.exit_code == 1

    def test_specific_model_verification(self):
        """Test verifying specific model only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "pyproject.toml").write_text(textwrap.dedent("""
                [tool.calabaria]
                schema = 1
                abi = "model-entrypoint@1"

                [[tool.calabaria.model]]
                class = "test.model1:Model1"
                files = ["src/model1.py"]

                [[tool.calabaria.model]]
                class = "test.model2:Model2"
                files = ["src/model2.py"]
            """))

            # Mock verification for specific model
            mock_results = {"test.model1:Model1": {"ok": True, "covered": ["src/model1.py"], "unexpected": []}}

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.verify_all_models', return_value=mock_results):

                result = self.runner.invoke(app, ["models", "verify", "--model", "test.model1:Model1"])

            assert result.exit_code == 0
            assert "All models passed verification" in result.stdout


class TestCLIErrorHandling:
    """Test error handling in CLI commands."""

    def setup_method(self):
        """Set up test runner."""
        self.runner = CliRunner()

    def test_discover_with_exception(self):
        """Test discovery with unexpected exception."""
        with patch('modelops_calabaria.cli.__main__.discover_models', side_effect=Exception("Discovery failed")):
            result = self.runner.invoke(app, ["models", "discover"])

        assert result.exit_code == 1
        assert "Error discovering models: Discovery failed" in result.stderr

    def test_export_with_exception(self):
        """Test export with write error."""
        with patch('modelops_calabaria.cli.__main__.write_model_config', side_effect=Exception("Write failed")):
            result = self.runner.invoke(app, [
                "models", "export",
                "test:Model",
                "--files", "test.py"
            ])

        assert result.exit_code == 1
        assert "Error writing configuration: Write failed" in result.stderr

    def test_verify_with_exception(self):
        """Test verification with unexpected error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            (tmpdir / "pyproject.toml").write_text(textwrap.dedent("""
                [tool.calabaria]
                schema = 1
                abi = "test@1"

                [[tool.calabaria.model]]
                id = "test"
                class = "test:Model"
                files = ["test.py"]
            """))

            with patch('pathlib.Path.cwd', return_value=tmpdir), \
                 patch('modelops_calabaria.cli.__main__.verify_all_models', side_effect=Exception("Verify failed")):

                result = self.runner.invoke(app, ["models", "verify"])

            assert result.exit_code == 1
            assert "Error during verification: Verify failed" in result.stderr

    def test_manifest_with_exception(self):
        """Test manifest building with error."""
        with patch('modelops_calabaria.cli.__main__.build_manifest', side_effect=Exception("Build failed")):
            result = self.runner.invoke(app, ["manifest", "build"])

        assert result.exit_code == 1
        assert "Error building manifest: Build failed" in result.stderr

    def test_keyboard_interrupt(self):
        """Test keyboard interrupt handling."""
        with patch('modelops_calabaria.cli.__main__.app', side_effect=KeyboardInterrupt()):
            from modelops_calabaria.cli.__main__ import cli_main
            import sys

            # Capture exit
            with pytest.raises(SystemExit) as exc_info:
                cli_main()

            assert exc_info.value.code == 1