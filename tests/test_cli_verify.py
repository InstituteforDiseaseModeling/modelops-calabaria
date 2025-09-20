"""Tests for CLI import boundary verification.

Tests subprocess-based import verification that ensures models
only import from their declared file dependencies.
"""

import pytest
from pathlib import Path
import tempfile
import textwrap
import sys
from unittest.mock import patch, MagicMock

from modelops_calabaria.cli.verify import (
    verify_model,
    verify_all_models,
    print_verification_summary,
    suggest_file_patterns,
)


class TestVerifyModel:
    """Tests for single model verification."""

    def test_verify_model_success(self):
        """Should verify model that imports only allowed files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a simple model file
            (tmpdir / "src").mkdir()
            model_file = tmpdir / "src" / "model.py"
            model_file.write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel

                class TestModel(BaseModel):
                    def build_sim(self, params, config):
                        return None

                    def run_sim(self, state, seed):
                        return {"result": 42}
            """))

            # Mock successful subprocess result
            mock_output = {
                "success": True,
                "loaded_files": [str(model_file.resolve())],
                "model_info": {"module": "src.model", "class": "TestModel", "has_space": True}
            }

            import json
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(mock_output),
                    stderr=""
                )

                # Change to temp directory
                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    result = verify_model("src.model:TestModel", {"src/model.py"})
                finally:
                    os.chdir(old_cwd)

                assert result["ok"] is True
                assert result["loaded"] == ["src/model.py"]
                assert result["unexpected"] == []
                assert result["covered"] == ["src/model.py"]

    def test_verify_model_unexpected_imports(self):
        """Should detect unexpected imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Mock output showing extra imports
            mock_output = {
                "success": True,
                "loaded_files": [
                    str((tmpdir / "src" / "model.py").resolve()),
                    str((tmpdir / "src" / "utils.py").resolve()),  # Not allowed
                ],
                "model_info": {"module": "src.model", "class": "TestModel", "has_space": True}
            }

            import json
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(mock_output),
                    stderr=""
                )

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    result = verify_model("src.model:TestModel", {"src/model.py"})
                finally:
                    os.chdir(old_cwd)

                assert result["ok"] is False
                assert "src/utils.py" in result["unexpected"]
                assert "src/model.py" in result["covered"]

    def test_verify_model_import_error(self):
        """Should handle import errors gracefully."""
        mock_error = {
            "success": False,
            "error": "ModuleNotFoundError: No module named 'nonexistent'",
            "traceback": "Traceback..."
        }

        import json
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1,
                stdout=json.dumps(mock_error),
                stderr=""
            )

            result = verify_model("nonexistent:Model", {"file.py"})

            assert result["ok"] is False
            assert "ModuleNotFoundError" in result["error"]
            assert result["unexpected"] == []

    def test_verify_model_timeout(self):
        """Should handle subprocess timeout."""
        with patch('subprocess.run') as mock_run:
            from subprocess import TimeoutExpired
            mock_run.side_effect = TimeoutExpired("python", 30)

            result = verify_model("test:Model", {"file.py"})

            assert result["ok"] is False
            assert "Timeout" in result["error"]

    def test_verify_model_malformed_output(self):
        """Should handle malformed JSON output."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="not valid json",
                stderr=""
            )

            result = verify_model("test:Model", {"file.py"})

            assert result["ok"] is False
            assert "Invalid JSON" in result["error"]

    def test_filters_external_dependencies(self):
        """Should filter out external dependencies."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Mock output with mix of project and external files
            mock_output = {
                "success": True,
                "loaded_files": [
                    str((tmpdir / "src" / "model.py").resolve()),  # Project file
                    "/usr/lib/python3.11/json/__init__.py",  # External
                    "/opt/homebrew/lib/python3.11/site-packages/polars/__init__.py",  # External
                ],
                "model_info": {"module": "src.model", "class": "TestModel", "has_space": True}
            }

            import json
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(mock_output),
                    stderr=""
                )

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    result = verify_model("src.model:TestModel", {"src/model.py"})
                finally:
                    os.chdir(old_cwd)

                # Should only track project files
                assert result["loaded"] == ["src/model.py"]
                assert len(result["external_deps"]) >= 2
                external_files = " ".join(result["external_deps"])
                assert "json" in external_files
                assert "polars" in external_files


class TestVerifyAllModels:
    """Tests for verifying multiple models."""

    def test_verify_all_models_success(self):
        """Should verify all models in configuration."""
        config = {
            "model": [
                {
                    "class": "src.model1:Model1",
                    "files": ["src/model1.py"]
                },
                {
                    "class": "src.model2:Model2",
                    "files": ["src/model2.py"]
                }
            ]
        }

        # Mock resolve_file_patterns
        def mock_resolve(patterns):
            if "model1.py" in patterns[0]:
                return [Path("src/model1.py")]
            else:
                return [Path("src/model2.py")]

        # Mock successful verification results
        def mock_verify(class_path, allowed_files):
            if "model1" in class_path:
                return {"ok": True, "covered": ["src/model1.py"], "unexpected": []}
            else:
                return {"ok": True, "covered": ["src/model2.py"], "unexpected": []}

        with patch('modelops_calabaria.cli.verify.resolve_file_patterns', side_effect=mock_resolve), \
             patch('modelops_calabaria.cli.verify.verify_model', side_effect=mock_verify):

            results = verify_all_models(config)

            assert len(results) == 2
            assert results["src.model1:Model1"]["ok"] is True
            assert results["src.model2:Model2"]["ok"] is True

    def test_verify_all_models_mixed_results(self):
        """Should handle mixed success/failure results."""
        config = {
            "model": [
                {
                    "class": "src.good:GoodModel",
                    "files": ["src/good.py"]
                },
                {
                    "class": "src.bad:BadModel",
                    "files": ["src/bad.py"]
                }
            ]
        }

        def mock_resolve(patterns):
            return [Path(patterns[0])]

        def mock_verify(class_path, allowed_files):
            if "good" in class_path:
                return {"ok": True, "covered": ["src/good.py"], "unexpected": []}
            else:
                return {"ok": False, "covered": ["src/bad.py"], "unexpected": ["src/utils.py"]}

        with patch('modelops_calabaria.cli.verify.resolve_file_patterns', side_effect=mock_resolve), \
             patch('modelops_calabaria.cli.verify.verify_model', side_effect=mock_verify):

            results = verify_all_models(config)

            assert results["src.good:GoodModel"]["ok"] is True
            assert results["src.bad:BadModel"]["ok"] is False
            assert "src/utils.py" in results["src.bad:BadModel"]["unexpected"]


class TestPrintVerificationSummary:
    """Tests for verification summary printing."""

    def test_print_summary_all_passed(self, capsys):
        """Should print summary for all passed models."""
        results = {
            "src.model1:Model1": {"ok": True, "covered": ["file1.py"], "unused": []},
            "src.model2:Model2": {"ok": True, "covered": ["file2.py"], "unused": []}
        }

        print_verification_summary(results)
        captured = capsys.readouterr()

        assert "Total models: 2" in captured.out
        assert "Passed: 2" in captured.out
        assert "Failed: 0" in captured.out

    def test_print_summary_mixed_results(self, capsys):
        """Should print summary with failures."""
        results = {
            "src.good:GoodModel": {"ok": True, "covered": ["good.py"], "unused": []},
            "src.bad:BadModel": {"ok": False, "covered": ["bad.py"], "unexpected": ["extra.py"]},
            "src.ugly:UglyModel": {"ok": False, "covered": [], "unexpected": ["wrong1.py", "wrong2.py"]}
        }

        print_verification_summary(results)
        captured = capsys.readouterr()

        assert "Total models: 3" in captured.out
        assert "Passed: 1" in captured.out
        assert "Failed: 2" in captured.out
        assert "src.bad:BadModel: 1 unexpected" in captured.out
        assert "src.ugly:UglyModel: 2 unexpected" in captured.out

    def test_print_summary_with_unused_files(self, capsys):
        """Should report unused files."""
        results = {
            "src.model:Model": {"ok": True, "covered": ["used.py"], "unused": ["unused1.py", "unused2.py"]}
        }

        print_verification_summary(results)
        captured = capsys.readouterr()

        assert "Files actually used: 1" in captured.out
        assert "Files allowed but unused: 2" in captured.out


class TestSuggestFilePatterns:
    """Tests for file pattern suggestions."""

    def test_suggest_single_files(self):
        """Should suggest exact paths for single files."""
        unexpected = ["src/model/core.py", "src/utils/helper.py"]

        suggestions = suggest_file_patterns(unexpected)

        assert "src/model/core.py" in suggestions
        assert "src/utils/helper.py" in suggestions

    def test_suggest_directory_patterns(self):
        """Should suggest directory patterns for multiple files in same dir."""
        unexpected = [
            "src/model/core.py",
            "src/model/utils.py",
            "src/model/helpers.py"
        ]

        suggestions = suggest_file_patterns(unexpected)

        # Should suggest directory pattern instead of individual files
        assert "src/model/**" in suggestions
        # Should not suggest individual files
        assert "src/model/core.py" not in suggestions

    def test_suggest_mixed_patterns(self):
        """Should handle mix of single files and directory groups."""
        unexpected = [
            "src/model/core.py",       # Part of group
            "src/model/utils.py",      # Part of group
            "src/single_file.py",      # Single file
            "tests/test_helper.py",    # Single file
        ]

        suggestions = suggest_file_patterns(unexpected)

        assert "src/model/**" in suggestions      # Directory pattern
        assert "src/single_file.py" in suggestions   # Single file
        assert "tests/test_helper.py" in suggestions # Single file

    def test_suggest_removes_duplicates(self):
        """Should remove duplicate suggestions."""
        unexpected = [
            "src/model/a.py",
            "src/model/b.py",
            "src/other/c.py",
            "src/other/d.py"
        ]

        suggestions = suggest_file_patterns(unexpected)

        # Should not have duplicates
        assert len(suggestions) == len(set(suggestions))
        assert suggestions == sorted(suggestions)


class TestRealIntegration:
    """Real integration tests using actual models."""

    def test_verify_example_seir_model(self):
        """Test verification with real StochasticSEIR model."""
        # Change to examples directory where the models are
        examples_dir = Path(__file__).parent.parent / "examples" / "epi_models"
        if not examples_dir.exists():
            pytest.skip("Example models not available")

        import os
        old_cwd = os.getcwd()
        old_env = os.environ.copy()

        try:
            os.chdir(examples_dir)
            # Set PYTHONPATH so subprocess can find models
            src_path = str(examples_dir / "src")
            current_pythonpath = os.environ.get("PYTHONPATH", "")
            new_pythonpath = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path
            os.environ["PYTHONPATH"] = new_pythonpath

            # Test the actual SEIR model
            allowed_files = {"src/models/seir.py", "src/models/__init__.py"}
            result = verify_model("models.seir:StochasticSEIR", allowed_files)

            # Should pass verification
            assert result["ok"] is True, f"Verification failed: {result.get('error', 'Unknown error')}"
            assert "src/models/seir.py" in result["covered"]
            assert "src/models/__init__.py" in result["covered"]
            assert len(result["unexpected"]) == 0, f"Unexpected imports: {result['unexpected']}"

            # Check external dependencies were tracked
            assert len(result["external_deps"]) > 0
            external_deps = " ".join(result["external_deps"])
            assert "numpy" in external_deps or "polars" in external_deps

        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)

    def test_verify_example_age_seir_model(self):
        """Test verification with real AgeStratifiedSEIR model."""
        examples_dir = Path(__file__).parent.parent / "examples" / "epi_models"
        if not examples_dir.exists():
            pytest.skip("Example models not available")

        import os
        old_cwd = os.getcwd()
        old_env = os.environ.copy()

        try:
            os.chdir(examples_dir)
            # Set PYTHONPATH so subprocess can find models
            src_path = str(examples_dir / "src")
            current_pythonpath = os.environ.get("PYTHONPATH", "")
            new_pythonpath = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path
            os.environ["PYTHONPATH"] = new_pythonpath

            # Test the age-stratified model
            allowed_files = {"src/models/seir_age.py", "src/models/__init__.py"}
            result = verify_model("models.seir_age:AgeStratifiedSEIR", allowed_files)

            assert result["ok"] is True, f"Verification failed: {result.get('error', 'Unknown error')}"
            assert "src/models/seir_age.py" in result["covered"]
            assert len(result["unexpected"]) == 0

        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)

    def test_verify_all_example_models(self):
        """Test verifying all models from example project config."""
        examples_dir = Path(__file__).parent.parent / "examples" / "epi_models"
        if not examples_dir.exists():
            pytest.skip("Example models not available")

        import os
        import tomllib
        old_cwd = os.getcwd()
        old_env = os.environ.copy()

        try:
            os.chdir(examples_dir)
            # Set PYTHONPATH so subprocess can find models
            src_path = str(examples_dir / "src")
            current_pythonpath = os.environ.get("PYTHONPATH", "")
            new_pythonpath = f"{src_path}:{current_pythonpath}" if current_pythonpath else src_path
            os.environ["PYTHONPATH"] = new_pythonpath

            # Read the actual pyproject.toml from examples
            with open("pyproject.toml", "rb") as f:
                config = tomllib.load(f)

            assert "tool" in config, f"No [tool] section found in config: {list(config.keys())}"
            assert "calabaria" in config["tool"], f"No [tool.calabaria] section found: {list(config['tool'].keys())}"
            calabria_config = config["tool"]["calabaria"]
            results = verify_all_models(calabria_config)

            # Both models should pass
            assert len(results) == 2
            for class_path, result in results.items():
                assert result["ok"] is True, f"Model {class_path} failed: {result.get('error', 'Unknown')}"
                assert len(result["unexpected"]) == 0, f"Model {class_path} has unexpected imports: {result['unexpected']}"

        finally:
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)

    def test_verify_model_with_extra_imports(self):
        """Test verification when model imports files not in allowed list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            src_dir = tmpdir / "src"
            src_dir.mkdir()

            # Create model that imports an extra utility
            (src_dir / "__init__.py").write_text("")
            (src_dir / "utils.py").write_text("def helper(): return 42")
            (src_dir / "model.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel
                from modelops_calabaria import ParameterSpace, ParameterSpec
                from . import utils  # This will cause unexpected import

                class TestModel(BaseModel):
                    @classmethod
                    def parameter_space(cls):
                        return ParameterSpace([
                            ParameterSpec("x", 0.0, 1.0, "float", doc="Test param")
                        ])

                    def build_sim(self, params, config):
                        return {"value": utils.helper()}

                    def run_sim(self, state, seed):
                        return {"result": state["value"]}
            """))

            import os
            old_cwd = os.getcwd()
            old_env = os.environ.copy()

            try:
                os.chdir(tmpdir)
                # Set PYTHONPATH for subprocess
                current_pythonpath = os.environ.get("PYTHONPATH", "")
                new_pythonpath = f"{src_dir}:{current_pythonpath}" if current_pythonpath else str(src_dir)
                os.environ["PYTHONPATH"] = new_pythonpath

                # Only allow model.py, not utils.py
                allowed_files = {"src/model.py"}
                result = verify_model("src.model:TestModel", allowed_files)

                # Should fail due to unexpected import
                assert result["ok"] is False
                assert "src/utils.py" in result["unexpected"]
                assert "src/model.py" in result["covered"]

            finally:
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_env)

    def test_verify_model_import_error(self):
        """Test verification when model fails to import."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            src_dir = tmpdir / "src"
            src_dir.mkdir()

            # Create model with import error
            (src_dir / "broken.py").write_text(textwrap.dedent("""
                from nonexistent_module import something  # This will fail
                from modelops_calabaria.base_model import BaseModel

                class BrokenModel(BaseModel):
                    def build_sim(self, params, config):
                        pass
                    def run_sim(self, state, seed):
                        pass
            """))

            import os
            old_cwd = os.getcwd()
            old_env = os.environ.copy()

            try:
                os.chdir(tmpdir)
                # Set PYTHONPATH for subprocess
                current_pythonpath = os.environ.get("PYTHONPATH", "")
                new_pythonpath = f"{src_dir}:{current_pythonpath}" if current_pythonpath else str(src_dir)
                os.environ["PYTHONPATH"] = new_pythonpath

                allowed_files = {"src/broken.py"}
                result = verify_model("src.broken:BrokenModel", allowed_files)

                # Should fail with import error
                assert result["ok"] is False
                assert "error" in result
                assert "nonexistent_module" in result["error"]

            finally:
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_env)

    def test_verify_model_with_relative_imports(self):
        """Test verification with complex relative imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create nested package structure
            models_dir = tmpdir / "models"
            models_dir.mkdir()
            (models_dir / "__init__.py").write_text("")

            common_dir = models_dir / "common"
            common_dir.mkdir()
            (common_dir / "__init__.py").write_text("")
            (common_dir / "base.py").write_text("class CommonBase: pass")

            sir_dir = models_dir / "sir"
            sir_dir.mkdir()
            (sir_dir / "__init__.py").write_text("")
            (sir_dir / "model.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel
                from modelops_calabaria import ParameterSpace, ParameterSpec
                from ..common.base import CommonBase

                class SIRModel(BaseModel, CommonBase):
                    @classmethod
                    def parameter_space(cls):
                        return ParameterSpace([
                            ParameterSpec("beta", 0.1, 1.0, "float", doc="Transmission rate")
                        ])

                    def build_sim(self, params, config):
                        return {}

                    def run_sim(self, state, seed):
                        return {"infections": [1, 2, 3]}
            """))

            import os
            old_cwd = os.getcwd()
            old_env = os.environ.copy()

            try:
                os.chdir(tmpdir)
                # Set PYTHONPATH for subprocess
                current_pythonpath = os.environ.get("PYTHONPATH", "")
                new_pythonpath = f"{tmpdir}:{current_pythonpath}" if current_pythonpath else str(tmpdir)
                os.environ["PYTHONPATH"] = new_pythonpath

                # Allow all files in the package
                allowed_files = {
                    "models/__init__.py",
                    "models/sir/__init__.py",
                    "models/sir/model.py",
                    "models/common/__init__.py",
                    "models/common/base.py"
                }
                result = verify_model("models.sir.model:SIRModel", allowed_files)

                # Should pass verification
                assert result["ok"] is True, f"Verification failed: {result.get('error', 'Unknown error')}"
                assert "models/sir/model.py" in result["covered"]
                assert "models/common/base.py" in result["covered"]

            finally:
                os.chdir(old_cwd)
                os.environ.clear()
                os.environ.update(old_env)


class TestIntegration:
    """Integration tests for verification system."""

    def test_verification_subprocess_script(self):
        """Test the verification script format."""
        # The actual verification script should be able to import and test models
        class_path = "test.model:TestModel"

        # Extract the script from verify_model function
        from modelops_calabaria.cli.verify import verify_model
        import inspect

        # Get the source code of verify_model to examine the script
        source = inspect.getsource(verify_model)
        assert 'verification_script = f"""' in source
        assert 'import sys' in source
        assert 'import json' in source
        assert 'import importlib' in source

    def test_full_verification_flow(self):
        """Test complete verification with mocked but realistic data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create mock project structure
            (tmpdir / "src").mkdir()
            (tmpdir / "pyproject.toml").write_text(textwrap.dedent("""
                [tool.calabaria]
                schema = 1
                abi = "model-entrypoint@1"

                [[tool.calabaria.model]]
                class = "src.model:TestModel"
                files = ["src/model.py", "src/utils.py"]
            """))

            # Mock successful verification
            mock_output = {
                "success": True,
                "loaded_files": [
                    str((tmpdir / "src" / "model.py").resolve()),
                    str((tmpdir / "src" / "utils.py").resolve()),
                ],
                "model_info": {"module": "src.model", "class": "TestModel", "has_space": True}
            }

            import json
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0,
                    stdout=json.dumps(mock_output),
                    stderr=""
                )

                # Mock config reading
                config = {
                    "schema": 1,
                    "abi": "model-entrypoint@1",
                    "model": [{
                        "class": "src.model:TestModel",
                        "files": ["src/model.py", "src/utils.py"]
                    }]
                }

                # Mock file resolution
                def mock_resolve(patterns):
                    return [Path("src/model.py"), Path("src/utils.py")]

                with patch('modelops_calabaria.cli.verify.resolve_file_patterns', side_effect=mock_resolve):
                    import os
                    old_cwd = os.getcwd()
                    try:
                        os.chdir(tmpdir)
                        results = verify_all_models(config)
                    finally:
                        os.chdir(old_cwd)

                    assert len(results) == 1
                    assert results["src.model:TestModel"]["ok"] is True
                    assert set(results["src.model:TestModel"]["covered"]) == {"src/model.py", "src/utils.py"}