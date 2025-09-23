"""Tests for CLI manifest generation.

Tests manifest building from pyproject.toml configuration,
including model metadata extraction and deterministic digests.
"""

import pytest
from pathlib import Path
import tempfile
import textwrap
import json
from unittest.mock import patch, MagicMock

from modelops_calabaria.cli.manifest import (
    build_model_metadata,
    compute_model_digest,
    build_manifest,
    check_manifest_drift,
    print_manifest_summary,
)


class TestBuildModelMetadata:
    """Tests for building metadata for a single model."""

    def test_build_metadata_success(self):
        """Should build metadata for valid model."""
        # Mock the model class and instance
        mock_space = MagicMock()
        # Need at least one parameter spec
        mock_param_spec = MagicMock()
        mock_param_spec.name = "test_param"
        mock_param_spec.kind = "float"
        mock_param_spec.min = 0.0
        mock_param_spec.max = 1.0
        mock_param_spec.doc = "Test parameter"
        mock_param_spec.to_dict.return_value = {"name": "test_param", "kind": "float", "min": 0, "max": 1}
        mock_space.specs = [mock_param_spec]
        mock_space.to_dict.return_value = {"parameters": ["test_param"]}

        mock_instance = MagicMock()
        mock_instance._scenarios = {"baseline": MagicMock(), "lockdown": MagicMock()}
        mock_instance._outputs = {"infections": MagicMock(), "deaths": MagicMock()}

        # Create a proper mock class that can pass issubclass
        from modelops_calabaria.base_model import BaseModel

        class MockTestModel(BaseModel):
            SPACE = mock_space  # Set the parameter space

            def __init__(self, space):
                # Initialize BaseModel properly
                super().__init__(space)
                # Override with our mock data
                self._scenarios = mock_instance._scenarios
                self._outputs = mock_instance._outputs

            def build_sim(self, params, config):
                pass

            def run_sim(self, state, seed):
                return {}

        # Mock the module and import
        mock_module = MagicMock()
        mock_module.TestModel = MockTestModel

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test files
            file1 = tmpdir / "model.py"
            file2 = tmpdir / "utils.py"
            file1.write_text("def model(): pass")
            file2.write_text("def utility(): pass")
            files = [file1, file2]

            # Import hashing module to patch it directly
            from modelops_calabaria.cli import hashing

            with patch('importlib.import_module', return_value=mock_module), \
                 patch.object(hashing, 'token_hash', side_effect=["hash1", "hash2"]), \
                 patch.object(hashing, 'code_sig', return_value="code_signature"), \
                 patch.object(hashing, 'content_hash', return_value="space_signature"), \
                 patch.object(Path, 'cwd', return_value=tmpdir):

                # Use absolute paths to avoid path issues
                metadata = build_model_metadata("test.model:TestModel", files)

                assert metadata["class"] == "test.model:TestModel"
                assert len(metadata["files"]) == 2
                assert metadata["code_sig"] == "code_signature"
                assert metadata["space_sig"] == "space_signature"
                assert set(metadata["scenarios"]) == {"baseline", "lockdown"}
                assert set(metadata["outputs"]) == {"deaths", "infections"}

    def test_build_metadata_import_error(self):
        """Should handle import errors gracefully."""
        with pytest.raises(ImportError, match="Cannot import"):
            build_model_metadata("nonexistent.module:NonExistentModel", [])

    def test_build_metadata_not_basemodel(self):
        """Should reject classes that don't inherit from BaseModel."""
        # Create a real class that's not a BaseModel subclass
        class NotAModel:
            pass

        mock_module = MagicMock()
        mock_module.NotAModel = NotAModel

        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(TypeError, match="not a BaseModel subclass"):
                build_model_metadata("test.module:NotAModel", [])

    def test_build_metadata_missing_space(self):
        """Should handle models without parameter space."""
        from modelops_calabaria.base_model import BaseModel

        # Create a real BaseModel subclass without parameter space
        class BadModel(BaseModel):
            def build_sim(self, params, config):
                pass
            def run_sim(self, state, seed):
                return {}
            # Intentionally no SPACE or parameter_space method

        mock_module = MagicMock()
        mock_module.BadModel = BadModel

        with patch('importlib.import_module', return_value=mock_module):
            with pytest.raises(AttributeError, match="must define SPACE or parameter_space"):
                build_model_metadata("test.module:BadModel", [])

    def test_handles_parameter_specs(self):
        """Should properly serialize parameter specifications."""
        from modelops_calabaria.cli import hashing
        from modelops_calabaria.base_model import BaseModel

        # Create mock parameter specs with required attributes
        mock_spec1 = MagicMock()
        mock_spec1.name = "param1"
        mock_spec1.kind = "float"
        mock_spec1.min = 0.0
        mock_spec1.max = 1.0
        mock_spec1.doc = "Test parameter 1"

        mock_spec2 = MagicMock()
        mock_spec2.name = "param2"
        mock_spec2.kind = "int"
        mock_spec2.min = 1
        mock_spec2.max = 10
        mock_spec2.doc = "Test parameter 2"

        mock_space = MagicMock()
        mock_space.specs = [mock_spec1, mock_spec2]
        mock_space.to_dict.return_value = {"parameters": ["spec1", "spec2"]}

        # Create a real BaseModel subclass with the mocked space
        class TestModel(BaseModel):
            SPACE = mock_space

            def build_sim(self, params, config):
                pass

            def run_sim(self, state, seed):
                return {}

            def _seal(self):
                self._scenarios = {}
                self._outputs = {}

        mock_module = MagicMock()
        mock_module.TestModel = TestModel

        # Mock SerializedParameterSpec
        mock_serialized_spec = MagicMock()
        mock_serialized_spec.to_json.return_value = {"name": "test.model:TestModel", "kind": "float"}

        with patch('importlib.import_module', return_value=mock_module), \
             patch.object(hashing, 'token_hash', return_value="hash"), \
             patch.object(hashing, 'code_sig', return_value="sig"), \
             patch.object(hashing, 'content_hash', return_value="content"):

            # Use an absolute path that can be made relative to cwd
            test_file = Path.cwd() / "test.py"
            metadata = build_model_metadata("test.module:TestModel", [test_file])

            assert len(metadata["param_specs"]) == 2
            # Verify parameter specs were serialized correctly
            param_names = [spec["name"] for spec in metadata["param_specs"]]
            assert "param1" in param_names
            assert "param2" in param_names


class TestComputeModelDigest:
    """Tests for model digest computation."""

    def test_compute_digest_deterministic(self):
        """Should produce deterministic digest."""
        model_metadata = {
            "code_sig": "code123",
            "space_sig": "space456"
        }

        digest1 = compute_model_digest(model_metadata, "abi@1", ">=3.11", "lock_hash")
        digest2 = compute_model_digest(model_metadata, "abi@1", ">=3.11", "lock_hash")

        assert digest1 == digest2
        assert len(digest1) == 64  # 64 hex chars (BLAKE2b)

    def test_compute_digest_different_inputs(self):
        """Should produce different digests for different inputs."""
        base_metadata = {"code_sig": "code123", "space_sig": "space456"}

        # Same inputs should be same
        digest1 = compute_model_digest(base_metadata, "abi@1", ">=3.11", "lock1")
        digest2 = compute_model_digest(base_metadata, "abi@1", ">=3.11", "lock1")
        assert digest1 == digest2

        # Different code sig
        different_metadata = {"code_sig": "code456", "space_sig": "space456"}
        digest3 = compute_model_digest(different_metadata, "abi@1", ">=3.11", "lock1")
        assert digest1 != digest3

        # Different ABI
        digest4 = compute_model_digest(base_metadata, "abi@2", ">=3.11", "lock1")
        assert digest1 != digest4

        # Different Python version
        digest5 = compute_model_digest(base_metadata, "abi@1", ">=3.12", "lock1")
        assert digest1 != digest5

        # Different lock hash
        digest6 = compute_model_digest(base_metadata, "abi@1", ">=3.11", "lock2")
        assert digest1 != digest6


class TestBuildManifest:
    """Tests for complete manifest building."""

    def test_build_manifest_success(self):
        """Should build complete manifest successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create pyproject.toml
            pyproject = tmpdir / "pyproject.toml"
            pyproject.write_text(textwrap.dedent("""
                [tool.calabaria]
                schema = 1
                abi = "model-entrypoint@1"
                requires_python = ">=3.11"

                [[tool.calabaria.model]]
                class = "test.model:TestModel"
                files = ["src/model.py"]
            """))

            # Mock all the dependencies
            with patch('modelops_calabaria.cli.config.read_pyproject') as mock_read, \
                 patch('modelops_calabaria.cli.config.validate_config', return_value=[]), \
                 patch('modelops_calabaria.cli.config.get_uv_lock_hash', return_value="lock_hash"), \
                 patch('modelops_calabaria.cli.config.resolve_file_patterns', return_value=[Path("src/model.py")]), \
                 patch('modelops_calabaria.cli.manifest.build_model_metadata') as mock_build_meta:

                mock_read.return_value = {
                    "schema": 1,
                    "abi": "model-entrypoint@1",
                    "requires_python": ">=3.11",
                    "model": [{
                        "class": "test.model:TestModel",
                        "files": ["src/model.py"]
                    }]
                }

                mock_build_meta.return_value = {
                    "class": "test.model:TestModel",
                    "files": [{"path": "src/model.py", "sha256": "filehash"}],
                    "code_sig": "codesig",
                    "space_sig": "spacesig",
                    "scenarios": ["baseline"],
                    "outputs": ["result"],
                    "param_specs": [{"name": "test_param", "kind": "float", "min": 0, "max": 1}]
                }

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    manifest, bundle_id = build_manifest(check_only=True)
                finally:
                    os.chdir(old_cwd)

                # Verify manifest structure
                assert manifest["schema"] == 1
                assert manifest["abi"] == "model-entrypoint@1"
                assert manifest["requires_python"] == ">=3.11"
                assert manifest["uv_lock_sha256"] == "lock_hash"
                assert "bundle_id" in manifest

                # Verify model metadata
                assert "test.model:TestModel" in manifest["models"]
                model = manifest["models"]["test.model:TestModel"]
                assert "model_digest" in model

                # Bundle ID should be deterministic
                assert len(bundle_id) == 64  # 64 hex chars (BLAKE2b)

    def test_build_manifest_no_pyproject(self):
        """Should handle missing pyproject.toml."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with pytest.raises(FileNotFoundError, match="pyproject.toml not found"):
                    build_manifest()
            finally:
                os.chdir(old_cwd)

    def test_build_manifest_no_calabria_config(self):
        """Should handle missing calabaria configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create empty pyproject.toml
            (tmpdir / "pyproject.toml").write_text("[project]\nname = 'test'")

            with patch('modelops_calabaria.cli.config.read_pyproject', return_value={}):
                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    with pytest.raises(ValueError, match="No.*calabaria.*configuration"):
                        build_manifest()
                finally:
                    os.chdir(old_cwd)

    def test_build_manifest_validation_errors(self):
        """Should handle configuration validation errors."""
        with patch('modelops_calabaria.cli.config.read_pyproject', return_value={"invalid": "config"}), \
             patch('modelops_calabaria.cli.config.validate_config', return_value=["Error 1", "Error 2"]):

            with pytest.raises(ValueError, match="Configuration errors"):
                build_manifest()

    def test_build_manifest_writes_file(self):
        """Should write manifest.json to disk when not check_only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            with patch('modelops_calabaria.cli.config.read_pyproject') as mock_read, \
                 patch('modelops_calabaria.cli.config.validate_config', return_value=[]), \
                 patch('modelops_calabaria.cli.config.get_uv_lock_hash', return_value="lock"), \
                 patch('modelops_calabaria.cli.config.resolve_file_patterns', return_value=[]), \
                 patch('modelops_calabaria.cli.manifest.build_model_metadata', return_value={
                     "class": "test:Test", "files": [], "code_sig": "sig", "space_sig": "sig2",
                     "scenarios": ["baseline"], "outputs": ["output1"],
                     "param_specs": [{"name": "param1", "kind": "float", "min": 0, "max": 1}]
                 }):

                mock_read.return_value = {
                    "schema": 1, "abi": "test@1",
                    "model": [{"class": "test:Test", "files": ["test.py"]}]
                }

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    manifest, bundle_id = build_manifest(check_only=False)

                    # Should create manifest.json
                    manifest_file = tmpdir / "manifest.json"
                    assert manifest_file.exists()

                    # Should be valid JSON
                    with open(manifest_file) as f:
                        loaded_manifest = json.load(f)
                    assert loaded_manifest["bundle_id"] == bundle_id

                finally:
                    os.chdir(old_cwd)


class TestCheckManifestDrift:
    """Tests for manifest drift detection."""

    def test_check_drift_no_manifest(self):
        """Should detect missing manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = check_manifest_drift()
                assert result is False
            finally:
                os.chdir(old_cwd)

    def test_check_drift_up_to_date(self):
        """Should detect when manifest is up to date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create existing manifest
            manifest = {"test.model:TestModel": "data", "version": "1.0"}
            manifest_file = tmpdir / "manifest.json"
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, sort_keys=True)

            # Mock build_manifest to return same data
            with patch('modelops_calabaria.cli.manifest.build_manifest') as mock_build:
                mock_build.return_value = (manifest, "bundle123")

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    result = check_manifest_drift()
                    assert result is True
                finally:
                    os.chdir(old_cwd)

    def test_check_drift_deterministic(self):
        """Should return True when manifest generation is deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Mock build_manifest to return the same data both times
            manifest_data = {"consistent": "data"}
            with patch('modelops_calabaria.cli.manifest.build_manifest') as mock_build:
                mock_build.return_value = (manifest_data, "bundle456")

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    result = check_manifest_drift()
                    assert result is True  # Deterministic generation
                    assert mock_build.call_count == 2  # Called twice for comparison
                finally:
                    os.chdir(old_cwd)

    def test_check_drift_nondeterministic(self):
        """Should return False when manifest generation is non-deterministic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Mock build_manifest to return different data each time
            manifests = [
                ({"first": "call"}, "bundle1"),
                ({"second": "call"}, "bundle2")
            ]
            with patch('modelops_calabaria.cli.manifest.build_manifest') as mock_build:
                mock_build.side_effect = manifests

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    result = check_manifest_drift()
                    assert result is False  # Non-deterministic generation
                    assert mock_build.call_count == 2  # Called twice for comparison
                finally:
                    os.chdir(old_cwd)

    def test_check_drift_error_handling(self):
        """Should handle errors gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create manifest file
            (tmpdir / "manifest.json").write_text("{}")

            # Mock build_manifest to raise error
            with patch('modelops_calabaria.cli.manifest.build_manifest') as mock_build:
                mock_build.side_effect = Exception("Build error")

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)
                    result = check_manifest_drift()
                    assert result is False
                finally:
                    os.chdir(old_cwd)


class TestPrintManifestSummary:
    """Tests for manifest summary printing."""

    def test_print_summary_basic(self, capsys):
        """Should print basic manifest information."""
        manifest = {
            "schema": 1,
            "abi": "model-entrypoint@1",
            "bundle_id": "abc123def456" + "0" * 48,
            "requires_python": ">=3.11",
            "uv_lock_sha256": "sha256:lock123" + "0" * 50,
            "models": {
                "test.model:TestModel": {
                    "class": "test.model:TestModel",
                    "files": [{"path": "src/model.py", "sha256": "hash1"}],
                    "scenarios": ["baseline", "lockdown"],
                    "outputs": ["infections", "deaths"],
                    "model_digest": "digest123" + "0" * 50
                }
            }
        }

        print_manifest_summary(manifest)
        captured = capsys.readouterr()

        assert "Schema: 1" in captured.out
        assert "ABI: model-entrypoint@1" in captured.out
        assert "Bundle ID: abc123def456" in captured.out
        assert "Models: 1" in captured.out
        assert "test.model:TestModel:" in captured.out
        assert "Files: 1" in captured.out
        assert "Scenarios: 2" in captured.out
        assert "Outputs: 2" in captured.out
        assert "Digest: digest123" in captured.out
        assert "Python: >=3.11" in captured.out
        assert "UV Lock: sha256:lock123" in captured.out

    def test_print_summary_no_python_version(self, capsys):
        """Should handle missing Python version."""
        manifest = {
            "schema": 1,
            "abi": "test@1",
            "bundle_id": "bundle123" + "0" * 54,
            "uv_lock_sha256": "lock456" + "0" * 57,
            "models": {}
        }

        print_manifest_summary(manifest)
        captured = capsys.readouterr()

        assert "Models: 0" in captured.out
        assert "Python:" not in captured.out  # Should not show if empty


class TestIntegration:
    """Integration tests for manifest system."""

    def test_full_manifest_pipeline(self):
        """Test complete manifest generation pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create realistic project structure
            (tmpdir / "src").mkdir()
            (tmpdir / "pyproject.toml").write_text(textwrap.dedent("""
                [tool.calabaria]
                schema = 1
                abi = "model-entrypoint@1"
                requires_python = ">=3.11"

                [[tool.calabaria.model]]
                src.sir:SIRModel"
                class = "src.sir:SIRModel"
                files = ["src/sir.py", "src/common.py"]
            """))

            # Create source files
            (tmpdir / "src" / "sir.py").write_text("# SIR model implementation")
            (tmpdir / "src" / "common.py").write_text("# Common utilities")

            # Mock all external dependencies
            mock_metadata = {
                "class": "src.sir:SIRModel",
                "files": [
                    {"path": "src/sir.py", "sha256": "sir_hash"},
                    {"path": "src/common.py", "sha256": "common_hash"}
                ],
                "code_sig": "code_signature",
                "space_sig": "space_signature",
                "scenarios": ["baseline", "intervention"],
                "outputs": ["infections", "recoveries"],
                "param_specs": [
                    {"name": "beta", "kind": "float", "min": 0.0, "max": 1.0}
                ]
            }

            with patch('modelops_calabaria.cli.config.read_pyproject') as mock_read, \
                 patch('modelops_calabaria.cli.config.validate_config', return_value=[]), \
                 patch('modelops_calabaria.cli.config.get_uv_lock_hash', return_value="sha256:lock_hash"), \
                 patch('modelops_calabaria.cli.config.resolve_file_patterns') as mock_resolve, \
                 patch('modelops_calabaria.cli.manifest.build_model_metadata', return_value=mock_metadata):

                mock_read.return_value = {
                    "schema": 1,
                    "abi": "model-entrypoint@1",
                    "requires_python": ">=3.11",
                    "model": [{
                        "id": "src.sir:SIRModel",
                        "class": "src.sir:SIRModel",
                        "files": ["src/sir.py", "src/common.py"]
                    }]
                }

                mock_resolve.return_value = [Path("src/sir.py"), Path("src/common.py")]

                import os
                old_cwd = os.getcwd()
                try:
                    os.chdir(tmpdir)

                    # Generate manifest
                    manifest, bundle_id = build_manifest(check_only=True)

                    # Verify complete structure
                    assert manifest["schema"] == 1
                    assert manifest["abi"] == "model-entrypoint@1"
                    assert manifest["requires_python"] == ">=3.11"
                    assert manifest["bundle_id"] == bundle_id
                    assert len(bundle_id) == 64  # 64 hex chars (BLAKE2b)

                    # Verify model entry
                    assert "src.sir:SIRModel" in manifest["models"]
                    sir_model = manifest["models"]["src.sir:SIRModel"]
                    assert sir_model["class"] == "src.sir:SIRModel"
                    assert len(sir_model["files"]) == 2
                    assert len(sir_model["scenarios"]) == 2
                    assert len(sir_model["outputs"]) == 2
                    assert "model_digest" in sir_model

                    # Test drift detection
                    # First time should be up to date with itself
                    with patch('modelops_calabaria.cli.manifest.build_manifest') as mock_check:
                        mock_check.return_value = (manifest, bundle_id)
                        assert check_manifest_drift() is True

                finally:
                    os.chdir(old_cwd)