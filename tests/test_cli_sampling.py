"""Tests for CLI sampling commands."""

import json
import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from modelops_calabaria.cli.sampling import sobol_command, grid_command, _job_to_dict
from modelops_contracts import SimJob, SimBatch, SimTask


class TestCLISampling:
    """Tests for sampling CLI commands."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock manifest
        self.manifest = {
            "models": {
                "models.test:TestModel": {
                    "scenarios": ["baseline", "variant"],
                    "outputs": ["result", "stats"],
                    "param_specs": [
                        {
                            "name": "alpha",
                            "kind": "float",
                            "min": 0.1,
                            "max": 1.0,
                            "doc": "Alpha parameter"
                        },
                        {
                            "name": "beta",
                            "kind": "float",
                            "min": 0.5,
                            "max": 2.0,
                            "doc": "Beta parameter"
                        },
                        {
                            "name": "count",
                            "kind": "int",
                            "min": 10.0,  # Will be converted to int
                            "max": 100.0,  # Will be converted to int
                            "doc": "Count parameter"
                        }
                    ]
                }
            }
        }

    @patch("modelops_calabaria.cli.sampling.Path.exists")
    def test_sobol_command_basic(self, mock_exists):
        """Test basic Sobol job generation."""
        mock_exists.return_value = True

        # Use the real open but intercept specific files
        real_open = open

        def selective_mock_open(filename, mode='r', *args, **kwargs):
            # Convert to string to handle Path objects
            filename_str = str(filename)

            # Only mock manifest.json, let everything else through
            if 'manifest.json' in filename_str:
                if mode == 'r':
                    # Create a real file-like object with our test manifest
                    from io import StringIO
                    return StringIO(json.dumps(self.manifest))
                else:
                    # For writing, use a real temp file
                    return real_open(filename, mode, *args, **kwargs)
            else:
                # Let scipy and other libraries access their files normally
                return real_open(filename, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_mock_open):
            # Capture the output by patching json.dump
            with patch('json.dump') as mock_json_dump:
                captured_data = []
                mock_json_dump.side_effect = lambda data, f, **kwargs: captured_data.append(data)

                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / "test_job.json"

                    # Run the command
                    sobol_command(
                        model_class="models.test:TestModel",
                        scenario="baseline",
                        n_samples=8,
                        bundle_ref="sha256:" + "a" * 64,
                        output=str(output_path),
                        seed=42,
                        scramble=False
                    )

                    # Check output was generated
                    assert len(captured_data) == 1
                    job_data = captured_data[0]

                    assert job_data["job_id"].startswith("job-")
                    assert job_data["bundle_ref"] == "sha256:" + "a" * 64
                    assert len(job_data["batches"]) == 1

                    batch = job_data["batches"][0]
                    assert batch["batch_id"].startswith("sobol-")
                    assert batch["sampling_method"] == "sobol"
                    assert len(batch["tasks"]) == 8

                    # Check first task structure
                    task = batch["tasks"][0]
                    assert task["entrypoint"] == "models.test/baseline"
                    assert task["bundle_ref"] == "sha256:" + "a" * 64
                    assert task["seed"] == 42
                    assert "alpha" in task["params"]["values"]
                    assert "beta" in task["params"]["values"]
                    assert "count" in task["params"]["values"]
                    assert task["outputs"] == ["result", "stats"]

    @patch("modelops_calabaria.cli.sampling.Path.exists")
    def test_grid_command_basic(self, mock_exists):
        """Test basic Grid job generation."""
        mock_exists.return_value = True

        # Use the same selective mocking approach
        real_open = open

        def selective_mock_open(filename, mode='r', *args, **kwargs):
            filename_str = str(filename)
            if 'manifest.json' in filename_str:
                if mode == 'r':
                    from io import StringIO
                    return StringIO(json.dumps(self.manifest))
                else:
                    return real_open(filename, mode, *args, **kwargs)
            else:
                return real_open(filename, mode, *args, **kwargs)

        with patch("builtins.open", side_effect=selective_mock_open):
            with patch('json.dump') as mock_json_dump:
                captured_data = []
                mock_json_dump.side_effect = lambda data, f, **kwargs: captured_data.append(data)

                with tempfile.TemporaryDirectory() as tmpdir:
                    output_path = Path(tmpdir) / "grid_job.json"

                    # Run the command
                    grid_command(
                        model_class="models.test:TestModel",
                        scenario="variant",
                        grid_points=2,
                        bundle_ref="sha256:" + "b" * 64,
                        output=str(output_path),
                        seed=100
                    )

                    # Check output
                    assert len(captured_data) == 1
                    job_data = captured_data[0]

                    assert job_data["job_id"].startswith("job-")
                    assert job_data["bundle_ref"] == "sha256:" + "b" * 64

                    batch = job_data["batches"][0]
                    assert batch["batch_id"].startswith("grid-")
                    assert batch["sampling_method"] == "grid"
                    # 2 points per param, 3 params = 2^3 = 8 tasks
                    assert len(batch["tasks"]) == 8

                    # Check tasks have incrementing seeds
                    for i, task in enumerate(batch["tasks"]):
                        assert task["seed"] == 100 + i

    def test_job_to_dict_numpy_conversion(self):
        """Test that _job_to_dict properly converts numpy types."""
        import numpy as np
        from modelops_contracts import UniqueParameterSet

        # Create a task with numpy scalar types in params
        # Note: arrays aren't valid parameter values, only scalars
        params_with_numpy = {
            "alpha": np.float64(0.5),
            "beta": np.float32(1.5),
            "count": np.int64(42),
            "flag": np.bool_(True)
        }

        # Convert numpy types to Python types for UniqueParameterSet
        clean_params = {}
        for k, v in params_with_numpy.items():
            if isinstance(v, (np.integer, np.floating, np.bool_)):
                clean_params[k] = v.item()
            else:
                clean_params[k] = v

        param_set = UniqueParameterSet.from_dict(clean_params)

        task = SimTask(
            bundle_ref="sha256:" + "a" * 64,
            entrypoint="test.model/scenario",
            params=param_set,
            seed=42
        )

        batch = SimBatch(
            batch_id="test-batch",
            tasks=[task],
            sampling_method="test"
        )

        job = SimJob(
            job_id="test-job",
            batches=[batch],
            bundle_ref="sha256:" + "a" * 64
        )

        # Convert to dict
        job_dict = _job_to_dict(job)

        # Check that numpy types were converted
        task_params = job_dict["batches"][0]["tasks"][0]["params"]["values"]

        assert isinstance(task_params["alpha"], float)
        assert isinstance(task_params["beta"], float)
        assert isinstance(task_params["count"], int)
        assert isinstance(task_params["flag"], bool)

        # Should be JSON serializable
        json_str = json.dumps(job_dict)
        assert json_str  # Should not raise

    @patch("modelops_calabaria.cli.sampling.Path.exists")
    def test_missing_manifest_error(self, mock_exists):
        """Test error when manifest.json is missing."""
        mock_exists.return_value = False

        from typer import Exit
        with pytest.raises(Exit) as exc_info:
            sobol_command(
                model_class="models.test:TestModel",
                scenario="baseline",
                n_samples=8,
                bundle_ref="sha256:" + "a" * 64,
                output="job.json"
            )

        assert exc_info.value.exit_code == 1

    @patch("modelops_calabaria.cli.sampling.Path.exists")
    @patch("builtins.open")
    def test_invalid_model_error(self, mock_open, mock_exists):
        """Test error when model not in manifest."""
        mock_exists.return_value = True

        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.read.return_value = json.dumps(self.manifest)
        mock_open.return_value = mock_file

        from typer import Exit
        with pytest.raises(Exit) as exc_info:
            sobol_command(
                model_class="models.nonexistent:Model",
                scenario="baseline",
                n_samples=8,
                bundle_ref="sha256:" + "a" * 64,
                output="job.json"
            )

        assert exc_info.value.exit_code == 1

    @patch("modelops_calabaria.cli.sampling.Path.exists")
    @patch("builtins.open")
    def test_invalid_scenario_error(self, mock_open, mock_exists):
        """Test error when scenario not available for model."""
        mock_exists.return_value = True

        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.__exit__.return_value = None
        mock_file.read.return_value = json.dumps(self.manifest)
        mock_open.return_value = mock_file

        from typer import Exit
        with pytest.raises(Exit) as exc_info:
            sobol_command(
                model_class="models.test:TestModel",
                scenario="nonexistent",
                n_samples=8,
                bundle_ref="sha256:" + "a" * 64,
                output="job.json"
            )

        assert exc_info.value.exit_code == 1