"""Tests for calibration CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import typer
from modelops_calabaria.cli.calibration import optuna_command
from modelops_calabaria.parameters import ParameterSpace, ParameterSpec


class MockModel:
    @classmethod
    def parameter_space(cls):
        return ParameterSpace(specs=[
            ParameterSpec(name="beta", min=0.1, max=1.0, kind="float", doc="Transmission"),
            ParameterSpec(name="gamma", min=0.05, max=0.5, kind="float", doc="Recovery"),
        ])


def _write_registry(root: Path):
    registry = root / ".modelops-bundle"
    registry.mkdir()
    (registry / "registry.yaml").write_text(
        "\n".join(
            [
                "version: '1.0'",
                "models:",
                "  sir:",
                "    entrypoint: models.sir:StarsimSIR",
                "    path: models/sir.py",
                "    class_name: StarsimSIR",
                "targets:",
                "  incidence:",
                "    path: targets/incidence.py",
                "    entrypoint: targets.incidence:incidence_target",
                "    model_output: incidence",
                "    data: []",
                "  prevalence:",
                "    path: targets/incidence.py",
                "    entrypoint: targets.incidence:prevalence_target",
                "    model_output: prevalence",
                "    data: []",
                "target_sets:",
                "  all_targets:",
                "    targets:",
                "      - incidence",
                "      - prevalence",
                "",
            ]
        )
    )


@patch("modelops_calabaria.cli.calibration.load_symbol", return_value=MockModel)
def test_optuna_uses_target_set(mock_load_symbol):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_registry(root)
        observed = root / "data.csv"
        observed.write_text("value\n1\n")
        output_path = root / "custom.json"

        optuna_command(
            model_class="sir",
            observed_data=str(observed),
            parameters="beta:0.1:0.5,gamma:0.05:0.2",
            target_set="all_targets",
            output=str(output_path),
            project_root=str(root),
        )

        data = json.loads(output_path.read_text())
        assert data["target_data"]["target_entrypoints"] == [
            "targets.incidence:incidence_target",
            "targets.incidence:prevalence_target",
        ]
        assert data["metadata"]["target_set"] == "all_targets"
        assert data["metadata"]["target_ids"] == ["incidence", "prevalence"]


@patch("modelops_calabaria.cli.calibration.load_symbol", return_value=MockModel)
def test_optuna_selects_specific_targets(mock_load_symbol):
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_registry(root)
        observed = root / "data.csv"
        observed.write_text("value\n1\n")

        prev = os.getcwd()
        os.chdir(root)
        try:
            optuna_command(
                model_class="sir",
                observed_data=str(observed),
                parameters="beta:0.1:0.5,gamma:0.05:0.2",
                targets=["incidence"],
                project_root=str(root),
            )
        finally:
            os.chdir(prev)

        spec_path = root / "calibration.json"
        assert spec_path.exists()
        data = json.loads(spec_path.read_text())
        assert data["target_data"]["target_entrypoints"] == ["targets.incidence:incidence_target"]
        assert data["metadata"]["target_ids"] == ["incidence"]


def test_optuna_errors_without_registry():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        observed = root / "data.csv"
        observed.write_text("value\n1\n")

        with patch("modelops_calabaria.cli.calibration.load_symbol", return_value=MockModel):
            with pytest.raises(typer.Exit):
                optuna_command(
                    model_class="models.sir:StarsimSIR",
                    observed_data=str(observed),
                    parameters="beta:0.1:0.5,gamma:0.05:0.2",
                    project_root=str(root),
                )
