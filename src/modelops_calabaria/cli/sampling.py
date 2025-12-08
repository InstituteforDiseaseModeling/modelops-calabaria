"""Sampling commands for generating simulation studies."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

import typer
from typer.models import OptionInfo

from modelops_contracts import SimulationStudy

from ..parameters import ParameterSpace, ParameterSpec
from ..sampling.sobol import SobolSampler
from ..sampling.grid import GridSampler
from ..utils import load_symbol
from ..utils.text import default_output_path
from ..utils.registry import resolve_model_identifier


def _parse_tags(tag_values: List[str]) -> Dict[str, str]:
    tags: Dict[str, str] = {}
    for raw in tag_values:
        if "=" not in raw:
            typer.echo(f"Warning: ignoring malformed tag '{raw}' (expected key=value)", err=True)
            continue
        key, value = raw.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            typer.echo(f"Warning: tag key missing in '{raw}'", err=True)
            continue
        tags[key] = value
    return tags


def _format_param_line(spec: ParameterSpec) -> str:
    if spec.lower == spec.upper:
        return f"    • {spec.name} = {spec.lower}"
    return f"    • {spec.name} ∈ [{spec.lower}, {spec.upper}]"


def _normalize_option_value(value):
    """Support calling the Typer command functions directly in tests."""
    return value.default if isinstance(value, OptionInfo) else value


def _print_study_summary(
    study_name: str,
    model_label: str,
    scenario: str,
    sampling_desc: str,
    parameter_space: ParameterSpace,
    n_sets: int,
    n_replicates: int,
    tags: Dict[str, str],
    output_path: Path,
):
    total_runs = n_sets * n_replicates
    tag_text = ", ".join(f"{k}={v}" for k, v in tags.items()) if tags else "-"

    typer.echo("\nStudy Summary")
    typer.echo(f"  Name       : {study_name}")
    typer.echo(f"  Model      : {model_label} (scenario={scenario})")
    typer.echo(f"  Sampling   : {sampling_desc}")
    typer.echo(f"  Parameters : {n_sets} sets × {n_replicates} replicates = {total_runs:,} simulations")
    typer.echo(f"  Tags       : {tag_text}")
    typer.echo(f"  Output     : {output_path}")
    if parameter_space.specs:
        typer.echo("  Parameter Space:")
        for spec in parameter_space.specs:
            typer.echo(_format_param_line(spec))
    else:
        typer.echo("  Parameter Space: (none)")


def sobol_command(
    model_class: str = typer.Argument(..., help="Model class from manifest (e.g., models.seir:StochasticSEIR)"),
    scenario: str = typer.Option("baseline", "--scenario", "-s", help="Scenario name"),
    n_samples: int = typer.Option(100, "--n-samples", "-n", help="Number of samples to generate"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output filename (defaults to <name>.json or study.json when --name is missing)",
    ),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    scramble: bool = typer.Option(True, "--scramble/--no-scramble", help="Use scrambled Sobol sequence"),
    n_replicates: int = typer.Option(1, "--n-replicates", "-r", help="Number of replicates per parameter set"),
    name: Optional[str] = typer.Option(None, "--name", help="Optional study name stored in metadata"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Attach metadata tags (k=v)"),
    project_root: Optional[str] = typer.Option(None, "--project-root", help="Project root to add to sys.path (default: cwd)"),
    no_cwd_import: bool = typer.Option(False, "--no-cwd-import", help="Do not add project root to sys.path during import"),
):
    """Generate SimulationStudy using Sobol sampling.

    This creates a study specification without bundle references.
    The bundle is added at submission time via mops jobs submit.
    """
    project_root = _normalize_option_value(project_root)
    scenario = _normalize_option_value(scenario)
    n_samples = _normalize_option_value(n_samples)
    output = _normalize_option_value(output)
    seed = _normalize_option_value(seed)
    scramble = _normalize_option_value(scramble)
    n_replicates = _normalize_option_value(n_replicates)
    name = _normalize_option_value(name)
    tags = _normalize_option_value(tags)
    no_cwd_import = _normalize_option_value(no_cwd_import)

    try:
        resolved_model, _ = resolve_model_identifier(model_class, project_root)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if resolved_model != model_class:
        typer.echo(f"Resolved model id '{model_class}' → {resolved_model}")

    try:
        model_cls = load_symbol(
            resolved_model,
            project_root=project_root,
            allow_cwd_import=(not no_cwd_import)
        )

        # Get parameter space from the model
        parameter_space = model_cls.parameter_space()
    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        typer.echo(f"Error: Could not import model '{model_class}': {e}", err=True)
        raise typer.Exit(1)

    # For now, skip scenario validation (models can implement scenario checking)
    # In the future, models could have a scenarios() class method

    # Create sampler from the parameter space
    sampler = SobolSampler(parameter_space, scramble=scramble, seed=seed)

    # Generate samples
    samples = sampler.sample(n_samples)

    typer.echo(f"Generated {len(samples)} Sobol samples for {len(parameter_space.specs)} parameters")

    # Create parameter sets as plain dicts (converting numpy types to Python types)
    # Note: We use plain dicts here, not ParameterSet from contracts
    # Calabaria's ParameterSet requires a ParameterSpace reference for validation
    parameter_sets = []
    for params in samples:
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_params[k] = v.item()
            elif isinstance(v, np.bool_):
                clean_params[k] = bool(v)
            else:
                clean_params[k] = v
        parameter_sets.append(clean_params)

    # Create study (no bundle reference needed)
    # Convert file path format (models/seir.py) to module format (models.seir)
    model_path = resolved_model.split(":")[0]
    if model_path.endswith(".py"):
        # Remove .py extension and replace / with .
        model_path = model_path[:-3].replace("/", ".")

    study_name = name or Path(output).stem
    tag_dict = _parse_tags(tags or [])

    output_path = Path(output) if output else default_output_path(study_name, ".json")
    if output is None:
        typer.echo(f"[info]Using default output path {output_path.name} (set --output to override)")

    study = SimulationStudy(
        model=model_path,  # Module path in proper format
        scenario=scenario,
        parameter_sets=parameter_sets,  # List of dicts
        sampling_method="sobol",
        n_replicates=n_replicates,
        outputs=None,  # Models can define outputs via model_outputs() method
        targets=None,
        metadata={
            "n_samples": n_samples,
            "scramble": scramble,
            "seed": seed,
            "model_class": resolved_model,
            "name": study_name,
            "tags": tag_dict,
        }
    )

    # Write output
    with open(output_path, "w") as f:
        study_dict = _study_to_dict(study)
        json.dump(study_dict, f, indent=2)

    typer.echo(f"✓ Generated SimulationStudy with {len(samples)} parameter sets")
    sampling_desc = f"Sobol (scramble={'on' if scramble else 'off'}, seed={seed if seed is not None else 'none'})"
    summary_label = resolved_model if resolved_model == model_class else f"{model_class} → {resolved_model}"
    _print_study_summary(
        study_name,
        summary_label,
        scenario,
        sampling_desc,
        parameter_space,
        len(parameter_sets),
        n_replicates,
        tag_dict,
        output_path,
    )


def grid_command(
    model_class: str = typer.Argument(..., help="Model class from manifest"),
    scenario: str = typer.Option("baseline", "--scenario", "-s", help="Scenario name"),
    grid_points: int = typer.Option(3, "--grid-points", "-g", help="Number of points per parameter"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output filename (defaults to <name>.json or study.json when --name is missing)",
    ),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    n_replicates: int = typer.Option(1, "--n-replicates", "-r", help="Number of replicates per parameter set"),
    name: Optional[str] = typer.Option(None, "--name", help="Optional study name stored in metadata"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", help="Attach metadata tags (k=v)"),
    project_root: Optional[str] = typer.Option(None, "--project-root", help="Project root to add to sys.path (default: cwd)"),
    no_cwd_import: bool = typer.Option(False, "--no-cwd-import", help="Do not add project root to sys.path during import"),
):
    """Generate SimulationStudy using Grid sampling."""
    project_root = _normalize_option_value(project_root)
    scenario = _normalize_option_value(scenario)
    grid_points = _normalize_option_value(grid_points)
    output = _normalize_option_value(output)
    seed = _normalize_option_value(seed)
    n_replicates = _normalize_option_value(n_replicates)
    name = _normalize_option_value(name)
    tags = _normalize_option_value(tags)
    no_cwd_import = _normalize_option_value(no_cwd_import)

    try:
        resolved_model, _ = resolve_model_identifier(model_class, project_root)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if resolved_model != model_class:
        typer.echo(f"Resolved model id '{model_class}' → {resolved_model}")

    try:
        model_cls = load_symbol(
            resolved_model,
            project_root=project_root,
            allow_cwd_import=(not no_cwd_import)
        )

        # Get parameter space from the model
        parameter_space = model_cls.parameter_space()
    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        typer.echo(f"Error: Could not import model '{model_class}': {e}", err=True)
        raise typer.Exit(1)

    # For now, skip scenario validation (models can implement scenario checking)
    # In the future, models could have a scenarios() class method

    # Create sampler from the parameter space
    sampler = GridSampler(parameter_space, n_points_per_param=grid_points)

    # Generate samples
    samples = sampler.sample(None)  # Grid determines its own size

    typer.echo(f"Generated {len(samples)} grid points for {len(parameter_space.specs)} parameters")

    # Create parameter sets as plain dicts (converting numpy types to Python types)
    # Note: We use plain dicts here, not ParameterSet from contracts
    parameter_sets = []
    for params in samples:
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (np.integer, np.floating)):
                clean_params[k] = v.item()
            elif isinstance(v, np.bool_):
                clean_params[k] = bool(v)
            else:
                clean_params[k] = v
        parameter_sets.append(clean_params)

    # Create study
    # Convert file path format (models/seir.py) to module format (models.seir)
    model_path = resolved_model.split(":")[0]
    if model_path.endswith(".py"):
        # Remove .py extension and replace / with .
        model_path = model_path[:-3].replace("/", ".")

    study_name = name or Path(output).stem
    tag_dict = _parse_tags(tags or [])

    output_path = Path(output) if output else default_output_path(study_name, ".json")
    if output is None:
        typer.echo(f"[info]Using default output path {output_path.name} (set --output to override)")

    study = SimulationStudy(
        model=model_path,
        scenario=scenario,
        parameter_sets=parameter_sets,  # List of dicts
        sampling_method="grid",
        n_replicates=n_replicates,
        outputs=None,
        targets=None,
        metadata={
            "grid_points": grid_points,
            "total_samples": len(samples),
            "model_class": resolved_model,
            "name": study_name,
            "tags": tag_dict,
        }
    )

    # Write output
    with open(output_path, "w") as f:
        study_dict = _study_to_dict(study)
        json.dump(study_dict, f, indent=2)

    typer.echo(f"✓ Generated SimulationStudy with {len(samples)} parameter sets")
    sampling_desc = f"Grid ({grid_points} points per parameter)"
    summary_label = resolved_model if resolved_model == model_class else f"{model_class} → {resolved_model}"
    _print_study_summary(
        study_name,
        summary_label,
        scenario,
        sampling_desc,
        parameter_space,
        len(parameter_sets),
        n_replicates,
        tag_dict,
        output_path,
    )


def _study_to_dict(study: SimulationStudy) -> dict:
    """Convert SimulationStudy to dictionary for JSON serialization."""
    result = {
        "model": study.model,
        "scenario": study.scenario,
        "parameter_sets": [
            {"params": ps} if isinstance(ps, dict) else {"params": ps.params}
            for ps in study.parameter_sets
        ],
        "sampling_method": study.sampling_method,
        "n_replicates": study.n_replicates,
        "outputs": study.outputs,
        "metadata": study.metadata
    }
    # Only include targets if they exist
    if study.targets:
        result["targets"] = study.targets
    return result
