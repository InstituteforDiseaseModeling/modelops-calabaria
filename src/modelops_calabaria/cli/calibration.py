"""Calibration commands for generating calibration specs."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from typer.models import OptionInfo

from modelops_contracts import CalibrationSpec

from ..utils import load_symbol
from ..utils.text import default_output_path
from ..utils.registry import resolve_model_identifier, resolve_target_entries


def _normalize_option(value):
    return value.default if isinstance(value, OptionInfo) else value


def optuna_command(
    model_class: str = typer.Argument(..., help="Model class or registered id (e.g., models.sir:StarsimSIR)"),
    observed_data: str = typer.Argument(..., help="Path to observed data file (parquet/csv)"),
    parameters: str = typer.Argument(..., help="Parameters to calibrate as name:lower:upper,... (e.g., beta:0.01:0.2,dur_inf:3:10)"),
    scenario: str = typer.Option("baseline", "--scenario", "-s", help="Scenario name"),
    target_set: Optional[str] = typer.Option(None, "--target-set", help="Target set defined in .modelops-bundle/registry.yaml"),
    targets: Optional[List[str]] = typer.Option(None, "--target", "-t", help="Specific target id(s) from registry"),
    max_iterations: int = typer.Option(1000000, "--max-iterations", "-m", help="Maximum ask/tell loop iterations (safety cap)"),
    max_trials: int = typer.Option(100, "--max-trials", help="Maximum Optuna trials"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Number of parallel evaluations"),
    n_replicates: int = typer.Option(3, "--n-replicates", "-r", help="Replicates per parameter set"),
    sampler: str = typer.Option("tpe", "--sampler", help="Optuna sampler type (tpe, random, grid)"),
    n_startup_trials: int = typer.Option(10, "--n-startup-trials", help="Number of random startup trials for TPE"),
    name: Optional[str] = typer.Option(None, "--name", help="Optional calibration spec name stored in metadata"),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output filename (defaults to <name>.json or calibration.json when --name is missing)",
    ),
    project_root: Optional[str] = typer.Option(None, "--project-root", help="Project root to add to sys.path (default: cwd)"),
    no_cwd_import: bool = typer.Option(False, "--no-cwd-import", help="Do not add project root to sys.path"),
):
    """Generate CalibrationSpec using Optuna optimization.

    Creates a calibration specification for Optuna-based parameter optimization.
    The spec can then be submitted via: mops jobs submit <spec.json>

    Examples:
        cb calibration optuna models.sir:StarsimSIR \\
            targets.incidence:daily_incidence \\
            data/observed.parquet \\
            beta:0.01:0.2,dur_inf:3:10
    """
    scenario = _normalize_option(scenario)
    project_root = _normalize_option(project_root)
    target_set = _normalize_option(target_set)
    target_values = _normalize_option(targets) or []
    output = _normalize_option(output)
    name = _normalize_option(name)
    max_iterations = _normalize_option(max_iterations)
    max_trials = _normalize_option(max_trials)
    batch_size = _normalize_option(batch_size)
    n_replicates = _normalize_option(n_replicates)
    sampler = _normalize_option(sampler)
    n_startup_trials = _normalize_option(n_startup_trials)
    no_cwd_import = _normalize_option(no_cwd_import)

    try:
        resolved_model, _ = resolve_model_identifier(model_class, project_root)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    if resolved_model != model_class:
        typer.echo(f"Resolved model id '{model_class}' → {resolved_model}")

    # Validate model can be imported
    try:
        model_cls = load_symbol(
            resolved_model,
            project_root=project_root,
            allow_cwd_import=(not no_cwd_import)
        )
        parameter_space = model_cls.parameter_space()
    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        typer.echo(f"Error: Could not import model '{resolved_model}': {e}", err=True)
        raise typer.Exit(1)

    # Parse parameters specification
    param_specs = _parse_parameter_specs(parameters, parameter_space)

    # Resolve targets from registry
    try:
        target_entries, resolved_set = resolve_target_entries(target_values, target_set, project_root)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(1)

    target_ids = [tid for tid, _ in target_entries]
    target_entrypoints = [entry.entrypoint for _, entry in target_entries]

    # Validate observed data file exists
    observed_path = Path(observed_data)
    if not observed_path.exists():
        typer.echo(f"Error: Observed data file not found: {observed_data}", err=True)
        raise typer.Exit(1)

    # Validate max_trials doesn't exceed max_iterations
    if max_trials > max_iterations:
        typer.echo(
            f"Error: max_trials ({max_trials}) exceeds max_iterations ({max_iterations}).\n"
            f"Either:\n"
            f"  1. Increase --max-iterations to at least {max_trials}, or\n"
            f"  2. Decrease --max-trials to at most {max_iterations}.\n\n"
            f"Note: max_iterations is a safety cap on the ask/tell loop, while max_trials\n"
            f"is the algorithm-specific limit for Optuna.",
            err=True
        )
        raise typer.Exit(1)

    # Convert model path format
    model_path = resolved_model.split(":")[0]
    if model_path.endswith(".py"):
        model_path = model_path[:-3].replace("/", ".")

    # Build algorithm configuration
    algorithm_config = {
        "max_trials": max_trials,
        "batch_size": batch_size,
        "n_replicates": n_replicates,
        "parameter_specs": param_specs,
        "sampler": {
            "type": sampler,
        }
    }

    # Add sampler-specific options
    if sampler == "tpe":
        algorithm_config["sampler"]["n_startup_trials"] = n_startup_trials

    # Add simulation entrypoint
    algorithm_config["simulation_entrypoint"] = model_class

    # Determine output path/name
    spec_name = name or "calibration"
    output_path = Path(output) if output else default_output_path(spec_name, ".json")
    if output is None:
        typer.echo(f"[info]Using default output path {output_path.name} (set --output to override)")

    # Create CalibrationSpec
    spec = CalibrationSpec(
        model=model_path,
        scenario=scenario,
        algorithm="optuna",
        target_data={
            "observed_file": str(observed_path),
            "target_entrypoints": target_entrypoints,
            "target_ids": target_ids,
        },
        max_iterations=max_iterations,
        algorithm_config=algorithm_config,
        outputs=None,  # Can be customized later
        metadata={
            "model_class": resolved_model,
            "n_parameters": len(param_specs),
            "created_by": "cb calibration optuna",
            "name": spec_name,
            "target_set": resolved_set,
            "target_ids": target_ids,
        }
    )

    # Write output
    with open(output_path, "w") as f:
        spec_dict = _spec_to_dict(spec)
        json.dump(spec_dict, f, indent=2)

    typer.echo(f"✓ Generated CalibrationSpec for Optuna optimization")
    typer.echo(f"  Model: {spec.model}/{spec.scenario}")
    typer.echo(f"  Parameters: {', '.join(param_specs.keys())}")
    typer.echo(f"  Targets: {', '.join(target_ids)}")
    if resolved_set:
        typer.echo(f"  Target set: {resolved_set}")
    typer.echo(f"  Max iterations: {max_iterations}")
    typer.echo(f"  Max trials: {max_trials}")
    typer.echo(f"  Output: {output_path}")
    typer.echo(f"\nNext step:")
    typer.echo(f"  mops jobs submit {output_path}")

def _parse_parameter_specs(param_string: str, parameter_space) -> Dict[str, Dict[str, Any]]:
    """Parse parameter specification string.

    Format: name1:lower1:upper1,name2:lower2:upper2,...

    Args:
        param_string: Parameter specification string
        parameter_space: Model's parameter space for validation

    Returns:
        Dictionary of parameter specifications
    """
    specs = {}

    for param_spec in param_string.split(","):
        parts = param_spec.strip().split(":")
        if len(parts) != 3:
            raise typer.BadParameter(
                f"Invalid parameter spec '{param_spec}'. "
                f"Expected format: name:lower:upper"
            )

        name, lower_str, upper_str = parts
        name = name.strip()

        # Check parameter exists in model
        param_names = [spec.name for spec in parameter_space.specs]
        if name not in param_names:
            typer.echo(
                f"Warning: Parameter '{name}' not found in model parameter space. "
                f"Available parameters: {', '.join(param_names)}",
                err=True
            )

        # Parse bounds as floats
        try:
            lower = float(lower_str)
            upper = float(upper_str)
        except ValueError:
            raise typer.BadParameter(
                f"Invalid bounds for '{name}': lower='{lower_str}', upper='{upper_str}'. "
                f"Bounds must be numeric."
            )

        if lower >= upper:
            raise typer.BadParameter(
                f"Invalid bounds for '{name}': lower ({lower}) must be less than upper ({upper})"
            )

        specs[name] = {
            "lower": lower,
            "upper": upper
        }

    return specs


def _spec_to_dict(spec: CalibrationSpec) -> dict:
    """Convert CalibrationSpec to dictionary for JSON serialization."""
    return {
        "model": spec.model,
        "scenario": spec.scenario,
        "algorithm": spec.algorithm,
        "target_data": spec.target_data,
        "max_iterations": spec.max_iterations,
        "convergence_criteria": spec.convergence_criteria,
        "algorithm_config": spec.algorithm_config,
        "outputs": spec.outputs,
        "metadata": spec.metadata
    }
