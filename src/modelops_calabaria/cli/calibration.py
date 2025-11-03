"""Calibration commands for generating calibration specs."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer

from modelops_contracts import CalibrationSpec

from ..utils import load_symbol


def optuna_command(
    model_class: str = typer.Argument(..., help="Model class from manifest (e.g., models.sir:StarsimSIR)"),
    targets: str = typer.Argument(..., help="Target entrypoints (e.g., targets.incidence:daily_incidence)"),
    observed_data: str = typer.Argument(..., help="Path to observed data file (parquet/csv)"),
    parameters: str = typer.Argument(..., help="Parameters to calibrate as name:lower:upper,... (e.g., beta:0.01:0.2,dur_inf:3:10)"),
    scenario: str = typer.Option("baseline", "--scenario", "-s", help="Scenario name"),
    max_iterations: int = typer.Option(100, "--max-iterations", "-m", help="Maximum optimization iterations"),
    max_trials: int = typer.Option(100, "--max-trials", help="Maximum Optuna trials"),
    batch_size: int = typer.Option(4, "--batch-size", "-b", help="Number of parallel evaluations"),
    n_replicates: int = typer.Option(3, "--n-replicates", "-r", help="Replicates per parameter set"),
    sampler: str = typer.Option("tpe", "--sampler", help="Optuna sampler type (tpe, random, grid)"),
    n_startup_trials: int = typer.Option(10, "--n-startup-trials", help="Number of random startup trials for TPE"),
    output: str = typer.Option("calibration_spec.json", "--output", "-o", help="Output filename"),
    project_root: Optional[str] = typer.Option(None, "--project-root", help="Project root to add to sys.path"),
    no_cwd_import: bool = typer.Option(False, "--no-cwd-import", help="Do not add project root to sys.path"),
):
    """Generate CalibrationSpec using Optuna optimization.

    Creates a calibration specification for Optuna-based parameter optimization.
    The spec can then be submitted via: mops jobs submit-calibration <spec.json>

    Examples:
        cb calibration optuna models.sir:StarsimSIR \\
            targets.incidence:daily_incidence \\
            data/observed.parquet \\
            beta:0.01:0.2,dur_inf:3:10
    """
    # Validate model can be imported
    try:
        model_cls = load_symbol(
            model_class,
            project_root=project_root,
            allow_cwd_import=(not no_cwd_import)
        )
        # Get parameter space to validate parameter names
        parameter_space = model_cls.parameter_space()
    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        typer.echo(f"Error: Could not import model '{model_class}': {e}", err=True)
        raise typer.Exit(1)

    # Parse parameters specification
    param_specs = _parse_parameter_specs(parameters, parameter_space)

    # Parse targets (comma-separated if multiple)
    target_list = [t.strip() for t in targets.split(",") if t.strip()]

    # Validate observed data file exists
    observed_path = Path(observed_data)
    if not observed_path.exists():
        typer.echo(f"Error: Observed data file not found: {observed_data}", err=True)
        raise typer.Exit(1)

    # Convert model path format
    model_path = model_class.split(":")[0]
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

    # Create CalibrationSpec
    spec = CalibrationSpec(
        model=model_path,
        scenario=scenario,
        algorithm="optuna",
        target_data={
            "observed_file": str(observed_path),
            "target_entrypoints": target_list
        },
        max_iterations=max_iterations,
        algorithm_config=algorithm_config,
        outputs=None,  # Can be customized later
        metadata={
            "model_class": model_class,
            "n_parameters": len(param_specs),
            "created_by": "cb calibration optuna"
        }
    )

    # Write output
    output_path = Path(output)
    with open(output_path, "w") as f:
        spec_dict = _spec_to_dict(spec)
        json.dump(spec_dict, f, indent=2)

    typer.echo(f"✓ Generated CalibrationSpec for Optuna optimization")
    typer.echo(f"  Model: {spec.model}/{spec.scenario}")
    typer.echo(f"  Parameters: {', '.join(param_specs.keys())}")
    typer.echo(f"  Targets: {', '.join(target_list)}")
    typer.echo(f"  Max iterations: {max_iterations}")
    typer.echo(f"  Max trials: {max_trials}")
    typer.echo(f"  Output: {output_path}")
    typer.echo(f"\nNext step:")
    typer.echo(f"  mops jobs submit {output_path}")


def abc_command(
    model_class: str = typer.Argument(..., help="Model class from manifest"),
    targets: str = typer.Argument(..., help="Target entrypoints"),
    observed_data: str = typer.Argument(..., help="Path to observed data file"),
    parameters: str = typer.Argument(..., help="Parameters as name:lower:upper,..."),
    scenario: str = typer.Option("baseline", "--scenario", "-s"),
    max_iterations: int = typer.Option(10, "--max-iterations", "-m"),
    particles: int = typer.Option(1000, "--particles", "-p", help="Number of particles"),
    epsilon_schedule: str = typer.Option("10,5,2,1", "--epsilon-schedule", help="Epsilon thresholds"),
    output: str = typer.Option("calibration_spec.json", "--output", "-o"),
    project_root: Optional[str] = typer.Option(None, "--project-root"),
    no_cwd_import: bool = typer.Option(False, "--no-cwd-import"),
):
    """Generate CalibrationSpec using ABC-SMC algorithm.

    Creates a calibration specification for Approximate Bayesian Computation.
    """
    # Validate model
    try:
        model_cls = load_symbol(
            model_class,
            project_root=project_root,
            allow_cwd_import=(not no_cwd_import)
        )
        parameter_space = model_cls.parameter_space()
    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        typer.echo(f"Error: Could not import model '{model_class}': {e}", err=True)
        raise typer.Exit(1)

    # Parse parameters
    param_specs = _parse_parameter_specs(parameters, parameter_space)

    # Parse targets
    target_list = [t.strip() for t in targets.split(",") if t.strip()]

    # Parse epsilon schedule
    epsilon_values = [float(e.strip()) for e in epsilon_schedule.split(",")]

    # Validate observed data
    observed_path = Path(observed_data)
    if not observed_path.exists():
        typer.echo(f"Error: Observed data file not found: {observed_data}", err=True)
        raise typer.Exit(1)

    # Convert model path
    model_path = model_class.split(":")[0]
    if model_path.endswith(".py"):
        model_path = model_path[:-3].replace("/", ".")

    # Build algorithm configuration
    algorithm_config = {
        "n_particles": particles,
        "epsilon_schedule": epsilon_values,
        "parameter_specs": param_specs,
        "simulation_entrypoint": model_class,
        "distance_function": "euclidean",  # Default
        "kernel": "uniform"  # Default perturbation kernel
    }

    # Create spec
    spec = CalibrationSpec(
        model=model_path,
        scenario=scenario,
        algorithm="abc-smc",
        target_data={
            "observed_file": str(observed_path),
            "target_entrypoints": target_list
        },
        max_iterations=max_iterations,
        algorithm_config=algorithm_config,
        metadata={
            "model_class": model_class,
            "n_parameters": len(param_specs),
            "created_by": "cb calibration abc"
        }
    )

    # Write output
    output_path = Path(output)
    with open(output_path, "w") as f:
        spec_dict = _spec_to_dict(spec)
        json.dump(spec_dict, f, indent=2)

    typer.echo(f"✓ Generated CalibrationSpec for ABC-SMC")
    typer.echo(f"  Model: {spec.model}/{spec.scenario}")
    typer.echo(f"  Parameters: {', '.join(param_specs.keys())}")
    typer.echo(f"  Particles: {particles}")
    typer.echo(f"  Epsilon schedule: {epsilon_values}")
    typer.echo(f"  Output: {output_path}")


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