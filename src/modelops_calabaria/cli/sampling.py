"""Sampling commands for generating simulation studies."""

import json
from pathlib import Path
from typing import Optional
import numpy as np

import typer

from modelops_contracts import SimulationStudy

from ..parameters import ParameterSpace, ParameterSpec
from ..sampling.sobol import SobolSampler
from ..sampling.grid import GridSampler


def sobol_command(
    model_class: str = typer.Argument(..., help="Model class from manifest (e.g., models.seir:StochasticSEIR)"),
    scenario: str = typer.Option("baseline", "--scenario", "-s", help="Scenario name"),
    n_samples: int = typer.Option(100, "--n-samples", "-n", help="Number of samples to generate"),
    output: str = typer.Option("study.json", "--output", "-o", help="Output filename"),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    scramble: bool = typer.Option(True, "--scramble/--no-scramble", help="Use scrambled Sobol sequence"),
):
    """Generate SimulationStudy using Sobol sampling.

    This creates a study specification without bundle references.
    The bundle is added at submission time via mops jobs submit.
    """
    # Dynamically import the model class
    try:
        module_path, class_name = model_class.rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)

        # Get parameter space from the model
        parameter_space = model_cls.parameter_space()
    except (ImportError, AttributeError, ValueError) as e:
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
    study = SimulationStudy(
        model=model_class.split(":")[0],  # Just the module path
        scenario=scenario,
        parameter_sets=parameter_sets,  # List of dicts
        sampling_method="sobol",
        n_replicates=1,  # Can be configured later
        outputs=None,  # Models can define outputs via model_outputs() method
        metadata={
            "n_samples": n_samples,
            "scramble": scramble,
            "seed": seed,
            "model_class": model_class
        }
    )

    # Write output
    output_path = Path(output)
    with open(output_path, "w") as f:
        study_dict = _study_to_dict(study)
        json.dump(study_dict, f, indent=2)

    typer.echo(f"✓ Generated SimulationStudy with {len(samples)} parameter sets")
    typer.echo(f"  Model: {study.model}/{study.scenario}")
    typer.echo(f"  Sampling: {study.sampling_method}")
    typer.echo(f"  Output: {output_path}")


def grid_command(
    model_class: str = typer.Argument(..., help="Model class from manifest"),
    scenario: str = typer.Option("baseline", "--scenario", "-s", help="Scenario name"),
    grid_points: int = typer.Option(3, "--grid-points", "-g", help="Number of points per parameter"),
    output: str = typer.Option("study.json", "--output", "-o", help="Output filename"),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for reproducibility"),
):
    """Generate SimulationStudy using Grid sampling."""
    # Dynamically import the model class
    try:
        module_path, class_name = model_class.rsplit(":", 1)
        import importlib
        module = importlib.import_module(module_path)
        model_cls = getattr(module, class_name)

        # Get parameter space from the model
        parameter_space = model_cls.parameter_space()
    except (ImportError, AttributeError, ValueError) as e:
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
    study = SimulationStudy(
        model=model_class.split(":")[0],
        scenario=scenario,
        parameter_sets=parameter_sets,  # List of dicts
        sampling_method="grid",
        n_replicates=1,
        outputs=None,  # Models can define outputs via model_outputs() method
        metadata={
            "grid_points": grid_points,
            "total_samples": len(samples),
            "model_class": model_class
        }
    )

    # Write output
    output_path = Path(output)
    with open(output_path, "w") as f:
        study_dict = _study_to_dict(study)
        json.dump(study_dict, f, indent=2)

    typer.echo(f"✓ Generated SimulationStudy with {len(samples)} parameter sets")
    typer.echo(f"  Model: {study.model}/{study.scenario}")
    typer.echo(f"  Output: {output_path}")


def _study_to_dict(study: SimulationStudy) -> dict:
    """Convert SimulationStudy to dictionary for JSON serialization."""
    return {
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