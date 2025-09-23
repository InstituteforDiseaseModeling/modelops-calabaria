"""Sampling commands for generating simulation jobs."""

import json
import uuid
from pathlib import Path
from typing import Optional
from dataclasses import asdict

import typer

from modelops_contracts import SimTask, SimBatch, SimJob

from ..parameters import ParameterSpace, ParameterSpec
from ..sampling.sobol import SobolSampler
from ..sampling.grid import GridSampler


def sobol_command(
    model_class: str = typer.Argument(..., help="Model class from manifest (e.g., models.seir:StochasticSEIR)"),
    scenario: str = typer.Option("baseline", "--scenario", "-s", help="Scenario name"),
    n_samples: int = typer.Option(100, "--n-samples", "-n", help="Number of samples to generate"),
    bundle_ref: str = typer.Option(..., "--bundle-ref", "-b", help="Bundle reference for code"),
    output: str = typer.Option("job.json", "--output", "-o", help="Output JSON file"),
    seed: Optional[int] = typer.Option(42, "--seed", help="Random seed for Sobol scrambling"),
    scramble: bool = typer.Option(True, "--scramble/--no-scramble", help="Scramble Sobol sequence"),
):
    """Generate a SimJob using Sobol sampling."""

    # Load manifest
    manifest_path = Path("manifest.json")
    if not manifest_path.exists():
        typer.echo("Error: manifest.json not found. Run 'cb manifest build' first.", err=True)
        raise typer.Exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get model from manifest
    if model_class not in manifest.get("models", {}):
        typer.echo(f"Error: Model '{model_class}' not found in manifest", err=True)
        typer.echo("Available models:", err=True)
        for model in manifest.get("models", {}).keys():
            typer.echo(f"  - {model}", err=True)
        raise typer.Exit(1)

    model_data = manifest["models"][model_class]

    # Check scenario exists
    if scenario not in model_data.get("scenarios", []):
        typer.echo(f"Error: Scenario '{scenario}' not found for model '{model_class}'", err=True)
        typer.echo(f"Available scenarios: {', '.join(model_data['scenarios'])}", err=True)
        raise typer.Exit(1)

    # Convert param specs to ParameterSpec objects
    param_specs = []
    for spec_dict in model_data.get("param_specs", []):
        # Handle both float and int types
        if spec_dict["kind"] in ["float", "int", "integer"]:
            # Convert bounds to int for integer parameters
            if spec_dict["kind"] in ["int", "integer"]:
                min_val = int(spec_dict["min"])
                max_val = int(spec_dict["max"])
            else:
                min_val = spec_dict["min"]
                max_val = spec_dict["max"]

            spec = ParameterSpec(
                name=spec_dict["name"],
                kind=spec_dict["kind"],
                min=min_val,
                max=max_val,
                doc=spec_dict.get("doc", "")
            )
        else:
            # For other types, we'd need to handle differently
            typer.echo(f"Warning: Skipping parameter '{spec_dict['name']}' with unsupported kind '{spec_dict['kind']}'")
            continue
        param_specs.append(spec)

    if not param_specs:
        typer.echo("Error: No valid parameters found for sampling", err=True)
        raise typer.Exit(1)

    # Create parameter space
    space = ParameterSpace(param_specs)

    # Generate samples
    typer.echo(f"Generating {n_samples} Sobol samples for {model_class}/{scenario}")
    sampler = SobolSampler(space, scramble=scramble, seed=seed)
    samples = sampler.sample(n_samples)

    # Create tasks
    tasks = []
    base_seed = seed or 42
    for i, params in enumerate(samples):
        task = SimTask.from_components(
            import_path=model_class.split(":")[0],  # Extract module path
            scenario=scenario,
            bundle_ref=bundle_ref,
            params=params,
            seed=base_seed + i,
            outputs=model_data.get("outputs", [])  # Include outputs from manifest
        )
        tasks.append(task)

    # Create batch
    batch_id = str(uuid.uuid4())[:8]
    batch = SimBatch(
        batch_id=f"sobol-{batch_id}",
        tasks=tasks,
        sampling_method="sobol",
        metadata={
            "n_samples": n_samples,
            "scramble": scramble,
            "seed": seed,
            "model_class": model_class,
            "scenario": scenario
        }
    )

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = SimJob(
        job_id=f"job-{job_id}",
        batches=[batch],
        bundle_ref=bundle_ref
    )

    # Write output
    output_path = Path(output)
    with open(output_path, "w") as f:
        # Convert to dict, handling nested dataclasses
        job_dict = _job_to_dict(job)
        json.dump(job_dict, f, indent=2)

    typer.echo(f"✓ Generated SimJob with {n_samples} tasks")
    typer.echo(f"  Job ID: {job.job_id}")
    typer.echo(f"  Bundle: {bundle_ref}")
    typer.echo(f"  Output: {output_path}")


def grid_command(
    model_class: str = typer.Argument(..., help="Model class from manifest (e.g., models.seir:StochasticSEIR)"),
    scenario: str = typer.Option("baseline", "--scenario", "-s", help="Scenario name"),
    grid_points: int = typer.Option(10, "--grid-points", "-g", help="Number of points per dimension"),
    bundle_ref: str = typer.Option(..., "--bundle-ref", "-b", help="Bundle reference for code"),
    output: str = typer.Option("job.json", "--output", "-o", help="Output JSON file"),
    seed: Optional[int] = typer.Option(42, "--seed", help="Base random seed"),
):
    """Generate a SimJob using Grid sampling."""

    # Load manifest
    manifest_path = Path("manifest.json")
    if not manifest_path.exists():
        typer.echo("Error: manifest.json not found. Run 'cb manifest build' first.", err=True)
        raise typer.Exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get model from manifest
    if model_class not in manifest.get("models", {}):
        typer.echo(f"Error: Model '{model_class}' not found in manifest", err=True)
        raise typer.Exit(1)

    model_data = manifest["models"][model_class]

    # Check scenario exists
    if scenario not in model_data.get("scenarios", []):
        typer.echo(f"Error: Scenario '{scenario}' not found", err=True)
        raise typer.Exit(1)

    # Convert param specs
    param_specs = []
    for spec_dict in model_data.get("param_specs", []):
        if spec_dict["kind"] in ["float", "int", "integer"]:
            # Convert bounds to int for integer parameters
            if spec_dict["kind"] in ["int", "integer"]:
                min_val = int(spec_dict["min"])
                max_val = int(spec_dict["max"])
            else:
                min_val = spec_dict["min"]
                max_val = spec_dict["max"]

            spec = ParameterSpec(
                name=spec_dict["name"],
                kind=spec_dict["kind"],
                min=min_val,
                max=max_val,
                doc=spec_dict.get("doc", "")
            )
            param_specs.append(spec)

    # Create parameter space
    space = ParameterSpace(param_specs)

    # Generate samples
    typer.echo(f"Generating grid with {grid_points} points per dimension")
    sampler = GridSampler(space, n_points_per_param=grid_points)
    samples = sampler.sample(None)  # Grid determines its own size

    typer.echo(f"Generated {len(samples)} grid points for {len(param_specs)} parameters")

    # Create tasks
    tasks = []
    base_seed = seed or 42
    for i, params in enumerate(samples):
        task = SimTask.from_components(
            import_path=model_class.split(":")[0],
            scenario=scenario,
            bundle_ref=bundle_ref,
            params=params,
            seed=base_seed + i,
            outputs=model_data.get("outputs", [])
        )
        tasks.append(task)

    # Create batch
    batch_id = str(uuid.uuid4())[:8]
    batch = SimBatch(
        batch_id=f"grid-{batch_id}",
        tasks=tasks,
        sampling_method="grid",
        metadata={
            "grid_points": grid_points,
            "total_samples": len(samples),
            "model_class": model_class,
            "scenario": scenario
        }
    )

    # Create job
    job_id = str(uuid.uuid4())[:8]
    job = SimJob(
        job_id=f"job-{job_id}",
        batches=[batch],
        bundle_ref=bundle_ref
    )

    # Write output
    output_path = Path(output)
    with open(output_path, "w") as f:
        job_dict = _job_to_dict(job)
        json.dump(job_dict, f, indent=2)

    typer.echo(f"✓ Generated SimJob with {len(samples)} tasks")
    typer.echo(f"  Job ID: {job.job_id}")
    typer.echo(f"  Output: {output_path}")


def _job_to_dict(job: SimJob) -> dict:
    """Convert SimJob to dict, handling nested dataclasses and numpy types."""
    import numpy as np

    def convert_value(v):
        """Convert numpy types to Python types."""
        if isinstance(v, (np.integer, np.floating)):
            return v.item()
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v

    return {
        "job_id": job.job_id,
        "bundle_ref": job.bundle_ref,
        "priority": job.priority,
        "resource_requirements": job.resource_requirements,
        "batches": [
            {
                "batch_id": batch.batch_id,
                "sampling_method": batch.sampling_method,
                "metadata": batch.metadata,
                "tasks": [
                    {
                        "entrypoint": str(task.entrypoint),
                        "bundle_ref": task.bundle_ref,
                        "params": {
                            "param_id": task.params.param_id,
                            "values": {k: convert_value(v) for k, v in task.params.params.items()},
                        },
                        "seed": task.seed,
                        "outputs": list(task.outputs) if task.outputs else None,
                        "config": dict(task.config) if task.config else None
                    }
                    for task in batch.tasks
                ]
            }
            for batch in job.batches
        ]
    }