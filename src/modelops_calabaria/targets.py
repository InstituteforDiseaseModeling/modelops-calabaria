"""
Target wire protocol for ModelOps-Calabaria.

Targets are defined in Python files with entrypoints and registered via modelops-bundle.
The wire function evaluates targets against simulation outputs and returns loss.
"""

import importlib
import io
from typing import Any, Dict

import polars as pl

from .core.target import Targets


def wire_target_function(
    entrypoint: str,
    sim_outputs: Dict[str, bytes]
) -> Dict[str, Any]:
    """
    Wire function for evaluating targets via entrypoint.

    Parameters
    ----------
    entrypoint : str
        Target entrypoint in format "module.path:function_name"
        e.g., "targets.prevalence:get_targets"
    sim_outputs : Dict[str, bytes]
        Simulation outputs as Arrow IPC bytes, keyed by output name

    Returns
    -------
    Dict[str, Any]
        Evaluation results including:
        - "total_loss": float
        - "target_losses": Dict[str, float] per-target losses
    """
    # Parse entrypoint
    if ":" not in entrypoint:
        raise ValueError(f"Invalid entrypoint format: {entrypoint}. Expected 'module:function'")
    module_path, function_name = entrypoint.split(":", 1)

    # Import and get the target function
    module = importlib.import_module(module_path)
    target_func = getattr(module, function_name)

    # Get the targets
    targets_result = target_func()

    # Ensure we have a Targets object
    # Handle different return types:
    # - Single Target from decorated function
    # - Targets collection from old-style get_targets()
    # - List of Target objects
    from .core.target import Target

    if isinstance(targets_result, Target):
        targets = Targets(targets=[targets_result])
    elif isinstance(targets_result, list):
        targets = Targets(targets=targets_result)
    else:
        targets = targets_result

    # Convert simulation outputs from bytes to DataFrames
    sim_output_dfs = {}
    for name, output_bytes in sim_outputs.items():
        if name != "metadata":  # Skip metadata
            sim_output_dfs[name] = pl.read_ipc(io.BytesIO(output_bytes))

    # Evaluate targets (single replicate for now, but evaluate_all expects a sequence)
    evaluations = targets.evaluate_all([sim_output_dfs])

    # Compute losses
    total_loss = 0.0
    target_losses = {}

    for eval_result in evaluations:
        if eval_result.weighted_loss is not None:
            total_loss += eval_result.weighted_loss
            target_losses[eval_result.name] = eval_result.loss

    return {
        "total_loss": total_loss,
        "target_losses": target_losses
    }