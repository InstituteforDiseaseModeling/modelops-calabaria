"""Stateless wire loader for creating wire functions from manifest data.

This module provides a pure functional approach to creating wire functions
without global state or registration side effects. Wire functions are created
on-demand from manifest data.

Key principles:
- No global state or registry
- No side effects at import time
- Manifest is the single source of truth
- Pure functions that can be cached if needed
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import io
from types import MappingProxyType

import polars as pl

from .parameters import ParameterSet, ParameterSpace, ParameterSpec, Scalar
from .scenarios import ScenarioSpec
from .wire_protocol import WireResponse, WireABI, SerializedParameterSpec


@dataclass(frozen=True)
class EntryRecord:
    """Immutable record describing a model's wire protocol interface.

    This is a pure value object with no side effects or global registration.
    Created from manifest data on demand.
    """
    class_path: str                                  # "pkg.mod:Class"
    model_digest: str                               # "sha256:..."
    abi_version: WireABI
    param_specs: Tuple[SerializedParameterSpec, ...]
    scenarios: Tuple[str, ...]
    outputs: Tuple[str, ...]


def entry_from_manifest(class_path: str, manifest: Dict) -> EntryRecord:
    """Create EntryRecord from manifest data.

    Args:
        class_path: Model class path in manifest
        manifest: Complete manifest dictionary

    Returns:
        EntryRecord for the model

    Raises:
        KeyError: If class_path not found in manifest
        ValueError: If manifest data is invalid
    """
    if class_path not in manifest["models"]:
        available = sorted(manifest["models"].keys())
        raise KeyError(f"Model {class_path} not found. Available: {available}")

    model_data = manifest["models"][class_path]

    # Convert parameter specs
    param_specs = tuple(
        SerializedParameterSpec(**spec_data)
        for spec_data in model_data["param_specs"]
    )

    return EntryRecord(
        class_path=class_path,
        model_digest=model_data["model_digest"],
        abi_version=WireABI.V1,  # Currently only V1 supported
        param_specs=param_specs,
        scenarios=tuple(model_data["scenarios"]),
        outputs=tuple(model_data["outputs"]),
    )


def make_wire(entry: EntryRecord) -> Callable:
    """Create a wire function from an EntryRecord.

    The wire function follows the V1 protocol:
    - Takes complete M-space parameters as dict
    - Applies scenario stack with last-write-wins semantics
    - Returns WireResponse with serialized outputs

    Args:
        entry: EntryRecord describing the model

    Returns:
        Wire function implementing the V1 protocol

    Raises:
        ImportError: If model class cannot be imported
        ValueError: If ABI version is unsupported
    """
    if entry.abi_version != WireABI.V1:
        raise ValueError(f"Unsupported ABI version: {entry.abi_version}")

    # Import model class using load_symbol for better path handling
    from .utils import load_symbol

    try:
        model_class = load_symbol(entry.class_path)
    except (ModuleNotFoundError, AttributeError, ValueError) as e:
        raise ImportError(f"Cannot import {entry.class_path}: {e}")

    # Validate it's a BaseModel subclass
    from .base_model import BaseModel
    if not issubclass(model_class, BaseModel):
        raise TypeError(f"{entry.class_path} is not a BaseModel subclass")

    # Reconstruct parameter space from serialized specs
    param_specs = [
        ParameterSpec(
            name=spec.name,
            min=int(spec.min) if spec.kind == "int" else spec.min,
            max=int(spec.max) if spec.kind == "int" else spec.max,
            kind=spec.kind,
            doc=spec.doc
        )
        for spec in entry.param_specs
    ]
    space = ParameterSpace(param_specs)

    def wire_v1(
        *,
        params_M: Dict[str, Scalar],
        seed: int,
        scenario_stack: Tuple[str, ...] = ("baseline",),
        outputs: Optional[List[str]] = None
    ) -> WireResponse:
        """V1 wire function implementation.

        Args:
            params_M: Complete M-space parameters
            seed: Random seed for reproducibility
            scenario_stack: Ordered scenarios to apply (LWW)
            outputs: Optional subset of outputs (None = all)

        Returns:
            WireResponse with serialized outputs and provenance
        """
        # Create fresh model instance (stateless execution)
        model = model_class(space)
        model._seal()  # Ensure registries are frozen

        # Validate all scenarios exist
        for scenario_name in scenario_stack:
            if scenario_name not in model._scenarios:
                available = sorted(model._scenarios.keys())
                raise ValueError(
                    f"Unknown scenario: {scenario_name}. "
                    f"Available: {available}"
                )

        # Convert to ParameterSet (validates completeness)
        pset = ParameterSet(space, params_M)

        # Apply scenario stack with LWW semantics
        config = dict(model.base_config)
        applied_scenarios = []

        for scenario_name in scenario_stack:
            if scenario_name in model._scenarios:
                spec = model._scenarios[scenario_name]
                pset, config = spec.apply(pset, MappingProxyType(config))
                applied_scenarios.append(scenario_name)

        # Run simulation pipeline
        state = model.build_sim(pset, MappingProxyType(config))
        raw = model.run_sim(state, seed)

        # Extract all outputs
        all_outputs = model.extract_outputs(raw, seed)

        # Filter outputs if requested
        if outputs is not None:
            unknown = set(outputs) - set(all_outputs.keys())
            if unknown:
                available = sorted(all_outputs.keys())
                raise ValueError(
                    f"Unknown outputs: {sorted(unknown)}. "
                    f"Available: {available}"
                )
            filtered_outputs = {k: v for k, v in all_outputs.items() if k in outputs}
        else:
            filtered_outputs = all_outputs

        # Serialize outputs to IPC format
        serialized_outputs = {}
        for name, df in filtered_outputs.items():
            buffer = io.BytesIO()
            df.write_ipc(buffer)
            serialized_outputs[name] = buffer.getvalue()

        # Build provenance
        provenance = {
            "model_digest": entry.model_digest,
            "class_path": entry.class_path,
            "abi": str(entry.abi_version),
            "params_M": dict(pset.values),
            "scenario_stack": list(scenario_stack),
            "applied_scenarios": applied_scenarios,
            "seed": seed,
        }

        return WireResponse(outputs=serialized_outputs, provenance=provenance)

    return wire_v1


def make_wire_from_manifest(class_path: str, manifest: Dict) -> Callable:
    """Convenience function to create wire directly from manifest.

    Args:
        class_path: Model class path in manifest
        manifest: Complete manifest dictionary

    Returns:
        Wire function for the model
    """
    entry = entry_from_manifest(class_path, manifest)
    return make_wire(entry)
