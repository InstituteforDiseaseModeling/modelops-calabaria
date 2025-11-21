"""Helpers for interacting with the local bundle registry."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from modelops_contracts import BundleRegistry, TargetEntry


def maybe_load_registry(project_root: Optional[str]) -> Tuple[BundleRegistry, Path] | None:
    """Load the bundle registry if it exists."""
    root = Path(project_root or os.getcwd()).resolve()
    registry_path = root / ".modelops-bundle" / "registry.yaml"
    if not registry_path.exists():
        return None
    registry = BundleRegistry.load(registry_path)
    return registry, registry_path


def load_registry(project_root: Optional[str]) -> Tuple[BundleRegistry, Path]:
    """Load the bundle registry or raise ValueError."""
    result = maybe_load_registry(project_root)
    if result is None:
        raise ValueError(
            "Bundle registry not found. Run 'modelops-bundle register-model/target' first "
            "or specify --project-root."
        )
    return result


def resolve_model_identifier(model_arg: str, project_root: Optional[str]) -> Tuple[str, Optional[str]]:
    """Resolve model identifier using the registry if needed.

    Returns:
        Tuple of (entrypoint string, registry_path if resolved via registry)
    """
    if ":" in model_arg:
        return model_arg, None

    result = maybe_load_registry(project_root)
    if not result:
        raise ValueError(
            f"Model '{model_arg}' must be specified as 'module_or_file:ClassName' "
            f"(example: models.sir:StarsimSIR) or match a registered bundle model id."
        )

    registry, registry_path = result
    entry = registry.models.get(model_arg)
    if not entry:
        available = ", ".join(sorted(registry.models.keys()))
        suffix = f" Available models: {available}" if available else ""
        raise ValueError(
            f"Model id '{model_arg}' was not found in {registry_path}.{suffix}"
        )
    return entry.entrypoint, str(registry_path)


def resolve_target_entries(
    target_ids: Iterable[str] | None,
    target_set: Optional[str],
    project_root: Optional[str],
) -> Tuple[List[Tuple[str, TargetEntry]], Optional[str]]:
    """Resolve target IDs using the registry."""
    registry, registry_path = load_registry(project_root)

    selected_ids: List[str]
    if target_set:
        target_set_obj = registry.target_sets.get(target_set)
        if not target_set_obj:
            available = ", ".join(sorted(registry.target_sets.keys()))
            suffix = f" Available sets: {available}" if available else ""
            raise ValueError(f"Target set '{target_set}' not found in {registry_path}.{suffix}")
        selected_ids = list(target_set_obj.targets)
    elif target_ids:
        selected_ids = list(dict.fromkeys(target_ids))
    else:
        selected_ids = sorted(registry.targets.keys())

    if not selected_ids:
        raise ValueError("No targets available in registry; register targets before calibrating.")

    missing = [tid for tid in selected_ids if tid not in registry.targets]
    if missing:
        raise ValueError(f"Unknown target id(s): {', '.join(missing)}")

    entries = [(tid, registry.targets[tid]) for tid in selected_ids]
    return entries, target_set
