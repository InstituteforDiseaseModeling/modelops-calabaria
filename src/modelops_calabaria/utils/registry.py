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

    target_sets = _load_target_sets(registry, Path(registry_path))

    def _extract_targets(entry):
        if hasattr(entry, "targets"):
            return list(entry.targets)
        if isinstance(entry, dict):
            return list(entry.get("targets", []))
        return []

    selected_ids: List[str]
    if target_set:
        target_set_obj = target_sets.get(target_set)
        if not target_set_obj:
            available = ", ".join(sorted(target_sets.keys()))
            suffix = f" Available sets: {available}" if available else ""
            raise ValueError(f"Target set '{target_set}' not found in {registry_path}.{suffix}")
        selected_ids = _extract_targets(target_set_obj)
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


def _load_target_sets(registry: BundleRegistry, registry_path: Path) -> dict:
    """Load target sets, supporting legacy registries and missing PyYAML."""
    target_sets = getattr(registry, "target_sets", {}) or {}
    if target_sets:
        return target_sets

    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        yaml = None

    try:
        raw_text = registry_path.read_text()
    except OSError:
        return {}

    if yaml is not None:
        try:
            raw = yaml.safe_load(raw_text) or {}
            return raw.get("target_sets", {}) or {}
        except Exception:
            pass

    return _parse_target_sets_block(raw_text)


def _parse_target_sets_block(text: str) -> dict:
    """Very small YAML subset parser for the target_sets block."""
    lines = text.splitlines()
    target_sets: dict[str, dict] = {}

    def leading_spaces(s: str) -> int:
        return len(s) - len(s.lstrip(" "))

    i = 0
    # Find the target_sets: section
    while i < len(lines) and lines[i].strip() != "target_sets:":
        i += 1
    if i == len(lines):
        return {}
    i += 1

    current: Optional[str] = None
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        indent = leading_spaces(line)
        if indent == 0:
            break
        if indent == 2 and stripped.endswith(":"):
            current = stripped[:-1].strip()
            target_sets[current] = {"targets": [], "weights": {}}
            i += 1
            continue
        if current is None:
            i += 1
            continue

        if indent == 4 and stripped.startswith("targets:"):
            i += 1
            while i < len(lines):
                entry_line = lines[i]
                entry_stripped = entry_line.strip()
                entry_indent = leading_spaces(entry_line)
                if not entry_stripped:
                    i += 1
                    continue
                if entry_indent <= 4:
                    break
                if entry_stripped.startswith("-"):
                    value = entry_stripped[1:].strip().strip("'\"")
                    if value:
                        target_sets[current]["targets"].append(value)
                    i += 1
                    continue
                break
            continue

        if indent == 4 and stripped.startswith("weights:"):
            i += 1
            while i < len(lines):
                weight_line = lines[i]
                weight_stripped = weight_line.strip()
                weight_indent = leading_spaces(weight_line)
                if not weight_stripped:
                    i += 1
                    continue
                if weight_indent <= 4:
                    break
                if ":" in weight_stripped:
                    name, raw_value = weight_stripped.split(":", 1)
                    value_text = raw_value.strip().strip("'\"")
                    try:
                        value = float(value_text)
                    except ValueError:
                        value = value_text
                    target_sets[current]["weights"][name.strip()] = value
                i += 1
            continue

        i += 1

    return target_sets
