"""Configuration handling for Calabria CLI.

Reads and writes pyproject.toml configuration for model exports.
"""

import glob
from pathlib import Path
from typing import Dict, List, Any, Optional
import tomllib
import toml


def read_pyproject() -> Dict[str, Any]:
    """Read pyproject.toml configuration.

    Returns:
        The [tool.calabaria] section, or empty dict if not found

    Raises:
        FileNotFoundError: If pyproject.toml doesn't exist
        tomllib.TOMLDecodeError: If TOML is malformed
    """
    root = Path.cwd()
    pyproject_path = root / "pyproject.toml"

    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found in current directory")

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    return data.get("tool", {}).get("calabaria", {})


def write_model_config(class_path: str, files: List[str]) -> None:
    """Add or update a model configuration in pyproject.toml.

    Args:
        class_path: Module path and class (e.g., "models.sir:SIRModel")
        files: List of glob patterns for model files
    """
    root = Path.cwd()
    pyproject_path = root / "pyproject.toml"

    # Read existing config or create new
    if pyproject_path.exists():
        with open(pyproject_path, "r", encoding="utf-8") as f:
            data = toml.load(f)
    else:
        data = {}

    # Ensure structure exists
    if "tool" not in data:
        data["tool"] = {}
    if "calabaria" not in data["tool"]:
        data["tool"]["calabaria"] = {
            "schema": 1,
            "abi": "model-entrypoint@1"
        }

    # Add model entry
    if "model" not in data["tool"]["calabaria"]:
        data["tool"]["calabaria"]["model"] = []

    models = data["tool"]["calabaria"]["model"]

    # Find existing model or add new
    existing_model = None
    for i, model in enumerate(models):
        if model.get("class") == class_path:
            existing_model = i
            break

    model_entry = {
        "class": class_path,
        "files": files
    }

    if existing_model is not None:
        models[existing_model] = model_entry
    else:
        models.append(model_entry)

    # Write back to file
    with open(pyproject_path, "w", encoding="utf-8") as f:
        toml.dump(data, f)


def resolve_file_patterns(patterns: List[str]) -> List[Path]:
    """Resolve glob patterns to actual Python files.

    Args:
        patterns: List of glob patterns (e.g., ["src/models/sir/**", "src/common/*.py"])

    Returns:
        List of resolved Python file paths, sorted deterministically

    Example:
        >>> patterns = ["src/models/**/*.py"]
        >>> files = resolve_file_patterns(patterns)
        >>> print([str(f) for f in files])
        ['src/models/sir/__init__.py', 'src/models/sir/core.py', ...]
    """
    root = Path.cwd()
    resolved = set()

    for pattern in patterns:
        # Make pattern relative to project root
        pattern_path = root / pattern if not Path(pattern).is_absolute() else Path(pattern)

        # Use glob to expand pattern
        matches = glob.glob(str(pattern_path), recursive=True)

        for match in matches:
            path = Path(match).resolve()
            # Only include Python files
            if path.is_file() and path.suffix == '.py':
                # Store relative to project root
                try:
                    relative_path = path.relative_to(root)
                    resolved.add(relative_path)
                except ValueError:
                    # File is outside project root, skip
                    continue

    return sorted(resolved)


def get_uv_lock_hash() -> Optional[str]:
    """Get hash of uv.lock file if it exists.

    Returns:
        SHA256 hash of uv.lock, or None if file doesn't exist
    """
    root = Path.cwd()
    uv_lock = root / "uv.lock"

    if not uv_lock.exists():
        return None

    from .hashing import sha256_bytes
    return sha256_bytes(uv_lock.read_bytes())


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate calabaria configuration.

    Args:
        config: The [tool.calabaria] configuration

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required fields
    if "schema" not in config:
        errors.append("Missing required field: schema")
    elif config["schema"] != 1:
        errors.append(f"Unsupported schema version: {config['schema']}")

    if "abi" not in config:
        errors.append("Missing required field: abi")

    # Validate models
    models = config.get("model", [])
    if not isinstance(models, list):
        errors.append("'model' must be a list")
        return errors

    seen_ids = set()
    for i, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"model[{i}] must be a dictionary")
            continue

        # Check required model fields
        for field in ["class", "files"]:
            if field not in model:
                errors.append(f"model[{i}] missing required field: {field}")

        # Check for duplicate class paths
        class_path = model.get("class")
        if class_path in seen_ids:
            errors.append(f"Duplicate model class: {class_path}")
        seen_ids.add(class_path)

        # Validate class format
        class_path = model.get("class", "")
        if ":" not in class_path:
            errors.append(f"model[{i}].class must be in format 'module:Class', got: {class_path}")

        # Validate files is a list
        files = model.get("files")
        if not isinstance(files, list):
            errors.append(f"model[{i}].files must be a list")

    return errors