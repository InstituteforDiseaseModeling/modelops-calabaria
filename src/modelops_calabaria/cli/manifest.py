"""Manifest builder for deterministic model exports.

Generates manifest.json from pyproject.toml configuration by:
1. Reading model configurations
2. Computing file hashes and signatures
3. Introspecting models for metadata
4. Creating deterministic JSON output
"""

import importlib
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

from . import config
from . import hashing

logger = logging.getLogger(__name__)


def build_model_metadata(class_path: str, files: List[Path]) -> Dict[str, Any]:
    """Build metadata for a single model by introspecting the class.

    Args:
        class_path: Module path like "models.sir:SIRModel"
        files: List of files that define this model

    Returns:
        Dictionary with model metadata

    Raises:
        ImportError: If model class cannot be imported
        AttributeError: If class doesn't have required attributes
    """
    module_path, class_name = class_path.split(":")

    try:
        # Import the model class
        module = importlib.import_module(module_path)
        model_class = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Cannot import {class_path}: {e}")

    # Check if it's a BaseModel subclass
    from ..base_model import BaseModel

    # First check if it's actually a class
    if not isinstance(model_class, type):
        raise TypeError(f"{class_path} is not a valid class")

    # Now safe to check subclass relationship
    if not issubclass(model_class, BaseModel):
        raise TypeError(f"{class_path} is not a BaseModel subclass")

    # Get or create parameter space
    if hasattr(model_class, 'SPACE'):
        space = model_class.SPACE
    elif hasattr(model_class, 'parameter_space'):
        space = model_class.parameter_space()
    else:
        raise AttributeError(f"{class_path} must define SPACE or parameter_space()")

    # Create temporary instance to get scenarios and outputs
    temp_instance = model_class(space)
    temp_instance._seal()

    # Extract metadata
    scenarios = sorted(temp_instance._scenarios.keys())
    outputs = sorted(temp_instance._outputs.keys())

    # Convert parameter specs to serializable form
    from ..wire_protocol import SerializedParameterSpec
    param_specs = [
        SerializedParameterSpec.from_spec(spec).to_json()
        for spec in space.specs
    ]

    # Compute file hashes
    file_records = []
    for file_path in sorted(files):
        file_hash = hashing.token_hash(file_path)
        # Handle both absolute and relative paths
        if file_path.is_absolute():
            relative_path = file_path.relative_to(Path.cwd())
        else:
            relative_path = file_path  # Already relative
        file_records.append((str(relative_path), file_hash))

    # Compute signatures
    code_sig_value = hashing.code_sig(file_records)
    space_sig_value = hashing.content_hash(space.to_dict())

    return {
        "class": class_path,
        "files": [{"path": path, "sha256": hash_val} for path, hash_val in file_records],
        "code_sig": code_sig_value,
        "space_sig": space_sig_value,
        "scenarios": scenarios,
        "outputs": outputs,
        "param_specs": param_specs,
    }


def compute_model_digest(model_metadata: Dict[str, Any],
                        abi: str,
                        requires_python: str,
                        uv_lock_sha: str) -> str:
    """Compute deterministic digest for a model.

    Args:
        model_metadata: Model metadata dictionary
        abi: ABI version string
        requires_python: Python version requirement
        uv_lock_sha: Hash of uv.lock file

    Returns:
        Model digest string
    """
    # Combine all components that affect model behavior
    digest_components = [
        model_metadata["code_sig"],
        model_metadata["space_sig"],
        abi,
        requires_python,
        uv_lock_sha
    ]

    digest_payload = "|".join(digest_components)
    return hashing.content_hash(digest_payload)


def build_manifest(check_only: bool = False) -> Tuple[Dict[str, Any], str]:
    """Build manifest.json from pyproject.toml configuration.

    Args:
        check_only: If True, don't write manifest.json, just return it

    Returns:
        Tuple of (manifest_dict, bundle_id)

    Raises:
        FileNotFoundError: If pyproject.toml not found
        ImportError: If any model cannot be imported
        ValueError: If configuration is invalid
    """
    # Read configuration
    try:
        configuration = config.read_pyproject()
    except FileNotFoundError:
        raise FileNotFoundError(
            "pyproject.toml not found. Run this command from your project root."
        )

    if not configuration:
        raise ValueError(
            "No [tool.calabaria] configuration found in pyproject.toml"
        )

    # Validate configuration
    errors = config.validate_config(configuration)
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

    # Extract configuration values
    abi = configuration.get("abi", "model-entrypoint@1")
    requires_python = configuration.get("requires_python", "")
    models_config = configuration.get("model", [])

    # Get uv.lock hash
    uv_lock_sha = config.get_uv_lock_hash() or "sha256:0000000000000000"

    # Build metadata for each model
    models = {}
    for model_config in models_config:
        class_path = model_config["class"]
        file_patterns = model_config["files"]

        logger.info(f"Building metadata for {class_path}")

        try:
            # Resolve file patterns
            files = config.resolve_file_patterns(file_patterns)
            if not files:
                logger.warning(f"No files found for patterns: {file_patterns}")

            # Build model metadata
            metadata = build_model_metadata(class_path, files)

            # Compute model digest
            model_digest = compute_model_digest(
                metadata, abi, requires_python, uv_lock_sha
            )
            metadata["model_digest"] = model_digest

            models[class_path] = metadata

        except Exception as e:
            logger.error(f"Failed to build metadata for {class_path}: {e}")
            raise

    # Build complete manifest
    manifest = {
        "schema": 1,
        "builder": {"name": "calabaria-cli", "version": "0.1.0"},
        "abi": abi,
        "requires_python": requires_python,
        "uv_lock_sha256": uv_lock_sha,
        "models": models
    }

    # Compute bundle ID from all model digests
    bundle_payload = "|".join(
        f"{class_path}:{metadata['model_digest']}"
        for class_path, metadata in sorted(models.items())
    )
    bundle_id = hashing.content_hash(bundle_payload)
    manifest["bundle_id"] = bundle_id

    # Write manifest unless check_only
    if not check_only:
        manifest_path = Path.cwd() / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        logger.info(f"Wrote manifest to {manifest_path}")

    return manifest, bundle_id


def check_manifest_drift() -> bool:
    """Check if manifest generation is deterministic.

    Returns:
        True if two consecutive builds produce identical manifests, False otherwise
    """
    try:
        # Build manifest twice and compare for determinism
        manifest1, _ = build_manifest(check_only=True)
        manifest2, _ = build_manifest(check_only=True)

        # Compare canonical JSON representations
        json1 = hashing.canonical_json(manifest1)
        json2 = hashing.canonical_json(manifest2)

        is_deterministic = json1 == json2

        if is_deterministic:
            logger.info("manifest.json generation is deterministic")
        else:
            logger.info("manifest.json generation is not deterministic")

        return is_deterministic

    except Exception as e:
        logger.error(f"Error checking manifest drift: {e}")
        return False


def print_manifest_summary(manifest: Dict[str, Any]) -> None:
    """Print a summary of the generated manifest.

    Args:
        manifest: The generated manifest dictionary
    """
    print(f"Manifest Summary:")
    print(f"  Schema: {manifest['schema']}")
    print(f"  ABI: {manifest['abi']}")
    print(f"  Bundle ID: {manifest['bundle_id']}")

    models = manifest.get("models", {})
    print(f"  Models: {len(models)}")

    for class_path, model_info in models.items():
        print(f"    {class_path}:")
        print(f"      Files: {len(model_info['files'])}")
        print(f"      Scenarios: {len(model_info['scenarios'])}")
        print(f"      Outputs: {len(model_info['outputs'])}")
        print(f"      Digest: {model_info['model_digest'][:16]}...")

    if manifest.get("requires_python"):
        print(f"  Python: {manifest['requires_python']}")

    print(f"  UV Lock: {manifest['uv_lock_sha256'][:16]}...")
