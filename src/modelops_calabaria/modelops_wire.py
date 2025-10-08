"""ModelOps wire function bridge for Calabaria models.

This module provides the wire function entry point that ModelOps expects,
bridging to Calabaria's wire_loader system for actual model execution.

The wire function is discovered by ModelOps via the 'modelops.wire' entry point
and handles model execution within bundles.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
import yaml

# Use standardized registry location from contracts
from modelops_contracts import REGISTRY_PATH

from .wire_loader import entry_from_manifest, make_wire

logger = logging.getLogger(__name__)


def wire_function(entrypoint: str, params: Dict[str, Any], seed: int) -> Dict[str, bytes]:
    """Execute a Calabaria model via the wire protocol.

    This is the main entry point that ModelOps calls to execute models.
    It bridges ModelOps' expectations to Calabaria's wire system.

    Args:
        entrypoint: Model entrypoint like "models.seir:StochasticSEIR"
        params: Model parameters as a dictionary
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping artifact names to serialized bytes
        Typically includes "table" with simulation results
    """
    logger.info(f"Wire function called: entrypoint={entrypoint}, seed={seed}")

    # Use standardized registry location
    registry_path = Path(REGISTRY_PATH)

    if not registry_path.exists():
        # If no registry, return minimal output
        logger.warning(f"No registry found at {REGISTRY_PATH}")
        return {
            "table": b"",  # Empty table
            "metadata": json.dumps({
                "error": f"No registry found at {REGISTRY_PATH}",
                "entrypoint": entrypoint,
                "seed": seed
            }).encode()
        }

    # Load registry (this contains model metadata from register-model)
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    # Convert entrypoint format if needed
    # ModelOps might send "models.seir/baseline" but registry has "models.seir:StochasticSEIR"
    if "/" in entrypoint:
        # Extract just the module/class part, ignore scenario for now
        entrypoint = entrypoint.split("/")[0]

    # Find matching model in registry
    model_key = None
    full_entrypoint = None
    for key, model_data in registry.get("models", {}).items():
        registry_entrypoint = model_data.get("entrypoint", "")

        # Exact match
        if registry_entrypoint == entrypoint:
            model_key = key
            full_entrypoint = registry_entrypoint
            break

        # If entrypoint is just module name (e.g., "models.seir"),
        # match against registry entries that start with it
        if ":" not in entrypoint and registry_entrypoint.startswith(entrypoint + ":"):
            model_key = key
            full_entrypoint = registry_entrypoint
            break

    if not model_key:
        logger.error(f"Model {entrypoint} not found in registry")
        return {
            "table": b"",
            "metadata": json.dumps({
                "error": f"Model {entrypoint} not found in registry",
                "available": list(registry.get("models", {}).keys())
            }).encode()
        }

    # Use the full entrypoint (with class name) for the manifest
    if not full_entrypoint:
        full_entrypoint = registry["models"][model_key].get("entrypoint", entrypoint)

    # Extract parameter specs from the model class
    param_specs = []
    try:
        # Import the model to get its parameter space
        module_name, class_name = full_entrypoint.split(":")
        import importlib
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        # Get parameter space if available
        if hasattr(model_class, 'parameter_space'):
            space = model_class.parameter_space()
            # Convert to serialized format for manifest
            for spec in space.specs:
                param_specs.append({
                    "name": spec.name,
                    "min": spec.min,
                    "max": spec.max,
                    "kind": spec.kind,
                    "doc": spec.doc
                })
    except Exception as e:
        logger.warning(f"Could not extract parameter specs from {full_entrypoint}: {e}")

    # Create a minimal manifest for wire_loader
    manifest = {
        "models": {
            full_entrypoint: {
                "model_digest": registry["models"][model_key].get("model_digest", "unknown"),
                "param_specs": param_specs,
                "scenarios": registry["models"][model_key].get("scenarios", []),
                "outputs": registry["models"][model_key].get("outputs", []),
            }
        }
    }

    try:
        # Create entry record from manifest using full entrypoint
        entry = entry_from_manifest(full_entrypoint, manifest)

        # Create wire function for this model
        wire_fn = make_wire(entry)

        # Execute the wire function with keyword arguments
        result = wire_fn(
            params_M=params,      # M-space parameters
            seed=seed,           # Random seed
            scenario_stack=(),   # Empty scenario stack for now
            outputs=None         # Default outputs
        )

        # Convert WireResponse to Dict[str, bytes]
        # Note: result.outputs already contains bytes (Arrow IPC format)
        outputs = {}
        for name, table_bytes in result.outputs.items():
            # Already serialized as bytes, no need to call write_parquet
            outputs[name] = table_bytes

        # Add metadata
        outputs["metadata"] = json.dumps({
            "entrypoint": entrypoint,
            "seed": seed,
            "params": params,
            "model_digest": entry.model_digest
        }).encode()

        logger.info(f"Wire function returning {len(outputs)} artifacts")
        return outputs

    except Exception as e:
        logger.error(f"Wire execution failed: {e}", exc_info=True)
        # Return error information
        return {
            "table": b"",
            "metadata": json.dumps({
                "error": str(e),
                "entrypoint": entrypoint,
                "seed": seed
            }).encode()
        }