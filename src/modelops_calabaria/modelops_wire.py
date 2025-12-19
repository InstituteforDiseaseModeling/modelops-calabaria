"""ModelOps wire function bridge for Calabaria models.

This module provides the wire function entry point that ModelOps expects,
bridging to Calabaria's wire_loader system for actual model execution.

The wire function is discovered by ModelOps via the 'modelops.wire' entry point
and handles model execution within bundles.
"""

import logging
from pathlib import Path
from typing import Dict, Any
import yaml

# Use standardized registry location from contracts
from modelops_contracts import REGISTRY_PATH

from .wire_loader import entry_from_manifest, make_wire

logger = logging.getLogger(__name__)


def _json_dumps(data: Any) -> bytes:
    """Helper to dump JSON with local import to avoid Python 3.13 scope issues."""
    import json as json_module
    return json_module.dumps(data).encode('utf-8')

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
            "metadata": _json_dumps({
                "error": f"No registry found at {REGISTRY_PATH}",
                "entrypoint": entrypoint,
                "seed": seed
            })
        }

    # Load registry (this contains model metadata from register-model)
    with open(registry_path) as f:
        registry = yaml.safe_load(f)

    # Convert entrypoint format if needed
    # ModelOps might send "models.seir/baseline" but registry has "models.seir:StochasticSEIR"
    if "/" in entrypoint:
        # Extract just the module/class part, ignore scenario for now
        entrypoint = entrypoint.split("/")[0]

    # Check if this is a target entrypoint
    is_target = False
    target_key = None

    # First check targets
    for key, target_data in registry.get("targets", {}).items():
        target_entrypoint = target_data.get("entrypoint", "")
        if target_entrypoint == entrypoint:
            is_target = True
            target_key = key
            full_entrypoint = target_entrypoint
            break

    # If not a target, find matching model in registry
    model_key = None
    if not is_target:
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

    if not model_key and not target_key:
        logger.error(f"Entrypoint {entrypoint} not found in registry")
        return {
            "table": b"",
            "metadata": _json_dumps({
                "error": f"Entrypoint {entrypoint} not found in registry",
                "available_models": list(registry.get("models", {}).keys()),
                "available_targets": list(registry.get("targets", {}).keys())
            })
        }

    # Handle targets differently - they just return Target objects
    if is_target:
        try:
            # Import and execute the target function
            module_name, func_name = full_entrypoint.split(":")
            import importlib
            module = importlib.import_module(module_name)
            target_func = getattr(module, func_name)

            # Execute target function (no params needed)
            target = target_func()

            # Convert Target to a simple response
            from io import BytesIO

            # Return target metadata and data
            buffer = BytesIO()
            target.data.write_ipc(buffer)
            return {
                "target_data": buffer.getvalue(),
                "metadata": _json_dumps({
                    "type": "target",
                    "entrypoint": entrypoint,
                    "model_output": target.model_output,
                    "alignment": type(target.alignment).__name__,
                    "evaluation": type(target.evaluation).__name__,
                    "weight": target.weight
                })
            }
        except Exception as e:
            logger.error(f"Target execution failed: {e}", exc_info=True)
            return {
                "table": b"",
                "metadata": _json_dumps({
                    "error": str(e),
                    "type": "target",
                    "entrypoint": entrypoint
                })
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
        if hasattr(model_class, 'PARAMS'):
            space = model_class.PARAMS
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

    # Create entry record from manifest using full entrypoint
    entry = entry_from_manifest(full_entrypoint, manifest)

    # Create wire function for this model (let ImportError/TypeError propagate)
    wire_fn = make_wire(entry)

    # Execute the wire function
    result = wire_fn(
        params_M=params,
        seed=seed,
        scenario_stack=(),
        outputs=None,
    )

    # Convert WireResponse to Dict[str, bytes]
    outputs = {name: table_bytes for name, table_bytes in result.outputs.items()}
    outputs["metadata"] = _json_dumps({
        "entrypoint": entrypoint,
        "seed": seed,
        "params": params,
        "model_digest": entry.model_digest,
    })
    logger.info(f"Wire function returning {len(outputs)} artifacts")
    return outputs
