"""Import boundary verification for models.

Verifies that models only import code from their declared dependencies
using subprocess isolation to track imports accurately.
"""

import json
import os
import pathlib
import subprocess
import sys
from typing import Set, Dict, List
import logging

from .config import resolve_file_patterns

logger = logging.getLogger(__name__)


def verify_model(class_path: str, allowed_files: Set[str]) -> Dict:
    """
    Verify that a model only imports code from its declared file dependencies.

    This runs the model in a clean subprocess to track all imports without
    contaminating the current Python environment.

    Args:
        class_path: Module path like "models.sir:SIRModel"
        allowed_files: Set of relative paths the model is allowed to import

    Returns:
        Dict with:
        - loaded: All project files that were imported
        - unexpected: Files imported but not in allowed_files
        - covered: Allowed files that were actually used
        - unused: Allowed files that were never imported
        - ok: Boolean, True if no unexpected imports
        - external_deps: Non-project dependencies loaded
        - model_info: Information about the model
        - error: Error message if verification failed
    """

    # Build the verification script as a string
    # We use an f-string to inject the specific model class path
    verification_script = f"""
import sys
import json
import importlib
import traceback

# This script runs in a fresh Python process with clean sys.modules
# So we can track exactly what gets imported by this model alone

try:
    # Parse the class path (injected via f-string)
    module_path, class_name = "{class_path}".split(":")

    # Import the module containing the model
    module = importlib.import_module(module_path)

    # Get the model class
    model_class = getattr(module, class_name)

    # Try to get the parameter space
    # Models might define it as class attribute or method
    if hasattr(model_class, 'SPACE'):
        space = model_class.SPACE
    elif hasattr(model_class, 'parameter_space'):
        space = model_class.parameter_space()
    else:
        # Try instantiating without space (model might have defaults)
        space = None

    # Create an instance to trigger all lazy imports
    if space is not None:
        model_instance = model_class(space=space)
    else:
        model_instance = model_class()

    # Seal the model (this triggers registration of scenarios/outputs)
    # This ensures all model code paths are exercised
    if hasattr(model_instance, '_seal'):
        model_instance._seal()

    # Now collect all modules that were loaded
    # These are ALL the dependencies this model needs
    loaded_files = []
    for module_name, module_obj in sys.modules.items():
        # Skip built-in modules (no __file__ attribute)
        if hasattr(module_obj, '__file__') and module_obj.__file__:
            file_path = module_obj.__file__
            # Only track .py files (skip .so, .pyd compiled extensions)
            if file_path.endswith('.py'):
                loaded_files.append(file_path)

    # Success! Print as JSON for parent process
    result = {{
        "success": True,
        "loaded_files": loaded_files,
        "model_info": {{
            "module": module_path,
            "class": class_name,
            "has_space": space is not None
        }}
    }}
    print(json.dumps(result))

except Exception as e:
    # If anything goes wrong, report the error
    result = {{
        "success": False,
        "error": str(e),
        "traceback": traceback.format_exc()
    }}
    print(json.dumps(result))
    sys.exit(1)
"""

    # Run the verification script in a subprocess
    # This gives us a clean, isolated Python environment
    try:
        # Restricted environment for security while preserving essential paths
        restricted_env = {
            "PATH": os.environ.get("PATH", ""),
            "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
            "PYTHONNOUSERSITE": "1",
            "CALABARIA_VERIFY": "1"
        }

        result = subprocess.run(
            [sys.executable, "-c", verification_script],
            capture_output=True,
            text=True,
            timeout=30,  # Prevent hanging on bad imports
            env=restricted_env,
            cwd=str(pathlib.Path.cwd())  # Preserve working directory
        )
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "error": f"Timeout verifying {class_path}",
            "traceback": "Verification timed out after 30 seconds",
            "loaded": [],
            "unexpected": []
        }

    # Parse the output
    if result.returncode != 0:
        # Script failed - try to parse error JSON
        try:
            error_data = json.loads(result.stdout)
            return {
                "ok": False,
                "error": error_data.get("error", "Unknown error"),
                "traceback": error_data.get("traceback", result.stderr),
                "loaded": [],
                "unexpected": []
            }
        except json.JSONDecodeError:
            # Couldn't even parse output - serious failure
            return {
                "ok": False,
                "error": f"Failed to import {class_path}",
                "traceback": result.stderr or result.stdout,
                "loaded": [],
                "unexpected": []
            }

    # Parse successful output
    try:
        output_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "error": f"Invalid JSON output: {e}",
            "traceback": result.stdout,
            "loaded": [],
            "unexpected": []
        }

    # Get the current working directory for relative path calculation
    project_root = pathlib.Path.cwd().resolve()

    # Process loaded files
    loaded_files = output_data.get("loaded_files", [])

    # Filter to only project-local files (not stdlib or site-packages)
    project_files = []
    external_deps = []

    for file_path in loaded_files:
        file_path = pathlib.Path(file_path).resolve()

        # Check if file is within project
        try:
            relative_path = file_path.relative_to(project_root)
            project_files.append(str(relative_path))
        except ValueError:
            # File is outside project (stdlib or installed package)
            # This is fine - we only track project-internal dependencies
            external_deps.append(str(file_path))

    # Convert allowed files to set for comparison
    allowed_set = set(allowed_files)
    project_set = set(project_files)

    # Find unexpected imports (project files not in allowed list)
    unexpected = sorted(project_set - allowed_set)

    # Find covered files (which of the allowed files were actually used)
    covered = sorted(project_set & allowed_set)

    # Find unused files (allowed but never imported)
    # This might indicate dead code or overly broad file patterns
    unused = sorted(allowed_set - project_set)

    return {
        "ok": len(unexpected) == 0,
        "loaded": sorted(project_files),
        "unexpected": unexpected,
        "covered": covered,
        "unused": unused,
        "external_deps": sorted(set(external_deps)),  # Dedup and sort
        "model_info": output_data.get("model_info", {})
    }


def verify_all_models(config: Dict) -> Dict[str, Dict]:
    """
    Verify all models defined in the configuration.

    Args:
        config: The parsed pyproject.toml [tool.calabaria] section

    Returns:
        Dict mapping model class path to verification results
    """
    results = {}

    for model_config in config.get("model", []):
        class_path = model_config["class"]
        file_patterns = model_config["files"]

        # Resolve file patterns to actual files
        allowed_files = {str(f) for f in resolve_file_patterns(file_patterns)}

        logger.info(f"Verifying {class_path}...")
        result = verify_model(class_path, allowed_files)
        results[class_path] = result

        if result["ok"]:
            logger.info(f"  ✓ OK - {len(result['covered'])} files used")
            if result.get("unused"):
                logger.warning(f"  ⚠ Warning: {len(result['unused'])} allowed files were not imported")
                for unused_file in result["unused"][:3]:  # Show first 3
                    logger.warning(f"    - {unused_file}")
        else:
            logger.error(f"  ✗ FAIL - {len(result['unexpected'])} unexpected imports:")
            for file in result['unexpected'][:5]:  # Show first 5
                logger.error(f"    - {file}")
            if len(result['unexpected']) > 5:
                logger.error(f"    ... and {len(result['unexpected']) - 5} more")

            if result.get("error"):
                logger.error(f"  Error: {result['error']}")

    return results


def print_verification_summary(results: Dict[str, Dict]) -> None:
    """Print a summary of verification results.

    Args:
        results: Results from verify_all_models()
    """
    total = len(results)
    passed = sum(1 for r in results.values() if r["ok"])
    failed = total - passed

    print(f"\nVerification Summary:")
    print(f"  Total models: {total}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed > 0:
        print(f"\nFailed models:")
        for class_path, result in results.items():
            if not result["ok"]:
                print(f"  {class_path}: {len(result['unexpected'])} unexpected imports")

    # Show overall statistics
    total_covered = sum(len(r.get("covered", [])) for r in results.values())
    total_unused = sum(len(r.get("unused", [])) for r in results.values())

    print(f"\nFile Usage:")
    print(f"  Files actually used: {total_covered}")
    if total_unused > 0:
        print(f"  Files allowed but unused: {total_unused}")


def suggest_file_patterns(unexpected_files: List[str]) -> List[str]:
    """Suggest file patterns to include unexpected imports.

    Args:
        unexpected_files: List of unexpected file paths

    Returns:
        List of suggested glob patterns
    """
    suggestions = []

    # Group files by directory
    directories = {}
    for file_path in unexpected_files:
        path = pathlib.Path(file_path)
        dir_path = path.parent
        if dir_path not in directories:
            directories[dir_path] = []
        directories[dir_path].append(path)

    # Suggest patterns for each directory
    for dir_path, files in directories.items():
        if len(files) == 1:
            # Single file - suggest exact path
            suggestions.append(str(files[0]))
        else:
            # Multiple files - suggest directory pattern
            suggestions.append(f"{dir_path}/**")

    return sorted(set(suggestions))  # Remove duplicates and sort