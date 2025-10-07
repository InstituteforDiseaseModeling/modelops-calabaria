"""Public API for modelops-calabaria functionality.

This module provides a clean, programmatic interface to all CLI functionality
for use by other packages and applications.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

# Import the actual implementations
from .cli import discover, verify, config
# manifest deprecated - use modelops-bundle register-model instead


class CalabariaCLI:
    """Thin facade over CLI functionality for programmatic use.

    Provides a clean, typed interface to all modelops-calabaria functionality
    that can be used programmatically by other packages.

    Examples:
        >>> cli = CalabariaCLI()
        >>> models = cli.discover()
        >>> manifest = cli.get_manifest()
        >>> bundle_id = cli.get_bundle_id()

        # With different directory
        >>> with CalabariaCLI(Path("examples/epi_models")) as cli:
        ...     results = cli.verify()
        ...     if any(not r["ok"] for r in results.values()):
        ...         print("Verification failed!")
    """

    def __init__(self, root_path: Optional[Path] = None):
        """Initialize with optional root path (defaults to cwd)."""
        self.root_path = root_path or Path.cwd()
        self._original_cwd = None

    def __enter__(self):
        """Context manager to temporarily change directory."""
        self._original_cwd = os.getcwd()
        os.chdir(self.root_path)
        return self

    def __exit__(self, *args):
        """Restore original directory."""
        if self._original_cwd:
            os.chdir(self._original_cwd)

    # Model Discovery
    def discover(self) -> List[Dict[str, Any]]:
        """Discover all BaseModel subclasses in the project.

        Returns:
            List of model metadata dictionaries with keys:
            - full_path: Module path like "models.seir:StochasticSEIR"
            - file_path: Path to the source file
            - line_number: Line where class is defined
            - methods: Dict with 'model_outputs' and 'model_scenarios' lists
        """
        return discover.discover_models()

    def suggest_configs(self, models: Optional[List[Dict]] = None) -> List[Dict[str, Any]]:
        """Suggest configurations for discovered models.

        Args:
            models: Optional list of models from discover(). If None, will discover automatically.

        Returns:
            List of suggested configurations with 'class' and 'files' keys.
        """
        if models is None:
            models = self.discover()
        return discover.suggest_model_config(models)

    # Configuration Management
    def read_config(self) -> Dict[str, Any]:
        """Read the [tool.calabaria] configuration from pyproject.toml.

        Returns:
            Configuration dictionary, or empty dict if no config found.
        """
        return config.read_pyproject()

    def add_model(self, class_path: str, file_patterns: List[str]) -> None:
        """Add a model to the configuration in pyproject.toml.

        Args:
            class_path: Module path like "models.seir:StochasticSEIR"
            file_patterns: List of glob patterns for model files

        Raises:
            ValueError: If class_path format is invalid
        """
        if ":" not in class_path:
            raise ValueError("class_path must be in format 'module:Class'")
        config.write_model_config(class_path, file_patterns)

    def validate_config(self) -> List[str]:
        """Validate configuration and return any errors.

        Returns:
            List of error messages, empty if no errors.
        """
        conf = self.read_config()
        if not conf:
            return ["No [tool.calabaria] configuration found in pyproject.toml"]
        return config.validate_config(conf)

    # Import Boundary Verification
    def verify(self, model_class: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Verify import boundaries for one or all models.

        Args:
            model_class: Optional specific model to verify. If None, verifies all.

        Returns:
            Dictionary mapping model class paths to verification results.
            Each result has keys: ok, loaded, unexpected, covered, unused, etc.

        Raises:
            ValueError: If no configuration found or model not found.
        """
        conf = self.read_config()
        if not conf:
            raise ValueError("No [tool.calabaria] configuration found in pyproject.toml")

        if model_class:
            # Verify single model
            model_configs = [m for m in conf.get("model", [])
                           if m["class"] == model_class]
            if not model_configs:
                raise ValueError(f"Model {model_class} not found in configuration")

            files = config.resolve_file_patterns(model_configs[0]["files"])
            allowed = {str(f) for f in files}
            result = verify.verify_model(model_class, allowed)
            return {model_class: result}
        else:
            # Verify all models
            return verify.verify_all_models(conf)

    def suggest_patterns(self, unexpected_files: List[str]) -> List[str]:
        """Suggest file patterns for unexpected imports.

        Args:
            unexpected_files: List of file paths that were unexpectedly imported

        Returns:
            List of suggested glob patterns to include those files
        """
        return verify.suggest_file_patterns(unexpected_files)

    # Manifest Building
    def build_manifest(self, save: bool = True) -> Tuple[Dict[str, Any], str]:
        """Build manifest and optionally save to file.

        Args:
            save: Whether to save manifest.json file (default: True)

        Returns:
            Tuple of (manifest_dict, bundle_id)

        Raises:
            FileNotFoundError: If pyproject.toml not found
            ValueError: If configuration is invalid
            ImportError: If any model cannot be imported
        """
        raise NotImplementedError("manifest deprecated - use modelops-bundle register-model") # return manifest.build_manifest(check_only=not save)

    def get_manifest(self) -> Dict[str, Any]:
        """Build and return just the manifest dictionary without saving.

        Returns:
            Manifest dictionary with schema, models, bundle_id, etc.
        """
        manifest_dict, _ = self.build_manifest(save=False)
        return manifest_dict

    def get_bundle_id(self) -> str:
        """Get the bundle ID for current project state.

        Returns:
            Bundle ID string like "sha256:abc123..."
        """
        _, bundle_id = self.build_manifest(save=False)
        return bundle_id

    def check_determinism(self) -> bool:
        """Check if manifest generation is deterministic.

        Builds the manifest twice and compares results.

        Returns:
            True if manifest generation is deterministic, False otherwise.
        """
        raise NotImplementedError("manifest deprecated - use modelops-bundle register-model") # return manifest.check_manifest_drift()


# Convenience functions for quick access without creating a CalabariaCLI instance
def quick_discover(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Quick discovery of models in a project.

    Args:
        path: Project root path (defaults to current directory)

    Returns:
        List of discovered models
    """
    with CalabariaCLI(path) as cli:
        return cli.discover()


def quick_manifest(path: Optional[Path] = None, save: bool = False) -> Dict[str, Any]:
    """Quick manifest generation for a project.

    Args:
        path: Project root path (defaults to current directory)
        save: Whether to save manifest.json file (default: False)

    Returns:
        Manifest dictionary
    """
    with CalabariaCLI(path) as cli:
        manifest_dict, _ = cli.build_manifest(save=save)
        return manifest_dict


def quick_verify(path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """Quick verification of all models in a project.

    Args:
        path: Project root path (defaults to current directory)

    Returns:
        Dictionary mapping model paths to verification results
    """
    with CalabariaCLI(path) as cli:
        return cli.verify()


def quick_bundle_id(path: Optional[Path] = None) -> str:
    """Quick bundle ID retrieval for a project.

    Args:
        path: Project root path (defaults to current directory)

    Returns:
        Bundle ID string
    """
    with CalabariaCLI(path) as cli:
        return cli.get_bundle_id()


# Export the main interface
__all__ = [
    "CalabariaCLI",
    "quick_discover",
    "quick_manifest",
    "quick_verify",
    "quick_bundle_id"
]