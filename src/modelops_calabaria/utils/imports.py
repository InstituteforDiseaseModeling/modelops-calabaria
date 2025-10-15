"""Import utilities for loading user code with proper path handling.

This module provides utilities for importing user models and targets,
automatically handling the common case where the current working directory
needs to be temporarily added to sys.path for imports to work.

Follows the pattern used by pytest, IPython, and streamlit to provide
"it just works" user experience while maintaining control and safety.
"""

from __future__ import annotations
from contextlib import contextmanager
from importlib import import_module, util
from pathlib import Path
import os
import sys
import types
import warnings
from typing import Any, Optional


@contextmanager
def _prepend_sys_path(path: str):
    """Context manager to temporarily prepend a path to sys.path.

    Args:
        path: Directory path to temporarily add to sys.path

    Yields:
        None: Control during the context with path prepended
    """
    p = str(Path(path).resolve())
    added = False
    if p and p not in sys.path:
        sys.path.insert(0, p)
        added = True
    try:
        yield
    finally:
        if added:
            # Best-effort cleanup; don't fail if already removed
            try:
                sys.path.remove(p)
            except ValueError:
                pass


def _import_from_file(pyfile: str) -> types.ModuleType:
    """Import a module directly from a Python file path.

    This bypasses sys.path entirely and loads the module from the specified file.

    Args:
        pyfile: Path to Python file (e.g., "./models/seir.py")

    Returns:
        Loaded module object

    Raises:
        ModuleNotFoundError: If file doesn't exist or can't be loaded
    """
    py = Path(pyfile).resolve()
    if not py.exists():
        raise ModuleNotFoundError(f"No such file: {py}")
    spec = util.spec_from_file_location(py.stem, py)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"Could not load module from {py}")
    mod = util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def load_symbol(
    qualified: str,
    project_root: Optional[str] = None,
    allow_cwd_import: bool = True
) -> Any:
    """Load a symbol from a qualified path with intelligent path handling.

    Supports both module paths ("pkg.mod:Symbol") and file paths
    ("./path/to/file.py:Symbol"). For module paths, will automatically
    add the project root to sys.path if needed (and remove it after).

    Args:
        qualified: Either 'module.path:Symbol' or '/path/to/file.py:Symbol'
        project_root: Project root directory to add to sys.path (default: cwd)
        allow_cwd_import: Whether to add project_root to sys.path if needed

    Returns:
        The loaded symbol (class, function, etc.)

    Raises:
        ValueError: If qualified string format is invalid
        ModuleNotFoundError: If module can't be imported
        AttributeError: If symbol doesn't exist in module

    Examples:
        >>> # Load from module path (may add cwd to sys.path)
        >>> ModelClass = load_symbol("models.seir:StochasticSEIR")

        >>> # Load from file path (doesn't touch sys.path)
        >>> ModelClass = load_symbol("./models/seir.py:StochasticSEIR")

        >>> # Disable auto-adding cwd (requires proper PYTHONPATH)
        >>> ModelClass = load_symbol("models.seir:StochasticSEIR", allow_cwd_import=False)
    """
    module_part, sep, symbol = qualified.partition(":")
    if not sep:
        raise ValueError(
            f"Expected 'module_or_file:Symbol' format, got: {qualified}"
        )

    # Path-style import (e.g., "./models/seir.py:StochasticSEIR")
    if module_part.endswith(".py") or "/" in module_part or "\\" in module_part:
        mod = _import_from_file(module_part)
        if not hasattr(mod, symbol):
            raise AttributeError(f"Module {module_part} has no attribute '{symbol}'")
        return getattr(mod, symbol)

    # Try normal import first (works for installed/editable packages)
    try:
        mod = import_module(module_part)
        if not hasattr(mod, symbol):
            raise AttributeError(f"Module {module_part} has no attribute '{symbol}'")
        return getattr(mod, symbol)
    except ModuleNotFoundError:
        if not allow_cwd_import:
            # Re-raise with more helpful message
            raise ModuleNotFoundError(
                f"Cannot import '{module_part}'. "
                f"Current directory is not in Python path. "
                f"Either set PYTHONPATH or use --project-root"
            )

    # Fallback: try with project root in sys.path
    root = Path(project_root or os.getcwd()).resolve()

    with _prepend_sys_path(str(root)):
        try:
            mod = import_module(module_part)
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                f"Cannot import '{module_part}' even with project root '{root}' in path. "
                f"Check that the module path is correct and the file exists."
            )

        if not hasattr(mod, symbol):
            raise AttributeError(f"Module {module_part} has no attribute '{symbol}'")

        obj = getattr(mod, symbol)

        # Safety check: warn if module resolved outside project root
        mfile = Path(getattr(mod, "__file__", "")).resolve()
        try:
            # Try to get relative path - if this succeeds, module is inside root
            _ = mfile.relative_to(root)
        except (ValueError, TypeError):
            # Module is outside project root - could be shadowing
            warnings.warn(
                f"Module '{module_part}' resolved to '{mfile}', "
                f"which is outside project root '{root}'. "
                f"This might indicate a shadowed import.",
                RuntimeWarning,
                stacklevel=2
            )

        return obj