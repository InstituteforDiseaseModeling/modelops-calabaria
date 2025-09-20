"""AST-based model discovery without executing imports.

Scans Python files to find BaseModel subclasses using only
the Abstract Syntax Tree, avoiding any code execution.
"""

import ast
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
import logging

logger = logging.getLogger(__name__)


class ModelDiscoveryVisitor(ast.NodeVisitor):
    """AST visitor that finds BaseModel subclasses."""

    def __init__(self, module_path: str, file_path: Optional[Path] = None):
        self.module_path = module_path
        self.file_path = file_path
        self.models: List[Dict[str, Any]] = []
        self.imports: Dict[str, str] = {}  # alias -> full_name
        self.from_imports: Dict[str, str] = {}  # name -> module

    def visit_Import(self, node: ast.Import) -> None:
        """Track import statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track from-import statements."""
        if node.module:
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                self.from_imports[name] = node.module
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Check if class inherits from BaseModel."""
        # Check all base classes
        for base in node.bases:
            if self._is_base_model(base):
                # Found a BaseModel subclass
                model_info = {
                    "class_name": node.name,
                    "module_path": self.module_path,
                    "full_path": f"{self.module_path}:{node.name}",
                    "file_path": str(self.file_path) if self.file_path else self.module_path.replace('.', '/') + '.py',
                    "line_number": node.lineno,
                    "decorators": self._extract_decorators(node),
                    "methods": self._extract_methods(node)
                }
                self.models.append(model_info)
                break

        self.generic_visit(node)

    def _is_base_model(self, base: ast.AST) -> bool:
        """Check if a base class reference is BaseModel."""
        # Handle direct name: could be BaseModel or an alias like BM
        if isinstance(base, ast.Name):
            name_used = base.id

            # Check if this name was imported from a module containing BaseModel
            if name_used in self.from_imports:
                # Get the module this name came from
                module = self.from_imports[name_used]
                # Check if it's likely a BaseModel import based on common module patterns
                return ("base_model" in module.lower() or
                        "basemodel" in module.lower() or
                        module.endswith(".base_model") or
                        module == "modelops_calabaria")

            # Check if it was imported as a full module import
            if name_used in self.imports:
                module = self.imports[name_used]
                return ("base_model" in module.lower() or
                        "basemodel" in module.lower() or
                        module == "modelops_calabaria")

        # Handle attribute access: package.BaseModel
        elif isinstance(base, ast.Attribute):
            if base.attr == "BaseModel":
                # Check if the module was imported
                if isinstance(base.value, ast.Name):
                    module_name = base.value.id
                    return module_name in self.imports

        return False

    def _extract_decorators(self, node: ast.ClassDef) -> List[str]:
        """Extract decorator names from class."""
        decorators = []
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                decorators.append(decorator.id)
            elif isinstance(decorator, ast.Attribute):
                decorators.append(self._get_attribute_name(decorator))
        return decorators

    def _extract_methods(self, node: ast.ClassDef) -> Dict[str, List[str]]:
        """Extract method information, focusing on decorated methods."""
        methods = {
            "model_outputs": [],
            "model_scenarios": [],
            "other_methods": []
        }

        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_decorators = []
                for dec in item.decorator_list:
                    if isinstance(dec, ast.Name):
                        method_decorators.append(dec.id)
                    elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                        method_decorators.append(dec.func.id)

                # Categorize method based on decorators
                if "model_output" in method_decorators:
                    methods["model_outputs"].append(item.name)
                elif "model_scenario" in method_decorators:
                    methods["model_scenarios"].append(item.name)
                else:
                    methods["other_methods"].append(item.name)

        return methods

    def _get_attribute_name(self, attr: ast.Attribute) -> str:
        """Get full attribute name like 'module.attr'."""
        parts = []
        current = attr
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))


def discover_models_in_file(file_path: Path, base_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Discover BaseModel subclasses in a single Python file.

    Args:
        file_path: Path to Python file to scan
        base_path: Base path for relative module path calculation (defaults to current directory)

    Returns:
        List of model info dictionaries

    Raises:
        SyntaxError: If the Python file has syntax errors
    """
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning(f"Skipping {file_path}: not valid UTF-8")
        return []

    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        logger.warning(f"Skipping {file_path}: syntax error at line {e.lineno}: {e.msg}")
        return []

    # Convert file path to module path
    # Use base_path if provided, otherwise current directory
    if base_path is None:
        base_path = Path.cwd()

    try:
        # Try to make it relative to base path
        relative_path = file_path.relative_to(base_path)
        module_parts = relative_path.with_suffix('').parts
    except ValueError:
        # If file is not under base path, use file name only
        module_parts = (file_path.stem,)

    # Handle __init__.py files
    if module_parts[-1] == '__init__':
        module_parts = module_parts[:-1]

    # Remove 'src' prefix if present
    if module_parts and module_parts[0] == 'src':
        module_parts = module_parts[1:]

    module_path = '.'.join(module_parts)

    # Calculate file path relative to base_path for display
    try:
        relative_file_path = file_path.relative_to(base_path or Path.cwd())
    except ValueError:
        relative_file_path = file_path.name

    visitor = ModelDiscoveryVisitor(module_path, relative_file_path)
    visitor.visit(tree)

    return visitor.models


def discover_models_in_directory(directory: Path,
                                 patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Discover BaseModel subclasses in a directory tree.

    Args:
        directory: Root directory to scan
        patterns: Optional list of glob patterns to filter files
                  (defaults to ["**/*.py"])

    Returns:
        List of all discovered models
    """
    if patterns is None:
        patterns = ["**/*.py"]

    models = []
    scanned_files = set()

    for pattern in patterns:
        for file_path in directory.glob(pattern):
            if file_path in scanned_files:
                continue
            scanned_files.add(file_path)

            if file_path.is_file() and file_path.suffix == '.py':
                try:
                    file_models = discover_models_in_file(file_path)
                    models.extend(file_models)
                except Exception as e:
                    logger.warning(f"Error scanning {file_path}: {e}")

    return models


def discover_models(root_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Discover all BaseModel subclasses in the project.

    Args:
        root_path: Project root path (defaults to current directory)

    Returns:
        List of discovered model information

    Example:
        >>> models = discover_models()
        >>> for model in models:
        ...     print(f"Found: {model['full_path']}")
        Found: models.sir:SIRModel
        Found: models.seir:SEIRModel
    """
    if root_path is None:
        root_path = Path.cwd()

    # Common patterns for Python projects
    search_patterns = [
        "src/**/*.py",
        "models/**/*.py",
        "**/*.py"
    ]

    # Remove duplicates by trying each pattern and collecting unique files
    all_models = []
    for pattern in search_patterns:
        try:
            models = discover_models_in_directory(root_path, [pattern])
            all_models.extend(models)
        except Exception as e:
            logger.debug(f"Pattern {pattern} failed: {e}")

    # Deduplicate by full_path
    seen = set()
    unique_models = []
    for model in all_models:
        key = model['full_path']
        if key not in seen:
            seen.add(key)
            unique_models.append(model)

    return unique_models


def suggest_model_config(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Suggest pyproject.toml configuration for discovered models.

    Args:
        models: List of discovered models

    Returns:
        List of suggested model configurations for pyproject.toml

    Example:
        >>> models = discover_models()
        >>> configs = suggest_model_config(models)
        >>> for config in configs:
        ...     print(f"cb models export {config['class']} --id {config['id']}")
    """
    suggestions = []

    for model in models:
        # Use class path as identifier
        suggested_id = f"{model['module_path']}:{model['class_name']}"

        # Guess file patterns based on module structure
        module_parts = model['module_path'].split('.')
        if len(module_parts) >= 2:
            # e.g., "models.sir" -> ["src/models/sir/**"]
            suggested_files = [f"src/{'/'.join(module_parts)}/**"]
        else:
            # Fallback to just the module directory
            suggested_files = [f"src/{module_parts[0]}/**"]

        suggestion = {
            "id": suggested_id,
            "class": model['full_path'],
            "files": suggested_files,
            "discovered_outputs": model['methods']['model_outputs'],
            "discovered_scenarios": model['methods']['model_scenarios']
        }

        suggestions.append(suggestion)

    return suggestions