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
        # Extract base class names for analysis
        base_class_names = []
        for base in node.bases:
            base_name = self._get_base_name(base)
            if base_name:
                base_class_names.append(base_name)

        # Check if any base class is BaseModel or likely BaseModel
        is_model_class = any(self._is_base_model(base) for base in node.bases)

        # Also check if it's likely a model class based on naming patterns
        if not is_model_class:
            is_model_class = self._is_likely_model_class(node.name, base_class_names)

        if is_model_class:
            # Found a BaseModel subclass
            model_info = {
                "class_name": node.name,
                "module_path": self.module_path,
                "full_path": f"{self.module_path}:{node.name}",
                "file_path": str(self.file_path) if self.file_path else self.module_path.replace('.', '/') + '.py',
                "line_number": node.lineno,
                "base_classes": base_class_names,
                "decorators": self._extract_decorators(node),
                "methods": self._extract_methods(node)
            }
            self.models.append(model_info)

        self.generic_visit(node)

    def _get_base_name(self, base: ast.AST) -> Optional[str]:
        """Extract the name of a base class."""
        if isinstance(base, ast.Name):
            return base.id
        elif isinstance(base, ast.Attribute):
            return self._get_attribute_name(base)
        return None

    def _is_likely_model_class(self, class_name: str, base_classes: List[str]) -> bool:
        """Check if class is likely a model based on naming patterns."""
        class_lower = class_name.lower()

        # Check class name patterns
        if "model" in class_lower and any(pattern in class_lower for pattern in [
            "sir", "seir", "compartment", "epidemi", "simulation", "dynamic"
        ]):
            return True

        # Check if it inherits from classes that sound like models
        for base in base_classes:
            if base and (
                "model" in base.lower() or
                "base" in base.lower() or
                any(pattern in base.lower() for pattern in [
                    "simulation", "epidemi", "compartment", "dynamic"
                ])
            ):
                return True

        return False

    def _is_base_model(self, base: ast.AST) -> bool:
        """Check if a base class reference is BaseModel."""
        # Handle direct name: could be BaseModel or an alias like BM
        if isinstance(base, ast.Name):
            name_used = base.id

            # Direct name match (common case: "class MyModel(BaseModel):")
            if name_used == "BaseModel":
                return True

            # Check if this name was imported from a module containing BaseModel
            if name_used in self.from_imports:
                module = self.from_imports[name_used]
                # More comprehensive module pattern matching
                return self._is_basemodel_module(module)

            # Check if it was imported as a full module import
            if name_used in self.imports:
                module = self.imports[name_used]
                return self._is_basemodel_module(module)

        # Handle attribute access: package.BaseModel or module.BaseModel
        elif isinstance(base, ast.Attribute):
            if base.attr == "BaseModel":
                if isinstance(base.value, ast.Name):
                    module_name = base.value.id
                    # Check if module was imported and likely contains BaseModel
                    if module_name in self.imports:
                        return self._is_basemodel_module(self.imports[module_name])
                    return True  # Assume valid if we can't verify
                elif isinstance(base.value, ast.Attribute):
                    # Handle nested attributes like "package.module.BaseModel"
                    return True  # Assume valid for now

        return False

    def _is_basemodel_module(self, module: str) -> bool:
        """Check if a module likely contains BaseModel."""
        module_lower = module.lower()
        return (
            "base_model" in module_lower or
            "basemodel" in module_lower or
            module.endswith(".base_model") or
            module.endswith(".BaseModel") or
            module == "modelops_calabaria" or
            module.startswith("modelops_calabaria.") or
            # Common patterns in scientific computing
            "model" in module_lower and any(pattern in module_lower for pattern in [
                "core", "base", "abstract", "framework"
            ])
        )

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
                method_decorators = self._extract_method_decorators(item)

                # Categorize method based on decorators (more flexible matching)
                if any(self._is_output_decorator(dec) for dec in method_decorators):
                    methods["model_outputs"].append(item.name)
                elif any(self._is_scenario_decorator(dec) for dec in method_decorators):
                    methods["model_scenarios"].append(item.name)
                else:
                    methods["other_methods"].append(item.name)

        return methods

    def _extract_method_decorators(self, func_node: ast.FunctionDef) -> List[str]:
        """Extract decorator names from method, handling various patterns."""
        decorators = []
        for dec in func_node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Name):
                    decorators.append(dec.func.id)
                elif isinstance(dec.func, ast.Attribute):
                    decorators.append(self._get_attribute_name(dec.func))
            elif isinstance(dec, ast.Attribute):
                decorators.append(self._get_attribute_name(dec))
        return decorators

    def _is_output_decorator(self, decorator_name: str) -> bool:
        """Check if decorator indicates a model output method."""
        return (
            "output" in decorator_name.lower() or
            decorator_name in ["model_output", "output", "result", "metric"]
        )

    def _is_scenario_decorator(self, decorator_name: str) -> bool:
        """Check if decorator indicates a model scenario method."""
        return (
            "scenario" in decorator_name.lower() or
            decorator_name in ["model_scenario", "scenario", "variant", "case"]
        )

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