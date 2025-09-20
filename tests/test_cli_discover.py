"""Tests for CLI model discovery functionality.

Tests AST-based model discovery that finds BaseModel subclasses
without executing imports.
"""

import pytest
from pathlib import Path
import tempfile
import textwrap

from modelops_calabaria.cli.discover import (
    ModelDiscoveryVisitor,
    discover_models_in_file,
    discover_models_in_directory,
    discover_models,
    suggest_model_config,
)


class TestModelDiscoveryVisitor:
    """Tests for AST visitor that finds BaseModel subclasses."""

    def test_finds_direct_basemodel_import(self):
        """Should find BaseModel imported directly."""
        code = textwrap.dedent("""
            from modelops_calabaria.base_model import BaseModel

            class MyModel(BaseModel):
                def build_sim(self, params, config):
                    pass

                def run_sim(self, state, seed):
                    pass
        """)

        # Parse and visit
        import ast
        tree = ast.parse(code)
        visitor = ModelDiscoveryVisitor("test.models")
        visitor.visit(tree)

        assert len(visitor.models) == 1
        model = visitor.models[0]
        assert model["class_name"] == "MyModel"
        assert model["module_path"] == "test.models"
        assert model["full_path"] == "test.models:MyModel"

    def test_finds_aliased_basemodel_import(self):
        """Should find BaseModel with alias."""
        code = textwrap.dedent("""
            from modelops_calabaria.base_model import BaseModel as BM

            class MyModel(BM):
                pass
        """)

        import ast
        tree = ast.parse(code)
        visitor = ModelDiscoveryVisitor("test.models")
        visitor.visit(tree)

        assert len(visitor.models) == 1
        assert visitor.models[0]["class_name"] == "MyModel"

    def test_finds_module_basemodel_import(self):
        """Should find BaseModel via module import."""
        code = textwrap.dedent("""
            import modelops_calabaria.base_model as base

            class MyModel(base.BaseModel):
                pass
        """)

        import ast
        tree = ast.parse(code)
        visitor = ModelDiscoveryVisitor("test.models")
        visitor.visit(tree)

        assert len(visitor.models) == 1
        assert visitor.models[0]["class_name"] == "MyModel"

    def test_ignores_non_basemodel_classes(self):
        """Should ignore classes that don't inherit from BaseModel."""
        code = textwrap.dedent("""
            class RegularClass:
                pass

            class AnotherClass(dict):
                pass
        """)

        import ast
        tree = ast.parse(code)
        visitor = ModelDiscoveryVisitor("test.models")
        visitor.visit(tree)

        assert len(visitor.models) == 0

    def test_extracts_decorated_methods(self):
        """Should extract methods with model decorators."""
        code = textwrap.dedent("""
            from modelops_calabaria.base_model import BaseModel

            class MyModel(BaseModel):
                @model_output
                def get_infections(self, raw, seed):
                    return raw['I']

                @model_scenario
                def lockdown_scenario(self):
                    return ScenarioSpec("lockdown")

                def regular_method(self):
                    pass
        """)

        import ast
        tree = ast.parse(code)
        visitor = ModelDiscoveryVisitor("test.models")
        visitor.visit(tree)

        assert len(visitor.models) == 1
        model = visitor.models[0]

        methods = model["methods"]
        assert "get_infections" in methods["model_outputs"]
        assert "lockdown_scenario" in methods["model_scenarios"]
        assert "regular_method" in methods["other_methods"]

    def test_handles_multiple_models(self):
        """Should find multiple models in same file."""
        code = textwrap.dedent("""
            from modelops_calabaria.base_model import BaseModel

            class SIRModel(BaseModel):
                pass

            class SEIRModel(BaseModel):
                pass

            class RegularClass:
                pass
        """)

        import ast
        tree = ast.parse(code)
        visitor = ModelDiscoveryVisitor("epidemiology.models")
        visitor.visit(tree)

        assert len(visitor.models) == 2
        names = [m["class_name"] for m in visitor.models]
        assert "SIRModel" in names
        assert "SEIRModel" in names


class TestDiscoverModelsInFile:
    """Tests for discovering models in a single file."""

    def test_discovers_model_in_file(self):
        """Should discover models in a Python file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a model file
            model_file = tmpdir / "src" / "models" / "sir.py"
            model_file.parent.mkdir(parents=True)
            model_file.write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel

                class SIRModel(BaseModel):
                    @model_output
                    def infections(self, raw, seed):
                        return raw['I']

                    def build_sim(self, params, config):
                        pass

                    def run_sim(self, state, seed):
                        pass
            """))

            models = discover_models_in_file(model_file, base_path=tmpdir)

            assert len(models) == 1
            model = models[0]
            assert model["class_name"] == "SIRModel"
            assert model["module_path"] == "models.sir"
            assert model["methods"]["model_outputs"] == ["infections"]

    def test_handles_syntax_errors_gracefully(self):
        """Should handle files with syntax errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            bad_file = tmpdir / "bad.py"
            bad_file.write_text("def broken_function(\n    pass")

            # Should not raise exception
            models = discover_models_in_file(bad_file, base_path=tmpdir)
            assert models == []

    def test_handles_non_utf8_files(self):
        """Should handle non-UTF8 files gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            binary_file = tmpdir / "binary.py"
            binary_file.write_bytes(b"\xff\xfe\x00\x00")  # Invalid UTF-8

            # Should not raise exception
            models = discover_models_in_file(binary_file, base_path=tmpdir)
            assert models == []

    def test_converts_file_path_to_module_path(self):
        """Should correctly convert file paths to module paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Test various path structures
            test_cases = [
                ("src/models/sir.py", "models.sir"),
                ("src/epidemiology/models/seir.py", "epidemiology.models.seir"),
                ("models/__init__.py", "models"),
                ("deep/nested/package/model.py", "deep.nested.package.model"),
            ]

            for file_path, expected_module in test_cases:
                model_file = tmpdir / file_path
                model_file.parent.mkdir(parents=True, exist_ok=True)
                model_file.write_text(textwrap.dedent("""
                    from modelops_calabaria.base_model import BaseModel

                    class TestModel(BaseModel):
                        pass
                """))

                models = discover_models_in_file(model_file, base_path=tmpdir)
                assert len(models) == 1
                assert models[0]["module_path"] == expected_module


class TestDiscoverModelsInDirectory:
    """Tests for discovering models in a directory tree."""

    def test_discovers_models_in_directory(self):
        """Should discover all models in directory tree."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create multiple model files
            (tmpdir / "models").mkdir()
            (tmpdir / "models" / "sir.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel

                class SIRModel(BaseModel):
                    pass
            """))

            (tmpdir / "models" / "seir.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel

                class SEIRModel(BaseModel):
                    pass
            """))

            # Non-model file
            (tmpdir / "models" / "utils.py").write_text("def utility(): pass")

            models = discover_models_in_directory(tmpdir)

            assert len(models) == 2
            names = [m["class_name"] for m in models]
            assert "SIRModel" in names
            assert "SEIRModel" in names

    def test_respects_file_patterns(self):
        """Should respect file pattern filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create files in different locations
            (tmpdir / "src").mkdir()
            (tmpdir / "src" / "model.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel
                class Model1(BaseModel): pass
            """))

            (tmpdir / "other").mkdir()
            (tmpdir / "other" / "model.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel
                class Model2(BaseModel): pass
            """))

            # Only search src directory
            models = discover_models_in_directory(tmpdir, patterns=["src/**/*.py"])

            assert len(models) == 1
            assert models[0]["class_name"] == "Model1"

    def test_deduplicates_files(self):
        """Should not scan the same file multiple times."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            model_file = tmpdir / "model.py"
            model_file.write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel
                class TestModel(BaseModel): pass
            """))

            # Use overlapping patterns
            patterns = ["*.py", "model.py", "**/*.py"]
            models = discover_models_in_directory(tmpdir, patterns=patterns)

            # Should only find one model despite multiple patterns
            assert len(models) == 1


class TestDiscoverModels:
    """Tests for project-wide model discovery."""

    def test_discovers_models_in_common_locations(self):
        """Should search common Python project locations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create models in different standard locations
            locations = ["src/models", "models", "myproject/models"]

            for i, location in enumerate(locations):
                dir_path = tmpdir / location
                dir_path.mkdir(parents=True)
                (dir_path / "model.py").write_text(textwrap.dedent(f"""
                    from modelops_calabaria.base_model import BaseModel
                    class Model{i}(BaseModel): pass
                """))

            # Change to temp directory to run discovery
            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                models = discover_models()
            finally:
                os.chdir(old_cwd)

            # Should find all models
            assert len(models) >= len(locations)  # May find extras due to **/*.py
            class_names = [m["class_name"] for m in models]
            for i in range(len(locations)):
                assert f"Model{i}" in class_names

    def test_deduplicates_across_patterns(self):
        """Should deduplicate models found by multiple patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create a model that would be found by multiple patterns
            (tmpdir / "src").mkdir()
            (tmpdir / "src" / "model.py").write_text(textwrap.dedent("""
                from modelops_calabaria.base_model import BaseModel
                class UniqueModel(BaseModel): pass
            """))

            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                models = discover_models()
            finally:
                os.chdir(old_cwd)

            # Should only have one instance of UniqueModel
            unique_models = [m for m in models if m["class_name"] == "UniqueModel"]
            assert len(unique_models) == 1


class TestSuggestModelConfig:
    """Tests for configuration suggestion functionality."""

    def test_suggests_basic_config(self):
        """Should suggest basic configuration for models."""
        models = [
            {
                "class_name": "SIRModel",
                "module_path": "models.sir",
                "full_path": "models.sir:SIRModel",
                "methods": {
                    "model_outputs": ["infections", "susceptible"],
                    "model_scenarios": ["baseline", "lockdown"],
                    "other_methods": ["build_sim", "run_sim"]
                }
            }
        ]

        suggestions = suggest_model_config(models)

        assert len(suggestions) == 1
        suggestion = suggestions[0]

        assert suggestion["id"] == "models.sir:SIRModel"
        assert suggestion["class"] == "models.sir:SIRModel"
        assert suggestion["files"] == ["src/models/sir/**"]
        assert suggestion["discovered_outputs"] == ["infections", "susceptible"]
        assert suggestion["discovered_scenarios"] == ["baseline", "lockdown"]

    def test_handles_different_module_structures(self):
        """Should suggest appropriate file patterns for different structures."""
        models = [
            {
                "class_name": "SimpleModel",
                "module_path": "model",  # Single module
                "full_path": "model:SimpleModel",
                "methods": {"model_outputs": [], "model_scenarios": [], "other_methods": []}
            },
            {
                "class_name": "DeepModel",
                "module_path": "deep.nested.package.model",
                "full_path": "deep.nested.package.model:DeepModel",
                "methods": {"model_outputs": [], "model_scenarios": [], "other_methods": []}
            }
        ]

        suggestions = suggest_model_config(models)

        assert len(suggestions) == 2

        # Single module should suggest src/module/**
        simple_suggestion = next(s for s in suggestions if s["class"] == "model:SimpleModel")
        assert simple_suggestion["files"] == ["src/model/**"]

        # Deep module should suggest full path
        deep_suggestion = next(s for s in suggestions if s["class"] == "deep.nested.package.model:DeepModel")
        assert deep_suggestion["files"] == ["src/deep/nested/package/model/**"]

    def test_generates_unique_ids(self):
        """Should generate unique IDs for different models."""
        models = [
            {
                "class_name": "ModelA",
                "module_path": "pkg.a",
                "full_path": "pkg.a:ModelA",
                "methods": {"model_outputs": [], "model_scenarios": [], "other_methods": []}
            },
            {
                "class_name": "ModelB",
                "module_path": "pkg.b",
                "full_path": "pkg.b:ModelB",
                "methods": {"model_outputs": [], "model_scenarios": [], "other_methods": []}
            }
        ]

        suggestions = suggest_model_config(models)

        assert len(suggestions) == 2
        ids = [s["id"] for s in suggestions]
        assert "pkg.a:ModelA" in ids
        assert "pkg.b:ModelB" in ids


class TestIntegration:
    """Integration tests for discovery functionality."""

    def test_full_discovery_pipeline(self):
        """Test complete discovery pipeline from files to suggestions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create realistic project structure
            src_dir = tmpdir / "src" / "epidemiology"
            src_dir.mkdir(parents=True)

            # SIR model
            (src_dir / "sir.py").write_text(textwrap.dedent("""
                '''SIR epidemiological model.'''
                from modelops_calabaria.base_model import BaseModel

                class SIRModel(BaseModel):
                    '''Susceptible-Infected-Recovered model.'''

                    @model_output
                    def infections(self, raw, seed):
                        '''Extract infection data.'''
                        return raw['I']

                    @model_scenario
                    def lockdown(self):
                        '''Lockdown scenario.'''
                        from modelops_calabria.scenarios import ScenarioSpec
                        return ScenarioSpec("lockdown")

                    def build_sim(self, params, config):
                        pass

                    def run_sim(self, state, seed):
                        pass
            """))

            import os
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Discover models
                models = discover_models()
                sir_models = [m for m in models if m["class_name"] == "SIRModel"]
                assert len(sir_models) == 1

                sir_model = sir_models[0]
                assert sir_model["module_path"] == "epidemiology.sir"
                assert sir_model["methods"]["model_outputs"] == ["infections"]
                assert sir_model["methods"]["model_scenarios"] == ["lockdown"]

                # Generate suggestions
                suggestions = suggest_model_config([sir_model])
                assert len(suggestions) == 1

                suggestion = suggestions[0]
                assert suggestion["id"] == "epidemiology.sir:SIRModel"
                assert suggestion["class"] == "epidemiology.sir:SIRModel"
                assert suggestion["files"] == ["src/epidemiology/sir/**"]

            finally:
                os.chdir(old_cwd)