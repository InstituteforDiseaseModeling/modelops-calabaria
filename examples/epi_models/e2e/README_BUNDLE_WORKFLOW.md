# ModelOps Bundle Workflow

> **Legacy Notice:** This walkthrough predates the simplified Calabaria CLI.
> Bundle discovery/manifests now live entirely in `modelops-bundle`; Calabaria
> no longer exposes the commands referenced below. The guide is preserved for context.

This document describes the new streamlined workflow for creating and managing model bundles using the enhanced `modelops-bundle` CLI.

## Overview

The modelops-bundle package now includes:
- **Model Discovery**: Automatically find models in your codebase
- **Interactive Selection**: Choose which models to include
- **Manifest Generation**: Create bundle metadata without executing code
- **Project Templates**: Quick start with pre-configured structure
- **Git-like Workflow**: Familiar commands (init, add, status, push)

## Quick Start

### 1. Initialize a New Project

```bash
# Create a new project directory with template
modelops-bundle init my-project

# Or initialize in existing directory
modelops-bundle init localhost:5555/my-models

# Or with explicit registry flag
modelops-bundle init --registry localhost:5555/my-models
```

When creating a new directory (`init my-project`), it automatically includes:
- `pyproject.toml` - Project configuration
- `README.md` - Documentation template
- `models/example.py` - Example model
- `.modelopsignore` - Ignore patterns
- `.modelops-bundle/` - Bundle state (git-ignored)

### 2. Discover Models

```bash
# Discover all models in the project
modelops-bundle discover

# Interactive mode - select models one by one
modelops-bundle discover --interactive

# Save discovered models to pyproject.toml
modelops-bundle discover --save
```

Models are identified by having:
- `simulate()` method - Run simulation
- `parameters()` method - Define parameters

### 3. Generate Manifest

```bash
# Generate manifest.json
modelops-bundle manifest

# Check if manifest is up-to-date
modelops-bundle manifest --check
```

The manifest contains:
- Model metadata (class paths, files)
- File hashes for integrity
- Bundle digest for versioning

### 4. Track Files

```bash
# Add files to the bundle
modelops-bundle add models/
modelops-bundle add data/*.csv

# Remove files from tracking
modelops-bundle remove old_model.py

# Check status
modelops-bundle status
```

### 5. Push to Registry

```bash
# Push bundle to registry
modelops-bundle push

# Dry run to see what would be pushed
modelops-bundle push --dry-run

# Push with specific tag
modelops-bundle push --tag v1.0.0
```

## Architecture Changes

### Separation of Concerns

**Before:**
- Calabaria handled both model discovery AND manifest generation
- Required importing model classes (executing code)
- Tight coupling between science and bundle management

**After:**
- `modelops-bundle` handles discovery and manifest ("what's in the bundle")
- `modelops-calabaria` handles science ("how to use the bundle")
- Discovery uses AST parsing (no code execution)
- Clean separation via manifest.json

### Key Components

```
modelops-bundle/
├── discovery.py      # AST-based model discovery
├── manifest.py       # Manifest generation
├── cli.py           # Enhanced CLI commands
└── ...

modelops-calabaria/
├── sampling.py      # Reads manifest.json
└── ...
```

## Configuration

### pyproject.toml

```toml
[tool.modelops-bundle]
models = [
    {
        id = "models.sir:SIRModel",
        class = "models.sir:SIRModel",
        files = ["models/**/*.py"]
    }
]
```

### manifest.json

```json
{
    "schema_version": "1.0",
    "bundle_digest": "sha256:abc123...",
    "models": {
        "models.sir:SIRModel": {
            "class": "models.sir:SIRModel",
            "files": ["models/sir.py"]
        }
    },
    "files": {
        "models/sir.py": {
            "sha256": "def456...",
            "size": 2048
        }
    }
}
```

## Benefits

1. **No Code Execution**: Discovery and manifest generation don't run model code
2. **Git-like Workflow**: Familiar commands for developers
3. **Interactive Mode**: User-friendly model selection
4. **Project Templates**: Quick project initialization
5. **Clean Separation**: Bundle management separate from science logic
6. **Deterministic**: Reproducible bundle digests

## Testing

Run the end-to-end test:

```bash
python test_bundle_workflow.py
```

This demonstrates the complete workflow from project creation to bundle push.

## Migration Guide

For existing projects:

1. Install updated packages:
   ```bash
   uv sync
   ```

2. Initialize bundle tracking:
   ```bash
   modelops-bundle init <registry>
   ```

3. Discover and save models:
   ```bash
   modelops-bundle discover --interactive --save
   ```

4. Generate manifest:
   ```bash
   modelops-bundle manifest
   ```

5. Track and push:
   ```bash
   modelops-bundle add .
   modelops-bundle push
   ```

## Next Steps

- Simplify `modelops-calabaria` sampling to read manifest.json
- Add support for model dependencies
- Implement bundle versioning
- Add CI/CD integration examples
