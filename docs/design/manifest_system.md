# Model Manifest System Design

> **Legacy Notice:** This document captures the deprecated Calabaria manifest/discovery flow.
> The modern CLI only supports sampling, calibration, and diagnostics commands; bundle manifests
> are now managed entirely by `modelops-bundle`.

The model manifest system provides **deterministic, content-addressed bundle
identification** for Calabaria models. It creates a cryptographic fingerprint
of models and their dependencies, enabling reproducible deployments and version
control based on actual content rather than arbitrary version numbers.

This is separate from the modelops-bundle manifest, which is invalidated with
any change to any code or data file. In addition to being a model entrypoint
spec, the model manifest does model code + dependency toked-based-hashing,
which is invalidated only for (potentially) meaningful changes to code.

## Overview

### Purpose
- **Content addressing**: Generate unique identifiers based on model content
- **Reproducibility**: Ensure identical models produce identical manifests
- **Change detection**: Detect when models or dependencies change
- **Deployment metadata**: Provide complete information for model execution

### Key Features
- **Format-agnostic hashing**: Code changes are detected semantically, not by formatting
- **Deterministic builds**: Multiple builds of the same content produce identical results
- **Dependency tracking**: Includes Python dependencies via `uv.lock` hash
- **Model introspection**: Automatically discovers scenarios, outputs, and parameter specs
- **Validation**: Ensures all models are valid BaseModel subclasses

## Architecture

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CLI Command   │───▶│ Manifest Builder │───▶│  manifest.json  │
│ cb manifest     │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   Model Metadata     │
                    │   Extractor          │
                    └──────────────────────┘
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
            ┌─────────────────┐    ┌─────────────────┐
            │   Token-based   │    │   Signature     │
            │   File Hasher   │    │   Generator     │
            └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Configuration Reading**: Parse `pyproject.toml` `[tool.calabaria]` section
2. **Model Discovery**: For each model configuration:
   - Import model class dynamically
   - Validate it's a BaseModel subclass
   - Create temporary instance and seal it
3. **Metadata Extraction**:
   - Extract scenarios (`@model_scenario` methods)
   - Extract outputs (`@model_output` methods)
   - Serialize parameter specifications
4. **File Processing**:
   - Resolve file patterns to actual file paths
   - Compute token-based hash for each file
   - Generate combined code signature
5. **Digest Computation**:
   - Combine code signature, parameter space, ABI version, and dependencies
   - Generate model-specific digest
6. **Bundle Generation**:
   - Combine all model digests into bundle ID
   - Write complete manifest

## Token-Based Hashing

### Problem Solved
Traditional file hashing breaks when code is reformatted:
```python
# These are semantically identical but hash differently with SHA256:
def foo(x,y): return x+y        # Raw hash: abc123...
def foo(x, y):                  # Raw hash: def456...
    return x + y
```

### Solution: Semantic Token Hashing

The system uses Python's `tokenize` module to hash based on semantic content:

```python
# Tokens to skip (formatting-only):
SKIP_TOKENS = {
    tokenize.COMMENT,     # Comments don't affect behavior
    tokenize.NL,          # Newlines within statements
    tokenize.NEWLINE,     # Statement-ending newlines
    tokenize.INDENT,      # Indentation changes
    tokenize.DEDENT,      # Dedentation changes
}

def token_hash(path: Path) -> str:
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(src).readline):
        if token.type in SKIP_TOKENS:
            continue
        # Keep only type and string, discard position info
        tokens.append((token.type, token.string))

    # Hash the canonical representation
    payload = canonical_json(tokens).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
```

### Benefits
- ✅ **Black formatting doesn't change hash**
- ✅ **Comment changes don't affect hash**
- ✅ **Whitespace normalization is automatic**
- ✅ **Tab vs spaces doesn't matter**
- ✅ **Semantic changes are always detected**

## Deterministic Builds

### Components Affecting Determinism

1. **Code Content**: Token-based hashes of all source files
2. **Parameter Space**: Serialized parameter specifications
3. **ABI Version**: Protocol compatibility version
4. **Python Requirements**: Version constraints from pyproject.toml
5. **Dependencies**: Hash of `uv.lock` file

### Signature Hierarchy

```
Bundle ID = hash(Model1_digest | Model2_digest | ...)
    │
    ├─ Model_digest = hash(code_sig | space_sig | abi | python | uv_lock)
    │     │
    │     ├─ code_sig = hash(file1_hash::path1 | file2_hash::path2 | ...)
    │     │     │
    │     │     └─ file_hash = token_hash(python_file)
    │     │
    │     └─ space_sig = hash(parameter_space_definition)
    │
    └─ [Repeat for each model]
```

### Deterministic JSON Serialization

```python
def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj,
        sort_keys=True,           # Deterministic key ordering
        separators=(",", ":"),    # Compact representation
        ensure_ascii=False        # Stable Unicode handling
    )
```

## Model Metadata Extraction

### Discovery Process

For each model, the system:

1. **Dynamic Import**:
   ```python
   module_path, class_name = "models.seir:StochasticSEIR".split(":")
   module = importlib.import_module(module_path)
   model_class = getattr(module, class_name)
   ```

2. **Validation**:
   ```python
   if not isinstance(model_class, type):
       raise TypeError(f"{class_path} is not a valid class")
   if not issubclass(model_class, BaseModel):
       raise TypeError(f"{class_path} is not a BaseModel subclass")
   ```

3. **Parameter Space Discovery**:
   ```python
   if hasattr(model_class, 'SPACE'):
       space = model_class.SPACE
   elif hasattr(model_class, 'parameter_space'):
       space = model_class.parameter_space()
   else:
       raise AttributeError(f"{class_path} must define SPACE or parameter_space()")
   ```

4. **Scenario and Output Discovery**:
   ```python
   temp_instance = model_class(space)
   temp_instance._seal()  # Triggers registration of @model_scenario/@model_output

   scenarios = sorted(temp_instance._scenarios.keys())
   outputs = sorted(temp_instance._outputs.keys())
   ```

### Extracted Metadata

Each model contributes:
- **Class path**: `"models.seir:StochasticSEIR"`
- **Source files**: `[{"path": "src/models/seir.py", "sha256": "..."}]`
- **Code signature**: Combined hash of all source files
- **Space signature**: Hash of parameter space definition
- **Scenarios**: `["baseline", "lockdown", "high_transmission"]`
- **Outputs**: `["prevalence", "incidence", "compartments", "summary"]`
- **Parameter specs**: Serialized parameter definitions
- **Model digest**: Combined hash of all above components

## Manifest Structure

### Schema Version 1

```json
{
  "schema": 1,                           // Manifest format version
  "builder": {                           // Tool information
    "name": "calabaria-cli",
    "version": "0.1.0"
  },
  "abi": "model-entrypoint@1",          // Protocol version
  "requires_python": ">=3.11",          // Python requirement
  "uv_lock_sha256": "sha256:...",       // Dependency lock hash
  "bundle_id": "sha256:...",            // Unique bundle identifier
  "models": {                           // Model definitions
    "models.seir:StochasticSEIR": {
      "class": "models.seir:StochasticSEIR",
      "files": [                        // Source files with hashes
        {
          "path": "src/models/seir.py",
          "sha256": "a1b2c3d4..."
        },
        {
          "path": "src/models/__init__.py",
          "sha256": "e5f6g7h8..."
        }
      ],
      "code_sig": "sha256:...",         // Combined code signature
      "space_sig": "sha256:...",        // Parameter space signature
      "scenarios": [                    // Available scenarios
        "baseline",
        "lockdown",
        "high_transmission"
      ],
      "outputs": [                      // Available outputs
        "prevalence",
        "incidence",
        "compartments",
        "summary"
      ],
      "param_specs": [                  // Parameter definitions
        {
          "name": "beta",
          "low": 0.1,
          "high": 2.0,
          "dtype": "float",
          "doc": "Transmission rate"
        }
      ],
      "model_digest": "sha256:..."      // Model-specific digest
    }
  }
}
```

### Key Properties

- **Self-contained**: Contains all metadata needed for execution
- **Verifiable**: Each hash can be recomputed and verified
- **Hierarchical**: Bundle ID depends on model digests, which depend on file hashes
- **Extensible**: Schema version allows for backward-compatible evolution

## CLI Usage

### Building Manifests

```bash
# Basic build
cb manifest build

# Verbose output with summary
cb manifest build --verbose

# Check if manifest needs rebuilding (for CI)
cb manifest check
```

### Example Output

```
$ cb manifest build --verbose
INFO:Building metadata for models.seir:StochasticSEIR
INFO:Building metadata for models.seir_age:AgeStratifiedSEIR
Generated manifest.json
Bundle ID: sha256:79968dd6ece1b99df249b58ec24cc754992f51d85f1f2e4e3b935ef9c9476185

Manifest Summary:
  Schema: 1
  ABI: model-entrypoint@1
  Bundle ID: sha256:79968dd6ece1b99df249b58ec24cc754992f51d85f1f2e4e3b935ef9c9476185
  Models: 2
    models.seir:StochasticSEIR:
      Files: 2
      Scenarios: 3
      Outputs: 4
      Digest: sha256:a1b2c3d4e5f6...
    models.seir_age:AgeStratifiedSEIR:
      Files: 2
      Scenarios: 3
      Outputs: 4
      Digest: sha256:f6e5d4c3b2a1...
  Python: >=3.11
  UV Lock: sha256:1234567890ab...
```

## Error Handling

### Common Issues

1. **Import Failures**:
   ```
   Cannot import models.seir:StochasticSEIR: No module named 'models'
   ```
   **Solution**: Ensure `PYTHONPATH` includes source directory

2. **Missing Parameter Space**:
   ```
   models.seir:StochasticSEIR must define SPACE or parameter_space()
   ```
   **Solution**: Add `parameter_space()` classmethod or `SPACE` attribute

3. **Invalid Model Class**:
   ```
   models.seir:StochasticSEIR is not a BaseModel subclass
   ```
   **Solution**: Ensure class inherits from BaseModel

4. **Configuration Errors**:
   ```
   No [tool.calabaria] configuration found in pyproject.toml
   ```
   **Solution**: Add proper configuration section

### Validation

The system validates:
- ✅ Required configuration fields are present
- ✅ All models can be imported successfully
- ✅ All models are BaseModel subclasses
- ✅ All models define parameter spaces
- ✅ File patterns resolve to actual files

## Integration with ModelOps

### Bundle Management

The manifest serves as the **source of truth** for:
- **Code packaging**: Which files to include in deployment bundles
- **Version identification**: Content-based versioning instead of tags
- **Dependency resolution**: Exact Python environment requirements
- **Capability discovery**: Available scenarios and outputs
- **Parameter validation**: Expected parameter specifications

### Deployment Pipeline

```
1. Developer: cb manifest build
2. CI/CD: Validates manifest generation is deterministic
3. Bundle Service: Uses manifest to package deployment artifacts
4. Execution Service: Uses manifest to validate model compatibility
5. Monitoring: Tracks deployments by bundle ID
```

### Cache Keys

Bundle IDs serve as cache keys throughout the system:
- **Build Cache**: `bundle_id` → compiled artifacts
- **Result Cache**: `(bundle_id, params_hash)` → simulation results
- **Model Registry**: `bundle_id` → model metadata and capabilities

This ensures that cache hits only occur for genuinely identical model versions, eliminating subtle bugs from version mismatches.
