# Calabaria Model Export & Bundle Implementation Plan

## Objectives (MVP)

1. **Explicit exports; no app logic in user code**
   Users list cloud-visible models + files in config (not Python). Calabaria CLI builds a manifest and a bundle (curated subset of the repo).

2. **Deterministic, low-churn artifacts**
   Stable ordering; token-based code hashing (ignores whitespace/comments); no timestamps/abs paths.

3. **Correct invalidation & cache keys**
   Per-model `model_digest` flips only when that model's code/fileset or environment changes; results keyed by `run_key`.

4. **Great UX** `cb models discover/export/verify`, `cb manifest build`,
   `mops-bundle push` (outside package), `mops workspace visible` (outside
   package), `mops sim submit <job.json>` (outside package).

## High-Level Architecture

```
User Repo
 ├─ src/models/...          # science code
 ├─ pyproject.toml          # explicit export list
 ├─ uv.lock                 # deps
 └─ (optional) .calabaria/  # local state (not required)

Calabaria CLI (client)
 ├─ reads config (exports)
 ├─ verifies import boundary
 ├─ builds manifest.json (deterministic)
 └─ pushes curated bundle → Cloud

Cloud
 ├─ validates bundle + manifest
 ├─ maps bundle → {model_id → model_digest}
 ├─ caches/looks up results by run_key
 └─ runs via EntryRecord → wire_v1 → BaseModel
```

## Core Design Principles

- **Explicit over implicit**: Models are explicitly listed in `pyproject.toml`
- **Deterministic builds**: Same code → same digest, always
- **Per-model isolation**: Adding model A doesn't affect model B's digest
- **Token-based hashing**: Whitespace/comment changes don't trigger rebuilds
- **Import boundary enforcement**: Models can only import declared dependencies

## Configuration Format

### pyproject.toml

```toml
[tool.calabaria]
schema = 1
abi = "model-entrypoint@1"

# Optional: declare Python requirement for reproducibility
requires_python = "==3.12.5"

[[tool.calabaria.model]]
id = "sir@v1"
class = "models.sir:SIRModel"
files = [
  "src/models/sir/**",
  "src/models/common/**/*.py"
]

[[tool.calabaria.model]]
id = "seir@v1"
class = "models.seir:SEIRModel"
files = [
  "src/models/seir/**",
  "src/models/common/**/*.py"
]
```

- **id**: Human-readable stable identifier (doesn't change as code evolves)
- **class**: Module path and class name (`module:ClassName`)
- **files**: Glob patterns defining exactly which code determines the model

## Manifest Format

### manifest.json (deterministic, committable)

```json
{
  "schema": 1,
  "builder": {"name": "calabaria-cli", "version": "0.1.0"},
  "abi": "model-entrypoint@1",
  "requires_python": "==3.12.5",
  "uv_lock_sha256": "sha256:...",

  "models": {
    "sir@v1": {
      "class": "models.sir:SIRModel",
      "files": [
        {"path": "src/models/sir/__init__.py", "sha256": "..."},
        {"path": "src/models/sir/core.py", "sha256": "..."},
        {"path": "src/models/common/numerics.py", "sha256": "..."}
      ],
      "code_sig": "sha256:...",      // token-hash over listed files
      "space_sig": "sha256:...",     // stable JSON of ParameterSpace
      "model_digest": "sha256:...",  // H(code_sig||space_sig||abi||py||uvlock)
      "scenarios": ["baseline", "..."],
      "outputs": ["incidence", "..."],
      "param_specs": [
        {"name": "beta", "min": 0.1, "max": 1.0, "kind": "float", "doc": ""},
        ...
      ]
    },
    "seir@v1": { "...": "..." }
  },

  "bundle_id": "sha256:..."  // H(sorted(model_id,model_digest)...)
}
```

- **bundle_id**: Envelope hash for the entire bundle (not used for result caching)
- **model_digest**: The cache/reproduction identity per model

## Hashing Strategy

### Token Hashing (ignores whitespace/comments)
```python
# calabaria_cli/hashing.py
import hashlib, tokenize, io, json, pathlib
from typing import Iterable

SKIP = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
        tokenize.INDENT, tokenize.DEDENT}

def token_hash(path: pathlib.Path) -> str:
    """Hash Python file based on tokens only, ignoring formatting."""
    src = path.read_text(encoding="utf-8")
    toks = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in SKIP:
            continue
        # Strip positions; keep type+string only
        toks.append((tok.type, tok.string))
    payload = json.dumps(toks, separators=(",",":"), ensure_ascii=False).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()

def code_sig(file_records: Iterable[tuple[str,str]]) -> str:
    """Create signature from multiple file hashes."""
    joined = "|".join(f"{p}::{h}" for p,h in sorted(file_records))
    return "sha256:" + hashlib.sha256(joined.encode()).hexdigest()

def canonical_json(obj) -> str:
    """Deterministic JSON serialization."""
    return json.dumps(obj, sort_keys=True, separators=(",",":"))
```

### Digest Computation
- **code_sig**: `sha256(concat(sorted(file_path||file_token_hash)))`
- **space_sig**: `sha256(canonical_json(ParameterSpace))`
- **model_digest**: `sha256(code_sig||space_sig||abi||requires_python||uv_lock_sha)`

## CLI Commands

### Core Commands

```bash
# Discover models via AST (no imports)
cb models discover

# Export model to pyproject.toml
cb models export models.sir:SIRModel --id sir@v1 \
  --files "src/models/sir/**" "src/models/common/**/*.py"

# Verify import boundaries and parameter spaces
cb models verify

# Build manifest.json deterministically
cb manifest build [--check]  # --check exits 1 if differs from committed

# Push bundle to cloud
mops-bundle push

# Show cloud-visible models
mops workspace cloud visible

# Submit a run
mops sim submit --model sir@v1 --params params.json \
  --data-version typhoid-2025-09 --seed 123 --reps 50
```

### CLI Implementation Skeleton

```python
# calabaria_cli/__main__.py
import typer
app = typer.Typer(help="Calabaria CLI")

@app.command()
def models_discover():
    """AST scan for BaseModel subclasses (no imports)."""
    # Use ast.parse() to find classes inheriting from BaseModel
    ...

@app.command()
def models_export(
    class_: str = typer.Argument(...),
    id: str = typer.Option(..., "--id"),
    files: list[str] = typer.Option(..., "--files")
):
    """Append/modify [tool.calabaria.model] in pyproject.toml."""
    # Edit pyproject.toml programmatically
    ...

@app.command()
def models_verify():
    """Verify imports ⊆ declared files ∪ stdlib ∪ site-packages."""
    # Spawn subprocess to import and track loaded modules
    ...

@app.command()
def manifest_build(check: bool = typer.Option(False, "--check")):
    """Build manifest.json deterministically."""
    # Read config, hash files, introspect models
    ...
```

## Import Boundary Verification

The verification system ensures models only import code from their declared dependencies. This is critical for reproducible builds and security.

### Why Subprocess Isolation?

We run verification in a **completely fresh Python subprocess** to track imports cleanly:

1. **Clean slate**: Each model starts with only Python stdlib loaded
2. **Accurate tracking**: We see EXACTLY what this model imports
3. **No contamination**: Each verification is independent
4. **Security**: Bad model code is contained in subprocess

### Complete Implementation

```python
# calabaria_cli/verify.py
import subprocess
import sys
import json
import pathlib
from typing import Set, Dict, List

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
        - errors: Any import errors encountered
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
    # This gives us a clean Python environment
    result = subprocess.run(
        [sys.executable, "-c", verification_script],
        capture_output=True,
        text=True,
        timeout=30  # Prevent hanging on bad imports
    )

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
        Dict mapping model ID to verification results
    """
    results = {}

    for model_config in config.get("model", []):
        model_id = model_config["id"]
        class_path = model_config["class"]
        file_patterns = model_config["files"]

        # Resolve file patterns to actual files
        allowed_files = resolve_file_patterns(file_patterns)

        print(f"Verifying {model_id} ({class_path})...")
        result = verify_model(class_path, allowed_files)
        results[model_id] = result

        if result["ok"]:
            print(f"  ✓ OK - {len(result['covered'])} files used")
            if result["unused"]:
                print(f"  ⚠ Warning: {len(result['unused'])} allowed files were not imported")
        else:
            print(f"  ✗ FAIL - {len(result['unexpected'])} unexpected imports:")
            for file in result['unexpected'][:5]:  # Show first 5
                print(f"    - {file}")
            if len(result['unexpected']) > 5:
                print(f"    ... and {len(result['unexpected']) - 5} more")

    return results


def resolve_file_patterns(patterns: List[str]) -> Set[str]:
    """
    Resolve glob patterns to actual file paths.

    Args:
        patterns: List of glob patterns like ["src/models/sir/**", "src/common/*.py"]

    Returns:
        Set of relative file paths
    """
    import glob

    resolved = set()
    for pattern in patterns:
        # Handle recursive ** patterns
        matches = glob.glob(pattern, recursive=True)
        for match in matches:
            path = pathlib.Path(match)
            if path.is_file() and path.suffix == '.py':
                resolved.add(str(path))

    return resolved
```

### Key Benefits

1. **Complete Isolation**: Each model verification runs in a fresh process
2. **Accurate Dependency Tracking**: See EXACTLY what a model imports
3. **Security Boundary**: Malicious code is contained in subprocess (with timeout)
4. **Detailed Reporting**:
   - What files were actually used (covered)
   - What files were allowed but unused (potential dead code)
   - What external packages were imported (for documentation)
   - What unexpected imports occurred (violations)
5. **Error Handling**: Graceful handling of import errors with detailed diagnostics

This enforces security boundaries while providing excellent debugging information for developers.

## Worker Integration

### Architecture: Protocol vs Implementation

The type separation between packages is intentional and correct:

- **modelops-contracts**: Protocol types (`EntryPointId`, `SimTask`, `SimReturn`)
  - Define WHAT gets passed between systems
  - Minimal, stable interface
  - Other systems can implement the same protocol differently

- **modelops-calabaria**: Implementation types (`EntryRecord`, `WireResponse`)
  - Define HOW Calabria implements the protocol internally
  - Rich metadata for Calabaria-specific features
  - Not exposed in the public contract

- **modelops**: Infrastructure orchestration
  - Uses protocol types only
  - Never sees Calabaria internals
  - Can work with any system that implements SimTask → SimReturn

This separation enables evolution: Calabaria can change its internal types without affecting the protocol.

### Manifest → EntryRecord

```python
# worker/entry_factory.py
from modelops_calabaria.wire_protocol import EntryRecord, SerializedParameterSpec, WireABI

def entry_from_manifest(model_rec: dict, abi: str) -> EntryRecord:
    """Create EntryRecord from manifest model record."""
    mpath, cname = model_rec["class"].split(":")
    specs = tuple(SerializedParameterSpec(**p) for p in model_rec["param_specs"])

    return EntryRecord(
        id=f"{mpath}.{cname}",
        model_hash=model_rec["model_digest"],  # Use manifest digest
        abi_version=WireABI(abi),
        module_name=mpath,
        class_name=cname,
        scenarios=tuple(model_rec["scenarios"]),
        outputs=tuple(model_rec["outputs"]),
        param_specs=specs,
        alias=None
    )

def run_job(manifest, model_id, params, seed, outputs=None):
    """Execute model from manifest."""
    model_rec = manifest["models"][model_id]
    entry = entry_from_manifest(model_rec, manifest["abi"])
    wire = entry.get_wire_factory()()  # Lazy import happens here
    return wire(params_M=params, seed=seed, outputs=outputs)
```

### Run Key (Cache Key)

```python
def run_key(model_digest: str, params: dict, data_version: str,
           seed: int, reps: int, runtime: dict) -> str:
    """Compute deterministic cache key for a run."""
    payload = "|".join([
        model_digest,
        canonical_json(params),
        data_version,
        f"seed={seed},reps={reps}",
        canonical_json(runtime)
    ]).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()
```

## Minor Type Adjustments

### 1. EntryRecord.model_hash Override

```python
# base_model.py
@classmethod
def compile_entrypoint(cls, space: ParameterSpace, *,
                       model_hash: str | None = None) -> 'EntryRecord':
    """Allow override of model_hash for manifest-based creation."""
    ...
    if model_hash is None:
        model_hash = hashlib.sha256(
            f"{cls.__module__}.{cls.__name__}".encode()
        ).hexdigest()[:8]
    ...
```

### 2. ParameterSpace Serialization

```python
# parameters.py
@dataclass(frozen=True)
class ParameterSpace:
    specs: tuple[ParameterSpec, ...]

    def to_dict(self) -> dict:
        """Stable dictionary representation."""
        return {
            "parameters": [s.to_dict() for s in self.specs]
        }
```

## CI/CD Integration

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
- repo: local
  hooks:
    - id: verify-models
      name: Verify model imports
      entry: cb models verify
      language: system
      pass_filenames: false

    - id: check-manifest
      name: Check manifest consistency
      entry: cb manifest build --check
      language: system
      pass_filenames: false
```

### CI Workflow

```yaml
# .github/workflows/manifest.yml
- name: Build manifest
  run: cb manifest build

- name: Check for drift
  run: |
    git diff --exit-code manifest.json || {
      echo "manifest.json is out of date!"
      echo "Run: cb manifest build"
      exit 1
    }
```

## Implementation Checklist

### Phase 1: Core Infrastructure
- [ ] Token-based hashing utilities
- [ ] pyproject.toml reader/writer
- [ ] AST-based model discovery
- [ ] Manifest builder with introspection
- [ ] Import boundary verifier

### Phase 2: CLI Commands
- [ ] `models discover` (AST scan)
- [ ] `models export` (edit config)
- [ ] `models verify` (boundary check)
- [ ] `manifest build` (deterministic)
- [ ] Basic error handling

### Phase 3: Integration
- [ ] Worker entry factory
- [ ] Run key computation
- [ ] BaseModel.compile_entrypoint override
- [ ] ParameterSpace.to_dict() helper

### Phase 4: Developer Experience
- [ ] Pre-commit hooks
- [ ] CI manifest checks
- [ ] Helpful error messages
- [ ] Documentation

## Future Work

### Cloud API
```
POST /api/bundles
  - Validate tar contents
  - Recompute/verify manifest
  - Store bundle_id → {model_id → model_digest} mapping
  - Return accepted models list

GET /api/bundles/{bundle_id}
  - Return cloud-visible models

POST /api/runs
  - Accept: bundle_id, model_id, params, data_version, seed, reps
  - Compute run_key for caching
  - Return cached result or enqueue job
  - Worker: uv sync --locked, build EntryRecord, execute wire
```

### Bundle Push Implementation

```python
# calabaria_cli/push.py
def bundle_push(manifest: dict, server_url: str):
    """Create and upload curated bundle."""
    # Collect all files from manifest
    paths = set()
    for m in manifest["models"].values():
        for f in m["files"]:
            paths.add(f["path"])
    paths |= {"pyproject.toml", "uv.lock", "manifest.json"}

    # Create tar and upload
    tar_bytes = make_tar(sorted(paths))
    r = requests.post(
        f"{server_url}/api/bundles",
        headers={"X-Bundle-Id": manifest["bundle_id"]},
        data=tar_bytes
    )
    ...
```

### Advanced Features
- Model versioning (sir@v1 → sir@v2)
- Dependency graph analysis
- Incremental manifest updates
- Bundle compression options
- Parallel verification
- Result streaming
- Provenance tracking

## Benefits of This Approach

1. **No heavy closures**: Models discovered from manifest, not Python imports
2. **Perfect cache invalidation**: Each model has independent digest
3. **Deterministic builds**: Token hashing ensures stability
4. **Import isolation**: Models can't accidentally depend on undeclared code
5. **Git-friendly**: Manifest is human-readable and diffable
6. **Fast iteration**: Change one model without affecting others
7. **Cloud-agnostic**: Manifest is just JSON, works with any backend

## Migration Path

1. Keep existing `compile_entrypoint()` for local development
2. Add manifest generation as parallel system
3. Gradually migrate cloud workers to use manifest
4. Eventually deprecate global REGISTRY in favor of manifest

This design achieves all objectives while maintaining compatibility with the existing wire protocol infrastructure.
