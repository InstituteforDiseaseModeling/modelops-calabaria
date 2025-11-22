# Wire Protocol Architecture: Dependency-Free Execution

> **Legacy Notice:** Operational examples in this document mention the old
> `cb manifest` workflow. Those commands have been removed; ModelOps-bundle
> now handles manifests end-to-end. The protocol architecture described here
> remains accurate.

## The Critical Principle: ModelOps Never Imports Calabaria

**ModelOps is pure infrastructure**. It orchestrates distributed computation but has ZERO knowledge of science frameworks. The user's code (including Calabaria) runs in isolated processes, while ModelOps only handles the protocol boundary.

## Architecture Overview

```ascii
ModelOps Infrastructure          User's Bundle (Self-Contained)
┌─────────────────────────┐     ┌──────────────────────────────┐
│                         │     │                              │
│ ┌─────────────────────┐ │     │  requirements.txt:           │
│ │ Worker Process      │ │     │  ├─ modelops-calabaria       │
│ │ ┌─────────────────┐ │ │     │  ├─ polars                   │
│ │ │ Only imports:   │ │ │     │  └─ numpy                    │
│ │ │ - json          │ │ │     │                              │
│ │ │ - subprocess    │ │ │     │  src/models/sir.py:          │
│ │ │ - pathlib       │ │ │     │  ┌─────────────────────────┐ │
│ │ │                 │ │ │     │  │ from modelops_calabaria │ │
│ │ │ NO Calabaria!   │ │ │     │  │ import BaseModel        │ │
│ │ └─────────────────┘ │ │     │  │                         │ │
│ │                     │ │     │  │ class SIRModel(BaseModel) │
│ │ Reads manifest.json │ │     │  └─────────────────────────┘ │
│ │ Creates subprocess  │ │     │                              │
│ │ Sends JSON-RPC      │ │     │  manifest.json (metadata):   │
│ └─────────────────────┘ │     │  ┌─────────────────────────┐ │
│                         │     │  │ "sir@v1": {             │ │
│ ┌─────────────────────┐ │     │  │   "class": "models.sir:SIRModel"
│ │ Subprocess Pool     │ │     │  │   "param_specs": [...], │ │
│ │ ┌───────────────────┐ │     │  │   "scenarios": [...],   │ │
│ │ │ sys.path.insert   │ │     │  │   "outputs": [...]      │ │
│ │ │ (bundle_path)     │ │     │  │ }                       │ │
│ │ │                   │ │     │  └─────────────────────────┘ │
│ │ │ import models.sir │◄┼─────┼───── Bundle Path             │
│ │ │ # NOW has         │ │     │                              │
│ │ │ # Calabaria!      │ │     └──────────────────────────────┘
│ │ └───────────────────┘ │  
│ └─────────────────────┘ │
└─────────────────────────┘
```

## Why This Architecture is Critical

### 1. **Separation of Concerns**

- **ModelOps**: Infrastructure orchestration (Kubernetes, networking, storage)
- **Calabaria**: Science framework (parameter spaces, scenarios, models)
- **User Code**: Domain models (epidemiology, economics, etc.)

### 2. **Framework Agnostic**

ModelOps can run ANY science framework that implements the protocol:
- Calabaria models
- PyTorch models
- TensorFlow models
- Custom simulation frameworks
- R scripts (via subprocess)

### 3. **Dependency Isolation**

- ModelOps has minimal dependencies (just Python stdlib + basic libraries)
- User bundles are self-contained with their own dependencies
- No version conflicts between infrastructure and user code
- Easy to upgrade frameworks without touching infrastructure

## How the Wire Protocol Works

### Step 1: Bundle Creation (Development Time)

```python
# User's requirements.txt
modelops-calabaria==0.1.0
polars==0.20.0
numpy==1.24.0

# Calabria CLI compiles manifest
cb manifest build
# Creates manifest.json with all metadata needed for discovery
```

### Step 2: Cloud Discovery (Runtime)

```python
# ModelOps Worker Process - NO Calabria imports!
import json
import subprocess
from pathlib import Path

def execute_simulation(bundle_path: Path, sim_task: SimTask):
    # 1. Read manifest (pure JSON - no imports needed)
    with open(bundle_path / "manifest.json") as f:
        manifest = json.load(f)

    # 2. Parse entrypoint to find model
    model_id = extract_model_id(sim_task.entrypoint)
    model_info = manifest["models"][model_id]

    # 3. Create subprocess with bundle in Python path
    proc = subprocess.Popen([
        "python", "-c", "import json_rpc_server; json_rpc_server.run()"
    ], env={
        "PYTHONPATH": str(bundle_path),  # Bundle contains Calabaria
        "MODEL_CLASS": model_info["class"],
        "MANIFEST_PATH": str(bundle_path / "manifest.json")
    })

    # 4. Send simulation request via JSON-RPC
    return send_jsonrpc_request(proc, sim_task)
```

### Step 3: User Code Execution (Subprocess)

```python
# This runs in the subprocess - NOW Calabaria is available
import sys
import json
import os

# Bundle is in Python path, so we can import Calabaria
from modelops_calabaria.wire_protocol import EntryRecord, SerializedParameterSpec

def json_rpc_server():
    # 1. Load manifest and reconstruct EntryRecord
    with open(os.environ["MANIFEST_PATH"]) as f:
        manifest = json.load(f)

    model_class_path = os.environ["MODEL_CLASS"]
    module_path, class_name = model_class_path.split(":")

    # 2. Import user's model (which imports Calabaria)
    module = __import__(module_path, fromlist=[class_name])
    model_class = getattr(module, class_name)

    # 3. Create EntryRecord from manifest data
    entry = EntryRecord.from_manifest_data(manifest, model_class_path)

    # 4. Get wire function and execute
    wire_fn = entry.get_wire_factory()()

    # 5. Handle JSON-RPC requests
    while True:
        request = json.loads(sys.stdin.readline())
        result = wire_fn(**request["params"])
        response = {"result": result.to_json()}
        print(json.dumps(response))
```

## Key Benefits

### 1. **Zero Coupling**

```ascii
Dependencies Flow:

User Bundle ──depends on──> Calabaria ──depends on──> ModelOps-Contracts
     ▲                                                        ▲
     │                                                        │
ModelOps ──depends on──> ModelOps-Contracts ◄─────────────────┘

ModelOps NEVER depends on Calabaria!
ModelOps only depends on the stable protocol.
```

### 2. **Framework Evolution**
- Calabaria can change internals without affecting ModelOps
- New frameworks can implement the same protocol
- Infrastructure remains stable while science code evolves

### 3. **Bundle Self-Containment**
```
Bundle Contents:
├─ manifest.json          # Metadata (JSON, no Python)
├─ src/models/            # User's model code
├─ requirements.txt       # Including modelops-calabaria
├─ pyproject.toml         # Project config
└─ uv.lock               # Exact dependencies
```

When ModelOps unpacks this bundle and adds it to `PYTHONPATH`, the subprocess
can import everything it needs, including Calabria.

## Implementation Details

### Manifest as Bridge

The manifest.json serves as the bridge between infrastructure (ModelOps) and
implementation (Calabaria):

```json
{
  "models": {
    "sir@v1": {
      "class": "models.sir:SIRModel",
      "model_digest": "sha256:abc123...",
      "param_specs": [
        {"name": "beta", "min": 0.1, "max": 1.0, "kind": "float"},
        {"name": "gamma", "min": 0.05, "max": 0.5, "kind": "float"}
      ],
      "scenarios": ["baseline", "lockdown"],
      "outputs": ["infected", "recovered"]
    }
  }
}
```

ModelOps can read this JSON without importing any Python code, extract the
metadata it needs, and then spawn a subprocess where Calabaria is available.

### EntryRecord Reconstruction

```python
# In subprocess where Calabaria is available
from modelops_calabaria.wire_protocol import EntryRecord, SerializedParameterSpec, WireABI

def entry_from_manifest(model_info: dict, class_path: str) -> EntryRecord:
    """Reconstruct EntryRecord from manifest data."""
    module_path, class_name = class_path.split(":")

    param_specs = tuple(
        SerializedParameterSpec(**spec_data)
        for spec_data in model_info["param_specs"]
    )

    return EntryRecord(
        id=f"{module_path}.{class_name}",
        model_hash=model_info["model_digest"],
        abi_version=WireABI.V1,
        module_name=module_path,
        class_name=class_name,
        scenarios=tuple(model_info["scenarios"]),
        outputs=tuple(model_info["outputs"]),
        param_specs=param_specs
    )
```

## Comparison with Anti-Pattern

### ❌ WRONG: ModelOps Imports Calabaria

```python
# This would be TERRIBLE architecture
from modelops_calabaria import BaseModel, EntryRecord  # NO!

class ModelOpsWorker:
    def execute(self, sim_task):
        # This creates tight coupling
        entry = EntryRecord.from_bundle(bundle)  # BAD!
        wire = entry.get_wire_factory()()
        return wire(sim_task.params, sim_task.seed)
```

Problems:
- ModelOps now depends on Calabaria
- Can't run other frameworks
- Version conflicts between infrastructure and user code
- Tight coupling prevents evolution

### ✅ CORRECT: Protocol-Only Interface

```python
# ModelOps Worker - framework agnostic
import json
import subprocess

class ModelOpsWorker:
    def execute(self, sim_task: SimTask) -> SimReturn:
        # Only uses protocol types from modelops-contracts
        bundle = self.storage.get_bundle(sim_task.bundle_ref)

        # Reads JSON manifest (no Python imports)
        manifest = json.loads((bundle / "manifest.json").read_text())

        # Spawns subprocess where user's framework is available
        result = self.subprocess_pool.execute(bundle, sim_task)

        # Returns protocol type
        return SimReturn.from_json(result)
```

Benefits:
- Zero coupling to any specific framework
- Can run Calabaria, PyTorch, TensorFlow, etc.
- Infrastructure evolves independently of science code
- Clean protocol boundary

## Summary

The wire protocol architecture achieves perfect separation:

1. **ModelOps** reads JSON manifests and orchestrates execution
2. **User bundles** are self-contained with all dependencies
3. **Subprocesses** have access to framework code via `PYTHONPATH`
4. **Protocol types** provide the stable interface boundary

This enables ModelOps to be truly framework-agnostic infrastructure while still
providing rich, framework-specific features through the user's own bundles.

The key insight: **the wire protocol is not shipped as code, it's reconstructed
from metadata at runtime**. This breaks the dependency cycle while maintaining
full functionality.
