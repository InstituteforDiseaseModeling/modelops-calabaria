# Hardened Wire Protocol and EntryRecord Design

## Executive Summary

This document specifies the production-ready wire protocol for Calabaria models. It addresses critical issues in the original design:

- **Heavy closures** → Lightweight import paths
- **No versioning** → ABI versioning for protocol evolution
- **Mutable registry** → Thread-safe, validated registry
- **Non-serializable** → Fully JSON-serializable records

The hardened design ensures models can be safely deployed, discovered, and executed in distributed cloud environments.

## Problems Solved

### 1. Closure Weight
**Problem**: Original wire functions were closures capturing entire model classes, making them non-serializable and memory-heavy.

**Solution**: Store import paths (`module.Class`) and create wire functions on-demand via factories.

### 2. Protocol Evolution
**Problem**: No way to evolve the wire function signature without breaking existing deployments.

**Solution**: Explicit `WireABI` versioning allows multiple protocol versions to coexist.

### 3. Registry Safety
**Problem**: Global mutable dictionary could be corrupted by concurrent access.

**Solution**: Thread-safe `ModelRegistry` class with validation and locking.

### 4. Serialization
**Problem**: ParameterSpace and other objects weren't guaranteed serializable.

**Solution**: `SerializedParameterSpec` and explicit JSON serialization for all registry data.

## Core Design Principles

1. **No Heavy Closures**: Use import paths and factories, not captured state
2. **Explicit Versioning**: Every wire protocol has an ABI version
3. **Fully Serializable**: Everything in EntryRecord must serialize to JSON
4. **Thread-Safe**: Registry operations are protected by locks
5. **Validated Construction**: Fail fast with clear errors during registration

## Implementation

### Wire ABI Versioning

```python
from enum import Enum
from typing import Protocol, runtime_checkable, Dict, Any, Optional, List, Tuple, Type, Callable
from dataclasses import dataclass, field, asdict
import threading
import io
from datetime import datetime
from types import MappingProxyType

class WireABI(str, Enum):
    """Wire protocol ABI versions.

    Each version defines a specific function signature and behavior.
    New versions can be added without breaking existing deployments.
    """
    V1 = "calabaria.wire.v1"  # (params_M, seed, scenario_stack, outputs)
    V2 = "calabaria.wire.v2"  # Future: might add config overrides

    def describe(self) -> str:
        """Human-readable description of this ABI version."""
        descriptions = {
            self.V1: "Basic wire: params, seed, scenarios, outputs",
            self.V2: "Extended wire: adds config overrides (future)",
        }
        return descriptions.get(self, "Unknown ABI version")
```

### Wire Function Protocol

```python
@runtime_checkable
class WireFunction(Protocol):
    """Protocol for wire functions - enables validation.

    All wire functions must match this signature regardless of ABI version.
    Different ABI versions may interpret the parameters differently.
    """
    def __call__(self,
                 *,
                 params_M: Dict[str, Scalar],
                 seed: int,
                 scenario_stack: Tuple[str, ...] = ("baseline",),
                 outputs: Optional[List[str]] = None) -> WireResponse:
        """Execute model with given parameters and configuration.

        Args:
            params_M: Complete M-space parameters (no missing allowed)
            seed: Random seed for reproducibility
            scenario_stack: Ordered list of scenarios to apply (LWW semantics)
            outputs: Optional subset of outputs to return (None = all)

        Returns:
            WireResponse with serialized outputs and provenance
        """
        ...
```

### Serializable Parameter Specification

```python
@dataclass(frozen=True)
class SerializedParameterSpec:
    """Serializable parameter specification.

    Unlike ParameterSpec, this is guaranteed to be JSON-serializable
    and contains no callable transforms or complex types.
    """
    name: str
    min: float
    max: float
    kind: str  # "float" or "int"
    doc: str = ""

    def __post_init__(self):
        """Validate specification."""
        if self.kind not in ("float", "int"):
            raise ValueError(f"Invalid kind: {self.kind}")
        if self.min > self.max:
            raise ValueError(f"Invalid bounds: {self.min} > {self.max}")

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'SerializedParameterSpec':
        """Reconstruct from JSON dict."""
        return cls(**data)

    @classmethod
    def from_spec(cls, spec: ParameterSpec) -> 'SerializedParameterSpec':
        """Convert from regular ParameterSpec."""
        return cls(
            name=spec.name,
            min=float(spec.min),
            max=float(spec.max),
            kind=spec.kind,
            doc=spec.doc
        )
```

### Hardened EntryRecord

```python
@dataclass(frozen=True)
class EntryRecord:
    """Hardened registry entry for a model class.

    Key improvements over original design:
    1. ABI version for wire protocol evolution
    2. Import path instead of heavy closure
    3. Serializable parameter specs
    4. Validation methods
    5. Factory pattern for wire creation
    """
    # Identity
    id: str                                      # e.g., "examples.sir.SIRModel@a1b2c3d4"
    model_hash: str                              # Hash of code + registry
    abi_version: WireABI                         # Wire protocol version

    # Import path (not closure)
    module_name: str                             # e.g., "examples.sir"
    class_name: str                              # e.g., "SIRModel"

    # Discovery (all serializable)
    scenarios: Tuple[str, ...]                   # Immutable, sorted
    outputs: Tuple[str, ...]                     # Immutable, sorted
    param_specs: Tuple[SerializedParameterSpec, ...]  # Serializable specs

    # Metadata
    alias: Optional[str] = None                  # Human-friendly name
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def __post_init__(self):
        """Validate everything is serializable and properly typed."""
        # Ensure tuples not lists (immutability)
        for field_name in ['scenarios', 'outputs', 'param_specs']:
            val = getattr(self, field_name)
            if not isinstance(val, tuple):
                object.__setattr__(self, field_name, tuple(val))

        # Validate all scenarios are strings
        if not all(isinstance(s, str) for s in self.scenarios):
            raise TypeError("All scenarios must be strings")

        # Validate all outputs are strings
        if not all(isinstance(o, str) for o in self.outputs):
            raise TypeError("All outputs must be strings")

        # Validate all param_specs are correct type
        if not all(isinstance(p, SerializedParameterSpec) for p in self.param_specs):
            raise TypeError("All param_specs must be SerializedParameterSpec")

        # Validate serializable by attempting serialization
        try:
            self.to_json()
        except Exception as e:
            raise ValueError(f"EntryRecord not serializable: {e}")

    def to_json(self) -> Dict[str, Any]:
        """Export to JSON for storage/transmission.

        This is used for:
        - Persisting registry to disk
        - Transmitting model metadata to cloud
        - UI/dashboard display
        """
        return {
            "id": self.id,
            "model_hash": self.model_hash,
            "abi_version": self.abi_version.value,
            "module_name": self.module_name,
            "class_name": self.class_name,
            "scenarios": list(self.scenarios),
            "outputs": list(self.outputs),
            "param_specs": [s.to_json() for s in self.param_specs],
            "alias": self.alias,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'EntryRecord':
        """Reconstruct from JSON dict."""
        return cls(
            id=data["id"],
            model_hash=data["model_hash"],
            abi_version=WireABI(data["abi_version"]),
            module_name=data["module_name"],
            class_name=data["class_name"],
            scenarios=tuple(data["scenarios"]),
            outputs=tuple(data["outputs"]),
            param_specs=tuple(
                SerializedParameterSpec.from_json(p)
                for p in data["param_specs"]
            ),
            alias=data.get("alias"),
            created_at=data["created_at"]
        )

    def get_wire_factory(self) -> Callable[[], WireFunction]:
        """Get factory that creates wire functions.

        Returns a factory (not the wire itself) to avoid heavy closures.
        The factory imports the model class and creates the wire on demand.

        This enables:
        - Lazy loading of model code
        - Multiple wire instances if needed
        - No heavy closures in registry
        """
        def factory() -> WireFunction:
            # Dynamic import at wire creation time
            import importlib
            module = importlib.import_module(self.module_name)
            model_class = getattr(module, self.class_name)

            # Validate it's a BaseModel subclass
            if not issubclass(model_class, BaseModel):
                raise TypeError(
                    f"{self.module_name}.{self.class_name} is not a BaseModel"
                )

            # Create wire function for this ABI version
            if self.abi_version == WireABI.V1:
                return self._make_v1_wire(model_class)
            else:
                raise ValueError(f"Unknown ABI version: {self.abi_version}")

        return factory

    @staticmethod
    def _make_v1_wire(model_class: Type[BaseModel]) -> WireFunction:
        """Create V1 wire function for a model class.

        V1 Protocol:
        - Takes complete M-space parameters
        - Applies scenario stack in order (LWW)
        - Returns IPC-serialized DataFrames
        """
        def wire_v1(*,
                    params_M: Dict[str, Scalar],
                    seed: int,
                    scenario_stack: Tuple[str, ...] = ("baseline",),
                    outputs: Optional[List[str]] = None) -> WireResponse:

            # Create fresh model instance (stateless execution)
            model = model_class()
            model._seal()  # Ensure registries are frozen

            # Validate all scenarios exist
            for scenario_name in scenario_stack:
                if scenario_name not in model._scenarios:
                    available = sorted(model._scenarios.keys())
                    raise ValueError(
                        f"Unknown scenario: {scenario_name}. "
                        f"Available: {available}"
                    )

            # Convert to ParameterSet (validates completeness)
            pset = ParameterSet(model.space, params_M)

            # Apply scenario stack with LWW semantics
            config = dict(model.base_config)
            for scenario_name in scenario_stack:
                spec = model._scenarios[scenario_name]
                pset, config = spec.apply(pset, MappingProxyType(config))

            # Run simulation pipeline
            state = model.build_sim(pset, MappingProxyType(config))
            raw = model.run_sim(state, seed)

            # Extract all outputs
            all_outputs = model.extract_outputs(raw, seed)

            # Filter outputs if requested
            if outputs is not None:
                unknown = set(outputs) - set(all_outputs.keys())
                if unknown:
                    available = sorted(all_outputs.keys())
                    raise ValueError(
                        f"Unknown outputs: {sorted(unknown)}. "
                        f"Available: {available}"
                    )
                filtered = {k: v for k, v in all_outputs.items() if k in outputs}
            else:
                filtered = all_outputs

            # Serialize DataFrames to Arrow IPC format
            serialized = {}
            for name, df in filtered.items():
                buffer = io.BytesIO()
                df.write_ipc(buffer)
                serialized[name] = buffer.getvalue()

            # Build provenance information
            import hashlib
            param_hash = hashlib.sha256(
                str(sorted(params_M.items())).encode()
            ).hexdigest()[:16]

            return WireResponse(
                outputs=serialized,
                provenance={
                    "model_id": f"{model_class.__module__}.{model_class.__name__}",
                    "model_hash": model_class.__hash__() if hasattr(model_class, '__hash__') else "unknown",
                    "param_hash": param_hash,
                    "scenario_stack": scenario_stack,
                    "outputs": list(filtered.keys()),
                    "seed": seed,
                    "abi_version": WireABI.V1.value,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

        # Attach metadata for introspection
        wire_v1.__name__ = f"wire_v1_{model_class.__name__}"
        wire_v1.__doc__ = f"V1 wire function for {model_class.__name__}"

        return wire_v1
```

### Thread-Safe Model Registry

```python
class ModelRegistry:
    """Thread-safe, validated model registry.

    This replaces the global ENTRY_REGISTRY dict with a proper class that:
    - Validates entries on registration
    - Provides thread-safe access
    - Supports serialization/persistence
    - Enables model discovery
    """

    def __init__(self):
        """Initialize empty registry."""
        self._entries: Dict[str, EntryRecord] = {}
        self._lock = threading.Lock()
        self._creation_order: List[str] = []  # Track registration order

    def register(self, record: EntryRecord) -> None:
        """Register a model entry with validation.

        Validates:
        - Entry is properly formed
        - Wire function can be created
        - No duplicate IDs

        Thread-safe for concurrent registration.
        """
        # Validate wire function can be created
        try:
            factory = record.get_wire_factory()
            wire = factory()
            if not isinstance(wire, WireFunction):
                raise TypeError(
                    f"Wire function doesn't match WireFunction protocol"
                )
        except ImportError as e:
            raise ImportError(
                f"Cannot import model {record.module_name}.{record.class_name}: {e}"
            )
        except Exception as e:
            raise ValueError(
                f"Cannot create wire function for {record.id}: {e}"
            )

        # Thread-safe registration
        with self._lock:
            if record.id in self._entries:
                existing = self._entries[record.id]
                if existing.model_hash == record.model_hash:
                    # Same model, same hash - idempotent registration
                    return
                else:
                    raise ValueError(
                        f"Model {record.id} already registered with different hash. "
                        f"Existing: {existing.model_hash}, "
                        f"New: {record.model_hash}"
                    )

            self._entries[record.id] = record
            self._creation_order.append(record.id)

    def get(self, entry_id: str) -> Optional[EntryRecord]:
        """Get entry by ID (thread-safe)."""
        with self._lock:
            return self._entries.get(entry_id)

    def get_wire(self, entry_id: str) -> WireFunction:
        """Get wire function for a model.

        Convenience method that:
        1. Looks up the entry
        2. Gets its factory
        3. Creates the wire function

        Raises:
            KeyError: If model not found
            ImportError: If model can't be imported
        """
        entry = self.get(entry_id)
        if entry is None:
            available = self.list_models()
            raise KeyError(
                f"Model {entry_id} not found. "
                f"Available: {available}"
            )

        factory = entry.get_wire_factory()
        return factory()

    def list_models(self) -> List[str]:
        """List all registered model IDs in registration order."""
        with self._lock:
            return list(self._creation_order)

    def list_by_module(self) -> Dict[str, List[str]]:
        """List models grouped by module."""
        with self._lock:
            by_module = {}
            for entry_id, entry in self._entries.items():
                module = entry.module_name
                if module not in by_module:
                    by_module[module] = []
                by_module[module].append(entry_id)
            return by_module

    def search(self,
               module_pattern: Optional[str] = None,
               class_pattern: Optional[str] = None,
               has_scenario: Optional[str] = None,
               has_output: Optional[str] = None) -> List[str]:
        """Search for models matching criteria.

        Args:
            module_pattern: Glob pattern for module name
            class_pattern: Glob pattern for class name
            has_scenario: Must have this scenario
            has_output: Must have this output

        Returns:
            List of matching model IDs
        """
        import fnmatch

        matches = []
        with self._lock:
            for entry_id, entry in self._entries.items():
                # Check module pattern
                if module_pattern and not fnmatch.fnmatch(entry.module_name, module_pattern):
                    continue

                # Check class pattern
                if class_pattern and not fnmatch.fnmatch(entry.class_name, class_pattern):
                    continue

                # Check scenario requirement
                if has_scenario and has_scenario not in entry.scenarios:
                    continue

                # Check output requirement
                if has_output and has_output not in entry.outputs:
                    continue

                matches.append(entry_id)

        return sorted(matches)

    def to_json(self) -> Dict[str, Any]:
        """Export entire registry as JSON."""
        with self._lock:
            return {
                "version": "1.0",
                "created_at": datetime.utcnow().isoformat(),
                "entries": {
                    entry_id: record.to_json()
                    for entry_id, record in self._entries.items()
                }
            }

    def from_json(self, data: Dict[str, Any]) -> None:
        """Load registry from JSON (replaces current contents)."""
        with self._lock:
            self._entries.clear()
            self._creation_order.clear()

            for entry_id, entry_data in data.get("entries", {}).items():
                record = EntryRecord.from_json(entry_data)
                # Don't use register() to avoid import validation
                self._entries[record.id] = record
                self._creation_order.append(record.id)

    def clear(self) -> None:
        """Clear all entries (useful for testing)."""
        with self._lock:
            self._entries.clear()
            self._creation_order.clear()

# Global registry instance
REGISTRY = ModelRegistry()
```

### Wire Response Type

```python
@dataclass(frozen=True)
class WireResponse:
    """Response from wire function execution.

    Contains both the simulation outputs and provenance information
    for tracking and debugging.
    """
    outputs: Dict[str, bytes]        # Output name → Arrow IPC bytes
    provenance: Dict[str, Any]       # Execution metadata

    def get_dataframe(self, output_name: str) -> 'pl.DataFrame':
        """Convenience method to deserialize a specific output.

        Args:
            output_name: Name of output to retrieve

        Returns:
            Polars DataFrame

        Raises:
            KeyError: If output not found
        """
        import polars as pl

        if output_name not in self.outputs:
            available = sorted(self.outputs.keys())
            raise KeyError(
                f"Output '{output_name}' not found. "
                f"Available: {available}"
            )

        return pl.read_ipc(io.BytesIO(self.outputs[output_name]))

    def get_all_dataframes(self) -> Dict[str, 'pl.DataFrame']:
        """Deserialize all outputs to DataFrames."""
        import polars as pl

        return {
            name: pl.read_ipc(io.BytesIO(data))
            for name, data in self.outputs.items()
        }
```

### Model Compilation Integration

```python
# Extension to BaseModel for registration
class BaseModel:
    """Base model with hardened compilation support."""

    @classmethod
    def compile_entrypoint(cls,
                          alias: Optional[str] = None,
                          register: bool = True) -> EntryRecord:
        """Compile model to hardened entry record.

        Args:
            alias: Human-friendly name for the model
            register: Whether to register in global registry

        Returns:
            EntryRecord for this model
        """
        # Create temporary instance for introspection
        temp_instance = cls()
        temp_instance._seal()  # Ensure everything is registered

        # Build serializable parameter specs
        param_specs = tuple(
            SerializedParameterSpec.from_spec(spec)
            for spec in temp_instance.space.specs
        )

        # Generate stable model hash
        import hashlib
        model_code = inspect.getsource(cls)
        model_hash = hashlib.sha256(model_code.encode()).hexdigest()[:16]

        # Create entry record
        record = EntryRecord(
            id=f"{cls.__module__}.{cls.__name__}@{model_hash[:8]}",
            model_hash=model_hash,
            abi_version=WireABI.V1,
            module_name=cls.__module__,
            class_name=cls.__name__,
            scenarios=tuple(sorted(temp_instance._scenarios.keys())),
            outputs=tuple(sorted(temp_instance._outputs.keys())),
            param_specs=param_specs,
            alias=alias or cls.__name__
        )

        # Register globally if requested
        if register:
            REGISTRY.register(record)

        return record
```

## Usage Examples

### Basic Model Registration

```python
from modelops_calabaria import BaseModel, ParameterSpec, ParameterSpace

class SIRModel(BaseModel):
    def __init__(self):
        space = ParameterSpace([
            ParameterSpec("beta", 0.0, 1.0, doc="Transmission rate"),
            ParameterSpec("gamma", 0.0, 1.0, doc="Recovery rate"),
            ParameterSpec("population", 100, 1_000_000, kind="int"),
        ])
        super().__init__(space)

    # ... implement build_sim, run_sim, etc ...

# Register the model
record = SIRModel.compile_entrypoint(alias="SIR Epidemic Model")
print(f"Registered: {record.id}")
```

### Using the Registry

```python
# List all models
models = REGISTRY.list_models()
print(f"Available models: {models}")

# Search for models
epidemic_models = REGISTRY.search(module_pattern="*epidemic*")
models_with_vaccination = REGISTRY.search(has_scenario="vaccination")

# Get and execute a model
entry = REGISTRY.get("examples.sir.SIRModel@a1b2c3d4")
wire = REGISTRY.get_wire(entry.id)

# Execute the wire function
response = wire(
    params_M={"beta": 0.3, "gamma": 0.1, "population": 10000},
    seed=42,
    scenario_stack=("baseline", "lockdown"),
    outputs=["incidence", "prevalence"]
)

# Get results
incidence_df = response.get_dataframe("incidence")
```

### Persisting the Registry

```python
# Save registry to JSON
registry_data = REGISTRY.to_json()
with open("model_registry.json", "w") as f:
    json.dump(registry_data, f, indent=2)

# Load registry from JSON
with open("model_registry.json", "r") as f:
    data = json.load(f)
    REGISTRY.from_json(data)
```

### Cloud Deployment

```python
def cloud_runner(entry_id: str, params: Dict, seed: int) -> WireResponse:
    """Example cloud runner using the registry."""
    # Get the wire function
    wire = REGISTRY.get_wire(entry_id)

    # Execute with cloud-provided parameters
    return wire(
        params_M=params,
        seed=seed,
        scenario_stack=("baseline",),  # Could come from job spec
        outputs=None  # Return all outputs
    )
```

## Migration Notes

### From Old Design

1. **Replace closure-based wires** with import path + factory pattern
2. **Add ABI version** to all EntryRecords
3. **Convert ParameterSpace** to SerializedParameterSpec list
4. **Replace global dict** with ModelRegistry instance
5. **Add validation** at registration time

### Compatibility

The hardened design is backward-compatible at the API level:
- `compile_entrypoint()` has the same signature
- Wire functions have the same calling convention
- Registry lookup works the same way

### Performance Impact

- **Positive**: No heavy closures in memory
- **Positive**: Lazy model loading via import
- **Neutral**: Small overhead for factory pattern
- **Positive**: JSON serialization enables caching

## Future Extensions

### Wire ABI V2
```python
# Future V2 wire with config overrides
def wire_v2(*,
            params_M: Dict[str, Scalar],
            seed: int,
            scenario_stack: Tuple[str, ...] = ("baseline",),
            outputs: Optional[List[str]] = None,
            config_overrides: Optional[Dict[str, Any]] = None) -> WireResponse:
    # V2 implementation with additional config support
    pass
```

### Registry Federation
```python
class FederatedRegistry:
    """Registry that can pull from multiple sources."""

    def __init__(self, registries: List[ModelRegistry]):
        self.registries = registries

    def search_all(self, **criteria) -> List[Tuple[str, str]]:
        """Search across all federated registries."""
        # Returns (registry_name, model_id) pairs
        pass
```

### Model Versioning
```python
@dataclass(frozen=True)
class VersionedEntryRecord(EntryRecord):
    """Entry with semantic versioning."""
    version: str  # e.g., "1.2.3"
    previous_versions: Tuple[str, ...] = ()  # Previous entry IDs
```

## Summary

This hardened design provides:

1. **Production Safety**: Thread-safe, validated, versioned
2. **Cloud Ready**: Fully serializable, no heavy closures
3. **Discoverable**: Rich search and introspection capabilities
4. **Evolvable**: ABI versioning for protocol evolution
5. **Performant**: Lazy loading, lightweight registry

The design maintains the simplicity of the original API while adding the robustness required for production deployment.
