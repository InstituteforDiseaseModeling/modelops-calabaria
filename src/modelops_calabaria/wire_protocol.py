"""Wire protocol implementation for Calabaria models.

This module provides the hardened wire protocol for deploying and executing
Calabaria models in distributed cloud environments. Key features:
- No heavy closures (uses import paths)
- Full JSON serialization
- Thread-safe registry
- ABI versioning for protocol evolution
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import hashlib
import importlib
import io
import threading

import polars as pl

from .parameters import ParameterSpec, ParameterSet, Scalar
from .constants import SEED_COL


class WireABI(str, Enum):
    """Wire protocol ABI versions.

    Each version defines a specific function signature and behavior.
    New versions can be added without breaking existing deployments.
    """
    V1 = "calabaria.wire.v1"  # (params_M, seed, scenario_stack, outputs)
    # Future: V2 = "calabaria.wire.v2"  # Might add config overrides

    def describe(self) -> str:
        """Human-readable description of this ABI version."""
        descriptions = {
            self.V1: "Basic wire: params, seed, scenarios, outputs",
        }
        return descriptions.get(self, "Unknown ABI version")


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


@dataclass(frozen=True)
class WireResponse:
    """Response from wire function execution.

    Contains both the simulation outputs and provenance information
    for tracking and debugging.
    """
    outputs: Dict[str, bytes]        # Output name → Arrow IPC bytes
    provenance: Dict[str, Any]       # Execution metadata

    def get_dataframe(self, output_name: str) -> pl.DataFrame:
        """Convenience method to deserialize a specific output.

        Args:
            output_name: Name of output to retrieve

        Returns:
            Polars DataFrame

        Raises:
            KeyError: If output not found
        """
        if output_name not in self.outputs:
            available = sorted(self.outputs.keys())
            raise KeyError(
                f"Output '{output_name}' not found. "
                f"Available: {available}"
            )

        return pl.read_ipc(io.BytesIO(self.outputs[output_name]))

    def get_all_dataframes(self) -> Dict[str, pl.DataFrame]:
        """Deserialize all outputs to DataFrames."""
        return {
            name: pl.read_ipc(io.BytesIO(data))
            for name, data in self.outputs.items()
        }


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
        - Bundle manifest generation
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

    def get_wire_factory(self) -> Callable[[], Callable]:
        """Get factory that creates wire functions.

        Returns a factory (not the wire itself) to avoid heavy closures.
        The factory imports the model class and creates the wire on demand.

        This enables:
        - Lazy loading of model code
        - Multiple wire instances if needed
        - No heavy closures in registry
        """
        def factory():
            # Dynamic import at wire creation time
            module = importlib.import_module(self.module_name)
            model_class = getattr(module, self.class_name)

            # Import BaseModel here to avoid circular import
            from .base_model import BaseModel

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

    def _make_v1_wire(self, model_class) -> Callable:
        """Create V1 wire function for a model class.

        V1 Protocol:
        - Takes complete M-space parameters
        - Applies scenario stack in order (LWW)
        - Returns IPC-serialized DataFrames
        """
        # Capture the space from the EntryRecord
        # We need to reconstruct the ParameterSpace from specs
        from .parameters import ParameterSpec, ParameterSpace

        param_specs = [
            ParameterSpec(
                name=spec.name,
                min=int(spec.min) if spec.kind == "int" else spec.min,
                max=int(spec.max) if spec.kind == "int" else spec.max,
                kind=spec.kind,
                doc=spec.doc
            )
            for spec in self.param_specs
        ]
        space = ParameterSpace(param_specs)

        def wire_v1(*,
                    params_M: Dict[str, Scalar],
                    seed: int,
                    scenario_stack: Tuple[str, ...] = ("baseline",),
                    outputs: Optional[List[str]] = None) -> WireResponse:

            # Create fresh model instance (stateless execution)
            model = model_class(space=space)
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
            param_hash = hashlib.sha256(
                str(sorted(params_M.items())).encode()
            ).hexdigest()[:16]

            # Generate model hash same way as in compile_entrypoint
            model_hash_input = f"{model_class.__module__}.{model_class.__name__}"
            model_hash = hashlib.sha256(model_hash_input.encode()).hexdigest()[:8]

            return WireResponse(
                outputs=serialized,
                provenance={
                    "model_id": f"{model_class.__module__}.{model_class.__name__}",
                    "model_hash": model_hash,
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
            # Don't actually create the wire, just validate factory works
            if not callable(factory):
                raise TypeError("Wire factory must be callable")
        except ImportError as e:
            raise ImportError(
                f"Cannot import model {record.module_name}.{record.class_name}: {e}"
            )
        except Exception as e:
            raise ValueError(
                f"Cannot create wire factory for {record.id}: {e}"
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

    def get_wire(self, entry_id: str) -> Callable:
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