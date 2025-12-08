"""Wire protocol core types for Calabaria models.

This module provides the essential wire protocol types:
- SerializedParameterSpec for parameter serialization
- WireResponse for execution results
- WireABI for protocol versioning

For stateless wire function creation, see wire_loader.py.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import io

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
    lower: Union[int, float]
    upper: Union[int, float]
    kind: str  # "float" or "int"
    doc: str = ""

    def __post_init__(self):
        """Validate specification."""
        if self.kind not in ("float", "int"):
            raise ValueError(f"Invalid kind: {self.kind}")
        if self.lower > self.upper:
            raise ValueError(f"Invalid bounds: {self.lower} > {self.upper}")

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'SerializedParameterSpec':
        """Reconstruct from JSON dict."""
        return cls(**data)

    @classmethod
    def from_spec(cls, spec: ParameterSpec) -> 'SerializedParameterSpec':
        """Convert from regular ParameterSpec."""
        # Preserve integer precision for int parameters
        if spec.kind == "int":
            lower_val = int(spec.lower)
            upper_val = int(spec.upper)
        else:
            lower_val = float(spec.lower)
            upper_val = float(spec.upper)

        return cls(
            name=spec.name,
            lower=lower_val,
            upper=upper_val,
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


