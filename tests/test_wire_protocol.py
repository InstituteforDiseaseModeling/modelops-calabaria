"""Test wire protocol implementation."""

import io

import polars as pl
import pytest

from modelops_calabaria import (
    WireABI,
    SerializedParameterSpec,
    WireResponse,
    ParameterSpec,
)
from modelops_calabaria.wire_loader import EntryRecord




def test_serialized_parameter_spec():
    """Test SerializedParameterSpec creation and JSON round-trip."""
    # Create from ParameterSpec
    spec = ParameterSpec("beta", min=0.1, max=1.0, kind="float", doc="Transmission")
    serialized = SerializedParameterSpec.from_spec(spec)

    assert serialized.name == "beta"
    assert serialized.min == 0.1
    assert serialized.max == 1.0
    assert serialized.kind == "float"
    assert serialized.doc == "Transmission"

    # Test JSON round-trip
    json_data = serialized.to_json()
    assert json_data["name"] == "beta"
    assert json_data["min"] == 0.1

    reconstructed = SerializedParameterSpec.from_json(json_data)
    assert reconstructed == serialized

    # Test validation
    with pytest.raises(ValueError, match="Invalid kind"):
        SerializedParameterSpec("bad", 0, 1, kind="complex")

    with pytest.raises(ValueError, match="Invalid bounds"):
        SerializedParameterSpec("bad", 10, 5, kind="float")


def test_wire_response():
    """Test WireResponse with DataFrame serialization."""
    # Create test DataFrames
    df1 = pl.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    df2 = pl.DataFrame({"a": [7, 8], "b": [9, 10]})

    # Serialize to Arrow IPC
    buf1 = io.BytesIO()
    df1.write_ipc(buf1)
    buf2 = io.BytesIO()
    df2.write_ipc(buf2)

    # Create response
    response = WireResponse(
        outputs={"out1": buf1.getvalue(), "out2": buf2.getvalue()},
        provenance={"model": "test", "seed": 42}
    )

    # Test single DataFrame retrieval
    retrieved = response.get_dataframe("out1")
    assert retrieved.equals(df1)

    # Test all DataFrames retrieval
    all_dfs = response.get_all_dataframes()
    assert len(all_dfs) == 2
    assert all_dfs["out1"].equals(df1)
    assert all_dfs["out2"].equals(df2)

    # Test missing output
    with pytest.raises(KeyError, match="Output 'missing' not found"):
        response.get_dataframe("missing")


def test_entry_record_creation():
    """Test EntryRecord creation and validation."""
    param_specs = (
        SerializedParameterSpec("beta", 0.1, 1.0, "float"),
        SerializedParameterSpec("gamma", 0.05, 0.5, "float"),
    )

    entry = EntryRecord(
        class_path="test.model:TestModel",
        model_digest="sha256:abcd1234",
        abi_version=WireABI.V1,
        param_specs=param_specs,
        scenarios=("baseline", "lockdown"),
        outputs=("infected", "recovered")
    )

    # Test attributes
    assert entry.class_path == "test.model:TestModel"
    assert entry.model_digest == "sha256:abcd1234"
    assert entry.abi_version == WireABI.V1
    assert entry.scenarios == ("baseline", "lockdown")
    assert entry.outputs == ("infected", "recovered")
    assert len(entry.param_specs) == 2

    # Test immutability
    with pytest.raises(AttributeError):
        entry.class_path = "different.path:Class"


