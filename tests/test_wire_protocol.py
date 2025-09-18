"""Test wire protocol implementation."""

import io
from typing import Dict, Any, Mapping

import polars as pl
import pytest

from modelops_calabaria import (
    BaseModel,
    ParameterSpec,
    ParameterSpace,
    ParameterSet,
    WireABI,
    SerializedParameterSpec,
    WireResponse,
    EntryRecord,
    REGISTRY,
    model_output,
    model_scenario,
    ScenarioSpec,
    SEED_COL,
)


class SimpleSIRModel(BaseModel):
    """Test model for wire protocol."""

    def __init__(self, space=None, base_config=None):
        """Initialize with SIR parameter space."""
        if space is None:
            space = ParameterSpace([
                ParameterSpec("beta", min=0.1, max=1.0, kind="float", doc="Transmission rate"),
                ParameterSpec("gamma", min=0.05, max=0.5, kind="float", doc="Recovery rate"),
                ParameterSpec("population", min=1000, max=10000, kind="int", doc="Total population"),
            ])
        super().__init__(space, base_config)

    @model_output("infected")
    def extract_infected(self, raw: pl.DataFrame, seed: int) -> pl.DataFrame:
        """Extract infected time series."""
        return raw.select(["time", "infected"])

    @model_output("recovered")
    def extract_recovered(self, raw: pl.DataFrame, seed: int) -> pl.DataFrame:
        """Extract recovered time series."""
        return raw.select(["time", "recovered"])

    @model_scenario("lockdown")
    def lockdown_scenario(self) -> ScenarioSpec:
        """Lockdown reduces transmission."""
        return ScenarioSpec(
            name="lockdown",
            doc="Lockdown reduces beta by 50%",
            param_patches={"beta": 0.2},
        )

    def build_sim(self, params: ParameterSet, config: Mapping[str, Any]) -> pl.DataFrame:
        """Build simple SIR simulation state."""
        # Just return a simple DataFrame for testing
        beta = params["beta"]
        gamma = params["gamma"]
        pop = params["population"]

        return pl.DataFrame({
            "time": range(10),
            "susceptible": [pop - i * 10 for i in range(10)],
            "infected": [i * 10 for i in range(10)],
            "recovered": [0] + [i * 5 for i in range(1, 10)],
            "beta": [beta] * 10,
            "gamma": [gamma] * 10,
        })

    def run_sim(self, state: pl.DataFrame, seed: int) -> pl.DataFrame:
        """Run simulation (just returns state for testing)."""
        # In real model this would run actual simulation
        return state


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
        id="test.model.TestModel@abcd1234",
        model_hash="abcd1234",
        abi_version=WireABI.V1,
        module_name="test.model",
        class_name="TestModel",
        scenarios=("baseline", "lockdown"),
        outputs=("infected", "recovered"),
        param_specs=param_specs,
        alias="Test SIR Model"
    )

    # Test JSON serialization
    json_data = entry.to_json()
    assert json_data["id"] == "test.model.TestModel@abcd1234"
    assert json_data["abi_version"] == "calabaria.wire.v1"
    assert json_data["scenarios"] == ["baseline", "lockdown"]
    assert len(json_data["param_specs"]) == 2

    # Test JSON deserialization
    reconstructed = EntryRecord.from_json(json_data)
    assert reconstructed.id == entry.id
    assert reconstructed.scenarios == entry.scenarios
    assert reconstructed.param_specs == entry.param_specs


def test_compile_entrypoint():
    """Test BaseModel.compile_entrypoint() method."""
    # Clear registry first
    REGISTRY.clear()

    # Compile the model
    model_class = SimpleSIRModel
    space = ParameterSpace([
        ParameterSpec("beta", min=0.1, max=1.0, kind="float"),
        ParameterSpec("gamma", min=0.05, max=0.5, kind="float"),
        ParameterSpec("population", min=1000, max=10000, kind="int"),
    ])

    entry = model_class.compile_entrypoint(space)

    # Check entry properties
    assert "SimpleSIRModel" in entry.id
    assert entry.module_name == "test_wire_protocol"
    assert entry.class_name == "SimpleSIRModel"
    assert entry.abi_version == WireABI.V1

    # Check scenarios (should include baseline + lockdown)
    assert "baseline" in entry.scenarios
    assert "lockdown" in entry.scenarios

    # Check outputs
    assert "infected" in entry.outputs
    assert "recovered" in entry.outputs

    # Check parameter specs
    assert len(entry.param_specs) == 3
    param_names = {spec.name for spec in entry.param_specs}
    assert param_names == {"beta", "gamma", "population"}

    # Check it was registered
    retrieved = REGISTRY.get(entry.id)
    assert retrieved == entry


def test_wire_function_v1():
    """Test V1 wire function execution."""
    # Clear registry and compile model
    REGISTRY.clear()
    model_class = SimpleSIRModel
    space = ParameterSpace([
        ParameterSpec("beta", min=0.1, max=1.0, kind="float"),
        ParameterSpec("gamma", min=0.05, max=0.5, kind="float"),
        ParameterSpec("population", min=1000, max=10000, kind="int"),
    ])

    entry = model_class.compile_entrypoint(space)

    # Get wire function
    wire_fn = REGISTRY.get_wire(entry.id)

    # Execute with baseline scenario
    params_M = {"beta": 0.3, "gamma": 0.1, "population": 5000}
    response = wire_fn(
        params_M=params_M,
        seed=42,
        scenario_stack=("baseline",),
        outputs=["infected"]
    )

    # Check response
    assert isinstance(response, WireResponse)
    assert "infected" in response.outputs
    assert response.provenance["seed"] == 42
    assert response.provenance["scenario_stack"] == ("baseline",)

    # Check DataFrame has seed column
    df = response.get_dataframe("infected")
    assert SEED_COL in df.columns
    assert df[SEED_COL][0] == 42

    # Test with lockdown scenario
    response2 = wire_fn(
        params_M=params_M,
        seed=43,
        scenario_stack=("baseline", "lockdown"),
        outputs=None  # Get all outputs
    )

    assert len(response2.outputs) == 2
    assert "infected" in response2.outputs
    assert "recovered" in response2.outputs

    # Test error on unknown scenario
    with pytest.raises(ValueError, match="Unknown scenario: invalid"):
        wire_fn(
            params_M=params_M,
            seed=44,
            scenario_stack=("invalid",)
        )

    # Test error on unknown output
    with pytest.raises(ValueError, match="Unknown outputs"):
        wire_fn(
            params_M=params_M,
            seed=45,
            scenario_stack=("baseline",),
            outputs=["nonexistent"]
        )


def test_registry_operations():
    """Test ModelRegistry operations."""
    # Start fresh
    REGISTRY.clear()

    # Register a model
    model_class = SimpleSIRModel
    space = ParameterSpace([
        ParameterSpec("beta", min=0.1, max=1.0, kind="float"),
        ParameterSpec("gamma", min=0.05, max=0.5, kind="float"),
        ParameterSpec("population", min=1000, max=10000, kind="int"),
    ])

    entry = model_class.compile_entrypoint(space)

    # Test list_models
    models = REGISTRY.list_models()
    assert entry.id in models

    # Test list_by_module
    by_module = REGISTRY.list_by_module()
    assert "test_wire_protocol" in by_module
    assert entry.id in by_module["test_wire_protocol"]

    # Test search
    matches = REGISTRY.search(class_pattern="*SIR*")
    assert entry.id in matches

    matches = REGISTRY.search(has_scenario="lockdown")
    assert entry.id in matches

    matches = REGISTRY.search(has_output="infected")
    assert entry.id in matches

    matches = REGISTRY.search(has_scenario="nonexistent")
    assert entry.id not in matches

    # Test JSON export/import
    json_data = REGISTRY.to_json()
    assert "entries" in json_data
    assert entry.id in json_data["entries"]

    # Clear and reload
    REGISTRY.clear()
    assert len(REGISTRY.list_models()) == 0

    REGISTRY.from_json(json_data)
    assert len(REGISTRY.list_models()) == 1
    assert REGISTRY.get(entry.id).id == entry.id


def test_registry_duplicate_handling():
    """Test handling of duplicate model registrations."""
    REGISTRY.clear()

    # Create and register model
    model_class = SimpleSIRModel
    space = ParameterSpace([
        ParameterSpec("beta", min=0.1, max=1.0, kind="float"),
        ParameterSpec("gamma", min=0.05, max=0.5, kind="float"),
        ParameterSpec("population", min=1000, max=10000, kind="int"),
    ])

    entry1 = model_class.compile_entrypoint(space)

    # Registering same model again should be idempotent
    entry2 = model_class.compile_entrypoint(space)
    assert entry1.id == entry2.id
    assert entry1.model_hash == entry2.model_hash

    # Only one entry in registry
    assert len(REGISTRY.list_models()) == 1