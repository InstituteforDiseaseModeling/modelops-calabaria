"""Tests for stateless wire loader functionality.

These tests verify that wire functions can be created on-demand from manifest data
without global state or registration side effects.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import json
import io

import polars as pl

from modelops_calabaria.wire_loader import (
    EntryRecord, entry_from_manifest, make_wire, make_wire_from_manifest
)
from modelops_calabaria.wire_protocol import WireABI, SerializedParameterSpec, WireResponse
from modelops_calabaria.parameters import (
    ParameterSpace, ParameterSpec, ConfigSpec, ConfigurationSpace, ConfigurationSet
)
from modelops_calabaria.base_model import BaseModel
from modelops_calabaria.decorators import model_output, model_scenario
from modelops_calabaria.scenarios import ScenarioSpec
from modelops_calabaria.constants import SEED_COL


class DummyModel(BaseModel):
    """Test model for wire loader tests."""

    def __init__(self, space):
        config_space = ConfigurationSpace([])
        base_config = ConfigurationSet(config_space, {})
        super().__init__(space, config_space, base_config)

    @model_output("infected")
    def infected(self, raw, seed):
        return pl.DataFrame({"count": [100, 90, 80]})

    @model_output("recovered")
    def recovered(self, raw, seed):
        return pl.DataFrame({"count": [0, 10, 20]})

    @model_scenario("lockdown")
    def lockdown(self):
        return ScenarioSpec("lockdown")

    def build_sim(self, params, config):
        return {"simulation_ready": True}

    def run_sim(self, state, seed):
        return {"data": "simulated"}


class TestEntryRecord:
    """Tests for EntryRecord value object."""

    def test_create_entry_record(self):
        """Should create immutable EntryRecord."""
        param_spec = SerializedParameterSpec(
            name="beta", lower=0.1, upper=1.0, kind="float", doc="Transmission rate"
        )

        entry = EntryRecord(
            class_path="test.module:TestModel",
            model_digest="sha256:abc123",
            abi_version=WireABI.V1,
            param_specs=(param_spec,),
            scenarios=("baseline", "lockdown"),
            outputs=("infected", "recovered")
        )

        assert entry.class_path == "test.module:TestModel"
        assert entry.model_digest == "sha256:abc123"
        assert entry.abi_version == WireABI.V1
        assert len(entry.param_specs) == 1
        assert entry.param_specs[0].name == "beta"
        assert entry.scenarios == ("baseline", "lockdown")
        assert entry.outputs == ("infected", "recovered")

    def test_entry_record_immutable(self):
        """Should be immutable dataclass."""
        entry = EntryRecord(
            class_path="test.module:TestModel",
            model_digest="sha256:abc123",
            abi_version=WireABI.V1,
            param_specs=(),
            scenarios=(),
            outputs=()
        )

        with pytest.raises(AttributeError):
            entry.class_path = "different.path"


class TestEntryFromManifest:
    """Tests for creating EntryRecord from manifest data."""

    def test_entry_from_manifest_success(self):
        """Should create EntryRecord from valid manifest."""
        manifest = {
            "models": {
                "models.sir:SIRModel": {
                    "model_digest": "sha256:def456",
                    "param_specs": [
                        {
                            "name": "beta",
                            "lower": 0.1,
                            "upper": 1.0,
                            "kind": "float",
                            "doc": "Transmission rate"
                        },
                        {
                            "name": "gamma",
                            "lower": 0.05,
                            "upper": 0.5,
                            "kind": "float",
                            "doc": "Recovery rate"
                        }
                    ],
                    "scenarios": ["baseline", "lockdown"],
                    "outputs": ["infected", "recovered"]
                }
            }
        }

        entry = entry_from_manifest("models.sir:SIRModel", manifest)

        assert entry.class_path == "models.sir:SIRModel"
        assert entry.model_digest == "sha256:def456"
        assert entry.abi_version == WireABI.V1
        assert len(entry.param_specs) == 2
        assert entry.param_specs[0].name == "beta"
        assert entry.param_specs[1].name == "gamma"
        assert entry.scenarios == ("baseline", "lockdown")
        assert entry.outputs == ("infected", "recovered")

    def test_entry_from_manifest_model_not_found(self):
        """Should raise KeyError if model not in manifest."""
        manifest = {"models": {"other.model:OtherModel": {}}}

        with pytest.raises(KeyError, match="Model models.sir:SIRModel not found"):
            entry_from_manifest("models.sir:SIRModel", manifest)


class TestMakeWire:
    """Tests for creating wire functions from EntryRecord."""

    def test_make_wire_success(self):
        """Should create working wire function."""
        # Create entry record
        param_specs = (
            SerializedParameterSpec(
                name="beta", lower=0.1, upper=1.0, kind="float", doc="Transmission rate"
            ),
            SerializedParameterSpec(
                name="gamma", lower=0.05, upper=0.5, kind="float", doc="Recovery rate"
            )
        )

        entry = EntryRecord(
            class_path="test_wire_loader:DummyModel",
            model_digest="sha256:test123",
            abi_version=WireABI.V1,
            param_specs=param_specs,
            scenarios=("baseline", "lockdown"),
            outputs=("infected", "recovered")
        )

        # Create wire function
        wire_fn = make_wire(entry)

        # Test execution
        params = {"beta": 0.3, "gamma": 0.1}
        response = wire_fn(
            params_M=params,
            seed=42,
            scenario_stack=("baseline",),
            outputs=["infected"]
        )

        assert isinstance(response, WireResponse)
        assert "infected" in response.outputs
        assert response.provenance["seed"] == 42
        assert response.provenance["model_digest"] == "sha256:test123"

    def test_make_wire_invalid_class_path(self):
        """Should raise ValueError for invalid class path."""
        entry = EntryRecord(
            class_path="invalid_format",  # Missing colon
            model_digest="sha256:test123",
            abi_version=WireABI.V1,
            param_specs=(),
            scenarios=(),
            outputs=()
        )

        with pytest.raises(ImportError, match="Expected 'module_or_file:Symbol' format"):
            make_wire(entry)

    def test_make_wire_unsupported_abi(self):
        """Should raise ValueError for unsupported ABI version."""
        # This would require creating a new WireABI enum value
        # For now, test with V1 since it's the only supported version
        pass

    def test_make_wire_import_error(self):
        """Should raise ImportError if model class cannot be imported."""
        entry = EntryRecord(
            class_path="nonexistent.module:NonexistentClass",
            model_digest="sha256:test123",
            abi_version=WireABI.V1,
            param_specs=(),
            scenarios=(),
            outputs=()
        )

        with pytest.raises(ImportError, match="Cannot import"):
            make_wire(entry)

    def test_make_wire_scenario_validation(self):
        """Should validate scenarios exist in model."""
        param_specs = (
            SerializedParameterSpec(
                name="beta", lower=0.1, upper=1.0, kind="float", doc="Transmission rate"
            ),
        )

        entry = EntryRecord(
            class_path="test_wire_loader:DummyModel",
            model_digest="sha256:test123",
            abi_version=WireABI.V1,
            param_specs=param_specs,
            scenarios=("baseline", "lockdown"),
            outputs=("infected",)
        )

        wire_fn = make_wire(entry)

        with pytest.raises(ValueError, match="Unknown scenario: invalid"):
            wire_fn(
                params_M={"beta": 0.3},
                seed=42,
                scenario_stack=("invalid",)
            )

    def test_make_wire_output_filtering(self):
        """Should filter outputs when requested."""
        param_specs = (
            SerializedParameterSpec(
                name="beta", lower=0.1, upper=1.0, kind="float", doc="Transmission rate"
            ),
        )

        entry = EntryRecord(
            class_path="test_wire_loader:DummyModel",
            model_digest="sha256:test123",
            abi_version=WireABI.V1,
            param_specs=param_specs,
            scenarios=("baseline",),
            outputs=("infected", "recovered")
        )

        wire_fn = make_wire(entry)

        # Test filtering to specific output
        response = wire_fn(
            params_M={"beta": 0.3},
            seed=42,
            outputs=["infected"]
        )

        assert "infected" in response.outputs
        assert "recovered" not in response.outputs

        # Test error on unknown output
        with pytest.raises(ValueError, match="Unknown outputs"):
            wire_fn(
                params_M={"beta": 0.3},
                seed=42,
                outputs=["nonexistent"]
            )


class TestMakeWireFromManifest:
    """Tests for convenience function to create wire from manifest."""

    def test_make_wire_from_manifest(self):
        """Should create wire function directly from manifest."""
        manifest = {
            "models": {
                "test_wire_loader:DummyModel": {
                    "model_digest": "sha256:manifest123",
                    "param_specs": [
                        {
                            "name": "beta",
                            "lower": 0.1,
                            "upper": 1.0,
                            "kind": "float",
                            "doc": "Transmission rate"
                        }
                    ],
                    "scenarios": ["baseline"],
                    "outputs": ["infected"]
                }
            }
        }

        wire_fn = make_wire_from_manifest("test_wire_loader:DummyModel", manifest)

        response = wire_fn(
            params_M={"beta": 0.3},
            seed=42
        )

        assert isinstance(response, WireResponse)
        assert "infected" in response.outputs
        assert response.provenance["model_digest"] == "sha256:manifest123"


class TestStatelessExecution:
    """Tests verifying stateless execution properties."""

    def test_no_global_state(self):
        """Should not create any global state."""
        # Import the module fresh to check for global state
        import importlib
        from modelops_calabaria import wire_loader
        importlib.reload(wire_loader)

        # Creating entries and wire functions should not modify module state
        entry = EntryRecord(
            class_path="test_wire_loader:DummyModel",
            model_digest="sha256:test123",
            abi_version=WireABI.V1,
            param_specs=(),
            scenarios=(),
            outputs=()
        )

        # Module should not have any registry or global state
        assert not hasattr(wire_loader, 'REGISTRY')
        assert not hasattr(wire_loader, 'registry')
        assert not hasattr(wire_loader, '_entries')

    def test_fresh_model_instances(self):
        """Should create fresh model instance for each call."""
        param_specs = (
            SerializedParameterSpec(
                name="beta", lower=0.1, upper=1.0, kind="float", doc="Transmission rate"
            ),
        )

        entry = EntryRecord(
            class_path="test_wire_loader:DummyModel",
            model_digest="sha256:test123",
            abi_version=WireABI.V1,
            param_specs=param_specs,
            scenarios=("baseline",),
            outputs=("infected",)
        )

        wire_fn = make_wire(entry)

        # Multiple calls should work independently
        response1 = wire_fn(params_M={"beta": 0.3}, seed=42)
        response2 = wire_fn(params_M={"beta": 0.5}, seed=43)

        assert response1.provenance["seed"] == 42
        assert response2.provenance["seed"] == 43
        assert response1.provenance["params_M"]["beta"] == 0.3
        assert response2.provenance["params_M"]["beta"] == 0.5