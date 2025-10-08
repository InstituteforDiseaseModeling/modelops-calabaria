"""Tests for targets wire protocol."""

import io
import tempfile
from pathlib import Path

import polars as pl
import pytest

from modelops_calabaria.core.target import Target, Targets
from modelops_calabaria.core.alignment import JoinAlignment
from modelops_calabaria.targets import wire_target_function


class SimpleEvaluation:
    """Simple evaluation strategy for testing."""

    def evaluate(self, aligned):
        from modelops_calabaria.core.evaluation.base import TargetEvaluation
        # Simple MSE calculation
        sim_col = f"value{aligned.data.columns[1][-5:]}"  # Get suffix
        obs_col = f"value{aligned.data.columns[2][-5:]}"  # Get suffix

        mse = ((aligned.data[sim_col] - aligned.data[obs_col]) ** 2).mean()

        return TargetEvaluation(
            name="test_target",
            loss=float(mse),
            weight=1.0,
            weighted_loss=float(mse)
        )

    def parameters(self):
        return None


def test_target_basic():
    """Test basic Target functionality."""
    observed = pl.DataFrame({"time": [1, 2, 3], "value": [10, 20, 30]})

    target = Target(
        model_output="output1",
        data=observed,
        alignment=JoinAlignment(on_cols="time", mode="exact"),
        evaluation=SimpleEvaluation(),
        weight=1.0
    )

    # Create simulated data
    sim_output = {
        "output1": pl.DataFrame({"time": [1, 2, 3], "value": [11, 19, 31]})
    }

    result = target.evaluate([sim_output])

    assert result.name == "test_target"
    assert result.loss is not None
    assert result.loss > 0  # Should have some loss
    assert result.weight == 1.0


def test_targets_evaluate_all():
    """Test Targets.evaluate_all functionality."""
    observed1 = pl.DataFrame({"time": [1, 2], "value": [10, 20]})
    observed2 = pl.DataFrame({"time": [1, 2], "count": [100, 200]})

    target1 = Target(
        model_output="output1",
        data=observed1,
        alignment=JoinAlignment(on_cols="time", mode="exact"),
        evaluation=SimpleEvaluation(),
        weight=1.0
    )

    # Note: Using different column name for second target
    class SimpleEvaluation2:
        def evaluate(self, aligned):
            from modelops_calabaria.core.evaluation.base import TargetEvaluation
            sim_col = f"count{aligned.data.columns[1][-5:]}"
            obs_col = f"count{aligned.data.columns[2][-5:]}"
            mse = ((aligned.data[sim_col] - aligned.data[obs_col]) ** 2).mean()
            return TargetEvaluation(
                name="target2",
                loss=float(mse),
                weight=0.5,
                weighted_loss=float(mse) * 0.5
            )
        def parameters(self):
            return None

    target2 = Target(
        model_output="output2",
        data=observed2,
        alignment=JoinAlignment(on_cols="time", mode="exact"),
        evaluation=SimpleEvaluation2(),
        weight=0.5
    )

    targets = Targets(targets=[target1, target2])

    # Create simulated outputs
    sim_outputs = {
        "output1": pl.DataFrame({"time": [1, 2], "value": [11, 19]}),
        "output2": pl.DataFrame({"time": [1, 2], "count": [99, 201]})
    }

    results = targets.evaluate_all([sim_outputs])

    assert len(results) == 2
    assert results[0].name == "test_target"
    assert results[1].name == "target2"


def test_wire_target_function():
    """Test the wire target function with a mock entrypoint."""

    # Create a temporary module with targets
    with tempfile.TemporaryDirectory() as tmpdir:
        target_file = Path(tmpdir) / "test_wire_targets.py"
        target_file.write_text("""
import polars as pl
from modelops_calabaria.core.target import Target, Targets
from modelops_calabaria.core.alignment import JoinAlignment

class TestEval:
    def evaluate(self, aligned):
        from modelops_calabaria.core.evaluation.base import TargetEvaluation
        return TargetEvaluation(
            name="wire_test",
            loss=1.5,
            weight=1.0,
            weighted_loss=1.5
        )
    def parameters(self):
        return None

def get_targets():
    observed = pl.DataFrame({"t": [1, 2], "y": [10, 20]})
    target = Target(
        model_output="sim_output",
        data=observed,
        alignment=JoinAlignment(on_cols="t", mode="exact"),
        evaluation=TestEval(),
        weight=1.0
    )
    return Targets(targets=[target])
""")

        # Add tmpdir to Python path
        import sys
        sys.path.insert(0, tmpdir)

        try:
            # Create simulated outputs as bytes (Arrow IPC format)
            sim_df = pl.DataFrame({"t": [1, 2], "y": [11, 19]})
            buffer = io.BytesIO()
            sim_df.write_ipc(buffer)
            sim_bytes = buffer.getvalue()

            sim_outputs = {
                "sim_output": sim_bytes,
                "metadata": b'{"test": true}'
            }

            # Call wire function
            result = wire_target_function("test_wire_targets:get_targets", sim_outputs)

            assert "total_loss" in result
            assert "target_losses" in result
            assert result["total_loss"] == 1.5
            assert result["target_losses"]["wire_test"] == 1.5

        finally:
            # Clean up sys.path
            sys.path.remove(tmpdir)


def test_wire_target_function_invalid_entrypoint():
    """Test wire function with invalid entrypoint."""
    sim_outputs = {"test": b"data"}

    # Test missing colon
    with pytest.raises(ValueError, match="Invalid entrypoint format"):
        wire_target_function("no_colon", sim_outputs)