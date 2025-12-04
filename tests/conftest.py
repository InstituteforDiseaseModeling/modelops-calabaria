"""Ensure local modelops-contracts sources are importable for tests."""

import sys
from pathlib import Path

import polars as pl
import pytest

from modelops_calabaria.core.alignment import AlignedData
from modelops_calabaria.core.constants import REPLICATE_COL, SUFFIX_OBS, SUFFIX_SIM

ROOT = Path(__file__).resolve().parents[1]
contracts_src = ROOT.parent / "modelops-contracts" / "src"
if contracts_src.exists():
    sys.path.insert(0, str(contracts_src))


# ============================================================================
# Fixtures for Loss Track Tests
# ============================================================================


@pytest.fixture
def simple_aligned_data():
    """Single replicate aligned data for loss tests."""
    data = pl.DataFrame(
        {
            "time": [1, 2, 3, 4, 5],
            f"value{SUFFIX_OBS}": [1.0, 2.0, 3.0, 4.0, 5.0],
            f"value{SUFFIX_SIM}": [1.1, 2.1, 2.9, 4.2, 4.8],
        }
    )
    return AlignedData(data=data, on_cols=["time"], replicate_col=None)


@pytest.fixture
def multi_replicate_aligned_data():
    """Multiple replicate aligned data for loss tests."""
    data = pl.DataFrame(
        {
            "time": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            REPLICATE_COL: [0, 1, 2, 0, 1, 2, 0, 1, 2],
            f"value{SUFFIX_OBS}": [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0],
            f"value{SUFFIX_SIM}": [1.1, 0.9, 1.2, 2.1, 1.9, 2.2, 2.9, 3.1, 2.8],
        }
    )
    return AlignedData(data=data, on_cols=["time"], replicate_col=REPLICATE_COL)


@pytest.fixture
def observed_data():
    """Observed data for loss target tests."""
    return pl.DataFrame({"time": [1, 2, 3], "value": [1.0, 2.0, 3.0]})


@pytest.fixture
def observed_data_alt():
    """Alternative observed data for multi-target tests."""
    return pl.DataFrame({"time": [1, 2, 3], "value": [1.5, 2.5, 3.5]})


@pytest.fixture
def mock_sim_outputs():
    """Mock simulation outputs for loss target tests."""
    return [
        {"output1": pl.DataFrame({"time": [1, 2, 3], "value": [1.1, 2.1, 2.9]})},
        {"output1": pl.DataFrame({"time": [1, 2, 3], "value": [0.9, 1.9, 3.1]})},
        {"output1": pl.DataFrame({"time": [1, 2, 3], "value": [1.2, 2.2, 2.8]})},
    ]


# ============================================================================
# Fixtures for Likelihood Track Tests
# ============================================================================


@pytest.fixture
def beta_binomial_aligned_data():
    """Aligned beta-binomial data with multiple replicates."""
    data = pl.DataFrame(
        {
            "time": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            REPLICATE_COL: [0, 1, 2, 0, 1, 2, 0, 1, 2],
            f"x{SUFFIX_OBS}": [5, 5, 5, 8, 8, 8, 3, 3, 3],
            f"n{SUFFIX_OBS}": [10, 10, 10, 10, 10, 10, 10, 10, 10],
            f"x{SUFFIX_SIM}": [4, 5, 6, 7, 8, 9, 2, 3, 4],
            f"n{SUFFIX_SIM}": [10, 10, 10, 10, 10, 10, 10, 10, 10],
        }
    )
    return AlignedData(data=data, on_cols=["time"], replicate_col=REPLICATE_COL)


@pytest.fixture
def binomial_aligned_data():
    """Aligned binomial data with multiple replicates."""
    data = pl.DataFrame(
        {
            "time": [1, 1, 1, 2, 2, 2],
            REPLICATE_COL: [0, 1, 2, 0, 1, 2],
            f"x{SUFFIX_OBS}": [5, 5, 5, 8, 8, 8],
            f"n{SUFFIX_OBS}": [10, 10, 10, 10, 10, 10],
            f"p{SUFFIX_SIM}": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        }
    )
    return AlignedData(data=data, on_cols=["time"], replicate_col=REPLICATE_COL)


@pytest.fixture
def beta_binomial_observed_data():
    """Observed beta-binomial data for likelihood target tests."""
    return pl.DataFrame({"time": [1, 2, 3], "x": [5, 8, 3], "n": [10, 10, 10]})


@pytest.fixture
def beta_binomial_observed_data_alt():
    """Alternative observed beta-binomial data for multi-target tests."""
    return pl.DataFrame({"time": [1, 2, 3], "x": [6, 7, 4], "n": [10, 10, 10]})


@pytest.fixture
def mock_sim_outputs_likelihood():
    """Mock simulation outputs for likelihood target tests."""
    return [
        {"output1": pl.DataFrame({"time": [1, 2, 3], "x": [4, 7, 2], "n": [10, 10, 10]})},
        {"output1": pl.DataFrame({"time": [1, 2, 3], "x": [5, 8, 3], "n": [10, 10, 10]})},
        {"output1": pl.DataFrame({"time": [1, 2, 3], "x": [6, 9, 4], "n": [10, 10, 10]})},
    ]


@pytest.fixture
def mock_single_replicate_sim_outputs():
    """Mock simulation output with single replicate."""
    return [
        {"output1": pl.DataFrame({"time": [1, 2, 3], "x": [5, 8, 3], "n": [10, 10, 10]})},
    ]


@pytest.fixture
def mock_empty_sim_outputs():
    """Mock empty simulation outputs for error testing."""
    return []


@pytest.fixture
def mock_sim_outputs_with_extreme_values():
    """Mock simulation outputs that might produce -inf log-likelihoods."""
    return [
        {"output1": pl.DataFrame({"time": [1, 2, 3], "x": [0, 0, 0], "n": [10, 10, 10]})},
        {"output1": pl.DataFrame({"time": [1, 2, 3], "x": [10, 10, 10], "n": [10, 10, 10]})},
        {"output1": pl.DataFrame({"time": [1, 2, 3], "x": [5, 5, 5], "n": [10, 10, 10]})},
    ]
