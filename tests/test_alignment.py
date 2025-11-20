"""Tests for alignment strategies."""

import polars as pl
import numpy as np

from modelops_calabaria.core.alignment import (
    AlignedData,
    AsofJoin,
    ExactJoin,
    JoinAlignment,
)
from modelops_calabaria.core.constants import (
    REPLICATE_COL,
    SUFFIX_OBS,
    SUFFIX_SIM,
)


def test_exact_join_basic_alignment():
    observed = pl.DataFrame({"timestep": [0, 1, 2], "infected": [10, 15, 20]})
    simulated = pl.DataFrame({"timestep": [0, 1, 2], "infected": [12, 14, 19]})

    aligner = ExactJoin(on_cols=["timestep"])
    aligned = aligner.align(observed, [simulated])

    assert isinstance(aligned, AlignedData)
    assert aligned.replicate_col == REPLICATE_COL
    assert aligned.data.shape == (3, 4)
    assert aligned.residuals("infected").to_list() == [-2, 1, 1]


def test_exact_join_column_suffixes():
    observed = pl.DataFrame({"t": [0, 1], "cases": [5, 10]})
    simulated = pl.DataFrame({"t": [0, 1], "cases": [4, 12]})

    aligner = ExactJoin(on_cols=["t"])
    aligned = aligner.align(observed, [simulated])

    assert f"cases{SUFFIX_SIM}" in aligned.data.columns
    assert f"cases{SUFFIX_OBS}" in aligned.data.columns
    assert aligned.residuals("cases").to_list() == [1, -2]


def test_asof_join_simple():
    observed = pl.DataFrame({"time": [1, 3, 5], "value": [10, 20, 30]}).sort("time")
    simulated = pl.DataFrame({"time": [2, 4, 6], "value": [9, 21, 31]}).sort("time")

    aligner = AsofJoin(on_column="time", strategy="backward")
    aligned = aligner.align(observed, [simulated])

    assert isinstance(aligned, AlignedData)
    assert aligned.data.shape[0] == 3
    assert aligned.residuals("value").to_list() == [11, 9]


def test_asof_join_with_grouping():
    observed = pl.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "time": [1, 3, 1, 3],
            "value": [10, 20, 100, 200],
        }
    ).sort(["region", "time"])
    simulated = pl.DataFrame(
        {
            "region": ["A", "A", "B", "B"],
            "time": [2, 4, 2, 4],
            "value": [9, 21, 99, 199],
        }
    ).sort(["region", "time"])

    aligner = AsofJoin(on_column="time", by=["region"], strategy="backward")
    aligned = aligner.align(observed, [simulated])

    assert aligned.data.shape[0] == 4
    assert aligned.residuals("value").to_list() == [11, 101]


def test_asof_join_sorts_unsorted_inputs():
    observed = pl.DataFrame(
        {
            "time": [3, 1, 5, 2],
            "value": [20, 10, 30, 15],
        }
    )
    simulated = pl.DataFrame(
        {
            "time": [2, 4, 6, 1],
            "value": [19, 29, 31, 9],
        }
    )

    aligner = AsofJoin(on_column="time", strategy="backward")
    unsorted_aligned = aligner.align(observed, [simulated])

    sorted_expected = aligner.align(observed.sort("time"), [simulated.sort("time")])
    assert unsorted_aligned.data.to_dicts() == sorted_expected.data.to_dicts()


def test_join_alignment_dispatch_exact():
    observed = pl.DataFrame({"t": [0, 1, 2], "y": [1.0, 2.0, 3.0]})
    simulated = pl.DataFrame({"t": [0, 1, 2], "y": [0.9, 2.1, 3.2]})

    aligner = JoinAlignment(on_cols="t", mode="exact")
    aligned = aligner.align(observed, [simulated])

    residuals = aligned.residuals("y").to_list()
    expected = [0.1, -0.1, -0.2]
    assert np.allclose(residuals, expected, atol=1e-8)


def test_join_alignment_dispatch_asof():
    observed = pl.DataFrame({"t": [1, 3, 5], "y": [10, 20, 30]}).sort("t")
    simulated = pl.DataFrame({"t": [2, 4, 6], "y": [9, 21, 31]}).sort("t")

    aligner = JoinAlignment(on_cols="t", mode="backward")
    aligned = aligner.align(observed, [simulated])

    assert aligned.residuals("y").to_list() == [11, 9]
