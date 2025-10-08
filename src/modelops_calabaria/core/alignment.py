"""
Alignment strategies for joining observed and simulated data.

This module defines alignment strategies for joining observed and simulated
data prior to loss evaluation.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Protocol

import polars as pl

from .constants import (
    LOSS_COL,
    REPLICATE_COL,
    SUFFIX_OBS,
    SUFFIX_SIM,
)

# Supported alignment modes
JoinMode = Literal["exact", "nearest", "backward", "forward"]


@dataclass(frozen=True)
class AlignedData:
    """
    Represents aligned observed and simulated data, post-join.
    Includes suffix conventions and optional replicate indexing.
    """

    data: pl.DataFrame
    on_cols: list[str]
    replicate_col: Optional[str] = None

    def sim_colname(self, col: str) -> str:
        return f"{col}{SUFFIX_SIM}"

    def obs_colname(self, col: str) -> str:
        return f"{col}{SUFFIX_OBS}"

    def get_sim_col(self, col: str) -> pl.Series:
        return self.data[self.sim_colname(col)]

    def get_sim_cols(self, cols: list[str]) -> pl.DataFrame:
        return self.data[[self.sim_colname(col) for col in cols]]

    def get_obs_col(self, col: str) -> pl.Series:
        return self.data[self.obs_colname(col)]

    def get_obs_cols(self, cols: list[str]) -> pl.DataFrame:
        return self.data[[self.obs_colname(col) for col in cols]]

    def get_loss_col(self) -> pl.Series:
        return self.data[LOSS_COL]

    def has_loss(self) -> bool:
        return LOSS_COL in self.data.columns

    def residuals(self, col: str) -> pl.Series:
        series = (self.get_obs_col(col) - self.get_sim_col(col)).alias("residual")
        cleaned = series.drop_nulls()
        return cleaned

    def residuals_raw(self, col: str) -> pl.Series:
        series = (self.get_obs_col(col) - self.get_sim_col(col)).alias("residual")
        return series

    def __repr__(self) -> str:
        return f"<AlignedData shape={self.data.shape} on={self.on_cols}>"


class AlignmentStrategy(Protocol):
    """
    Strategy interface for aligning observed and simulated data.
    The result should be a merged table suitable for residual computation.
    """

    def align(
        self,
        observed: pl.DataFrame,
        simulated: list[pl.DataFrame],
    ) -> AlignedData: ...

    @property
    def on(self) -> list[str]: ...


@dataclass
class ExactJoin:
    """
    Exact join strategy for observed and simulated data; all simulated replicates
    will be concatenated into a single dataframe with a replicate column.
    """

    on_cols: list[str]

    def align(
        self,
        observed: pl.DataFrame,
        simulated: list[pl.DataFrame],
    ) -> AlignedData:
        aligned_frames = []

        obs_renamed = observed.rename(
            {c: f"{c}{SUFFIX_OBS}" for c in observed.columns if c not in self.on_cols}
        )

        for i, s in enumerate(simulated):
            # Rename columns for simulated and observed
            sim_renamed = s.rename(
                {c: f"{c}{SUFFIX_SIM}" for c in s.columns if c not in self.on_cols}
            )

            joined = sim_renamed.join(obs_renamed, how="inner", on=self.on_cols)
            joined = joined.with_columns(pl.lit(i).alias(REPLICATE_COL))
            aligned_frames.append(joined)

        return AlignedData(
            data=pl.concat(aligned_frames),
            on_cols=self.on_cols,
            replicate_col=REPLICATE_COL,
        )

    @property
    def on(self) -> list[str]:
        return self.on_cols


@dataclass
class AsofJoin:
    """
    Approximate join on a single time-like column with optional grouping.
    Only supports one 'on' column as per Polars ASOF limitations.
    """

    on_column: str
    by: Optional[list[str]] = None
    strategy: Literal["nearest", "backward", "forward"] = "nearest"

    @property
    def on(self) -> list[str]:
        return ([self.on_column] + self.by) if self.by else [self.on_column]

    def align(self, observed, simulated) -> AlignedData:
        if isinstance(simulated, list):
            aligned = [
                self.align(observed, s).data.with_columns(
                    pl.lit(i).alias(REPLICATE_COL)
                )
                for i, s in enumerate(simulated)
            ]
            return AlignedData(
                data=pl.concat(aligned), on_cols=self.on, replicate_col=REPLICATE_COL
            )

        keep_sim = {self.on_column} | (set(self.by) if self.by else set())
        keep_obs = {self.on_column} | (set(self.by) if self.by else set())

        sim_renamed = simulated.rename(
            {c: f"{c}{SUFFIX_SIM}" for c in simulated.columns if c not in keep_sim}
        )
        obs_renamed = observed.rename(
            {c: f"{c}{SUFFIX_OBS}" for c in observed.columns if c not in keep_obs}
        )

        joined = obs_renamed.join_asof(
            sim_renamed,
            on=self.on_column,
            by=self.by,
            strategy=self.strategy,
            coalesce=False,
            check_sortedness=False,
        )
        return AlignedData(data=joined, on_cols=self.on)


@dataclass
class JoinAlignment(AlignmentStrategy):
    """
    Unified entry point for choosing between exact and approximate join strategies.
    """

    on_cols: str | list[str]
    mode: JoinMode = "exact"

    def to_strategy(self) -> AlignmentStrategy:
        return make_alignment(
            self.mode,
            on=[self.on_cols] if isinstance(self.on_cols, str) else self.on_cols,
        )

    def align(self, observed, simulated) -> AlignedData:
        return self.to_strategy().align(observed, simulated)

    @property
    def on(self) -> list[str]:
        return [self.on_cols] if isinstance(self.on_cols, str) else self.on_cols


def make_alignment(
    mode: JoinMode, on: list[str], by: Optional[list[str]] = None
) -> AlignmentStrategy:
    if mode == "exact":
        return ExactJoin(on_cols=on)
    elif mode in {"nearest", "backward", "forward"}:
        if len(on) != 1:
            raise ValueError(f"Asof joins require exactly one 'on' column, got: {on}")
        return AsofJoin(on_column=on[0], by=by, strategy=mode)
    else:
        raise ValueError(f"Unknown alignment mode: {mode}")