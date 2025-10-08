"""
Types and Type Aliases for ModelOps-Calabaria.
"""

from typing import Mapping, Union

import polars as pl

# Type alias for simulation outputs
SimOutput = Mapping[str, pl.DataFrame]  # canonical
RawSimOutput = Union[pl.DataFrame, SimOutput]  # author-facing