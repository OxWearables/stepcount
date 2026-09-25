"""Shared type contracts for Stepcount's numerical and model boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, TypedDict, Union

import numpy as np
import numpy.typing as npt


NDArray = npt.NDArray[Any]
if TYPE_CHECKING:
    Numeric = Union[int, float, np.number[Any]]
else:
    Numeric = Union[int, float, np.number]
FeatureDict = Dict[str, Numeric]
PeakParams = Dict[str, float]


class HMMParams(TypedDict):
    """Parameters consumed by the local Viterbi implementation."""

    prior: NDArray
    emission: NDArray
    transition: NDArray
    labels: NDArray
