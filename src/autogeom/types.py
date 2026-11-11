"""Array types and Pydantic fields for NumPy arrays."""

from typing import Annotated

import numpy as np
import pint
from numpy import typing as npt
from pydantic import BeforeValidator, PlainSerializer
from pydantic.functional_validators import SkipValidation

FloatArray = npt.NDArray[np.float64]


def _float_array_validator(obj: object) -> FloatArray:
    return np.array(obj, dtype=np.float64)


def _float_array_serializer(arr: FloatArray) -> list:
    return arr.tolist()


def _coordinates_validator(obj: object) -> FloatArray:
    if isinstance(obj, pint.Quantity):
        try:
            obj = obj.m_as("angstrom")
        except pint.DimensionalityError as err:
            msg = f"Expected coordinates with length units but got {obj.units:~}."
            raise ValueError(msg) from err

    arr = _float_array_validator(obj)
    if arr.ndim != 2 or arr.shape[-1] != 3:  # noqa: PLR2004
        msg = f"Expected array of shape (N, 3) but got {arr.shape}."
        raise ValueError(msg)

    return arr


FloatArrayField = Annotated[
    SkipValidation[FloatArray],
    BeforeValidator(_float_array_validator),
    PlainSerializer(_float_array_serializer, return_type=list),
]
"""Float array field, serialized as nested lists."""

CoordinatesField = Annotated[
    SkipValidation[FloatArray],
    BeforeValidator(_coordinates_validator),
    PlainSerializer(_float_array_serializer, return_type=list),
]
"""``(N, 3)`` coordinates field in Angstroms; `pint` quantities are converted."""
